"""V1 of motion_frame_pipeline: same detection, flow and plots, plus the COR2+HI1 image processing.

Differences from motion_frame_pipeline.make_frames (which is left unchanged):
- noise_model ('ring' default, 'global', 'none'): difference images are put in
  noise (sigma) units by single_noise_model.apply_noise_model, so detect_floor
  and bright_contrast are in sigma; 'none' keeps input image units.
- display_stretch / display_soft / display_limit / display_smooth_px: display
  only (detection unaffected), as in the COR2+HI1 frames.
- diff_panel: 'both' (default, as before), 'base' or 'running'.
- AR rectangle: ar_half_lon/ar_half_lat seeded every AR_seed_spacing_deg; AR loops
  need one footpoint inside it when trace_longARrooted_loops=1 (large AR-rooted
  loops kept), both when 0; no arcsec box.
- show_vectors: False hides velocity arrows and the speed colour bar.
- reuse_cache: 'latest' or a cache folder redraws only (no downloads, alignment
  or flow); the cache's noise model is switched to noise_model first.
V1 caches are named V1_<craft|LASCO>_<detector>_* so the original launchers'
caches are never modified.
"""
from pathlib import Path
import tempfile
import sunpy.map
from datamine_calib_helper import load_map  # truthful WCS for every source
from astropy.time import Time
from prepare_sources import fill_coronagraph,fill_disk,prepare_pfss,ensure_hv_coronagraph,ensure_hv_disk
from motion_inputs import make_motion_cache
from single_noise_model import apply_noise_model
from goes_cache import resolve_goes_pickle
from track_coronagraph_flows import render_frames

def make_frames_V1(mission,*, noise_model='ring', ring_step_rsun=1., ring_strength=.7, reuse_cache=None,
	diff_panel='both', display_stretch='asinh', display_soft=1., display_limit=40., display_smooth_px=1., show_vectors=True, plot_style=None, pfss_linewidth=1.5, base_region_span_rsun=None,
	sT, eT, dT, base_diff_DT,
	max_image_dt, reference_lookback, max_gap_seconds, input_folder,
	disk_max_minutes, download, helioviewer, pfss_needed,
	pfss_dir, gong_cache, PFSS_dT, rss,
	nrho, r_seed, n_lon, n_lat,
	focus_AR, NOAA_number, AR_time, SRS_file,
	AR_seed_spacing_deg=1., trace_longARrooted_loops=1, n_AR_open=None, AR_r_seed=1.01,
	AR_lookup_max_age, AR_closedB_col, AR_openB_col, closed_field_color,
	n_open, n_global_closed, n_closed, ar_half_lon,
	ar_half_lat, npix, r_inner,
	r_outer, spoke_pa, spoke_halfwidth, spoke_arrow_buffer_deg,
	mask_buffer_px, detect_sigma, detect_floor, detect_min_area,
	region_span_rsun, weak_bright_factor, flow_support_radius_rsun, bright_contrast,
	bright_min_area, arrow_min_area, edge_spacing, support_radius,
	min_support, min_displacement, user_datacsv, goes_folder,
	goes_download, goes_pickle, goes_csv, bright_border,
	dark_border, new_border, border_width, arrow_length_pixels, centroid_arrow_length_pixels,
	speed_bins, dpi, outdir, final_image_folder,
	LASCO=None, LASCO_imgfolder=None, need_LASCO_calib=None, SDO_fold=None,
	SDO_instrument=None, SDO_product=None, SDO_dT=None, spacecraft=None,
	instrument=None, STEREO_imgfolder=None, polar=None, EUVI_fold=None,
	EUVI_wave=None, EUVI_dT=None, detect_smooth_px=1.5, bright_contrast_snr=False,
	region_min_width_rsun=0., region_max_aspect=None, AR_loop_reach_deg=None):
	"""Prepare and render one event using settings from its launcher (V1 options in the module docstring)."""
	root=Path(input_folder).expanduser().resolve()
	def source(path):
		return root/Path(path).expanduser() if path is not None else None
	if Time(eT)<=Time(sT) or Time(base_diff_DT)>Time(sT) or dT<=0:
		raise ValueError('Require eT > sT, base_diff_DT <= sT, and positive dT.')
	mode=mission.upper()
	if mode not in ('LASCO','STEREO'):raise ValueError('mission must be LASCO or STEREO.')
	detector=(LASCO if mode=='LASCO' else instrument).upper()
	limits={'C2':(2.2,6.),'C3':(4.,30.),'COR1':(1.5,4.),'COR2':(2.5,15.)}
	if detector not in (('C2','C3') if mode=='LASCO' else ('COR1','COR2')):
		raise ValueError('Detector does not match mission.')
	craft=None if mode=='LASCO' else spacecraft.upper()
	if craft not in (None,'A','B'):raise ValueError('spacecraft must be A or B.')
	lo=limits[detector][0] if r_inner is None else r_inner
	hi=limits[detector][1] if r_outer is None else r_outer
	if not 0<=lo<hi:raise ValueError('Require 0 <= r_inner < r_outer.')
	folder=source(LASCO_imgfolder if mode=='LASCO' else STEREO_imgfolder)
	model=source(pfss_dir)
	if pfss_needed and model is None:raise ValueError('Set pfss_dir.')
	for p in (source(user_datacsv),):
		if p is None or not p.is_file():raise FileNotFoundError(f'Missing light-curve input: {p}')
	goes_p,goes_c=source(goes_pickle),source(goes_csv)
	if goes_p is None and goes_c is None:
		goes_p=resolve_goes_pickle(source(goes_folder),sT,eT,goes_download)
	if not (goes_p or goes_c).is_file():raise FileNotFoundError(goes_p or goes_c)
	print(f'GOES input: {goes_p or goes_c}',flush=True)
	panels={'both':('base_difference','running_difference'),'base':('base_difference',),'running':('running_difference',)}.get(str(diff_panel).lower())
	if panels is None:raise ValueError("diff_panel must be 'both', 'base' or 'running'.")
	dest=source(outdir);dest.mkdir(parents=True,exist_ok=True)
	cache_root=dest/'motion_cache';cache_root.mkdir(exist_ok=True)
	prefix=f'V1_{craft or "LASCO"}_{detector}_'
	if reuse_cache:
		if reuse_cache=='latest':
			found=sorted(cache_root.glob(prefix+'*'),key=lambda p:p.stat().st_mtime)
			if not found:raise FileNotFoundError(f'No {prefix}* cache in {cache_root}.')
			motion_cache=found[-1]
		else:
			motion_cache=Path(reuse_cache).expanduser()
			if not motion_cache.is_absolute():motion_cache=root/motion_cache
		if not (motion_cache/'times.json').is_file():raise FileNotFoundError(f'Not a motion cache: {motion_cache}')
		print(f'Redraw only, reusing {motion_cache}',flush=True)
		download=False
	if download:
		fill_coronagraph(folder,detector,craft,sT,eT,dT,base_diff_DT,reference_lookback,
			max_image_dt,need_LASCO_calib,polar,helioviewer)
	# Science FITS if they cover every cadence slot; otherwise Helioviewer maps_HV.
	# Route Level-1 -> Helioviewer -> Level-0.5 (STEREO always; LASCO when
	# need_LASCO_calib is True). need_LASCO_calib=False keeps Level-0.5 first.
	prefer_hv=bool(craft) or need_LASCO_calib is not False
	if not reuse_cache:
		ensure_hv_coronagraph(folder,detector,craft,sT,eT,dT,base_diff_DT,reference_lookback,
			max_image_dt,download,helioviewer,prefer_hv)
		motion_cache=Path(tempfile.mkdtemp(prefix=prefix,dir=cache_root))
		make_motion_cache(folder,detector,craft,sT,eT,dT,base_diff_DT,reference_lookback,
			max_image_dt,max_gap_seconds,npix,lo,hi,motion_cache,prefer_helioviewer=prefer_hv)
	apply_noise_model(motion_cache,noise_model,ring_step_rsun,ring_strength)
	if download:
		disk_folder=source(EUVI_fold if craft else SDO_fold)
		if disk_folder is not None:
			fill_disk(disk_folder,'EUVI' if craft else SDO_instrument,
				EUVI_wave if craft else SDO_product,load_map(motion_cache/'reference_grid.fits'),
				craft,sT,eT,EUVI_dT if craft else SDO_dT,disk_max_minutes,helioviewer)
	disk_folder=source(EUVI_fold if craft else SDO_fold)
	if disk_folder is not None:
		ensure_hv_disk(disk_folder,'EUVI' if craft else SDO_instrument,
			EUVI_wave if craft else SDO_product,load_map(motion_cache/'reference_grid.fits'),
			craft,sT,eT,EUVI_dT if craft else SDO_dT,disk_max_minutes)
	pfss_models=None
	if pfss_needed:
		if PFSS_dT is not None:
			pfss_models=prepare_pfss(model,sT,eT,PFSS_dT,download,source(gong_cache),
				dict(rss=rss,nrho=nrho,r_seed=r_seed,n_lon=n_lon,n_lat=n_lat))
		if not pfss_models and not any((p.parent/'metadata.json').is_file() for p in model.rglob('field_lines.npz')):
			raise FileNotFoundError(f'No saved PFSS model in {model}.')
	render_frames(motion_cache,
		panel_keys=panels,
		display_stretch=display_stretch,
		display_soft=display_soft,
		display_limit=display_limit,
		display_smooth_px=display_smooth_px,
		show_vectors=show_vectors,
		sT=sT,
		eT=eT,
		LASCO=detector,
		SDO_fold=source(SDO_fold),
		EUVI_fold=source(EUVI_fold),
		pfss_dir=model,
		outdir=dest,
		final_image_folder=source(final_image_folder) if final_image_folder else dest/'frames',
		SDO_instrument=SDO_instrument,
		SDO_product=SDO_product,
		EUVI_wave=EUVI_wave,
		focus_AR=focus_AR,
		ar_seed_spacing_deg=AR_seed_spacing_deg,
		ar_long_loops=bool(trace_longARrooted_loops),
		n_AR_open=n_AR_open,
		AR_r_seed=AR_r_seed,
		NOAA_number=NOAA_number,
		AR_time=AR_time,
		SRS_file=source(SRS_file),
		AR_lookup_max_age=AR_lookup_max_age,
		AR_closedB_col=AR_closedB_col,
		AR_openB_col=AR_openB_col,
		closed_field_color=closed_field_color,
		pfss_linewidth=pfss_linewidth, plot_style=plot_style,
		user_datacsv=source(user_datacsv),
		goes_pickle=goes_p,
		goes_csv_input=goes_c,
		edge_spacing=edge_spacing,
		support_radius=support_radius,
		min_support=min_support,
		min_displacement=min_displacement,
		arrow_length_pixels=arrow_length_pixels,
		centroid_arrow_length_pixels=centroid_arrow_length_pixels,
		speed_bins=speed_bins,
		pfss_needed=pfss_needed,
		pfss_models=pfss_models,
		disk_max_minutes=disk_max_minutes,
		bright_border=bright_border,
		dark_border=dark_border,
		new_border=new_border,
		border_width=border_width,
		n_open=n_open,
		n_global_closed=n_global_closed,
		n_closed=n_closed,
		ar_half_lon=ar_half_lon,
		ar_half_lat=ar_half_lat,
		ar_half_box_arcsec=None,
		spoke_pa=spoke_pa,
		spoke_halfwidth=spoke_halfwidth,
		spoke_arrow_buffer_deg=spoke_arrow_buffer_deg if spoke_halfwidth>0 else 0.,
		mask_buffer_px=mask_buffer_px,
		detect_sigma=detect_sigma,
		detect_min_area=detect_min_area,
		region_span_rsun=region_span_rsun,
		base_region_span_rsun=base_region_span_rsun,
		weak_bright_factor=weak_bright_factor,
		flow_support_radius_rsun=flow_support_radius_rsun,
		bright_contrast=bright_contrast,
		bright_contrast_snr=bright_contrast_snr,
		detect_smooth_px=detect_smooth_px,
		region_min_width_rsun=region_min_width_rsun,
		region_max_aspect=region_max_aspect,
		ar_loop_reach_deg=AR_loop_reach_deg,
		bright_min_area=bright_min_area,
		arrow_min_area=arrow_min_area,
		detect_floor=detect_floor,
		max_gap_seconds=max_gap_seconds,
		dpi=dpi)
	return motion_cache
