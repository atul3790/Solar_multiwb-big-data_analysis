"""Run motion detection and PFSS overlays on a new local FITS sequence."""
from pathlib import Path
import tempfile
import sunpy.map
from datamine_calib_helper import load_map  # truthful WCS for every source
from astropy.time import Time
from prepare_sources import fill_coronagraph,fill_disk,prepare_pfss,ensure_hv_coronagraph,ensure_hv_disk
from motion_inputs import make_motion_cache
from goes_cache import resolve_goes_pickle
from track_coronagraph_flows import render_frames

def make_frames(mission,*, plot_style=None, pfss_linewidth=1.5, base_region_span_rsun=None,
	sT, eT, dT, base_diff_DT,
	max_image_dt, reference_lookback, max_gap_seconds, input_folder,
	disk_max_minutes, download, helioviewer, pfss_needed,
	pfss_dir, gong_cache, PFSS_dT, rss,
	nrho, r_seed, n_lon, n_lat,
	focus_AR, NOAA_number, AR_time, SRS_file,
	AR_halfwidth_arcsec=400., AR_seed_spacing_arcsec=40., AR_r_seed=1.01,
	AR_lookup_max_age, AR_closedB_col, AR_openB_col, closed_field_color,
	n_open, n_global_closed, n_closed, ar_half_lon,
	ar_half_lat, ar_half_box_arcsec, npix, r_inner,
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
	EUVI_wave=None, EUVI_dT=None):
	"""Prepare and render one event using settings from its launcher."""
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
	if download:
		fill_coronagraph(folder,detector,craft,sT,eT,dT,base_diff_DT,reference_lookback,
			max_image_dt,need_LASCO_calib,polar,helioviewer)
	# Science FITS if they cover every cadence slot; otherwise Helioviewer maps_HV.
	# Route Level-1 -> Helioviewer -> Level-0.5 (STEREO always; LASCO when
	# need_LASCO_calib is True). need_LASCO_calib=False keeps Level-0.5 first.
	prefer_hv=bool(craft) or need_LASCO_calib is not False
	ensure_hv_coronagraph(folder,detector,craft,sT,eT,dT,base_diff_DT,reference_lookback,
		max_image_dt,download,helioviewer,prefer_hv)
	dest=source(outdir);dest.mkdir(parents=True,exist_ok=True)
	cache_root=dest/'motion_cache'
	cache_root.mkdir(exist_ok=True)
	motion_cache=Path(tempfile.mkdtemp(prefix=f'{craft or "LASCO"}_{detector}_',dir=cache_root))
	make_motion_cache(folder,detector,craft,sT,eT,dT,base_diff_DT,reference_lookback,
		max_image_dt,max_gap_seconds,npix,lo,hi,motion_cache,prefer_helioviewer=prefer_hv)
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
		AR_halfwidth_arcsec=AR_halfwidth_arcsec,
		AR_seed_spacing_arcsec=AR_seed_spacing_arcsec,
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
		ar_half_box_arcsec=ar_half_box_arcsec,
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
		bright_min_area=bright_min_area,
		arrow_min_area=arrow_min_area,
		detect_floor=detect_floor,
		max_gap_seconds=max_gap_seconds,
		dpi=dpi)
