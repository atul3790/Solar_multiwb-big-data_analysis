"""STEREO COR2 + HI1 moving-structure frames: download, combined cache, render.

Same steps as motion_frame_pipeline.make_frames for STEREO, with two changes:
HI1 sums are fetched through Multiwb_datamine.get_STEREO_imgs (route
Level-1 -> Helioviewer -> Level-0.5), and the motion cache is the combined
COR2+HI1 grid from combined_motion_inputs. Only one difference panel
(base or running) is drawn above the LAT/GOES light curves.
"""
from pathlib import Path
import tempfile
from astropy.time import Time
import astropy.units as u
from datamine_calib_helper import load_map
from prepare_sources import fill_coronagraph,fill_disk,prepare_pfss,ensure_hv_coronagraph,ensure_hv_disk
from combined_motion_inputs import make_combined_cache,apply_noise_model,hi_clean_flow
from goes_cache import resolve_goes_pickle
from track_coronagraph_flows import render_frames

def make_combined_frames(*,diff_panel,STEREO_root,spacecraft,sT,eT,dT,base_diff_DT,max_image_dt,
	reference_lookback,max_gap_seconds,input_folder,download,helioviewer,
	hi_max_image_dt,hi_lookback,hi_max_gap,hi_star_filter_px,hi_edge_px,seam_rsun,
	npix,r_inner,r_outer,spoke_pa,spoke_halfwidth,spoke_arrow_buffer_deg,
	EUVI_fold,EUVI_wave,EUVI_dT,disk_max_minutes,
	pfss_needed,pfss_dir,gong_cache,PFSS_dT,rss,nrho,r_seed,n_lon,n_lat,
	user_datacsv,goes_folder,goes_download,goes_pickle,goes_csv,outdir,final_image_folder,
	noise_model='ring',ring_step_rsun=3.,ring_strength=.7,hi_flow_clean=True,reuse_cache=None,
	**render):
	"""Prepare and render one event; render holds the remaining render_frames settings.

	reuse_cache: None builds a new motion cache (downloads, alignment, flow).
	'latest' or a cache folder path skips all of that and only redraws: the
	cache's noise model is switched to noise_model (reversible), then frames are
	rendered for sT-eT from it.
	"""
	root=Path(input_folder).expanduser().resolve()
	def source(path):
		return root/Path(path).expanduser() if path is not None else None
	if Time(eT)<=Time(sT) or Time(base_diff_DT)>Time(sT) or dT<=0:
		raise ValueError('Require eT > sT, base_diff_DT <= sT, and positive dT.')
	panel={'base':'base_difference','running':'running_difference'}.get(str(diff_panel).lower())
	if panel is None:raise ValueError("diff_panel must be 'base' or 'running'.")
	craft=spacecraft.upper()
	if craft not in ('A','B'):raise ValueError('spacecraft must be A or B.')
	lo=2.5 if r_inner is None else r_inner
	hi=15. if r_outer is None else r_outer
	if not 0<=lo<hi:raise ValueError('Require 0 <= r_inner < r_outer.')
	folder=source(STEREO_root);folder.mkdir(parents=True,exist_ok=True)
	model=source(pfss_dir)
	if pfss_needed and model is None:raise ValueError('Set pfss_dir.')
	if source(user_datacsv) is None or not source(user_datacsv).is_file():
		raise FileNotFoundError(f'Missing light-curve input: {source(user_datacsv)}')
	goes_p,goes_c=source(goes_pickle),source(goes_csv)
	if goes_p is None and goes_c is None:
		goes_p=resolve_goes_pickle(source(goes_folder),sT,eT,goes_download)
	if not (goes_p or goes_c).is_file():raise FileNotFoundError(goes_p or goes_c)
	print(f'GOES input: {goes_p or goes_c}',flush=True)
	dest=source(outdir);dest.mkdir(parents=True,exist_ok=True)
	cache_root=dest/'motion_cache';cache_root.mkdir(exist_ok=True)
	if reuse_cache:
		if reuse_cache=='latest':
			found=sorted(cache_root.glob(f'{craft}_COR2_HI1_*'),key=lambda p:p.stat().st_mtime)
			if not found:raise FileNotFoundError(f'No {craft}_COR2_HI1_* cache in {cache_root}.')
			motion_cache=found[-1]
		else:
			motion_cache=Path(reuse_cache).expanduser()
			if not motion_cache.is_absolute():motion_cache=root/motion_cache
		if not (motion_cache/'times.json').is_file():raise FileNotFoundError(f'Not a motion cache: {motion_cache}')
		print(f'Redraw only, reusing {motion_cache}',flush=True)
		download=False
	#### COR2: same route and coverage check as the single-instrument flow code ####
	if reuse_cache:pass
	elif download:
		fill_coronagraph(folder,'COR2',craft,sT,eT,dT,base_diff_DT,reference_lookback,
			max_image_dt,None,0,helioviewer)
	if not reuse_cache:
		ensure_hv_coronagraph(folder,'COR2',craft,sT,eT,dT,base_diff_DT,reference_lookback,
			max_image_dt,download,helioviewer,True)
	#### HI1: one window covering base, predecessor and event ####
	if download:
		import Multiwb_datamine as dm
		start,end,base=map(Time,(sT,eT,base_diff_DT))
		lo_t=min(base-hi_max_image_dt*u.s,start-hi_lookback*u.s)
		try:
			dm.get_STEREO_imgs(lo_t.isot,(end+hi_max_image_dt*u.s).isot,dT,spacecraft=craft,
				instruments=['HI1'],waves=[None],outdir=str(folder),nproc=1,helioviewer=False)
		except Exception as exc:
			print(f'HI1: download unsuccessful; continuing with local files: {exc}',flush=True)
	if reuse_cache:
		apply_noise_model(motion_cache,noise_model,ring_step_rsun,ring_strength)
		if hi_flow_clean:hi_clean_flow(motion_cache,hi_max_gap=hi_max_gap)  # upgrades older caches once
	else:
		motion_cache=Path(tempfile.mkdtemp(prefix=f'{craft}_COR2_HI1_',dir=cache_root))
		make_combined_cache(folder,folder,craft,sT,eT,dT,base_diff_DT,reference_lookback,max_image_dt,
			max_gap_seconds,npix,lo,hi,motion_cache,prefer_helioviewer=True,
			hi_max_image_dt=hi_max_image_dt,hi_lookback=hi_lookback,hi_max_gap=hi_max_gap,
			hi_star_filter_px=hi_star_filter_px,hi_edge_px=hi_edge_px,seam_rsun=seam_rsun,
			spoke_pa=spoke_pa,spoke_halfwidth=spoke_halfwidth,spoke_arrow_buffer_deg=spoke_arrow_buffer_deg,
			noise_model=noise_model,ring_step_rsun=ring_step_rsun,ring_strength=ring_strength,hi_flow_clean=hi_flow_clean)
	#### EUVI disk and PFSS, as before ####
	disk_folder=source(EUVI_fold)
	if disk_folder is not None:
		grid=load_map(motion_cache/'reference_grid.fits')
		if download:
			fill_disk(disk_folder,'EUVI',EUVI_wave,grid,craft,sT,eT,EUVI_dT,disk_max_minutes,helioviewer)
		ensure_hv_disk(disk_folder,'EUVI',EUVI_wave,grid,craft,sT,eT,EUVI_dT,disk_max_minutes)
	pfss_models=None
	if pfss_needed:
		if PFSS_dT is not None:
			pfss_models=prepare_pfss(model,sT,eT,PFSS_dT,download,source(gong_cache),
				dict(rss=rss,nrho=nrho,r_seed=r_seed,n_lon=n_lon,n_lat=n_lat))
		if not pfss_models and not any((p.parent/'metadata.json').is_file() for p in model.rglob('field_lines.npz')):
			raise FileNotFoundError(f'No saved PFSS model in {model}.')
	SRS_file=render.pop('SRS_file',None)
	# Spoke masking already applied to the COR2 zone of the cache.
	render_frames(motion_cache,panel_keys=(panel,),sT=sT,eT=eT,LASCO='COR2+HI1',SDO_fold=None,
		EUVI_fold=disk_folder,pfss_dir=model,outdir=dest,
		final_image_folder=source(final_image_folder) if final_image_folder else dest/'frames',
		SDO_instrument=None,SDO_product=None,EUVI_wave=EUVI_wave,SRS_file=source(SRS_file),
		user_datacsv=source(user_datacsv),goes_pickle=goes_p,goes_csv_input=goes_c,
		pfss_needed=pfss_needed,pfss_models=pfss_models,disk_max_minutes=disk_max_minutes,
		spoke_pa=spoke_pa,spoke_halfwidth=0.,spoke_arrow_buffer_deg=0.,
		max_gap_seconds=max_gap_seconds,**render)
	return motion_cache
