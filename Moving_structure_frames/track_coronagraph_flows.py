"""Render cached coronagraph intensity-flow vectors along bright and dark edges.

This demonstration does not re-estimate optical flow or edit core modules.
"""
from pathlib import Path
import copy,csv,json,io,warnings,pickle,re
import shutil,subprocess,tempfile
from functools import lru_cache
from astropy.coordinates import SkyCoord
import pandas as pd
import matplotlib.dates as mdates
from scipy.ndimage import map_coordinates,distance_transform_edt,binary_erosion,gaussian_filter
from contourpy import contour_generator
import numpy as np
from astropy.time import Time
import astropy.units as u
import sunpy.map
from datamine_calib_helper import load_map  # truthful WCS for every source
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from PIL import Image
from edge_motion import sample_edge_motion,contour_pixels,sample_centroid_motion
from matplotlib.patches import FancyArrowPatch,Polygon
from pfss_projection import _draw_pfss,select_ar_closed,select_ar_open
from pfss_fixed_seeds import select_trace_seeds,retrace_pfss,trace_ar_lines
from sunpy.coordinates import frames as solar_frames
from puddle_analysis import usable_grid,detect_signed,association,outward_support
from solar_disk_layer import disk_inventory,nearest_disk
from noaa_region import find_NOAA_location

def write_mp4(paths,destination):
	"""Encode the supplied PNGs in order, at 450 ms per frame. Requires FFmpeg with libx264."""
	ffmpeg=shutil.which('ffmpeg')
	if ffmpeg is None:
		raise RuntimeError('Install FFmpeg with libx264 and make ffmpeg available on PATH to write the MP4.')
	width=height=0
	for path in paths:
		with Image.open(path) as frame:
			width=max(width,frame.width)
			height=max(height,frame.height)
	width+=width%2
	height+=height%2
	destination=Path(destination)
	with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
		output=Path(tmp)/destination.name
		command=[ffmpeg,'-v','error','-f','rawvideo','-pixel_format','rgb24',
			'-video_size',f'{width}x{height}','-framerate','20/9','-i','pipe:0',
			'-c:v','libx264','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
		with subprocess.Popen(command,stdin=subprocess.PIPE) as process:
			try:
				for path in paths:
					with Image.open(path) as frame:
						canvas=Image.new('RGB',(width,height),'white')
						canvas.paste(frame.convert('RGB'),(0,0))
						process.stdin.write(canvas.tobytes())
				process.stdin.close()
			except BrokenPipeError as exc:
				raise RuntimeError('FFmpeg stopped while encoding the MP4. Check that libx264 is available.') from exc
			if process.wait():
				raise RuntimeError('FFmpeg could not encode the MP4.')
		output.replace(destination)

def is_stereo_grid(m):
	"""COR1, COR2 or the combined COR2+HI1 grid."""
	return str(m.detector).upper() in ('COR1','COR2','COR2+HI1')

def frame_map(ref,t,item):
	"""Keep the flow grid fixed; assign STEREO geometry from this exposure."""
	meta=copy.deepcopy(ref.meta)
	stereo='STEREO' in str(ref.observatory).upper() or is_stereo_grid(ref)
	if stereo and not is_stereo_grid(ref):
		raise ValueError('The STEREO flow cache needs a COR1/COR2 (or combined COR2+HI1) reference grid.')
	meta.update({'date-obs':t.isot,'date_obs':t.isot})
	if stereo:
		keys=('hgln_obs','hglt_obs','dsun_obs')
		try:
			lon,lat,distance=(float(item['observer'][key]) for key in keys)
		except (KeyError,TypeError,ValueError) as exc:
			raise ValueError(f'Missing STEREO exposure observer at {t.isot}; supply times.json observer longitude, latitude and distance.') from exc
		if not np.all(np.isfinite([lon,lat,distance])) or abs(lat)>90 or distance<=0:
			raise ValueError(f'Invalid STEREO exposure observer at {t.isot}')
		# Remove observer aliases inherited from the reference exposure.
		for observer_keys,_ in ref._supported_observer_coordinates:
			for key in observer_keys:
				meta.pop(key,None)
		# The cached Stonyhurst coordinates are defined at t.
		for key in ('date-avg','date_avg','date-beg','date_beg','date-end','date_end'):
			if key in meta:meta[key]=t.isot
		meta.update(dict(zip(keys,(lon,lat,distance))))
	m=sunpy.map.Map(ref.data,meta)
	if stereo:
		obs=m.observer_coordinate.transform_to(solar_frames.HeliographicStonyhurst(obstime=t))
		dlon=(obs.lon.to_value(u.deg)-lon+180)%360-180
		if abs(dlon)>.1 or abs(obs.lat.to_value(u.deg)-lat)>.1 or not np.isclose(obs.radius.to_value(u.m),distance,rtol=1e-5,atol=0):
			raise ValueError(f'STEREO observer WCS mismatch at {t.isot}: '
				f'expected ({lon:.6f}, {lat:.6f}, {distance:.3f} m), '
				f'got ({obs.lon.deg:.6f}, {obs.lat.deg:.6f}, {obs.radius.to_value(u.m):.3f} m)')
	return m

def render_frames(motion_cache,*, plot_style=None, pfss_linewidth=1.5, base_region_span_rsun=None,
	panel_keys=('base_difference','running_difference'),display_limit=None,
	display_stretch='linear',display_soft=1.,display_smooth_px=0.,ar_seed_spacing_deg=None,ar_long_loops=True,show_vectors=True,n_AR_open=None,
	detect_smooth_px=1.5,bright_contrast_snr=False,region_min_width_rsun=0.,region_max_aspect=None,ar_loop_reach_deg=None,
	AR_halfwidth_arcsec=400., AR_seed_spacing_arcsec=40., AR_r_seed=1.01,
	sT, eT, LASCO, SDO_fold, EUVI_fold, pfss_dir, outdir,
	final_image_folder, SDO_instrument, SDO_product, EUVI_wave, focus_AR, NOAA_number, AR_time,
	SRS_file, AR_lookup_max_age, AR_closedB_col, AR_openB_col, closed_field_color, user_datacsv, goes_pickle,
	goes_csv_input, edge_spacing, support_radius, min_support, min_displacement, arrow_length_pixels, centroid_arrow_length_pixels, speed_bins,
	pfss_needed, pfss_models, disk_max_minutes, bright_border, dark_border, new_border, border_width,
	n_open, n_global_closed, n_closed, ar_half_lon, ar_half_lat, ar_half_box_arcsec, spoke_pa,
	spoke_halfwidth, spoke_arrow_buffer_deg, mask_buffer_px, detect_sigma, detect_min_area, region_span_rsun, weak_bright_factor,
	flow_support_radius_rsun, bright_contrast, bright_min_area, arrow_min_area, detect_floor, max_gap_seconds, dpi):
	"""Render aligned motion products using settings supplied by the task script.

	n_AR_open: maximum number of open AR field lines (magenta) drawn, spread evenly
	over the selected set; None draws all. The cached AR lines are not reduced.
	show_vectors: False draws no edge or centroid velocity arrows and hides the
	speed colour bar (same frame layout); flow and the vector CSVs are still
	computed and saved.

	ar_seed_spacing_deg: None keeps the original AR handling (AR_halfwidth_arcsec
	seed square; loops with both footpoints in +-ar_half_lon/lat and inside
	ar_half_box_arcsec). A number (deg) switches to the AR-rectangle mode: seeds
	every ar_seed_spacing_deg over a patch covering +-ar_half_lon x +-ar_half_lat,
	no arcsec box. ar_long_loops=True keeps closed loops with at least one
	footpoint in the rectangle (large AR-rooted loops, tallest first up to
	n_closed); False needs both footpoints inside. All traced lines are cached
	as AR_field_lines_rooted.npz (selection applied at drawing), reused only for
	the same rectangle and spacing, so the toggle needs no retrace.
	ar_loop_reach_deg: closed AR loops need both ends within this distance (deg)
	of the NOAA centre; None: no limit (pfss_projection.select_ar_closed).

	Detection (puddle_analysis.detect_signed): detect_smooth_px smooths before
	thresholding. bright_contrast_snr=True makes bright_contrast a unitless local
	SNR, (95th percentile of region - ring mean)/ring std, ring 2-6 x smoothing;
	False keeps the old median difference in image units. region_min_width_rsun
	and region_max_aspect set the minimum breadth and maximum length/breadth of a
	region's fitted box (None: no limit); thin lines (HI1 artifacts) are dropped.
	"""
	rooted=ar_seed_spacing_deg is not None
	if rooted:
		rsun_1au=np.degrees(np.arctan(695700./1.495978707e8))*3600  # 959.6 arcsec
		AR_halfwidth_arcsec=rsun_1au*np.sin(np.radians(max(ar_half_lon,ar_half_lat)))
		AR_seed_spacing_arcsec=rsun_1au*np.radians(ar_seed_spacing_deg)
		ar_half_box_arcsec=None
		ar_rect=np.array([ar_half_lon,ar_half_lat,ar_seed_spacing_deg],float)
	ar_name='AR_field_lines_rooted.npz' if rooted else 'AR_field_lines.npz'
	ar_rule='one' if rooted and ar_long_loops else 'both'
	style=(plot_style or {}).get
	if base_region_span_rsun is None:base_region_span_rsun=region_span_rsun
	root=Path(outdir).expanduser()
	outdir_frames=Path(final_image_folder).expanduser()
	outdir_frames.mkdir(parents=True,exist_ok=True)
	with np.load(motion_cache/'coordinates.npz') as z:xr,yr=z['x_rsun'],z['y_rsun']
	ref=load_map(motion_cache/'reference_grid.fits')
	stereo=is_stereo_grid(ref)
	if not stereo and str(ref.detector).upper()!=LASCO.upper():
		raise ValueError(f'LASCO={LASCO} does not match the cached {ref.detector} grid.')
	folder,instr,product=(EUVI_fold,'EUVI',EUVI_wave) if stereo else (SDO_fold,SDO_instrument,SDO_product)
	inventory=disk_inventory(folder,instr,product,ref)
	if not inventory and folder is not None and instr.upper()=='HMI':
		inventory=disk_inventory(motion_cache,instr,product,ref)
	if not inventory and folder is not None:warnings.warn(f'No matching {instr} disk FITS in {folder}')
	load_disk=lru_cache(maxsize=2)(load_map)
	keys=['lon','lat','r','seg','pol']
	pfss_file=None
	ar_candidates=[]
	if pfss_needed:
		model=Path(pfss_dir).expanduser() if pfss_dir is not None else motion_cache
		if focus_AR and rooted:ar_candidates=sorted(model.rglob(ar_name))
		elif focus_AR:ar_candidates=sorted(p for p in model.rglob('AR*.npz') if p.name=='AR_field_lines.npz' or re.fullmatch(r'AR[_ -]?'+str(NOAA_number)+r'(?:[_ -]field_lines)?\.npz',p.name,re.I))
		candidates=[p for p in model.rglob('field_lines.npz') if (p.parent/'metadata.json').is_file()]
		if pfss_models:
			model_epoch,pfss_file=pfss_models[0]
		elif candidates:
			mid=Time(sT)+(Time(eT)-Time(sT))/2
			def epoch(p):
				return Time(json.loads((p.parent/'metadata.json').read_text())['model_time'])
			pfss_file=min(candidates,key=lambda p:abs((epoch(p)-mid).sec))
			model_epoch=epoch(pfss_file)
		else:
			pfss_file=motion_cache/'field_lines.npz'
			model_epoch=Time(json.loads((motion_cache/'PFSS_metadata.json').read_text())['model_time'])
	@lru_cache(maxsize=None)
	def initial_pfss_layers(pfss_file,model_epoch_iso):
		model_epoch=Time(model_epoch_iso)
		with np.load(pfss_file) as z:fields={k:z[k] for k in [*keys,'rss']}
		op=np.unique(fields['seg'][fields['pol']!=0])
		indices=np.arange(len(op)) if n_open is None else np.linspace(0,len(op),min(n_open,len(op)),endpoint=False,dtype=int)
		keep=(fields['pol']==0)|np.isin(fields['seg'],op[indices])
		ar_fields=None
		if focus_AR:
			cfg=json.loads((motion_cache/'AR_settings.json').read_text()) if (motion_cache/'AR_settings.json').is_file() else {}
			same_region=cfg.get('noaa')==NOAA_number
			if SRS_file is None and same_region and abs((Time(cfg['time'])-Time(AR_time)).sec)<=AR_lookup_max_age*3600:
				lon,lat,reported=cfg['lon'],cfg['lat'],Time(cfg['time'])
			else:
				lon,lat,reported,_=find_NOAA_location(NOAA_number,AR_time,root/'NOAA_SRS',SRS_file,AR_lookup_max_age)
				reported=Time(reported)
			c=SkyCoord(lon*u.deg,lat*u.deg,1*u.solRad,frame=solar_frames.HeliographicStonyhurst(obstime=reported))
			c=c.transform_to(solar_frames.HeliographicCarrington(observer='earth',obstime=reported))
			c=SkyCoord(c.lon,c.lat,c.radius,frame=solar_frames.HeliographicCarrington(observer='earth',obstime=model_epoch))
			c=c.transform_to(solar_frames.HeliographicStonyhurst(obstime=model_epoch))
			epoch_tag=model_epoch.strftime('%Y%m%d_%H%M%S')
			sibling=pfss_file.parent/ar_name
			matches=[p for p in ar_candidates if epoch_tag in p.relative_to(model).parts[:-1]]
			local=[p for p in ar_candidates if p.parent==pfss_file.parent]
			paths=list(dict.fromkeys([*local,sibling,*matches]))
			ar=fields;ar_source=None;generated=False
			for path in paths:
				if not path.is_file() or (path.parent==motion_cache and not same_region):continue
				try:
					with np.load(path) as z:
						if 'noaa' in z and int(z['noaa'])!=NOAA_number:continue
						if 'rss' in z and not np.allclose(z['rss'],fields['rss']):continue
						if rooted and ('ar_rect' not in z or not np.allclose(z['ar_rect'],ar_rect)):continue
						candidate={k:z[k] for k in keys}
						generated='global_source' in z and str(z['global_source'])==str(pfss_file.resolve())
						if generated:
							continue  # Sparse selection from global lines: trace densely instead.
					if any(v.ndim!=1 or v.shape!=candidate['seg'].shape for v in candidate.values()):
						raise ValueError('Inconsistent field-line arrays')
					if not candidate['seg'].size:continue
				except (OSError,ValueError,KeyError,TypeError) as exc:
					warnings.warn(f'Cannot use AR field lines {path}: {exc}')
					continue
				ar=candidate;ar_source=path
				print(f'Using AR field lines: {path}')
				break
			traced=False
			if ar_source is None:
				# No cached AR lines: trace a dense seed patch through this snapshot.
				try:
					ar={k:v for k,v in trace_ar_lines(pfss_file.parent,c.lon.deg,c.lat.deg,model_epoch,
						AR_halfwidth_arcsec,None,None,AR_r_seed,AR_seed_spacing_arcsec).items() if k in keys}
					traced=True
				except (OSError,ValueError,KeyError,RuntimeError) as exc:
					warnings.warn(f'AR tracing failed ({exc}); selecting AR lines from the global PFSS lines.')
					ar=fields
			use_ar_cache=ar_source is not None
			closed=select_ar_closed(ar,ref,model_epoch,c.lon.deg,c.lat.deg,ar_half_lon,ar_half_lat,ar_half_box_arcsec,n_closed,ar_rule,ar_loop_reach_deg)
			opened=select_ar_open(ar,model_epoch,c.lon.deg,c.lat.deg,ar_half_lon,ar_half_lat,ar_half_box_arcsec)
			ar_keep=closed|opened
			ar_fields={k:ar[k][ar_keep] for k in keys}|{'rss':fields['rss']}
			if ar is fields:
				# AR lines taken from the global set: do not draw them twice.
				keep &= ~np.isin(fields['seg'],ar['seg'][ar_keep])
			if not use_ar_cache:
				ar_dir=model/epoch_tag
				ar_dir.mkdir(parents=True,exist_ok=True)
				ar_path=ar_dir/ar_name;extra=dict(ar_rect=ar_rect) if rooted else {}
				if traced:
					np.savez_compressed(ar_path,**({k:ar[k] for k in keys}|{'rss':fields['rss']} if rooted else ar_fields),**extra,noaa=NOAA_number,traced_source=str(pfss_file.resolve()),
						halfwidth_arcsec=AR_halfwidth_arcsec,spacing_arcsec=AR_seed_spacing_arcsec,r_seed=AR_r_seed)
					print(f'Saved traced AR field lines: {ar_path}')
				elif not ar_path.exists():
					np.savez_compressed(ar_path,**ar_fields,**extra,noaa=NOAA_number,global_source=str(pfss_file.resolve()))
					print(f'Saved AR field lines: {ar_path}')
			print(f'NOAA {NOAA_number} at {model_epoch.isot}: '
				f'{len(np.unique(ar["seg"][opened]))} open traces, '
				f'{len(np.unique(ar["seg"][closed]))} closed loops selected')
			if n_AR_open is not None:
				# Drawing only: keep at most n_AR_open open AR lines, spread evenly over the set.
				ids=np.unique(ar_fields['seg'][ar_fields['pol']!=0])
				if len(ids)>n_AR_open:
					drop=np.setdiff1d(ids,ids[np.linspace(0,len(ids),int(n_AR_open),endpoint=False,dtype=int)])
					sel=~np.isin(ar_fields['seg'],drop)
					ar_fields={k:(v[sel] if k!='rss' else v) for k,v in ar_fields.items()}
					print(f'NOAA {NOAA_number}: drawing {int(n_AR_open)} of {len(ids)} open AR traces',flush=True)
		open_fields={k:fields[k][keep] for k in keys}|{'rss':fields['rss']}
		return open_fields,ar_fields
	fixed_seeds=None

	@lru_cache(maxsize=None)
	def pfss_layers(pfss_file,model_epoch_iso):
		nonlocal fixed_seeds
		if not pfss_models or len(pfss_models)==1:
			return initial_pfss_layers(pfss_file,model_epoch_iso)
		if fixed_seeds is None:
			global_lines,ar_lines=initial_pfss_layers(pfss_file,model_epoch_iso)
			metadata=json.loads((pfss_file.parent/'metadata.json').read_text())
			seed_radius=float(metadata['r_seed'])
			closed_ids=np.unique(global_lines['seg'][global_lines['pol']==0])
			closed_ids=sorted(closed_ids,key=lambda i:(-global_lines['r'][global_lines['seg']==i].max(),i))
			keep=(global_lines['pol']!=0)|np.isin(global_lines['seg'],closed_ids[:n_global_closed])
			global_seeds=select_trace_seeds({k:global_lines[k][keep] for k in keys},seed_radius)
			ar_seeds=select_trace_seeds(ar_lines,seed_radius) if ar_lines is not None else np.empty((0,3))
			fixed_seeds=(global_seeds,ar_seeds)
			np.savez_compressed(outdir_frames/'pfss_carrington_seeds.npz',global_seeds=global_seeds,
				ar_seeds=ar_seeds,initial_epoch=model_epoch_iso,columns=['lon_deg','lat_deg','radius_rsun'])
		global_lines,ar_lines=retrace_pfss(pfss_file,fixed_seeds)
		print(f'PFSS {model_epoch_iso}: retraced {len(np.unique(global_lines["seg"]))} global and '
			f'{len(np.unique(ar_lines["seg"]))} AR traces from fixed Carrington seeds')
		return global_lines,ar_lines

	lat_csv=Path(user_datacsv).expanduser()
	if not lat_csv.is_file():lat_csv=motion_cache/lat_csv.name
	goes_csv=goes_csv_input or motion_cache/'GOES_flux.csv'
	if goes_pickle is None:goes=pd.read_csv(goes_csv)
	else:
		with Path(goes_pickle).expanduser().open('rb') as f:g=pickle.load(f)
		goes=pd.DataFrame(dict(datetime=g['Time'],flux=g['Flux']))
	lat=pd.read_csv(lat_csv)
	for column,default in (('flux_error_lower',lat.get('flux_error',0.)),('flux_error_upper',lat.get('flux_error',0.)),('is_upper_limit',False),('time_error_seconds',0.)):
		if column not in lat:lat[column]=default
	for table in (goes,lat):table['datetime']=pd.to_datetime(table['datetime'],utc=True,format='mixed')
	frames=[];all_rows=[];region_rows=[];centroid_rows=[];centroids={};prior=None;prior_time=None
	for path in sorted(motion_cache.glob('structures_*.npz')):
		with np.load(path) as z:data={k:z[k].copy() for k in z.files}
		# Combined cache: base and running differences cover different HI1 pixels.
		if len(panel_keys)==1 and 'valid_'+panel_keys[0].split('_')[0] in data:
			data['valid']=data['valid_'+panel_keys[0].split('_')[0]]
		# Exact fractional times and pair duration come from the original inventory.
		times=json.loads((motion_cache/'times.json').read_text());item=times[path.stem]
		t,prev=Time(item['datetime']),Time(item['previous_time']);dt=(t-prev).sec
		if not Time(sT)<=t<=Time(eT):continue
		m=frame_map(ref,t,item)
		if m.data.shape!=xr.shape or data['valid'].shape!=xr.shape:
			raise ValueError('Difference, coordinate and reference grids must have identical shapes.')
		valid=usable_grid(xr,yr,data['valid'],spoke_pa,spoke_halfwidth,0)
		arrow_valid=usable_grid(xr,yr,data['valid'],spoke_pa,spoke_halfwidth+spoke_arrow_buffer_deg,0)&binary_erosion(data['valid'],iterations=mask_buffer_px)
		if 'arrow_zone' in data:arrow_valid&=data['arrow_zone']  # combined cache: spoke buffer on COR2 only
		y,x=np.indices(valid.shape);dx,dy=data['dx'],data['dy']
		q=data['quality']&arrow_valid&np.isfinite(dx)&np.isfinite(dy)
		for f in (.25,.5,.75,1.):
			q&=map_coordinates(arrow_valid.astype(np.uint8),[y+f*dy,x+f*dx],order=0,mode='constant',cval=0)>0
		panels={};new_prior={}
		support=outward_support(xr,yr,dx,dy,q,radius_rsun=flow_support_radius_rsun) if dt<=max_gap_seconds else None
		for key in panel_keys:
			labels,threshold=detect_signed(data[key],valid,detect_sigma,detect_min_area,detect_floor,bright_contrast,bright_min_area,
				grid=(xr,yr),min_span=base_region_span_rsun if key=='base_difference' else region_span_rsun,
				flow_support=support if key=='running_difference' else None,weak_factor=weak_bright_factor,
				smooth_px=detect_smooth_px,snr=bool(bright_contrast_snr),min_width=region_min_width_rsun or 0.,
				max_aspect=np.inf if region_max_aspect in (None,'') else float(region_max_aspect))
			vectors=[];new_masks=[]
			for kind,lab in zip(('bright','dark'),labels):
				old=prior[key,kind] if prior is not None and prior_time==prev.isot and dt<=max_gap_seconds else None
				new,regions=association(old,lab,dx,dy,q)
				if old is None:
					regions=[dict(component=k,pixels=int((lab==k).sum()),coverage=0.,overlap=0.,previous_components='',status='unknown') for k in range(1,int(lab.max())+1)]
				region_rows.extend(dict(datetime=t.isot,panel=key,sign=kind,threshold=threshold,**r) for r in regions)
				new_prior[key,kind]=lab;new_masks.append(new)
				if dt>max_gap_seconds:continue
				if centroid_arrow_length_pixels>0:
					for row in sample_centroid_motion(lab,dx,dy,q,xr,yr,dt,arrow_min_area,min_support,min_displacement,
						intensity=data[key],sign=1 if kind=='bright' else -1):
						px,py=row['x_pixel'],row['y_pixel'];vx,vy=row['dx_pixel'],row['dy_pixel']
						f=np.linspace(0,centroid_arrow_length_pixels/np.hypot(vx,vy),max(16,int(np.ceil(centroid_arrow_length_pixels))+1))
						if not map_coordinates(arrow_valid.astype(np.uint8),[py+f*vy,px+f*vx],order=0,mode='constant',cval=0).all():continue
						row.update(edge_type=kind,panel=key,datetime=t.isot,previous_time=prev.isot,pair_seconds=float(dt))
						centroid_rows.append(row)
						centroids.setdefault((t.isot,key),[]).append(row)
				sizes=np.bincount(lab.ravel());large=(lab>0)&(sizes[lab]>=arrow_min_area)
				if not large.any():continue
				edges=contour_generator(z=large.astype(float)).lines(.5)
				if not edges:continue
				edge_band=distance_transform_edt(~(large&~binary_erosion(large)))<=1.5
				q_edge=q&(map_coordinates(edge_band.astype(np.uint8),[y+dy,x+dx],order=0,mode='constant',cval=0)>0)
				for row in sample_edge_motion(np.concatenate(edges),dx,dy,q_edge,xr,yr,dt,spacing=edge_spacing,radius=support_radius,min_support=min_support,min_displacement=min_displacement):
					px,py=row['x_pixel'],row['y_pixel'];vx,vy=row['dx_pixel'],row['dy_pixel']
					if not arrow_valid[round(py),round(px)]:continue
					f=np.linspace(0,arrow_length_pixels/np.hypot(vx,vy),16)
					if not map_coordinates(arrow_valid.astype(np.uint8),[py+f*vy,px+f*vx],order=0,mode='constant',cval=0).all():continue
					if any(np.hypot(px-r['x_pixel'],py-r['y_pixel'])<edge_spacing for r in vectors):continue
					row.update(edge_type=kind,panel=key,datetime=t.isot,previous_time=prev.isot,pair_seconds=float(dt))
					vectors.append(row)
			panels[key]=(labels,new_masks,vectors);all_rows.extend(vectors)
		np.savez_compressed(outdir_frames/f'{path.stem}_masks.npz',valid=valid,arrow_valid=arrow_valid,**{key+'_'+name:arr for key,(labs,news,_) in panels.items() for name,arr in zip(('bright','dark','new_bright','new_dark'),[*labs,*news])})
		frames.append((m,t,dt,data,panels,item));prior,prior_time=new_prior,t.isot
	if not frames:raise ValueError('No cached motion frames in sT–eT.')
	cmap=plt.get_cmap('rainbow_r',len(speed_bins)-1)
	norm=BoundaryNorm(speed_bins,cmap.N)
	# Grey-scale half-range per panel. None: largest per-frame 99th percentile
	# (one stretch for the movie; a strong CME frame then flattens the rest).
	# 'median': median of the per-frame 99th percentiles. 'frame': each frame
	# its own 99th percentile. A number: fixed limit in image units.
	p99={key:[np.percentile(abs(f[3][key][f[3]['valid']]),99) for f in frames] for key in panel_keys}
	if display_limit is None:limits={k:max(v) for k,v in p99.items()}
	elif display_limit=='median':limits={k:float(np.median(v)) for k,v in p99.items()}
	elif display_limit=='frame':limits=None
	else:limits={k:float(display_limit) for k in panel_keys}
	paths=[];disk_sources=[]
	pfss_sources=[]
	for index,(m,t,dt,data,panels,item) in enumerate(frames):
		if pfss_needed:
			epoch,path=min(pfss_models,key=lambda row:abs((row[0]-t).sec)) if pfss_models else (model_epoch,pfss_file)
			open_fields,ar_fields=pfss_layers(path,epoch.isot)
			pfss_sources.append(dict(datetime=t.isot,pfss_epoch=epoch.isot,pfss_file=str(path)))
		disk,disk_path=nearest_disk(m,inventory,disk_max_minutes,load_disk)
		disk_sources.append(dict(datetime=t.isot,disk_file=str(disk_path) if disk_path else None))
		fig=plt.figure(figsize=(15,10),dpi=dpi,layout='constrained')
		fig.get_layout_engine().set(h_pad=.09,rect=(0,0,1,.875))
		g=fig.add_gridspec(2,len(panel_keys),height_ratios=[4,1.45]);axs=[fig.add_subplot(g[0,j]) for j in range(len(panel_keys))];lc=fig.add_subplot(g[1,:]);rt=lc.twinx()
		for ax,key in zip(axs,panel_keys):
			limit=limits[key] if limits is not None else p99[key][index];title='Quiet-base difference' if key=='base_difference' else 'Running difference'
			# Display only (detection and arrows use the unsmoothed linear data):
			# optional mask-normalised smoothing, then linear or asinh stretch.
			# asinh is linear within +/-display_soft and logarithmic beyond, so
			# faint structures gain contrast while bright fronts do not saturate.
			img=data[key]
			if display_smooth_px>0:
				w=gaussian_filter(data['valid'].astype(float),display_smooth_px)
				img=np.where(data['valid'],gaussian_filter(np.where(data['valid'],img,0.),display_smooth_px)/np.maximum(w,1e-6),0.)
			if display_stretch=='asinh':img,limit=np.arcsinh(img/display_soft),np.arcsinh(limit/display_soft)
			ax.set_facecolor('black');ax.imshow(np.ma.masked_where(~data['valid'],img),origin='lower',cmap='gray',vmin=-limit,vmax=limit)
			if disk is not None:
				image,extent,cm,norm_disk=disk
				ax.imshow(image,extent=extent,origin='lower',cmap=cm,norm=norm_disk,interpolation='nearest',zorder=1.5)
			if pfss_needed:
				if open_fields['seg'].size:_draw_pfss(ax,m,open_fields,lw=pfss_linewidth,alpha=.65,N_Bloop=np.nan if fixed_seeds is not None else n_global_closed,colors={0:closed_field_color,1:'#ff4d4d',-1:'#4da6ff'})
				if ar_fields is not None and ar_fields['seg'].size:_draw_pfss(ax,m,ar_fields,lw=pfss_linewidth,alpha=.85,colors={0:AR_closedB_col,1:AR_openB_col,-1:AR_openB_col},N_Bloop=np.nan if fixed_seeds is not None else n_closed)
			if 'hi_zone' in data and data['hi_zone'].any():
				# COR2 | HI1 seam (combined cache).
				ax.contour(data['cor2_zone'],levels=[.5],colors=['#8a8a8a'],linestyles=':',linewidths=.9)
			labels,new_masks,rows=panels[key]
			for lab,colour,line_style in zip(labels,(bright_border,dark_border),('-','--')):
				if lab.any():ax.contour(lab>0,levels=[.5],colors=[colour],linestyles=line_style,linewidths=border_width)
			for mask in new_masks:
				if mask.any():ax.contour(mask,levels=[.5],colors=[new_border],linestyles=':',linewidths=1.15)
			if rows and show_vectors:
				x=np.array([r['x_pixel'] for r in rows]);y=np.array([r['y_pixel'] for r in rows])
				dx=np.array([r['dx_pixel'] for r in rows]);dy=np.array([r['dy_pixel'] for r in rows])
				length=np.hypot(dx,dy)
				dx=arrow_length_pixels*dx/length;dy=arrow_length_pixels*dy/length
				speed=np.array([r['speed_projected_kms'] for r in rows])
				for px,py,vx,vy,spd in zip(x,y,dx,dy,speed):
					arrow=FancyArrowPatch((px,py),(px+vx,py+vy),arrowstyle='->',
						mutation_scale=10,shrinkA=0,shrinkB=0,color=cmap(norm(spd)),linewidth=1.15,zorder=12)
					arrow.set_path_effects([pe.Stroke(linewidth=1.73,foreground='black'),pe.Normal()])
					ax.add_patch(arrow)
			for row in (centroids.get((t.isot,key),()) if show_vectors else ()):
				px,py=row['x_pixel'],row['y_pixel']
				vx,vy=row['dx_pixel'],row['dy_pixel']
				scale=centroid_arrow_length_pixels/np.hypot(vx,vy)
				end=(px+scale*vx,py+scale*vy)
				colour=cmap(norm(row['speed_projected_kms']))
				direction=np.array([vx,vy])/np.hypot(vx,vy)
				base=end-min(12.,centroid_arrow_length_pixels*.5)*direction
				wing=min(8.,centroid_arrow_length_pixels/3)*np.array([-direction[1],direction[0]])
				shaft,=ax.plot([px,base[0]],[py,base[1]],color=colour,lw=2.25,zorder=13)
				shaft.set_path_effects([pe.Stroke(linewidth=3.38,foreground='black'),pe.Normal()])
				head=bright_border if row['edge_type']=='bright' else dark_border
				ax.add_patch(Polygon([base+wing,end,base-wing],facecolor=head,edgecolor='black',lw=.85,zorder=14))
			ax.set_title(style('base_title' if key=='base_difference' else 'running_title',title),fontsize=style('panel_title_size',style('xlabel_size',18)),pad=10)
			ny,nx=xr.shape
			for axis,coords,set_ticks in (('x',xr[ny//2],ax.set_xticks),('y',yr[:,nx//2],ax.set_yticks)):
				step=5 if np.ptp(coords)<50 else 10
				vals=(np.arange(np.ceil(coords.min()/step),np.floor(coords.max()/step)+1)*step
					if stereo else np.arange(-20,21,10))
				if style(axis+'ticks') is not None:vals=style(axis+'ticks')
				order=np.argsort(coords)
				set_ticks(np.interp(vals,coords[order],order),labels=[f'{v:g}' for v in vals])
			ax.tick_params(axis='x',labelsize=style('xtick_size',13));ax.tick_params(axis='y',labelsize=style('ytick_size',13))
			ax.set_xlabel(style('xlabel',r'Solar X ($R_\odot$)'),fontsize=style('xlabel_size',16))
			ax.set_ylabel(style('ylabel',r'Solar Y ($R_\odot$)'),fontsize=style('ylabel_size',16))
		centres=(np.asarray(speed_bins[:-1])+np.asarray(speed_bins[1:]))/2
		cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=list(axs),panchor=(.5,1),shrink=.78,pad=.02,boundaries=speed_bins,ticks=centres,spacing='uniform',extend='max')
		cb.ax.set_yticklabels([f'{a:g}–{b:g}' for a,b in zip(speed_bins[:-1],speed_bins[1:])])
		cb.set_label(style('colorbar_label','Speed magnitude |v| (km/s)'),fontsize=style('colorbar_label_size',13));cb.ax.tick_params(labelsize=style('colorbar_tick_size',15),length=0)
		cb.ax.text(.5,1.06,'Faster',transform=cb.ax.transAxes,ha='center',fontsize=11)
		cb.ax.text(.5,-.06,'Slower',transform=cb.ax.transAxes,ha='center',fontsize=11)
		if not show_vectors:cb.ax.set_visible(False)  # keep the panel layout
		gap=' | gap: arrows and associations withheld' if dt>max_gap_seconds else ''
		default_title=f'{m.instrument} {m.detector} | {t.strftime("%Y-%m-%d %H:%M:%S")} UT | $\\delta t$ = {dt/60:.1f} min{gap}'
		if item.get('hi_time'):
			hi_dt=f'$\\delta t$ = {item["hi_pair_seconds"]/60:.0f} min' if item.get('hi_pair_seconds') else 'no running pair'
			default_title=(f'{m.instrument} COR2 {t.strftime("%Y-%m-%d %H:%M")} UT ($\\delta t$ = {dt/60:.0f} min) + '
				f'HI1 {Time(item["hi_time"]).strftime("%H:%M")} UT ({hi_dt}){gap}')
		fig.text(.5,.985,default_title if style('title') is None else style('title'),
			ha='center',va='top',fontsize=style('title_size',18))
		fig.suptitle(('Narrow arrows: velocity along the bright and dark region contours. Arrow colors indicate speed.\n'
			'Thick long arrows: intensity-weighted velocity of the regions. Arrowhead: orange (dark region); yellow (bright region)') if show_vectors else
			'Contours: bright (solid) and dark (dashed) regions. Velocity vectors not shown.',
			x=.025,y=.946,ha='left',va='top',multialignment='left',fontsize=style('subtitle_size',15))
		legend=[Line2D([],[],color=bright_border,lw=border_width,label='Bright region'),
			Line2D([],[],color=dark_border,lw=border_width,ls='--',label='Dark region')]
		for handle,label in zip(legend,style('legend_labels',[h.get_label() for h in legend])):handle.set_label(label)
		if style('legend',True):axs[0].legend(handles=legend,loc=style('legend_loc','lower left'),fontsize=style('legend_size',style('xtick_size',15)),facecolor='#303030',labelcolor='white',borderpad=.35,labelspacing=.25,handlelength=1.6)
		rt.plot(goes.datetime,goes.flux.where(goes.flux>0),color='orange',lw=1.2,label='GOES XRS-B')
		det=~lat.is_upper_limit.astype(str).str.lower().isin(['true','1']);a=lat.loc[det];b=lat.loc[~det]
		lc.errorbar(mdates.date2num(a.datetime),a.flux,yerr=[a.flux_error_lower,a.flux_error_upper],xerr=a.time_error_seconds/86400,fmt='o',ms=4,color='#9c260b')
		lc.errorbar(mdates.date2num(b.datetime),b.flux,yerr=.25*b.flux,xerr=b.time_error_seconds/86400,uplims=True,fmt='o',ms=4,color='#9c260b')
		lc.axvline(t.to_datetime(),color='black',ls=':',lw=1)
		lc.set(xlim=[pd.Timestamp(sT),pd.Timestamp(eT)],yscale='log',ylabel='LAT >100 MeV flux\n(ph cm⁻² s⁻¹)',xlabel=f'Time +{sT} (UT)')
		_w=lat[(lat['datetime']>=pd.Timestamp(sT,tz='UTC'))&(lat['datetime']<=pd.Timestamp(eT,tz='UTC'))]  # LAT axis from points inside sT-eT
		lo,hi=(_w['flux']-_w['flux_error_lower']).where(lambda x:x>0,_w['flux']).min(),(_w['flux']+_w['flux_error_upper']).max()
		e10=lambda x:10**np.floor(np.log10(x))  # decade of x
		if np.isfinite(lo) and np.isfinite(hi) and 0<lo<=hi:  # flux -+ error, rounded out to 1 or 5 x 10^n
			_m=hi/e10(hi);lc.set_ylim((5 if lo/e10(lo)>=5 else 1)*e10(lo),(1 if _m<=1 else 5 if _m<=5 else 10)*e10(hi))
		else:lc.set_ylim(1e-8,1e-4)
		rt.set(yscale='log',ylabel='GOES SXR flux\n(W m⁻²)')
		lc.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
		lc.grid(alpha=.2)
		lc.text(.02,.98,'(b)',transform=lc.transAxes,va='top',fontsize=style('xlabel_size',18),bbox=dict(facecolor='white',edgecolor='none',pad=1))
		lc.xaxis.label.set_size(style('xlabel_size',18))
		lc.yaxis.label.set_size(style('ylabel_size',18));rt.yaxis.label.set_size(style('ylabel_size',18))
		lc.tick_params(axis='x',labelsize=style('xtick_size',15))
		for axis in (lc,rt):axis.tick_params(axis='y',labelsize=style('ytick_size',15))
		fig.canvas.draw()
		image_box=axs[-1].get_position();bar_box=cb.ax.get_position()
		fig.set_layout_engine(None)
		cb.ax.set_axes_locator(None);cb.ax.set_box_aspect(None);cb.ax.set_aspect('auto')
		renderer=fig.canvas.get_renderer()
		label_shift=(rt.yaxis.label.get_window_extent(renderer).x0-cb.ax.yaxis.label.get_window_extent(renderer).x0)/fig.bbox.width
		cb.ax.set_position([max(image_box.x1+.012,bar_box.x0+min(0,label_shift)),image_box.y0,bar_box.width,image_box.height])
		fig.text(.012,axs[0].get_position().y1+.012,'(a)',va='bottom',fontsize=style('xlabel_size',18))
		path=outdir_frames/f'edge_arrows_{t.strftime("%Y%m%d_%H%M%S")}.png'
		with io.BytesIO() as buffer:
			fig.savefig(buffer,format='png',dpi=dpi,bbox_inches='tight',pad_inches=.08);path.write_bytes(buffer.getvalue())
		plt.close(fig);paths.append(path)
		print(t.isot,[(k,[int(l.max()) for l in v[0]],len(v[2])) for k,v in panels.items()],flush=True)
	with (outdir_frames/'edge_vectors.csv').open('w',newline='') as fil:
		w=csv.DictWriter(fil,fieldnames=list(all_rows[0]) if all_rows else ['datetime','panel']);w.writeheader();w.writerows(all_rows)
	with (outdir_frames/'centroid_vectors.csv').open('w',newline='') as fil:
		w=csv.DictWriter(fil,fieldnames=list(centroid_rows[0]) if centroid_rows else ['datetime','panel'])
		w.writeheader();w.writerows(centroid_rows)
	with (outdir_frames/'regions.csv').open('w',newline='') as fil:
		w=csv.DictWriter(fil,fieldnames=list(region_rows[0]) if region_rows else ['datetime','panel']);w.writeheader();w.writerows(region_rows)
	pd.DataFrame(pfss_sources).to_csv(outdir_frames/'pfss_sources.csv',index=False)
	pd.DataFrame(disk_sources).to_csv(outdir_frames/'disk_sources.csv',index=False)
	movie_name=f'{str(ref.detector).upper().replace("+","_")}_edge_velocity.mp4'
	write_mp4(paths,outdir_frames/movie_name)
	(outdir_frames/'parameters.json').write_text(json.dumps(dict(region_span_rsun=region_span_rsun,base_region_span_rsun=base_region_span_rsun,weak_bright_factor=weak_bright_factor,flow_support_radius_rsun=flow_support_radius_rsun,speed_cmap='rainbow_r',speed_bins_kms=speed_bins,colourbar_spacing='categorical equal-sized bands; numeric intervals explicitly labelled',edge_spacing_pixels=edge_spacing,support_radius_pixels=support_radius,min_support=min_support,min_displacement_pixels=min_displacement,arrow_length_pixels=arrow_length_pixels,centroid_arrow_length_pixels=centroid_arrow_length_pixels,length_encodes_speed=False,flow_source='previous trial intensity TV-L1, source vectors advected to target positions',n_open=n_open,n_global_closed=n_global_closed,n_closed=n_closed,ar_half_lon=ar_half_lon,ar_half_lat=ar_half_lat,ar_half_box_arcsec=ar_half_box_arcsec,spoke_pa=spoke_pa,spoke_halfwidth=spoke_halfwidth,spoke_arrow_buffer_deg=spoke_arrow_buffer_deg,bright_contrast=bright_contrast,bright_contrast_snr=bool(bright_contrast_snr),detect_smooth_px=detect_smooth_px,region_min_width_rsun=region_min_width_rsun,region_max_aspect=region_max_aspect,bright_min_area=bright_min_area,arrow_min_area=arrow_min_area,mask_buffer_px=mask_buffer_px,detect_sigma=detect_sigma,detect_floor=detect_floor,detect_min_area=detect_min_area,max_gap_seconds=max_gap_seconds,disk_folder=str(folder) if folder is not None else None,NOAA_number=NOAA_number if focus_AR else None,pfss_file=str(pfss_file),disk_max_minutes=None if np.isinf(disk_max_minutes) else disk_max_minutes,bright_border=bright_border,dark_border=dark_border,border_width=border_width,physical_velocity_validated=False),indent=2))

if __name__=='__main__':
	raise SystemExit('Run the LASCO or STEREO frame maker instead.')
