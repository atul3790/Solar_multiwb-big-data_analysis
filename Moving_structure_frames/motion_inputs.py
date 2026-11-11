"""Build the renderer inputs from a local coronagraph FITS sequence."""
from pathlib import Path
import json
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import sunpy.map
from datamine_calib_helper import load_map  # truthful WCS for every source
from sunpy.coordinates import frames
from scipy import ndimage as ndi
from skimage.registration import optical_flow_tvl1

def flow(a,b,valid):
	def feature(im):
		w=ndi.gaussian_filter(valid.astype('float32'),8)
		bg=ndi.gaussian_filter(np.where(valid,im,0),8)/np.maximum(w,1e-6)
		return np.where(valid,im-bg,0)
	f,g=feature(a),feature(b)
	scale=max(float(np.percentile(np.abs(np.r_[f[valid],g[valid]]),95)),1.)
	f,g=[np.clip(i/scale,-1,1)*.5+.5 for i in (f,g)]
	dy,dx=optical_flow_tvl1(f,g,num_warp=5,num_iter=8,prefilter=True)
	by,bx=optical_flow_tvl1(g,f,num_warp=5,num_iter=8,prefilter=True)
	y,x=np.indices(a.shape);coords=[y+dy,x+dx]
	err=np.hypot(dx+ndi.map_coordinates(bx,coords,order=1),dy+ndi.map_coordinates(by,coords,order=1))
	q=valid&(err<1.5)&(ndi.map_coordinates(valid.astype(float),coords,order=0)>.5)
	return dx,dy,q

def spacecraft_id(value):
	return str(value).upper().replace('_','').replace('-','').replace(' ','').replace('AHEAD','A').replace('BEHIND','B')

def inventory(folder,detector,spacecraft):
	rows=[]
	for p in sorted(Path(folder).expanduser().rglob('*')):
		if not p.is_file() or not p.name.lower().endswith(('.fits','.fit','.fts','.fits.gz','.fits.fz','.fit.gz','.fts.gz')):continue
		try:
			for hdu in (fits.getheader(p,0),):
				if str(hdu.get('DETECTOR','')).upper()!=detector:continue
				if spacecraft and not any(spacecraft_id(hdu.get(k,''))=='STEREO'+spacecraft for k in ('OBSRVTRY','TELESCOP')):continue
				preview=bool(hdu.get('OV_HV')) or 'helioviewer' in str(hdu.get('OV_SRC','')).lower()
				if not preview and not str(hdu.get('BUNIT') or '').strip():continue
				t=hdu.get('DATE-OBS') or hdu.get('DATE_OBS')
				if t and len(str(t))==10 and (hdu.get('TIME-OBS') or hdu.get('TIME_OBS')):t=str(t).replace('/','-')+'T'+str(hdu.get('TIME-OBS') or hdu.get('TIME_OBS'))
				if t:rows.append((Time(t),p,hdu))
		except (OSError,ValueError,TypeError):continue
	return sorted(rows,key=lambda row:row[0].unix)

def product_groups(rows):
	"""Group inventory rows into product families that may be differenced together."""
	groups={}
	for row in rows:
		_,path,h=row
		fields=('OV_SRC','OV_CALSR','OV_CAL','OV_FULL','OV_NORM','LEVEL','DATA_LEV','SERIES','INSTRUME','POLAR','FILTER','BUNIT')
		meta=tuple(str(h.get(k,'')).strip().lower() for k in fields)
		hv=bool(h.get('OV_HV')) or 'helioviewer' in meta[0]
		if hv:
			# All Helioviewer-derived FITS are one display product, whichever code
			# wrote them: canonical tags, keyed only by instrument (field 8).
			meta=('helioviewer jp2','','helioviewer visualization; not photometric fits','','true','','','',meta[8],'','','ct')
		origin='' if any(meta[:2]) else str(path.parent)
		exposure=h.get('EXPTIME') if not hv and meta[-1] in ('dn','adu','ct','count','counts') else None
		# Ignore sub-0.1 ms header jitter when grouping raw-count exposures.
		if exposure is not None:exposure=round(float(exposure),4)
		groups.setdefault((hv,origin,meta,exposure),[]).append(row)
	return groups

def family_label(key,group):
	hv,origin,meta,exposure=key
	name=meta[0] or meta[1] or meta[2] or origin or 'header provenance'
	return f"{name} | BUNIT={meta[-1]}"+(f" | EXPTIME={exposure}" if exposure is not None else '')

def group_coverage(group,start,end,targets,base_time,reference_lookback,max_image_dt):
	"""Selected event rows, uncovered target indices, predecessors and base for one family."""
	window=[r for r in group if start<=r[0]<=end]
	selected={}
	if window:
		offset=np.array([(r[0]-start).sec for r in window])
		for t in targets:
			i=int(np.argmin(abs(offset-t)))
			if abs(offset[i]-t)<=max_image_dt:selected[i]=window[i]
	missing=[k for k,t in enumerate(targets) if not window or np.min(abs(np.array([(r[0]-start).sec for r in window])-t))>max_image_dt]
	selected=[selected[i] for i in sorted(selected)]
	if not selected:return [],missing,[],None
	first=selected[0][0]
	previous=[r for r in group if r[0]<first and (first-r[0]).sec<=reference_lookback]
	bases=[r for r in group if r[0]<=first and abs((r[0]-base_time).sec)<=max_image_dt]
	base=min(bases,key=lambda row:abs((row[0]-base_time).sec)) if bases else None
	return selected,missing,previous,base

def write_coverage_report(out,report,missing_times):
	"""coverage_report.csv: one row per product family; USED marks the chosen one."""
	import csv
	with (Path(out)/'coverage_report.csv').open('w',newline='') as f:
		w=csv.writer(f)
		w.writerow(['family','kind','files','covered_slots','has_predecessor','has_base','usable','chosen'])
		w.writerows(report)
		w.writerow([]);w.writerow(['uncovered_slots_in_chosen_family']+list(missing_times))
	for r in report:print('  coverage:',r[1],r[3],'slots',('<- USED' if r[-1] else ''),'|',r[0],flush=True)

def family_priority(key,prefer_helioviewer=True):
	"""0 Level-1 (MSB), then Helioviewer and Level-0.5 in the order the route asks for."""
	if key[0]:return 1 if prefer_helioviewer else 2
	if key[2][-1] in ('msb','bsun','b_sun','meansolarbrightness'):return 0
	return 2 if prefer_helioviewer else 1

def make_motion_cache(folder,detector,spacecraft,start,end,cadence,base_time,reference_lookback,max_image_dt,max_gap,npix,r_inner,r_outer,out,prefer_helioviewer=True):
	"""Select local exposures, align angular grids, and compute adjacent TV-L1 flow."""
	out=Path(out);out.mkdir(parents=True,exist_ok=True)
	start,end,base_time=map(Time,(start,end,base_time))
	if end<start or base_time>start or cadence<=0 or max_image_dt<0 or reference_lookback<=0 or npix<64:
		raise ValueError('Check interval, cadence, base-reference time and npix.')
	targets=np.arange(0,(end-start).sec+1e-6,cadence)
	key,selected,previous,base,_=select_family(folder,detector,spacecraft,start,end,targets,base_time,
		reference_lookback,max_image_dt,out,prefer_helioviewer)
	return _build_cache(key,selected,previous,base,detector,spacecraft,max_gap,npix,r_inner,r_outer,out)

def select_family(folder,detector,spacecraft,start,end,targets,base_time,reference_lookback,max_image_dt,out,prefer_helioviewer=True):
	"""Choose one product family (route + coverage); write coverage_report.csv in out.

	Returns (key, selected rows, predecessor rows, base row, uncovered target indices).
	"""
	rows=inventory(folder,detector,spacecraft)
	if not rows:raise FileNotFoundError(f'No {detector} FITS with BUNIT or Helioviewer provenance under {folder}.')
	groups=product_groups(rows)
	best=None;report=[]
	for key,group in groups.items():
		selected,missing,previous,base=group_coverage(group,start,end,targets,base_time,reference_lookback,max_image_dt)
		ok=bool(selected) and bool(previous) and base is not None
		kind=('Level-1','Helioviewer','Level-0.5')[[0,1,2][family_priority(key,True)]]
		report.append([family_label(key,group),kind,len(group),
			f'{len(targets)-len(missing)}/{len(targets)}',bool(previous),base is not None,ok,''])
		if not ok:continue
		# Route Level-1 -> Helioviewer -> Level-0.5 (Level-0.5 before Helioviewer when
		# prefer_helioviewer=False): the first family covering every slot wins;
		# otherwise the one covering most slots, ties by that priority.
		covered=len(targets)-len(missing)
		score=(covered==len(targets),-family_priority(key,prefer_helioviewer) if covered==len(targets) else covered,
			-family_priority(key,prefer_helioviewer))
		if best is None or score>best[0]:best=(score,key,selected,previous,base,missing,len(report)-1)
	if best is None:
		write_coverage_report(out,report,[])
		raise ValueError('No single source/unit group has event frames, a preceding exposure and a base exposure near base_diff_DT.')
	report[best[6]][-1]='USED'
	write_coverage_report(out,report,[(start+targets[i]*u.s).isot for i in best[5]])
	_,key,selected,previous,base,missing,_=best
	return key,selected,previous,base,missing

def _build_cache(key,selected,previous,base,detector,spacecraft,max_gap,npix,r_inner,r_outer,out):
	"""Align the chosen family on one angular grid and save differences and TV-L1 flow."""
	print(f'{detector}: using one FITS group ({key[2][0] or key[2][1] or key[2][2] or key[1] or "header provenance"}, BUNIT={key[2][-1]}, {len(selected)} event exposures)',flush=True)
	ref=load_map(selected[0][1])
	preview=lambda m:bool(m.meta.get('ov_hv')) or 'helioviewer' in str(m.meta.get('ov_src','')).lower()
	unit=str(ref.unit);pol=ref.meta.get('polar')
	if not preview(ref) and ref.unit is None:raise ValueError(f'{selected[0][1]} has unrecognized BUNIT={ref.meta.get("bunit")!r}. Use prepared FITS with a known BUNIT.')
	if max(ref.data.shape)>npix:ref=ref.resample([npix,npix]*u.pix)
	if np.abs(ref.rotation_matrix-np.eye(2)).max()>1e-3:ref=ref.rotate()
	yy,xx=np.indices(ref.data.shape)
	coords=ref.pixel_to_world(xx*u.pix,yy*u.pix)
	xr=coords.Tx.to_value(u.arcsec)/ref.rsun_obs.to_value(u.arcsec)
	yr=coords.Ty.to_value(u.arcsec)/ref.rsun_obs.to_value(u.arcsec)
	radius=np.hypot(xr,yr)
	valid0=(radius>=r_inner)&(radius<=r_outer)
	def align(row):
		t,p,_=row;m=load_map(p)
		if str(m.unit)!=unit or preview(m)!=preview(ref) or m.meta.get('polar')!=pol:
			raise ValueError(f'Mixed image units, source type or polarizer at {p}. Use a homogeneous FITS sequence.')
		if not preview(m) and unit.lower() in ('dn','adu','ct','count','counts'):
			if not np.isclose(m.exposure_time.to_value(u.s),ref.exposure_time.to_value(u.s),rtol=0,atol=1e-4):
				raise ValueError('Raw count exposures differ; provide consistently prepared FITS.')
		# Align by helioprojective angle in each exposure, not photospheric reprojection.
		from astropy.coordinates import SkyCoord
		w=SkyCoord(coords.Tx,coords.Ty,frame=m.coordinate_frame)
		sx,sy=m.world_to_pixel(w)
		img=ndi.map_coordinates(np.asarray(m.data,dtype='float32'),[sy.value,sx.value],order=1,mode='constant',cval=np.nan)
		valid=np.isfinite(img)&valid0
		obs={}
		if spacecraft:
			if not all(k in m.meta for k in ('hgln_obs','hglt_obs','dsun_obs')):
				raise ValueError(f'{p} needs verified Stonyhurst HGLN_OBS, HGLT_OBS and DSUN_OBS for STEREO.')
			o=m.observer_coordinate.transform_to(frames.HeliographicStonyhurst(obstime=t))
			obs=dict(observer=dict(hgln_obs=o.lon.deg,hglt_obs=o.lat.deg,dsun_obs=o.radius.to_value(u.m)))
		return np.where(valid,img,0),valid,obs
	ref.save(out/'reference_grid.fits',overwrite=True)
	np.savez_compressed(out/'coordinates.npz',x_rsun=xr,y_rsun=yr)
	b,bv,_=align(base)
	prev=previous[-1];a,av,_=align(prev)
	times={}
	for row in selected:
		t,p,_=row;im,iv,obs=align(row)
		dt=(t-prev[0]).sec;valid=iv&av&bv
		if not valid.any():raise ValueError(f'No valid overlap at {t.isot}. Check image WCS and radial mask.')
		dx,dy,q=flow(a,im,valid) if dt<=max_gap else (np.zeros_like(im),np.zeros_like(im),np.zeros_like(valid))
		tag=t.strftime('structures_%Y%m%d_%H%M%S')
		np.savez_compressed(out/(tag+'.npz'),valid=valid,dx=dx,dy=dy,quality=q,base_difference=im-b,running_difference=im-a)
		times[tag]=dict(datetime=t.isot,previous_time=prev[0].isot,source=str(p),base_source=str(base[1]),base_time=base[0].isot,**obs)
		print(f'{t.isot} | Δt={dt/60:.1f} min',flush=True)
		prev,a,av=row,im,iv
	(out/'times.json').write_text(json.dumps(times,indent=2))
	return out
