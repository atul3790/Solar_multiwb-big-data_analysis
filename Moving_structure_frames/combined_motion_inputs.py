"""COR2 + HI1 motion cache: both instruments on one helioprojective grid.

The grid is a plate-carree helioprojective (HPLN/HPLT-CAR) map seen from the
STEREO spacecraft, wide enough for the COR2 disk (r <= r_outer) and the whole
HI1 field. COR2 owns r <= seam_rsun; HI1 owns the pixels beyond it.

Each instrument is differenced only against itself (COR2 against COR2, HI1
against HI1, one product family each) and then divided by its own robust noise
scale (median over frames of 1.4826*MAD of its running differences). The saved
base/running differences are therefore in units of that noise ("sigma"), so
one detection threshold applies across the seam. TV-L1 flow is computed
separately inside each zone; HI1 displacements are rescaled from the HI1 pair
duration to the COR2 pair duration so the renderer's speeds stay correct.

HI1 runs at a 40-min cadence: each COR2 slot uses the nearest HI1 sum within
hi1_max_image_dt, so one HI1 pair can repeat over several COR2 slots.

Output files match motion_inputs.make_motion_cache (reference_grid.fits,
coordinates.npz, structures_*.npz, times.json) plus zone masks, so
track_coronagraph_flows.render_frames draws them unchanged.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
import sunpy.map
from scipy import ndimage as ndi
from sunpy.coordinates import frames
from datamine_calib_helper import load_map,_hi_level
from motion_inputs import flow,select_family,spacecraft_id
from puddle_analysis import usable_grid

FITS_SUFFIX=('.fits','.fit','.fts','.fits.gz','.fit.gz','.fts.gz')

def hi_inventory(folder,detector,spacecraft):
	"""(mid-exposure Time, path, level) for HI science sums under folder.

	Level '0.5': SSC sums (N_IMAGES>1, >=512 px). Level '1': hi_prep products.
	The same exposure saved twice (raw/ and maps/) is listed once.
	"""
	rows={}
	for p in sorted(Path(folder).expanduser().rglob('*')):
		if not p.is_file() or not p.name.lower().endswith(FITS_SUFFIX):continue
		try:
			h=fits.getheader(p)
			if str(h.get('DETECTOR','')).strip().upper()!=detector:continue
			if not any(spacecraft_id(h.get(k,''))=='STEREO'+spacecraft for k in ('OBSRVTRY','TELESCOP')):continue
			level=_hi_level({k.lower():v for k,v in h.items()})
			if level=='0.5' and (int(h.get('N_IMAGES',1) or 1)<=1 or min(int(h.get('NAXIS1',0)),int(h.get('NAXIS2',0)))<512):continue
			t=Time(h.get('DATE-AVG') or h.get('DATE-OBS'))
			rows.setdefault((level,t.isot[:19]),(t,p,level))
		except (OSError,ValueError,TypeError):continue
	return sorted(rows.values(),key=lambda r:r[0].unix)

def choose_hi_family(rows,slot_times,tolerance):
	"""Level-1 if it covers every slot; otherwise the level covering most slots (ties: Level-1)."""
	best=None
	for level in ('1','0.5'):
		group=[r for r in rows if r[2]==level]
		covered=sum(any(abs((r[0]-t).sec)<=tolerance for r in group) for t in slot_times)
		score=(covered==len(slot_times),covered,level=='1')
		print(f'  HI coverage: Level-{level} {covered}/{len(slot_times)} slots',flush=True)
		if group and (best is None or score>best[0]):best=(score,level,group)
	if best is None:raise FileNotFoundError('No HI science FITS found for the combined cache.')
	return best[1],best[2]

def combined_grid(cor2,hi,r_outer,npix,pad_rsun=1.):
	"""CAR helioprojective map covering the COR2 disk to r_outer and the HI field."""
	rs=cor2.rsun_obs.to_value(u.deg)
	ny,nx=hi.data.shape
	e=np.arange(0,max(nx,ny),8)
	px=np.r_[e[e<nx],np.full((e<ny).sum(),nx-1),e[e<nx],np.zeros((e<ny).sum())]
	py=np.r_[np.zeros((e<nx).sum()),e[e<ny],np.full((e<nx).sum(),ny-1),e[e<ny]]
	c=hi.pixel_to_world(px*u.pix,py*u.pix)
	tx,ty=c.Tx.to_value(u.deg),c.Ty.to_value(u.deg)
	edge=(r_outer+pad_rsun)*rs
	xmin,xmax=min(tx.min(),-edge),max(tx.max(),edge)
	ymin,ymax=min(ty.min(),-edge),max(ty.max(),edge)
	scale=max(xmax-xmin,ymax-ymin)/npix
	nx,ny=int(np.ceil((xmax-xmin)/scale)),int(np.ceil((ymax-ymin)/scale))
	o=cor2.observer_coordinate.transform_to(frames.HeliographicStonyhurst(obstime=cor2.date))
	header={'naxis':2,'naxis1':nx,'naxis2':ny,'ctype1':'HPLN-CAR','ctype2':'HPLT-CAR',
		'cunit1':'deg','cunit2':'deg','cdelt1':scale,'cdelt2':scale,'crval1':0.,'crval2':0.,
		'crpix1':-xmin/scale+1,'crpix2':-ymin/scale+1,'date-obs':cor2.date.isot,
		'hgln_obs':o.lon.deg,'hglt_obs':o.lat.deg,'dsun_obs':o.radius.to_value(u.m),
		'rsun_ref':cor2.meta.get('rsun_ref',695700000.),'obsrvtry':cor2.meta.get('obsrvtry','STEREO_A'),
		'telescop':'STEREO','instrume':'SECCHI','detector':'COR2+HI1','bunit':'',
		'ov_note':'COR2 inside seam, HI1 outside; differences in per-instrument noise units'}
	return sunpy.map.Map(np.zeros((ny,nx),dtype='float32'),header)

def sample(m,data,coords):
	"""Image values at the grid's helioprojective angles (same spacecraft), NaN outside."""
	w=SkyCoord(coords.Tx,coords.Ty,frame=m.coordinate_frame)
	sx,sy=m.world_to_pixel(w)
	return ndi.map_coordinates(np.asarray(data,dtype='float32'),[sy.value,sx.value],order=1,mode='constant',cval=np.nan)

def zone_flow(a,b,valid,pad=6):
	"""motion_inputs.flow on the bounding box of valid; zeros elsewhere."""
	dx=np.zeros(a.shape,'float32');dy=np.zeros(a.shape,'float32');q=np.zeros(a.shape,bool)
	if not valid.any():return dx,dy,q
	ys,xs=np.nonzero(valid)
	y0,y1=max(ys.min()-pad,0),min(ys.max()+pad+1,a.shape[0])
	x0,x1=max(xs.min()-pad,0),min(xs.max()+pad+1,a.shape[1])
	c=(slice(y0,y1),slice(x0,x1))
	fx,fy,fq=flow(np.where(valid,a,0)[c],np.where(valid,b,0)[c],valid[c])
	dx[c],dy[c],q[c]=fx,fy,fq
	return dx,dy,q

def robust_sigma(values):
	values=values[np.isfinite(values)]
	if not values.size:return np.nan
	return 1.4826*float(np.median(np.abs(values-np.median(values))))

def make_combined_cache(cor2_folder,hi_folder,spacecraft,start,end,cadence,base_time,reference_lookback,
	max_image_dt,max_gap,npix,r_inner,r_outer,out,prefer_helioviewer=True,hi_detector='HI1',
	hi_max_image_dt=1500.,hi_lookback=3600.,hi_max_gap=5400.,hi_star_filter_px=5,hi_edge_px=8,
	seam_rsun=None,spoke_pa=0.,spoke_halfwidth=0.,spoke_arrow_buffer_deg=0.,noise_model='ring',ring_step_rsun=3.,ring_strength=.7,hi_flow_clean=True):
	"""Build the COR2+HI1 cache in out; return out.

	noise_model: 'ring' (default) then divides each zone by its noise-versus-
	radius profile (see apply_noise_model); 'global' keeps one scale per instrument.

	cor2_folder/hi_folder: searched recursively (may be the same STEREO root).
	r_inner/r_outer: COR2 radial limits (Rsun). seam_rsun: COR2/HI1 boundary
	(default r_outer). Spoke settings mask the COR2 zone only.
	hi_max_image_dt: largest HI-to-COR2 slot offset (s). hi_lookback: largest gap
	to the HI predecessor (s). hi_max_gap: longer HI pairs give no HI flow.
	hi_star_filter_px: median filter (native HI pixels) that removes stars; 0 off.
	hi_edge_px: grid pixels trimmed from the HI field edge.
	"""
	out=Path(out);out.mkdir(parents=True,exist_ok=True)
	start,end,base_time=map(Time,(start,end,base_time))
	seam=r_outer if seam_rsun is None else seam_rsun
	targets=np.arange(0,(end-start).sec+1e-6,cadence)
	#### COR2: one family, as in the single-instrument pipeline ####
	key,selected,previous,base,_=select_family(cor2_folder,'COR2',spacecraft,start,end,targets,base_time,
		reference_lookback,max_image_dt,out,prefer_helioviewer)
	(out/'coverage_report.csv').replace(out/'coverage_report_COR2.csv')
	preview=lambda m:bool(m.meta.get('ov_hv')) or 'helioviewer' in str(m.meta.get('ov_src','')).lower()
	print(f'COR2: {len(selected)} event exposures ({"Helioviewer" if key[0] else key[2][-1]})',flush=True)
	#### HI: one family, nearest sum per COR2 slot ####
	hi_rows=hi_inventory(hi_folder,hi_detector,spacecraft)
	level,hi_rows=choose_hi_family(hi_rows,[r[0] for r in selected],hi_max_image_dt)
	print(f'{hi_detector}: using Level-{level} ({len(hi_rows)} sums)',flush=True)
	def nearest_hi(t,before=False):
		pool=[r for r in hi_rows if abs((r[0]-t).sec)<=hi_max_image_dt and (not before or r[0]<=t)]
		return min(pool,key=lambda r:abs((r[0]-t).sec)) if pool else None
	def hi_predecessor(row):
		pool=[r for r in hi_rows if r[0]<row[0] and (row[0]-r[0]).sec<=hi_lookback]
		return pool[-1] if pool else None
	hi_base=nearest_hi(base_time)
	if hi_base is None:print(f'{hi_detector}: no sum within {hi_max_image_dt:.0f} s of base_diff_DT; HI base difference left blank',flush=True)
	@__import__('functools').lru_cache(maxsize=4)
	def hi_image(path):
		m=load_map(path);d=np.asarray(m.data,dtype=float)
		if level=='0.5':d=d/m.exposure_time.to_value(u.s)  # DN -> DN/s (sum exposures differ by ms)
		if hi_star_filter_px and hi_star_filter_px>1:d=ndi.median_filter(d,size=int(hi_star_filter_px))
		return m,d
	#### Grid ####
	cor2_ref=load_map(selected[0][1])
	first_hi=next((nearest_hi(r[0]) for r in selected if nearest_hi(r[0]) is not None),hi_rows[0])
	grid=combined_grid(cor2_ref,hi_image(first_hi[1])[0],r_outer,npix)
	ny,nx=grid.data.shape
	yy,xx=np.indices((ny,nx))
	coords=grid.pixel_to_world(xx*u.pix,yy*u.pix)
	rs=cor2_ref.rsun_obs.to_value(u.arcsec)
	xr=coords.Tx.to_value(u.arcsec)/rs;yr=coords.Ty.to_value(u.arcsec)/rs
	radius=np.hypot(xr,yr)
	ring=(radius>=r_inner)&(radius<=min(r_outer,seam))
	cor2_zone=usable_grid(xr,yr,ring,spoke_pa,spoke_halfwidth,0) if spoke_halfwidth>0 else ring
	cor2_arrow=usable_grid(xr,yr,ring,spoke_pa,spoke_halfwidth+spoke_arrow_buffer_deg,0) if spoke_halfwidth>0 else ring
	hi_cover=np.isfinite(sample(*hi_image(first_hi[1]),coords))
	hi_zone=ndi.binary_erosion(hi_cover,iterations=int(hi_edge_px))&(radius>seam)
	print(f'Grid {nx}x{ny} px, {grid.scale[0].to_value(u.deg/u.pix)/cor2_ref.rsun_obs.to_value(u.deg):.3f} Rsun/px; '
		f'x {xr.min():.1f}..{xr.max():.1f}, y {yr.min():.1f}..{yr.max():.1f} Rsun',flush=True)
	grid.save(out/'reference_grid.fits',overwrite=True)
	np.savez_compressed(out/'coordinates.npz',x_rsun=xr,y_rsun=yr)
	#### COR2 alignment, as motion_inputs ####
	unit,pol=str(cor2_ref.unit),cor2_ref.meta.get('polar')
	def cor2_image(row):
		t,p,_=row;m=load_map(p)
		if str(m.unit)!=unit or preview(m)!=preview(cor2_ref) or m.meta.get('polar')!=pol:
			raise ValueError(f'Mixed COR2 units, source type or polarizer at {p}.')
		img=sample(m,m.data,coords)
		o=m.observer_coordinate.transform_to(frames.HeliographicStonyhurst(obstime=t))
		return img,dict(hgln_obs=o.lon.deg,hglt_obs=o.lat.deg,dsun_obs=o.radius.to_value(u.m))
	b,_=cor2_image(base)
	prev=previous[-1];a,_=cor2_image(prev)
	hb=sample(*hi_image(hi_base[1]),coords) if hi_base is not None else None
	hi_flows={};times={};sig={'COR2':[],'HI':[]}
	for row in selected:
		t,p,_=row;im,obs=cor2_image(row)
		dt=(t-prev[0]).sec
		cv=cor2_zone&np.isfinite(im)&np.isfinite(a)&np.isfinite(b)
		run=np.where(cv,im-a,0.).astype('float32');bas=np.where(cv,im-b,0.).astype('float32')
		if dt<=max_gap:dx,dy,q=zone_flow(np.nan_to_num(a),np.nan_to_num(im),cv)
		else:dx,dy,q=(np.zeros(cv.shape,'float32'),np.zeros(cv.shape,'float32'),np.zeros(cv.shape,bool))
		arrow=cor2_arrow&cv
		hrow=nearest_hi(t);hprev=hi_predecessor(hrow) if hrow is not None else None
		# HI1 base difference needs only this sum and the base sum; the running
		# difference and flow also need the predecessor. Kept separate so a
		# missing predecessor does not blank HI1 in the base-difference frames.
		hv=np.zeros(cv.shape,bool);hb_ok=np.zeros(cv.shape,bool);hinfo={}
		if hrow is not None:
			h=sample(*hi_image(hrow[1]),coords)
			if hb is not None:
				hb_ok=hi_zone&np.isfinite(h)&np.isfinite(hb)
				bas=np.where(hb_ok,h-hb,bas).astype('float32')
			hinfo=dict(hi_source=str(hrow[1]),hi_time=hrow[0].isot,hi_previous_time=None,hi_pair_seconds=None,
				hi_base_time=hi_base[0].isot if hi_base is not None else None,hi_level=level)
		if hrow is not None and hprev is not None:
			hp=sample(*hi_image(hprev[1]),coords)
			hv=hi_zone&np.isfinite(h)&np.isfinite(hp)
			run=np.where(hv,h-hp,run).astype('float32')
			hdt=(hrow[0]-hprev[0]).sec
			pair=(str(hprev[1]),str(hrow[1]))
			if pair not in hi_flows:
				hi_flows.clear()
				hi_flows[pair]=zone_flow(np.nan_to_num(hp),np.nan_to_num(h),hv) if hdt<=hi_max_gap else None
			hf=hi_flows[pair]
			if hf is not None and dt<=max_gap:
				f=dt/hdt  # HI displacement over the COR2 pair duration
				dx=np.where(hv,hf[0]*f,dx);dy=np.where(hv,hf[1]*f,dy);q=np.where(hv,hf[2],q)
			arrow=arrow|hv
			sig['HI'].append(robust_sigma((h-hp)[hv]))
			hinfo.update(hi_previous_time=hprev[0].isot,hi_pair_seconds=hdt)
		sig['COR2'].append(robust_sigma((im-a)[cv]))
		valid=cv|hv|hb_ok
		if not valid.any():raise ValueError(f'No valid COR2/HI overlap at {t.isot}.')
		tag=t.strftime('structures_%Y%m%d_%H%M%S')
		np.savez_compressed(out/(tag+'.npz'),valid=valid,valid_running=cv|hv,valid_base=cv|hb_ok,
			dx=dx.astype('float32'),dy=dy.astype('float32'),quality=q&(cv|hv),
			base_difference=bas,running_difference=run,cor2_zone=cv,hi_zone=hv|hb_ok,arrow_zone=arrow)
		times[tag]=dict(datetime=t.isot,previous_time=prev[0].isot,source=str(p),base_source=str(base[1]),
			base_time=base[0].isot,observer=obs,**hinfo)
		hi_txt=(' | no HI1 sum' if not hinfo else f' | HI1 {hinfo["hi_time"][11:16]} '+
			(f'Δt={hinfo["hi_pair_seconds"]/60:.0f} min' if hinfo['hi_pair_seconds'] else 'no predecessor (base only)'))
		print(f'{t.isot} | COR2 Δt={dt/60:.1f} min'+hi_txt,flush=True)
		prev,a=row,im
	#### Per-instrument noise units ####
	scale={k:float(np.nanmedian(v)) if len(v) and np.isfinite(v).any() else 1. for k,v in sig.items()}
	for k in scale:
		if not np.isfinite(scale[k]) or scale[k]<=0:scale[k]=1.
	for tag in times:
		with np.load(out/(tag+'.npz')) as z:d={k:z[k] for k in z.files}
		for name in ('base_difference','running_difference'):
			d[name]=np.where(d['cor2_zone'],d[name]/scale['COR2'],np.where(d['hi_zone'],d[name]/scale['HI'],0.)).astype('float32')
		np.savez_compressed(out/(tag+'.npz'),**d)
	(out/'combined_scales.json').write_text(json.dumps(dict(COR2_sigma=scale['COR2'],COR2_unit='ct (Helioviewer)' if key[0] else unit,
		HI_sigma=scale['HI'],HI_unit='DN/s' if level=='0.5' else 'Level-1 BUNIT',HI_level=level,seam_rsun=seam,
		note='Differences saved divided by these noise scales'),indent=2))
	(out/'times.json').write_text(json.dumps(times,indent=2))
	print(f'Noise scales: COR2 {scale["COR2"]:.4g}, {hi_detector} {scale["HI"]:.4g} (differences now in sigma units)',flush=True)
	apply_noise_model(out,noise_model,ring_step_rsun,ring_strength)
	if hi_flow_clean:hi_clean_flow(out,hi_max_gap=hi_max_gap)
	return out

ZONES=('cor2_zone','hi_zone')

DIFFS=(('running_difference','valid_running'),('base_difference','valid_base'))

def ring_noise_profile(out,step=3.,key='running_difference',valid_key='valid_running'):
	"""Per zone: radius-bin centres (Rsun) and the median over all frames of the
	robust sigma (1.4826*MAD) of difference `key` in each ring.

	A time median, so a CME filling a ring in some frames does not raise that
	ring's noise level. Base and running differences get separate profiles:
	base differences carry hours of background drift and are 2-4x noisier.
	Rings with fewer than 200 valid pixels are skipped.
	"""
	out=Path(out)
	with np.load(out/'coordinates.npz') as c:R=np.hypot(c['x_rsun'],c['y_rsun'])
	edges=np.arange(0,np.ceil(R.max()/step)*step+step,step);centres=(edges[:-1]+edges[1:])/2
	rows={z:[] for z in ZONES}
	for tag in json.loads((out/'times.json').read_text()):
		with np.load(out/(tag+'.npz')) as z:
			d=z[key];v=z[valid_key] if valid_key in z.files else z['valid']
			for name in ZONES:
				m=z[name]&v;row=[]
				for a,b in zip(edges[:-1],edges[1:]):
					s=m&(R>=a)&(R<b)
					row.append(robust_sigma(d[s]) if s.sum()>=200 else np.nan)
				rows[name].append(row)
	profile={}
	for name,r in rows.items():
		with warnings.catch_warnings():
			warnings.simplefilter('ignore',RuntimeWarning)  # rings outside this zone are all-NaN
			sig=np.nanmedian(np.array(r,dtype=float),axis=0) if r else np.full(len(centres),np.nan)
		ok=np.isfinite(sig)&(sig>0)
		profile[name]=(centres[ok].tolist(),sig[ok].tolist())
	return profile

def _profile_map(R,zones,profile,valid):
	"""Per-pixel noise from the zone profiles (1 where no profile)."""
	noise=np.ones(R.shape)
	for name in ZONES:
		r,sig=profile.get(name,([],[]))
		m=zones[name]&valid
		if len(r) and m.any():noise[m]=np.interp(R[m],r,sig)
	return noise

def apply_noise_model(out,noise_model='ring',step=3.,strength=.7):
	"""Set the cache's noise normalisation in place: 'ring' or 'global'.

	'global': differences divided by one robust scale per instrument (as built).
	'ring': each difference (running, base) additionally divided by its own
	zone noise-versus-radius profile, so every pixel is in local-noise units and
	faint outer HI1 structures are detected as readily as bright inner ones.
	strength (0-1) applies profile**strength: 1 = full local-noise units,
	0 = global. Below 1 the inner HI1 edge, whose ring noise includes real
	structure variance, keeps more of its bright features. Applied profiles are
	stored in combined_scales.json (ring_profiles), so the change is reversible
	and re-applying the same model does nothing. Works on existing caches.
	"""
	if noise_model not in ('ring','global'):raise ValueError("noise_model must be 'ring' or 'global'.")
	out=Path(out);info=json.loads((out/'combined_scales.json').read_text())
	current=info.get('noise_model','global')
	same=(current==noise_model and (noise_model=='global' or (info.get('ring_step_rsun')==step and
		info.get('ring_strength',1.)==strength and 'ring_profiles' in info)))
	if same:
		print(f'Noise model already {noise_model}',flush=True);return out
	with np.load(out/'coordinates.npz') as c:R=np.hypot(c['x_rsun'],c['y_rsun'])
	tags=list(json.loads((out/'times.json').read_text()))
	if current=='ring':
		# Undo what is stored, back to global units. Older caches stored one
		# 'ring_profile' (from running differences) applied to both differences.
		old=info.get('ring_profiles') or {name:info.get('ring_profile') for name,_ in DIFFS}
		if any(v is None for v in old.values()):raise ValueError('Cache marked ring-normalised but has no stored profile.')
		power=float(info.get('ring_strength',1.))
		for tag in tags:
			with np.load(out/(tag+'.npz')) as z:d={k:z[k] for k in z.files}
			for name,_ in DIFFS:d[name]=(d[name]*_profile_map(R,d,old[name],d['valid'])**power).astype('float32')
			np.savez_compressed(out/(tag+'.npz'),**d)
		for k in ('ring_profile','ring_profiles','ring_step_rsun','ring_strength'):info.pop(k,None)
		info['noise_model']='global'
	if noise_model=='ring':
		profiles={name:ring_noise_profile(out,step,name,vkey) for name,vkey in DIFFS}
		for tag in tags:
			with np.load(out/(tag+'.npz')) as z:d={k:z[k] for k in z.files}
			for name,_ in DIFFS:d[name]=(d[name]/_profile_map(R,d,profiles[name],d['valid'])**float(strength)).astype('float32')
			np.savez_compressed(out/(tag+'.npz'),**d)
		info.update(noise_model='ring',ring_step_rsun=step,ring_strength=strength,ring_profiles=profiles,
			note='Each difference divided by the per-instrument scale, then by its ring_profiles entry**ring_strength')
		for diff,prof in profiles.items():
			for name,(r,sig) in prof.items():
				if r:print(f'Ring noise {diff} {name}: {sig[0]:.3g} at {r[0]:.0f} Rsun -> {sig[-1]:.3g} at {r[-1]:.0f} Rsun',flush=True)
	(out/'combined_scales.json').write_text(json.dumps(info,indent=2))
	print(f'Noise model set to {noise_model}'+(f' (strength {strength})' if noise_model=='ring' else '')+f': {out}',flush=True)
	return out

def star_drift(m_from,m_to,grid,points=((-30,0),(-60,0))):
	"""Star-field shift (dx, dy) in grid pixels between two HI exposures.

	Stars are fixed in ICRS; the spacecraft's orbital motion moves them across
	the Sun-centred frame (~1 px per 40 min for HI1-A on a 0.1 Rsun grid,
	sunward). Computed from the two exposures' observer positions.
	"""
	rs=grid.rsun_obs.to_value(u.arcsec);pxs=grid.scale[0].to_value(u.arcsec/u.pix)/rs;v=[]
	for X,Y in points:
		s=SkyCoord(X*rs*u.arcsec,Y*rs*u.arcsec,distance=1e6*u.AU,
			frame=frames.Helioprojective(observer=m_from.observer_coordinate,obstime=m_from.date)).transform_to('icrs')
		s=SkyCoord(s.ra,s.dec,distance=1e6*u.AU,frame='icrs').transform_to(
			frames.Helioprojective(observer=m_to.observer_coordinate,obstime=m_to.date))
		v.append((s.Tx.to_value(u.arcsec)/rs-X,s.Ty.to_value(u.arcsec)/rs-Y))
	return np.mean(v,axis=0)/pxs

def hi_clean_flow(out,star_filter_px=3,bg_percentile=10.,neighbours=3,hi_max_gap=5400.,force=False):
	"""Recompute the HI flow of a cache from background-cleaned HI images.

	Raw HI intensity is dominated by stars and diffuse starlight, which drift
	sunward with the spacecraft orbit; TV-L1 on it tracks that drift, so HI
	arrows pointed sunward even where structures move outward. Here, for every
	HI sum used by the cache:
	  1. F-corona (Sun-fixed): the bg_percentile percentile over all sums is subtracted;
	  2. star field: the median of the +-neighbours sums, each shifted by the
	     star drift into this sum's star frame, is subtracted;
	then TV-L1 runs on consecutive cleaned sums inside the HI zone. Displacements
	are rescaled to each frame's COR2 pair duration as before. COR2 flow and all
	differences are untouched. Works on existing caches; recorded as
	hi_flow='clean' in combined_scales.json, so repeating it does nothing.
	"""
	out=Path(out);info=json.loads((out/'combined_scales.json').read_text())
	if info.get('hi_flow')=='clean' and not force:
		print('HI flow already from cleaned images',flush=True);return out
	times=json.loads((out/'times.json').read_text())
	srcs=[it['hi_source'] for it in times.values() if it.get('hi_source')]
	if not srcs:print('No HI frames in cache',flush=True);return out
	level=info.get('HI_level','0.5')
	folder=Path(srcs[0]).parent
	if folder.name=='maps':folder=folder.parent  # also raw/ (e.g. sums before the first saved map)
	spacecraft=spacecraft_id(load_map(out/'reference_grid.fits').meta.get('obsrvtry','STEREO_A'))[-1]
	rows=[r for r in hi_inventory(folder,'HI1',spacecraft) if r[2]==level]
	if not rows:raise FileNotFoundError(f'No Level-{level} HI1 sums in {folder}')
	grid=load_map(out/'reference_grid.fits');ny,nx=grid.data.shape
	yy,xx=np.indices((ny,nx));coords=grid.pixel_to_world(xx*u.pix,yy*u.pix)
	with np.load(out/'coordinates.npz') as c:R=np.hypot(c['x_rsun'],c['y_rsun'])
	t0=min(Time(it['hi_previous_time']) for it in times.values() if it.get('hi_previous_time'))
	t1=max(Time(it['hi_time']) for it in times.values() if it.get('hi_time'))
	pad=(neighbours+1)*3*3600.
	rows=[r for r in rows if (t0-r[0]).sec<=pad and (r[0]-t1).sec<=pad]
	print(f'HI clean flow: {len(rows)} sums, F-corona p{bg_percentile:g}, star median of +-{neighbours}',flush=True)
	maps=[];stack=[]
	for t,p,_ in rows:
		m=load_map(p);d=np.asarray(m.data,dtype=float)
		if level=='0.5':d=d/m.exposure_time.to_value(u.s)
		if star_filter_px and star_filter_px>1:d=ndi.median_filter(d,size=int(star_filter_px))
		maps.append(m);stack.append(sample(m,d,coords))
	stack=np.array(stack,dtype='float32')
	with warnings.catch_warnings():
		warnings.simplefilter('ignore',RuntimeWarning)
		F=np.nanpercentile(stack,bg_percentile,axis=0) if len(stack)>=5 else np.zeros((ny,nx),'float32')
	E=stack-F
	index={r[0].isot[:19]:i for i,r in enumerate(rows)}
	cleaned={}
	def clean(i):
		if i not in cleaned:
			nb=[]
			for j in range(max(0,i-neighbours),min(len(rows),i+neighbours+1)):
				if j==i:continue
				dx,dy=star_drift(maps[j],maps[i],grid)
				nb.append(ndi.shift(np.nan_to_num(E[j]),(dy,dx),order=1,mode='nearest'))
			cleaned[i]=E[i]-np.median(nb,axis=0) if len(nb)>=2 else E[i]
		return cleaned[i]
	pairs={}
	for tag,it in times.items():
		if not it.get('hi_previous_time'):continue
		i,j=index.get(it['hi_time'][:19]),index.get(it['hi_previous_time'][:19])
		if i is None or j is None:
			warnings.warn(f'{tag}: HI sums not found for clean flow; HI flow left blank');i=j=None
		with np.load(out/(tag+'.npz')) as z:d={k:z[k] for k in z.files}
		hv=d['hi_zone']&(d['valid_running'] if 'valid_running' in d else d['valid'])
		dt=(Time(it['datetime'])-Time(it['previous_time'])).sec;hdt=float(it['hi_pair_seconds'])
		if i is not None and hdt<=hi_max_gap:
			if (j,i) not in pairs:
				pairs.clear();a,b=clean(j),clean(i)
				ok=hv&np.isfinite(a)&np.isfinite(b)
				pairs[(j,i)]=zone_flow(np.nan_to_num(a),np.nan_to_num(b),ok)
			fx,fy,fq=pairs[(j,i)];f=dt/hdt
			d['dx']=np.where(hv,fx*f,d['dx']).astype('float32');d['dy']=np.where(hv,fy*f,d['dy']).astype('float32')
			d['quality']=np.where(hv,fq,d['quality'])
		else:
			d['quality']=np.where(hv,False,d['quality'])
		np.savez_compressed(out/(tag+'.npz'),**d)
	info.update(hi_flow='clean',hi_flow_note=f'TV-L1 on HI sums minus F-corona (p{bg_percentile:g}) minus star-aligned median of +-{neighbours} sums')
	(out/'combined_scales.json').write_text(json.dumps(info,indent=2))
	print('HI flow recomputed from cleaned images',flush=True)
	return out
