"""Motion cache from the CDAW LASCO C2 PNG images (use_CDAW_pngs).

CDAW (https://cdaw.gsfc.nasa.gov/images/soho/lasco/YYYY/MM/DD/) serves, per C2
exposure, a running-difference image (*_lasc2rdf.png, ~11 grey levels) and a
total-intensity composite with an AIA 193 inset (*_lasc2aia193.png, blue colour
table). Neither has coordinates or physical units. This module downloads them
and writes a cache in the motion_inputs format, so the usual renderer draws it:
  running_difference  the CDAW running difference (grey level - mid grey);
  base_difference     intensity - intensity of the frame nearest base_diff_DT;
  dx, dy, quality     TV-L1 on the intensity images (motion_inputs.flow, with
                      flow_input and flow_bin), never on running differences.
Intensity is the luminance of the colour-mapped composite (monotonic along the
blue-white colour table; the brightest streamer cores saturate).
Geometry: Sun-centred, solar north up; the white 1 Rsun limb circle of the
running difference fixes the centre and scale (measured on the first frame).
The grid is the 512 x 512 PNG grid; the observer is SOHO (0.99 AU, Earth
longitude). Values are display units, so speeds are approximate and region
detection relies on the noise model ('ring' or 'global').

C2 only: CDAW has no C3 intensity composite. recompute_flow is not supported
for these caches (sources are PNGs).

Importable: list_cdaw, download_cdaw, cdaw_geometry, make_cdaw_cache.
"""
from pathlib import Path
import json,re,time,urllib.request
import numpy as np
import astropy.units as u
from astropy.time import Time
from scipy import optimize
from PIL import Image
import sunpy.map
from sunpy.coordinates import sun
from motion_inputs import flow,flow_pair,check_flow_input,auto_flow_bin,_set_state

URL='https://cdaw.gsfc.nasa.gov/images/soho/lasco/{:%Y/%m/%d}/'
PRODUCTS={'C2':('lasc2rdf','lasc2aia193')}
TEXT_ROWS=486  # PNG rows from here down hold the time labels

def _get(url,tries=4):
	"""Bytes at url; retries dropped connections (CDAW resets some requests)."""
	for k in range(tries):
		try:
			with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=60) as r:return r.read()
		except OSError:
			if k==tries-1:break
			time.sleep(2*(k+1))
	import subprocess  # CDAW keeps resetting some Python connections; curl usually gets through
	r=subprocess.run(['curl','-sfL','--retry','3','--max-time','60',url],capture_output=True)
	if r.returncode:raise OSError(f'download failed: {url}')
	return r.stdout

def list_cdaw(day,product):
	"""(Time, file name) of one product in the CDAW folder of one UTC day."""
	html=_get(URL.format(day.datetime)).decode()
	names=sorted(set(re.findall(rf'href="(\d{{8}}_\d{{6}}_{product}\.png)"',html)))
	return [(Time.strptime(n[:15],'%Y%m%d_%H%M%S'),n) for n in names]

def download_cdaw(folder,t0,t1,detector='C2',download=True):
	"""Rows (time, running-difference path, intensity path or None) for t0..t1.

	Inputs: folder (files saved and reused here), t0, t1 (Time), detector.
	download=False uses only files already in folder. Returns rows sorted by time.
	"""
	rdf,inten=PRODUCTS[detector];folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
	found={}
	if download:
		day=Time(t0.datetime.date().isoformat())
		while day<=t1:
			for kind in (rdf,inten):
				try:items=list_cdaw(day,kind)
				except OSError as exc:print(f'CDAW listing failed for {day.isot[:10]}: {exc}',flush=True);items=[]
				for t,n in items:
					if t0<=t<=t1 and not (folder/n).is_file():
						try:(folder/n).write_bytes(_get(URL.format(t.datetime)+n))
						except OSError as exc:print(f'CDAW download failed: {n}: {exc}',flush=True)
			day=day+1*u.day
	for p in folder.glob(f'*_{rdf}.png'):
		t=Time.strptime(p.name[:15],'%Y%m%d_%H%M%S')
		if t0<=t<=t1:
			q=folder/p.name.replace(rdf,inten);found[t.isot]=(t,p,q if q.is_file() else None)
	for p in folder.glob(f'*_{inten}.png'):  # intensity frames without a running difference (base, predecessor)
		t=Time.strptime(p.name[:15],'%Y%m%d_%H%M%S')
		if t0<=t<=t1 and t.isot not in found:found[t.isot]=(t,None,p)
	return sorted(found.values(),key=lambda r:r[0].unix)

def _grey(path):return np.asarray(Image.open(path).convert('L'),dtype='float32')

def _intensity(path):
	"""Luminance of the colour-mapped composite (0-255), PNG orientation."""
	a=np.asarray(Image.open(path).convert('RGB'),dtype='float32')
	return .299*a[...,0]+.587*a[...,1]+.114*a[...,2]

def cdaw_geometry(rdf_path):
	"""(x centre, y centre, Rsun radius) in PNG pixels from the white limb circle."""
	a=_grey(rdf_path);y,x=np.indices(a.shape);r=np.hypot(x-a.shape[1]/2,y-a.shape[0]/2)
	ys,xs=np.nonzero((a>=250)&(r>25)&(r<70))
	if len(xs)<100:raise ValueError(f'No 1 Rsun limb circle found in {rdf_path}.')
	return tuple(optimize.least_squares(lambda p:np.hypot(xs-p[0],ys-p[1])-p[2],[a.shape[1]/2,a.shape[0]/2,40.]).x)

def _grid(t,shape,cx,cy,R,detector):
	"""Reference sunpy map (FITS orientation) and x/y in Rsun for the PNG grid."""
	ny,nx=shape;d=(.99*sun.earth_distance(t)).to(u.m)
	rs=np.degrees(np.arcsin(695700e3/d.value))*3600  # angular radius from SOHO, arcsec
	header={'naxis':2,'naxis1':nx,'naxis2':ny,'ctype1':'HPLN-TAN','ctype2':'HPLT-TAN','cunit1':'arcsec','cunit2':'arcsec',
		'cdelt1':rs/R,'cdelt2':rs/R,'crval1':0.,'crval2':0.,'crpix1':cx+1,'crpix2':(ny-1-cy)+1,'date-obs':t.isot,
		'hgln_obs':0.,'hglt_obs':sun.B0(t).to_value(u.deg),'dsun_obs':d.value,'rsun_ref':695700e3,'rsun_obs':rs,
		'telescop':'SOHO','instrume':'LASCO','detector':detector,'obsrvtry':'SOHO','bunit':'',
		'ov_src':'CDAW PNG','ov_note':'CDAW running difference / intensity composite; display units'}
	yy,xx=np.indices(shape)
	return sunpy.map.Map(np.zeros(shape,dtype='float32'),header),(xx-cx)/R,(yy-(ny-1-cy))/R

def make_cdaw_cache(folder,start,end,cadence,base_time,reference_lookback,max_image_dt,max_gap,r_inner,r_outer,out,
	detector='C2',download=True,flow_input='intensity',flow_bin='auto'):
	"""Download CDAW PNGs and write a motion cache in out; return out.

	Inputs as motion_inputs.make_motion_cache (times UTC, seconds, Rsun). One
	running-difference frame per cadence slot within max_image_dt. The flow uses
	the intensity composites of the frame and of the previous exposure; pairs
	longer than max_gap get no flow. Saves reference_grid.fits, coordinates.npz,
	structures_*.npz, times.json and noise_scales.json (source 'CDAW PNG').
	"""
	check_flow_input(flow_input)
	if detector not in PRODUCTS:raise ValueError('use_CDAW_pngs supports LASCO C2 only (CDAW has no C3 intensity composite).')
	out=Path(out);out.mkdir(parents=True,exist_ok=True)
	start,end,base_time=map(Time,(start,end,base_time))
	lo=min(start-reference_lookback*u.s,base_time-max_image_dt*u.s)
	rows=download_cdaw(folder,lo,end,detector,download)
	event=[r for r in rows if start<=r[0]<=end and r[1] is not None and r[2] is not None]
	if not event:raise FileNotFoundError(f'No CDAW {detector} running-difference + intensity pairs in {start.isot}..{end.isot} under {folder}.')
	targets=np.arange(0,(end-start).sec+1e-6,cadence);offset=np.array([(r[0]-start).sec for r in event])
	keep=sorted({int(np.argmin(abs(offset-t))) for t in targets if np.min(abs(offset-t))<=max_image_dt})
	selected=[event[i] for i in keep]
	inten=[r for r in rows if r[2] is not None]
	bases=[r for r in inten if abs((r[0]-base_time).sec)<=max_image_dt]
	if not bases:raise FileNotFoundError(f'No CDAW {detector} intensity image within {max_image_dt} s of base_diff_DT.')
	base=min(bases,key=lambda r:abs((r[0]-base_time).sec))
	cx,cy,R=cdaw_geometry(selected[0][1])
	print(f'CDAW {detector}: {len(selected)} frames; Sun centre ({cx:.1f}, {cy:.1f}) px, 1 Rsun = {R:.2f} px',flush=True)
	shape=_grey(selected[0][1]).shape
	ref,xr,yr=_grid(selected[0][0],shape,cx,cy,R,detector);ref.save(out/'reference_grid.fits',overwrite=True)
	np.savez_compressed(out/'coordinates.npz',x_rsun=xr,y_rsun=yr)
	radius=np.hypot(xr,yr);rows_ok=np.zeros(shape,bool);rows_ok[shape[0]-TEXT_ROWS:]=True  # FITS rows above the text band
	edge=min(cy,TEXT_ROWS-1-cy,cx,shape[1]-1-cx)/R  # largest circle inside the image and above the labels
	valid0=(radius>=max(r_inner,2.35))&(radius<=min(r_outer,edge))&rows_ok
	flip=lambda a:a[::-1].astype('float32')
	b=flip(_intensity(base[2]))
	if flow_bin=='auto':
		dts=[(q[0]-p[0]).sec for p,q in zip(inten[:-1],inten[1:]) if start-reference_lookback*u.s<=q[0]<=end]
		flow_bin=auto_flow_bin(1/R,float(np.median(dts)) if dts else 720.)
	flow_bin=int(flow_bin)
	if flow_bin>1:print(f'TV-L1 on {flow_bin}x{flow_bin} pixel blocks',flush=True)
	times={}
	for t,rdf,ip in selected:
		prev=[r for r in inten if r[0]<t and (t-r[0]).sec<=reference_lookback]
		run=flip(_grey(rdf));im=flip(_intensity(ip))
		valid=valid0.copy();run=run-np.median(run[valid])
		if prev:
			pt,_,pp=prev[-1];a=flip(_intensity(pp));dt=(t-pt).sec
			dx,dy,q=flow(*flow_pair(a,im,b,flow_input),valid,flow_bin) if dt<=max_gap else (np.zeros(shape,'float32'),np.zeros(shape,'float32'),np.zeros(shape,bool))
		else:
			pt,dt=t-max_gap*u.s-1*u.s,np.inf;dx,dy,q=np.zeros(shape,'float32'),np.zeros(shape,'float32'),np.zeros(shape,bool)
		tag=t.strftime('structures_%Y%m%d_%H%M%S')
		np.savez_compressed(out/(tag+'.npz'),valid=valid,dx=dx.astype('float32'),dy=dy.astype('float32'),quality=q,
			base_difference=np.where(valid,im-b,0.).astype('float32'),running_difference=np.where(valid,run,0.).astype('float32'))
		times[tag]=dict(datetime=t.isot,previous_time=pt.isot,source=str(rdf),intensity_source=str(ip),
			previous_intensity_source=str(prev[-1][2]) if prev else None,base_source=str(base[2]),base_time=base[0].isot)
		print(f'{t.isot} | Δt={dt/60:.1f} min',flush=True)
	(out/'times.json').write_text(json.dumps(times,indent=2))
	_set_state(out,flow_input,flow_bin)
	f=out/'noise_scales.json';info=json.loads(f.read_text());info.update(source='CDAW PNG',cdaw_centre_px=[cx,cy],cdaw_rsun_px=R);f.write_text(json.dumps(info,indent=2))
	return out
