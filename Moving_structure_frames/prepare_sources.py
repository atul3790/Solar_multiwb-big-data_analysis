"""Fill gaps in the existing frame-maker databases and prepare PFSS epochs."""
from pathlib import Path
import inspect
import json
import warnings
import numpy as np
from astropy.time import Time
import astropy.units as u

def fill_gaps(rows,targets,tolerance,cadence,fetch):
	"""Request consecutive missing bins together; reuse observations already close enough."""
	missing=[i for i,t in enumerate(targets) if not any(abs((r[0]-t).sec)<=tolerance for r in rows)]
	groups=[]
	for i in missing:
		if groups and i==groups[-1][-1]+1:groups[-1].append(i)
		else:groups.append([i])
	for group in groups:
		lo=targets[group[0]]-cadence/2*u.s
		hi=targets[group[-1]]+cadence/2*u.s
		fetch(lo.isot,hi.isot)
	return len(missing)

def stereo_root(folder,craft):
	"""STEREO database root for folder (which may be the root or a STEREO_<sc>/... folder inside it).

	get_STEREO_imgs saves into <root>/STEREO_<sc>/<instrument>[/<wave>], so it
	must be given the root; passing an instrument folder nests the downloads.
	"""
	p=Path(folder).expanduser();tag=f'STEREO_{craft}'
	return (Path(*p.parts[:p.parts.index(tag)]) if p.parts.index(tag) else Path('.')) if tag in p.parts else p

def fill_coronagraph(folder,detector,craft,sT,eT,dT,base,lookback,tolerance,calib,polar,helioviewer):
	from motion_inputs import inventory
	start,end,b=map(Time,(sT,eT,base))
	rows=inventory(folder,detector,craft)
	def fetch(lo,hi):
		import Multiwb_datamine as dm
		if craft:
			dm.get_STEREO_imgs(lo,hi,dT,spacecraft=craft,instruments=[detector],waves=[None],
				outdir=str(stereo_root(folder,craft)),nproc=1,polar=polar,helioviewer=helioviewer)
		else:
			try:
				dm.get_LASCO_imgs(lo,hi,dT,detector,outdir=str(folder),need_calib=calib)
			except (OSError,RuntimeError,ValueError) as exc:
				print(f'{detector}: download unsuccessful for {lo} -- {hi}; continuing: {exc}',flush=True)
	targets=start+np.arange(0,(end-start).sec+1e-6,dT)*u.s
	fill_gaps([r for r in rows if start<=r[0]<=end],targets,tolerance,dT,fetch)
	if not any(abs((r[0]-b).sec)<=tolerance and r[0]<=start for r in rows):
		fetch((b-max(tolerance,1)*u.s).isot,min(start,(b+max(tolerance,1)*u.s)).isot)
	rows=inventory(folder,detector,craft)
	first=min((r[0] for r in rows if start<=r[0]<=end),default=start)
	if not any(0<(first-r[0]).sec<=lookback for r in rows):
		fetch((first-lookback*u.s).isot,first.isot)

def fill_disk(folder,instrument,product,target,craft,sT,eT,cadence,max_minutes,helioviewer):
	from solar_disk_layer import disk_inventory
	start,end=map(Time,(sT,eT))
	step=min(cadence,max_minutes*60)
	if not np.isfinite(step) or step<=0:raise ValueError('Disk cadence and time tolerance must be positive.')
	targets=start+np.r_[np.arange(0,(end-start).sec,step),(end-start).sec]*u.s
	rows=disk_inventory(folder,instrument,product,target)
	def fetch(lo,hi):
		try:
			import Multiwb_datamine as dm
			if craft:
				dm.get_STEREO_imgs(lo,hi,step,spacecraft=craft,instruments=['EUVI'],waves=[[product]],
					outdir=str(stereo_root(folder,craft)),nproc=1,helioviewer=helioviewer)
			elif 'instruments' in inspect.signature(dm.get_SDO_imgs).parameters:
				dm.get_SDO_imgs(lo,hi,step,instruments=[instrument],waves=[[product]],outdir=str(folder),nproc=1)
			else:
				dm.get_SDO_imgs(lo,hi,step,instrument=instrument,wave=float(product) if instrument=='AIA' else product,outdir=str(folder))
		except Exception as exc:warnings.warn(f'{instrument} download failed; disk overlay may be omitted: {exc}')
	fill_gaps(rows,targets,min(step/2,max_minutes*60),step,fetch)

def _jp2_candidates(folder,lo,hi,match):
	"""Cached JP2s under folder whose observation time lies in [lo,hi] and match(map)."""
	import re
	from datetime import datetime
	from datamine_calib_helper import load_map
	out=[]
	for p in sorted(Path(folder).expanduser().rglob('*.jp2')):
		stamp=re.search(r'(\d{8}T\d{6})',p.name)
		if stamp:
			t=Time(datetime.strptime(stamp[1],'%Y%m%dT%H%M%S'))
			if t<lo-3600*u.s or t>hi+3600*u.s:continue
		try:
			m=load_map(p)
			if lo<=m.date<=hi and match(m):out.append(p)
		except Exception as exc:warnings.warn(f'Skipping JP2 {p}: {exc}')
	return out

def ensure_hv_coronagraph(folder,detector,craft,sT,eT,dT,base,lookback,max_image_dt,download,helioviewer,prefer_helioviewer=True):
	"""Coverage check for the route Level-1 -> Helioviewer -> Level-0.5.

	Families ranked above Helioviewer (Level-1; also Level-0.5 when
	prefer_helioviewer=False) are counted per slot within max_image_dt.
	If none covers every slot, Helioviewer JP2s are fetched for
	the whole interval (download and helioviewer enabled), plus the predecessor
	window and the base reference, then converted to FITS in maps_HV beside the
	instrument's other products (STEREO: <root>/STEREO_<sc>/<detector>/maps_HV).
	make_motion_cache then uses the family covering most slots (ties: science).
	"""
	from motion_inputs import inventory,product_groups,group_coverage,family_priority
	start,end,b=map(Time,(sT,eT,base))
	targets=np.arange(0,(end-start).sec+1e-6,dT)
	rows=inventory(folder,detector,craft)
	groups=product_groups(rows)
	# Helioviewer is needed unless a family ranked above it covers every slot
	# (Level-1; or also Level-0.5 when prefer_helioviewer=False).
	above=[g for k,g in groups.items() if not k[0] and family_priority(k,prefer_helioviewer)<family_priority((True,),prefer_helioviewer)]
	science=max((len(targets)-len(group_coverage(g,start,end,targets,b,lookback,max_image_dt)[1]) for g in above),default=0)
	print(f'{detector}: FITS ranked above Helioviewer cover {science}/{len(targets)} cadence slots',flush=True)
	if science==len(targets):return 'science'
	# Event slots on the same grid as make_motion_cache, then predecessor and base.
	windows=[(start,end),(start-lookback*u.s,start),(b-max_image_dt*u.s,b+max_image_dt*u.s)]
	if download and helioviewer:
		import Multiwb_datamine as dm
		if craft:
			jp2dir=stereo_root(folder,craft)/f'STEREO_{craft}'/detector/'raw'/'helioviewer'
		else:
			jp2dir=Path(folder)/'raw'/f'helioviewer_{detector}'
		for lo,hi in windows:
			try:
				dm.get_helioviewer_jp2(lo.isot,hi.isot,dT,'LASCO' if not craft else detector,
					wave=detector if not craft else None,spacecraft=craft or 'A',outdir=jp2dir)
			except Exception as exc:warnings.warn(f'{detector}: Helioviewer JP2 fetch failed for {lo.isot}--{hi.isot}: {exc}')
	from motion_inputs import spacecraft_id
	def match(m):
		if str(m.detector).strip().upper()!=detector:return False
		return not craft or spacecraft_id(m.meta.get('obsrvtry',m.observatory))=='STEREO'+craft
	lo,hi=min(w[0] for w in windows),max(w[1] for w in windows)
	jp2s=_jp2_candidates(folder,lo,hi,match)
	have=[r[0] for k,g in groups.items() if k[0] for r in g]
	from datamine_calib_helper import jp2_to_fits
	hv=stereo_root(folder,craft)/f'STEREO_{craft}'/detector/'maps_HV' if craft else Path(folder)/'maps_HV'
	made=jp2_to_fits(jp2s,hv,skip_times=have)
	print(f'{detector}: {len(jp2s)} Helioviewer JP2s found; {len(made)} new or reused FITS in maps_HV',flush=True)
	return 'helioviewer'

def ensure_hv_disk(folder,instrument,product,target,craft,sT,eT,cadence,max_minutes):
	"""Disk underlay: convert cached Helioviewer JP2s to maps_HV for slots the science FITS miss.

	The disk layer is never differenced, so per-frame mixing is safe; the
	nearest image within disk_max_minutes is drawn whatever its family.
	"""
	if folder is None or not Path(folder).expanduser().is_dir():return 0
	from solar_disk_layer import disk_inventory
	from datamine_calib_helper import jp2_to_fits
	start,end=map(Time,(sT,eT))
	step=min(cadence,max_minutes*60)
	targets=start+np.r_[np.arange(0,(end-start).sec,step),(end-start).sec]*u.s
	rows=disk_inventory(folder,instrument,product,target)
	tol=min(step/2,max_minutes*60)
	gaps=[t for t in targets if not any(abs((r[0]-t).sec)<=tol for r in rows)]
	print(f'{instrument} {product}: disk FITS cover {len(targets)-len(gaps)}/{len(targets)} slots',flush=True)
	if not gaps:return 0
	instrument=instrument.upper()
	def match(m):
		ins=(str(m.meta.get('instrume',''))+' '+str(m.detector)).upper()
		if instrument not in ins:return False
		if instrument in ('AIA','EUVI') and int(round(m.wavelength.to_value(u.angstrom)))!=int(product):return False
		if instrument=='EUVI' and craft:
			obs=str(m.meta.get('obsrvtry',m.observatory)).upper().replace('_','').replace('-','').replace(' ','')
			if obs not in ('STEREO'+craft,'STEREO'+('AHEAD' if craft=='A' else 'BEHIND')):return False
		if instrument=='HMI':
			kind='continuum' if 'continuum' in str(m.meta.get('content','')).lower() else 'magnetogram'
			if kind!=str(product).lower():return False
		return True
	jp2s=[p for p in _jp2_candidates(folder,start-tol*u.s,end+tol*u.s,match)]
	from datamine_calib_helper import load_map
	keep=[p for p in jp2s if any(abs((load_map(p).date-t).sec)<=tol for t in gaps)]
	hv=stereo_root(folder,craft)/f'STEREO_{craft}'/'EUVI'/str(product)/'maps_HV' if instrument=='EUVI' and craft else Path(folder)/'maps_HV'
	made=jp2_to_fits(keep,hv,skip_times=[r[0] for r in rows])
	print(f'{instrument} {product}: {len(made)} Helioviewer FITS in maps_HV fill disk gaps',flush=True)
	return len(made)

def prepare_pfss(folder,sT,eT,hours,build,gong_cache,kwargs):
	"""Use UTC-anchored PFSS epochs, including the bracketing snapshots."""
	if not np.isfinite(hours) or hours<=0:raise ValueError('PFSS_dT must be positive hours.')
	start,end=map(Time,(sT,eT))
	step=hours*3600
	models=[]
	for seconds in np.arange(np.floor(start.unix/step)*step,np.ceil(end.unix/step)*step+step/2,step):
		epoch=Time(seconds,format='unix')
		target=Path(folder)/epoch.strftime('%Y%m%d_%H%M%S')
		def ready():
			try:
				meta=json.loads((target/'metadata.json').read_text())
				return (target/'field_lines.npz').is_file() and abs((Time(meta['model_time'])-epoch).sec)<1 and all(meta.get(k)==v for k,v in kwargs.items())
			except (OSError,ValueError,KeyError,TypeError):return False
		if not ready():
			if all((target/name).is_file() for name in ('metadata.json','field_lines.npz')):
				raise ValueError(f'Existing PFSS snapshot has incompatible metadata: {target}. Match its parameters or choose another pfss_dir.')
			if not build:raise FileNotFoundError(f'Missing matching PFSS snapshot {target}. Enable download or set PFSS_dT=None to use an existing model.')
			from solar_overlay import build_pfss
			build_pfss(epoch.isot,epoch.isot,target,cache_dir=str(gong_cache),**kwargs)
			if not ready():raise RuntimeError(f'PFSS build did not produce a matching snapshot: {target}')
		models.append((epoch,target/'field_lines.npz'))
	return models
