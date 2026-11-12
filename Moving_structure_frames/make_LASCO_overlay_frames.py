"""AR-focused LASCO/HMI/PFSS + GOES/LAT frames from an existing database.
Core overlay, PFSS and acquisition modules are used unchanged.
"""
from pathlib import Path
from datetime import datetime as dt, timedelta
from multiprocess import get_context
import sys
import csv
import json
import pickle
import copy
import inspect
import shutil
import glob
import re
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp
import astropy.units as u
from astropy.time import Time
from astropy.constants import R_sun
from astropy.coordinates import SkyCoord
import sunpy.map
from sunpy.coordinates import frames
import solar_overlay_helper as oh
from pfss_projection import ar_rooted_lines  # shared AR selection with the flow frames
import solar_overlay as so
import datamine_calib_helper as dh
import Multiwb_datamine as dm

#################### INPUT ##################################################################
basedir='/path/to/event_folder'  # Event folder: all relative paths below live inside it
sT='YYYY-MM-DDThh:mm:ss'  # Event start, UTC
eT='YYYY-MM-DDThh:mm:ss'  # Event end, UTC
dT=600                             # LASCO and output frame cadence, seconds
SDO_dT=7200                        # AIA/HMI download and coverage cadence, seconds
diff_type='base_diff'            # 'running_diff' or 'base_diff'
first_frame_intensity=False        # True: first frame is actual LASCO; False: difference
base_diff_DT='YYYY-MM-DDThh:mm:ss'  # Quiet base-difference reference, UTC, at or before sT
LASCO='C3'						# Which LASCO coronagraph to use
LASCO_imgfolder='./LASCO/C3'    # Path to LASCO database within which C2 image folder either exist or should be save into, searched recursively
SDO_fold='./SDO'                     # SDO database root, searched recursively
SDO_instrument='HMI'               # 'AIA' or 'HMI': ONE overlaid image
SDO_product='magnetogram'          # AIA: 94,171,...; HMI: 'continuum'/'magnetogram'
pfss_dir='./PFSS'    # Saved core build_pfss output
user_datacsv='LAT_fluxLC.csv'
goes_pickle=None                    # Existing module GOES .p, or None to fetch/cache
goes_folder='./GOES'  # GOES cache: read if present, downloads saved here
SRS_folder='./NOAA_SRS'  # NOAA SRS reports: read here, downloads saved here (shared with STEREO and flow codes)
outdir='./LASCO_LAT_frames'
final_image_folder='./LASCO_LAT_frames/pfss_lasC3bd_HMI_LC' # Final overlay + light-curve PNG destination
need_LASCO_calib=True                # Retrieval preference for missing LASCO images

nproc=4                           # Number of parallel workers; 1 runs serially

### PFSS parameters ####
PFSS_dT=6                          # PFSS snapshot cadence, HOURS
save_PFSS_maps=True     # True: save AR + full-Sun PFSS maps (on the disk image, or a blank Stonyhurst-grid disk) at the sT-eT midpoint in basedir as PFSS_*.png; False: skip (no redraw)
rss= 2.5
nrho=35
r_seed=1.05
n_lon=36
n_lat=18
#############
### INPUT for AR region - focussed PFSS line drawing ####
focus_AR=True                      # True: add AR traces; False: global PFSS only
NOAA_number=None  # Set the NOAA active-region number (e.g. 11402) when focus_AR is on
AR_time=sT                          # Requested lookup epoch (UTC)
ar_half_lon=12.          # AR rectangle half-width in longitude, deg (Stonyhurst, about the NOAA centre)
ar_half_lat=12.          # AR rectangle half-width in latitude, deg
AR_seed_spacing_deg=1.   # Seed spacing over the AR rectangle, deg (+-12 deg at 1 deg = 25 x 25 seeds)
trace_longARrooted_loops=1 # 1: loops with one footpoint in the rectangle (large AR loops kept); 0: both footpoints inside
AR_r_seed=1.01                      # Radius in solar radii, for low closed loops
########
### PFSS plot making input parameters 
AR_closedB_col='lime'               # Closed AR-seeded lines (focus_AR=True)
AR_openB_col='magenta'              # Open AR-seeded lines, both polarities
closed_field_color='white'         # Matplotlib colour for closed PFSS lines
n_global_closed=20       # Global closed PFSS loops drawn
n_closed=30              # Max AR closed loops drawn (tallest first with trace_longARrooted_loops=1); np.nan: all
n_AR_open=5              # Max open AR field lines drawn (magenta), spread evenly; np.nan: all
max_image_dt=600                     # Maximum image offset from target, seconds
reference_lookback=3600              # Pre-sT predecessor search window, seconds
npix=1024
dpi=90
########################################################################
#----------------------------------------------------------------------#

download=True                      # True fills missing images/PFSS; False reuses database
PFSS_dT*=3600                      # Convert hours to seconds internally
img_folder=LASCO_imgfolder
pfss_kwargs={'rss':rss,'nrho':nrho,'r_seed':r_seed,'n_lon':n_lon,'n_lat':n_lat}
lat_csv=Path(user_datacsv)  # inside basedir
def helioviewer_requests(folder,detector):
	"""Infer previous request (start,end,cadence) tuples from JP2 names only."""
	times=[]
	for filename in glob.glob(str(Path(folder)/'**'/'*'),recursive=True):
		p=Path(filename)
		if p.suffix.lower() not in ('.jp2','.j2k') or detector not in str(p).upper():
			continue
		match=re.search(r'(\d{8}T\d{6})',p.name)
		if match:
			times.append(dt.strptime(match[1],'%Y%m%dT%H%M%S'))
	times=sorted(set(times))
	tuples=[]
	if len(times)==1:
		tuples=[(times[0],times[0],0.)]
	elif len(times)>1:
		a=0
		step=(times[1]-times[0]).total_seconds()
		for i in range(1,len(times)-1):
			newstep=(times[i+1]-times[i]).total_seconds()
			if abs(newstep-step)>1:
				tuples.append((times[a],times[i],step))
				a=i
				step=newstep
		tuples.append((times[a],times[-1],step))
	return times,tuples


def scan_image_cache(folder,instrument,key,log_dir,lo,hi):
	"""Read matching cache images; time-limit LASCO, retain all SDO candidates."""
	folder=Path(folder).expanduser().resolve()
	files=glob.glob(str(folder/'**'/'*'),recursive=True)
	files=sorted(f for f in files if f.lower().endswith(
		('.fits','.fit','.fts','.fits.gz','.fit.gz','.fts.gz','.fits.fz','.jp2','.j2k')))
	files.sort(key=lambda f:f.lower().endswith(('.jp2','.j2k')))  # Finished FITS first.
	entries=[]
	seen={}
	with (Path(log_dir)/f'cache_inventory_{instrument}_{key}.csv').open('w',newline='') as stream:
		writer=csv.writer(stream)
		writer.writerow(['file','datetime','source','usable_file'])
		for filename in files:
			path=Path(filename)
			jp2=path.suffix.lower() in ('.jp2','.j2k')
			if jp2:
				if instrument=='LASCO' and key not in filename.upper():
					continue
				stamp=re.search(r'(\d{8}T\d{6})',path.name)
				if instrument=='LASCO' and stamp and not lo<=Time(dt.strptime(stamp[1],'%Y%m%dT%H%M%S'))<=hi:
					continue
			try:
				m=dh.load_map(path)
				if instrument=='LASCO' and not lo<=m.date<=hi:
					continue
				ins=str(m.meta.get('instrume','')).upper()
				det=str(m.meta.get('detector','')).upper()
				if not (jp2 and instrument=='LASCO') and instrument not in ins and instrument not in det:
					continue
				if instrument=='LASCO' and not jp2 and key not in det and key not in ins:
					continue
				if instrument=='AIA' and int(round(m.wavelength.to_value(u.angstrom)))!=int(key):
					continue
				if instrument=='HMI':
					product=' '.join(str(m.meta.get(k,'')) for k in ('content','physobs','bunit','series')).lower()
					tags=('continuum','intensity','hmi.ic_') if key=='continuum' else ('magnet','gauss','hmi.m_')
					if not any(tag in product for tag in tags):
						continue
				rank=(0 if dh._lasco_science_ready(m) else 1 if jp2 or m.meta.get('ov_hv',False) else 2) if instrument=='LASCO' else (1 if jp2 else 0)
				identity=(m.date.isot,rank)
				usable=seen.get(identity,path)
				if jp2 and instrument=='LASCO' and identity not in seen:
					meta=dh._hv_tags(copy.deepcopy(m.meta),path)  # Same tags as every JP2 -> FITS route.
					meta.update({'instrume':'LASCO','detector':key})
					maps=folder/'maps'
					maps.mkdir(parents=True,exist_ok=True)
					usable=maps/f'LASCO_{key}_HV_{m.date.strftime("%Y%m%d_%H%M%S")}.fits'
					if not usable.is_file():
						dh._prepare_lasco(sunpy.map.Map(m.data,meta),key).save(usable,overwrite=False)
				if identity not in seen:
					entries.append((m.date,usable,rank))
					seen[identity]=usable
				writer.writerow([path,m.date.isot,rank,usable])
			except (OSError,ValueError,KeyError,TypeError,ImportError,RuntimeError) as exc:
				print('Cannot read cached image:',path,';',exc)
	entries.sort(key=lambda r:r[0].mjd)
	print(f'{instrument}: {len(entries)} cached exposures in the requested window')
	return entries


def find_NOAA_location(number,requested,cache_dir,local_file=None,max_age=36):
	"""Return Stonyhurst lon, lat, validity epoch and NOAA SRS provenance.

	Use the nearest LOCATION-VALIDITY epoch, not report issue time. SRS often
	lists four-digit region identifiers; full NOAA numbers are matched too.
	Local SRS text is accepted for offline use. Missing/old reports raise.
	"""
	if number is None or int(number)!=number or int(number)<=0:
		raise ValueError('Set NOAA_number to the NOAA active-region number')
	number=int(number)
	target=dt.fromisoformat(requested.replace('Z','+00:00'))
	if target.tzinfo is not None:
		target=pd.Timestamp(target).tz_convert('UTC').tz_localize(None).to_pydatetime()
	cache=Path(cache_dir)
	cache.mkdir(parents=True,exist_ok=True)
	reports=[]
	failures=[]
	for offset in ([0] if local_file is not None else [-1,0,1]):
		day=target+timedelta(days=offset)
		url=f'https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_region_summaries/{day:%Y/%m/%Y%m%d}SRS.txt'
		path=Path(local_file) if local_file is not None else cache/f'{day:%Y%m%d}_000000_SRS.txt'
		try:
			if not path.is_file():
				if local_file is not None:
					raise FileNotFoundError(path)
				with urllib.request.urlopen(url,timeout=30) as response:
					text=response.read().decode('utf-8',errors='replace')
				if 'Solar Region Summary' not in text and 'SOLAR REGION SUMMARY' not in text:
					raise ValueError('Response is not a NOAA SRS report')
				path.write_text(text)
			text=path.read_text()
			issued=re.search(r':Issued:\s*(\d{4}\s+\w{3}\s+\d{1,2}\s+\d{4})\s+UTC',text,re.I)
			if issued:
				issue=dt.strptime(issued[1],'%Y %b %d %H%M')
			else:
				issued=re.search(r'Issued at (\d{4})Z on (\d{1,2} \w{3} \d{4})',text,re.I)
				if issued is None:
					raise ValueError('Missing issue date in SRS header')
				issue=dt.strptime(issued[2]+' '+issued[1],'%d %b %Y %H%M')
			valid=re.search(r'(?:Locations|Positions) Valid at\s+(\d{2})/(\d{2})(\d{2})Z',text,re.I)
			if valid is None:
				raise ValueError('Missing location-validity epoch in SRS header')
			dd,hh,mm=map(int,valid.groups())
			if not (0<=hh<=24 and 0<=mm<60 and (hh<24 or mm==0)):
				raise ValueError('Invalid SRS validity time')
			epochs=[]
			for shift in (-2,-1,0,1):
				candidate=issue.replace(hour=0,minute=0,second=0,microsecond=0)+timedelta(days=shift)
				if candidate.day==dd:
					epochs.append(candidate+timedelta(hours=hh,minutes=mm))
			if not epochs:
				raise ValueError('Cannot resolve SRS validity date')
			epoch=min(epochs,key=lambda t:abs((t-issue).total_seconds()))
			# Section II lists departed regions; only I and IA supply disk positions.
			disk=re.split(r'^\s*II\.',text,flags=re.M)[0]
			for match in re.finditer(r'^\s*(\d{4,5})\s+([NS])(\d{2})([EW])(\d{2})\b',disk,re.M):
				n=int(match[1])
				if n!=number and not (n<10000 and number>=10000 and n==number%10000):
					continue
				latitude=int(match[3])*(1 if match[2]=='N' else -1)
				longitude=int(match[5])*(1 if match[4]=='W' else -1)
				age=abs((epoch-target).total_seconds())/3600
				if age<=max_age:
					reports.append(dict(noaa=number,lon=longitude,lat=latitude,
						epoch=epoch.isoformat(),issue=issue.isoformat(),offset_hours=age,
						file=str(path.resolve()),source=str(path.resolve()) if local_file else url))
		except (OSError,ValueError) as exc:
			failures.append(f'{path.name}: {exc}')
	if not reports:
		raise ValueError(f'NOAA {number}: no disk location within {max_age} h of {requested}. '
			'Check the number/date or supply SRS_file. '+'; '.join(failures))
	best=min(reports,key=lambda row:row['offset_hours'])
	(cache/'AR_location.json').write_text(json.dumps(best,indent=2))
	print(f'NOAA {number}: lon={best["lon"]:+g}, lat={best["lat"]:+g}; valid {best["epoch"]} UT')
	return best['lon'],best['lat'],best['epoch'],best


def route_rank(rank):
	"""Scan rank (0 Level-1, 1 Helioviewer, 2 Level-0.5) -> route order.

	need_LASCO_calib=True: Level-1 -> Helioviewer -> Level-0.5.
	need_LASCO_calib=False: Level-1 -> Level-0.5 -> Helioviewer.
	"""
	return rank if globals().get('need_LASCO_calib',True) else {0:0,1:2,2:1}.get(rank,rank)


def choose_image(records,target,tolerance,lasco=False):
	"""Within tolerance of a target: route order first, then nearest in time."""
	available=[r for r in records if abs((r[0]-target).to_value(u.s))<=tolerance]
	if not available:
		return None
	return min(available,key=lambda r:(route_rank(r[2]) if lasco else 0,abs((r[0]-target).to_value(u.s))))


def covering_goes(folder,start,end,tol_s=2.):
	"""GOES XRS-B flux (DataFrame 'flux', UTC index) from any cached pickle in folder
	spanning start-end: module Time/Flux dicts or flux DataFrames. The closest-fitting
	file wins; None if none covers the window (then the caller downloads)."""
	start,end=pd.to_datetime(str(start),utc=True),pd.to_datetime(str(end),utc=True)
	tol=pd.Timedelta(seconds=tol_s);best=None
	for p in sorted(Path(folder).expanduser().glob('*.p')):
		try:
			g=pd.read_pickle(p)
			if isinstance(g,dict) and 'Time' in g and 'Flux' in g:
				df=pd.DataFrame({'flux':np.asarray(g['Flux'],float)},index=pd.to_datetime(g['Time'],utc=True))
			elif isinstance(g,pd.DataFrame) and 'flux' in g.columns:
				df=g[['flux']].copy();df.index=pd.to_datetime(df.index,utc=True)
			else:continue
			t=df.index[np.isfinite(df['flux'].values)&(df['flux'].values>0)]
			if len(t)<2 or t.min()>start+tol or t.max()<end-tol:continue
			score=abs(t.min()-start)+abs(t.max()-end)
			if best is None or score<best[0]:best=(score,df,p)
		except Exception:continue
	if best:print(f'GOES: reusing {best[2]}',flush=True)
	return best[1] if best else None

def save_lightcurve_frames(frames,goes,flux_csv,sT,eT,outdir,
	figsize=(10,12),dpi=90,flux_label='LAT >100 MeV',
	flux_ylabel=r'LAT flux (ph cm$^{-2}$ s$^{-1}$)'):
	"""Combine existing overlay PNGs with GOES and optional UTC flux/error CSV.

	goes: DataFrame indexed by datetime, with 'flux' in W/m2.
	CSV: datetime,flux,flux_error; optional asymmetric flux_error_lower/upper,
	is_upper_limit and time_error_seconds. LAT flux is in ph cm^-2 s^-1.
	Returns final PNG paths; exposure timestamps are parsed from overlay names.
	"""
	start,end=pd.Timestamp(sT,tz='UTC'),pd.Timestamp(eT,tz='UTC')
	folder=Path(outdir)
	folder.mkdir(parents=True,exist_ok=True)
	goes=goes.copy()
	goes.index=pd.to_datetime(goes.index,utc=True)
	goes=goes.sort_index().loc[start:end]
	if goes.empty:
		raise ValueError('No GOES samples in the event interval')
	lat=None
	if flux_csv is not None:
		lat=pd.read_csv(flux_csv)
		lat['datetime']=pd.to_datetime(lat['datetime'],utc=True)
		lat=lat.loc[lat['datetime'].between(start,end)].sort_values('datetime')
		if 'is_upper_limit' not in lat:
			lat['is_upper_limit']=False
		lat['is_upper_limit']=lat['is_upper_limit'].astype(str).str.lower().isin(['true','1'])
		if not np.all(np.isfinite(lat['flux'])&(lat['flux']>0)):
			raise ValueError('CSV flux must be finite and positive for the log axis')
	written=[]
	for frame in frames:
		stamp=pd.Timestamp(dt.strptime(Path(frame).stem.removeprefix('overlay_'),
			'%Y%m%d_%H%M%S'),tz='UTC')
		fig=plt.figure(figsize=figsize,dpi=dpi)
		try:
			grid=fig.add_gridspec(2,1,height_ratios=[3,1.35],hspace=0.06)
			image_ax=fig.add_subplot(grid[0])
			image_ax.imshow(plt.imread(frame))
			image_ax.axis('off')
			ax=fig.add_subplot(grid[1])
			gax=ax.twinx() if lat is not None else ax
			gax.plot(goes.index,goes['flux'].where(goes['flux']>0),
				color='orange',lw=1.3,label='GOES XRS-B')
			gax.set_yscale('log')
			gax.set_ylim(1e-7,1e-3)
			gax.set_ylabel(r'GOES flux (W m$^{-2}$)',fontsize=17)
			gax.tick_params(labelsize=14)
			if lat is not None:
				det=lat.loc[~lat['is_upper_limit']]
				if not det.empty:
					yerr=det[['flux_error_lower','flux_error_upper']].to_numpy().T if {
						'flux_error_lower','flux_error_upper'}.issubset(det.columns) else det['flux_error'].to_numpy()
					xerr=det['time_error_seconds'].to_numpy()/86400 if 'time_error_seconds' in det else None
					ax.errorbar(mdates.date2num(det['datetime'].tolist()),det['flux'],yerr=yerr,xerr=xerr,
						fmt='o',color='red',ms=5,capsize=3,lw=1,label=flux_label)
				ul=lat.loc[lat['is_upper_limit']]
				if not ul.empty:
					# Arrow length is only a plotting choice, not a measured error.
					ax.errorbar(mdates.date2num(ul['datetime'].tolist()),ul['flux'],yerr=0.3*ul['flux'],
						xerr=ul['time_error_seconds'].to_numpy()/86400 if 'time_error_seconds' in ul else None,uplims=True,
						fmt='v',color='blue',ms=5,capsize=3,lw=1,label='LAT upper limit')
				ax.set_yscale('log')
				ax.set_ylim(1e-8,1e-4)
				ax.set_ylabel(flux_ylabel,fontsize=17)
			ax.axvline(stamp,color='black',ls='--',lw=1.5)
			ax.text(0.02,0.97,stamp.strftime('%Y-%m-%d %H:%M:%S UT'),
				transform=ax.transAxes,ha='left',va='top',fontsize=12)
			ax.set_xlim(start,end)
			ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[1,4,7,10,13,16,19,22]))
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
			date=start.strftime('%Y-%m-%d')
			if start.date()!=end.date():
				date+=' — '+end.strftime('%Y-%m-%d')
			ax.set_xlabel('Time (UT) '+date,fontsize=17)
			ax.tick_params(labelsize=14)
			ax.grid(alpha=0.2)
			handles,labels=ax.get_legend_handles_labels()
			if gax is not ax:
				h,l=gax.get_legend_handles_labels()
				handles+=h
				labels+=l
			ax.legend(handles,labels,loc='upper right',fontsize=11)
			fig.subplots_adjust(left=0.13,right=0.86,bottom=0.08,top=0.99)
			path=folder/f'overlay_{stamp:%Y%m%d_%H%M%S}.png'
			fig.savefig(path,dpi=dpi)  # Fixed dimensions for movie assembly.
			written.append(str(path))
		finally:
			plt.close(fig)
	return written

def _init_frame_worker(settings):
	"""Load shared rendering settings once in each worker."""
	globals().update(settings)
	globals()['_pfss_cache']={}
	plt.switch_backend('Agg')

def _render_overlay_frame(job,settings):
	"""Render one frame using its predetermined running-difference reference."""
	row,predecessor,h=job
	diff_type=settings['diff_type']
	root=settings['root']
	LASCO=settings['LASCO']
	SDO_label=settings['SDO_label']
	SDO_instrument=settings['SDO_instrument']
	SDO_product=settings['SDO_product']
	npix=settings['npix']
	dpi=settings['dpi']
	pfss_models=settings['pfss_models']
	focus_AR=settings['focus_AR']
	N_global_closed=settings['n_global_closed']
	field_colors={1:'#ff4d4d',-1:'#4da6ff',0:settings['closed_field_color']}
	_pfss_cache=globals().setdefault('_pfss_cache',{})
	# Each worker loads a snapshot only when first needed.
	model=min(pfss_models,key=lambda item:abs((row[0]-Time(item[0])).to_value(u.s)))
	folder=model[1]
	if folder not in _pfss_cache:
		lines=oh._load_pfss(folder)
		ar=None
		path=settings.get('AR_selected',{}).get(folder)
		if focus_AR and path and Path(path).is_file():
			with np.load(path) as data:
				ar={key:data[key].copy() for key in ('lon','lat','r','seg','pol')}
				ar['time']=Time(str(data['time'].item()))
				ar['rss']=float(data['rss'])
		_pfss_cache[folder]=(lines,ar)
	global_lines,ar_lines=_pfss_cache[folder]
	records=[]
	t,path,rank=row
	source=('verified Level-1/MSB','Helioviewer display','other preview')[rank]
	print(t.isot,LASCO,source)
	if h is None:
		records.append([t.isot,SDO_label,'','','','omitted: no cached selected SDO image'])
	m=dh._prepare_lasco(dh.load_map(path),LASCO)
	refpath=''
	if predecessor is not None:
		refpath=predecessor[1]
		ref=dh._prepare_lasco(dh.load_map(refpath),LASCO)
		preview=oh._is_preview_image(m,path)
		ref_preview=oh._is_preview_image(ref,refpath)
		if preview!=ref_preview or rank!=predecessor[2] or (not preview and str(m.unit)!=str(ref.unit)):
			records.append([t.isot,'LASCO/'+LASCO,source,path,refpath,'skipped: incompatible difference products'])
			return None,records
		meta=copy.deepcopy(m.meta)
		meta.pop('blank',None)
		if preview:
			meta.update({'ov_hv':True,'ov_src':'Helioviewer JP2 difference','bunit':'display'})
		meta['diff_ref'],meta['diff_typ']=ref.date.isot,diff_type.replace('_diff','')
		meta['diff_dt']=(m.date-ref.date).to_value(u.s)
		m=sunpy.map.Map(np.asarray(m.data,float)-oh._regrid(ref,m),meta)
		m.save(root/'difference_FITS'/LASCO/diff_type/f'LASCO_{LASCO}_{t.strftime("%Y%m%d_%H%M%S")}.fits',overwrite=True)
	tmap=oh._target_map(m,npix)
	d=np.asarray(tmap.data,float)
	good=np.isfinite(d)
	if refpath:
		limit=max(np.nanpercentile(abs(d),99),1e-30)
		scaled=np.clip(0.5+d/(2*limit),0,1)
	else:
		lo,hi=np.nanpercentile(d,[1,99.5])
		scaled=np.arcsinh(10*np.clip((d-lo)/max(hi-lo,1e-30),0,1))/np.arcsinh(10)
	rgb=plt.get_cmap('gray')(np.nan_to_num(scaled))[...,:3]
	image=rgb
	if h is not None:
		hmap=dh.load_map(h[1])
		b=oh._regrid(hmap,tmap)
		bvalid=np.isfinite(b)
		preview=oh._is_preview_image(hmap,h[1])
		weight=np.ones_like(b)  # Do not make faint disk emission transparent.
		if SDO_instrument=='HMI' and SDO_product=='magnetogram' and not preview:
			scaled_b=np.clip(0.5+np.nan_to_num(b)/1000,0,1)
			cmap='RdBu_r'
		elif SDO_instrument=='AIA':
			cmap,norm=dh._image_display_settings(hmap,'AIA',SDO_product,preview)
			scaled_b=norm(np.ma.masked_invalid(b))
		else:
			lo,hi=np.nanpercentile(b,[1,99.5])
			scaled_b=np.clip((b-lo)/max(hi-lo,1e-30),0,1)
			cmap='gray'
		brgb=plt.get_cmap(cmap)(np.nan_to_num(scaled_b))[...,:3]
		image=oh._composite([(rgb,np.ones_like(d),good),(brgb,weight,bvalid)],0.9)
	fig=Figure(figsize=(8,8),dpi=dpi)
	ax=fig.add_subplot(projection=tmap)
	ax.imshow(image,origin='lower')
	oh._draw_pfss(ax,tmap,global_lines,N_Bloop=N_global_closed,colors=field_colors)
	if ar_lines is not None:
		ar_colors={0:settings['AR_closedB_col'],1:settings['AR_openB_col'],-1:settings['AR_openB_col']}
		oh._draw_pfss(ax,tmap,ar_lines,N_Bloop=np.nan,colors=ar_colors)  # already selected and capped
	ax.text(.02,.98,t.strftime('%Y-%m-%d %H:%M:%S UT'),transform=ax.transAxes,
		va='top',color='white',bbox=dict(facecolor='black',alpha=.6))
	ax.set_title(f'LASCO {LASCO} '+(diff_type.replace('_',' ') if refpath else 'intensity')+f' + {SDO_label} + PFSS')
	png=root/'overlay_images'/f'overlay_{t.strftime("%Y%m%d_%H%M%S")}.png'
	fig.savefig(png,dpi=dpi)
	fig.clear()
	records.append([t.isot,'LASCO/'+LASCO,source,path,refpath,'saved'])
	if h is not None:
		records.append([h[0].isot,SDO_label,'SDO image',h[1],'','saved'])
	return str(png),records

def _safe_overlay_frame(task):
	job,settings=task
	try:
		return _render_overlay_frame(job,settings)
	except Exception as exc:
		t,path,rank=job[0]
		print(f'Frame {t.isot} skipped: {exc}')
		return None,[[t.isot,'frame','','', '',f'skipped: {exc}']]

def _render_lightcurve_frame(task):
	job,settings=task
	return save_lightcurve_frames([job],settings['_frame_goes'],settings['lat_csv'],
		settings['sT'],settings['eT'],Path(settings['final_image_folder']).expanduser(),dpi=settings['dpi'])

if __name__=='__main__':
	_unset=[k for k in ('basedir','sT','eT','base_diff_DT') if k in globals() and ('YYYY' in str(globals()[k]) or str(globals()[k]).startswith('/path/to'))]
	if _unset:raise ValueError('Set these in the INPUT block first: '+', '.join(_unset))
	if save_PFSS_maps:  # AR + full-Sun PFSS maps at the sT-eT midpoint, saved in basedir
		import solar_overlay as _so;from astropy.time import Time as _T
		try:_so.pfss_context_maps((_T(sT)+(_T(eT)-_T(sT))/2).isot,'SDO',SDO_instrument if SDO_fold else None,SDO_product,img_folder=SDO_fold,basedir=basedir,pfss_dir=pfss_dir,rss=rss,nrho=nrho,r_seed=r_seed,n_lon=n_lon,n_lat=n_lat,PFSS_dT=PFSS_dT/3600,SRS_folder=SRS_folder,closed_field_color=closed_field_color,NOAA_number=NOAA_number if focus_AR else None,ar_half_lon=ar_half_lon,ar_half_lat=ar_half_lat,AR_seed_spacing_deg=AR_seed_spacing_deg,trace_longARrooted_loops=bool(trace_longARrooted_loops),AR_r_seed=AR_r_seed,n_closed=n_closed,n_AR_open=None if n_AR_open in ('',None) or n_AR_open!=n_AR_open else int(n_AR_open),n_global_closed=n_global_closed,AR_closedB_col=AR_closedB_col,AR_openB_col=AR_openB_col)
		except Exception as exc:print('PFSS maps not saved:',exc)
	import os
	cwd=os.getcwd()
	os.chdir(basedir)  # all relative paths are inside the event folder
	try:
		# IPython %run may omit this attribute; spawn expects it to exist.
		if not hasattr(sys.modules['__main__'],'__spec__'):
			sys.modules['__main__'].__spec__=None
		if not isinstance(nproc,int) or nproc<1:
			raise ValueError('nproc must be a positive integer')
		if diff_type not in ('running_diff','base_diff'):
			raise ValueError('diff_type must be running_diff or base_diff')
		if focus_AR and NOAA_number is None:
			raise ValueError('Set NOAA_number when focus_AR=True')
		root=Path(outdir).expanduser().resolve()
		if focus_AR:
			AR_lon,AR_lat,AR_time,AR_report=find_NOAA_location(
				NOAA_number,AR_time,Path(SRS_folder).expanduser().resolve())
		(root/'overlay_images').mkdir(parents=True,exist_ok=True)
		(root/'difference_FITS'/LASCO/diff_type).mkdir(parents=True,exist_ok=True)
		start,end=Time(sT),Time(eT)
		if end<=start or not np.isfinite(dT) or dT<=0 or not np.isfinite(SDO_dT) or SDO_dT<=0:
			raise ValueError('Require eT>sT, dT>0 and SDO_dT>0')
		targets=list(oh._target_times(sT,eT,dT))
		requested,previous=helioviewer_requests(img_folder,LASCO)
		with (root/'LASCO_request_ranges.csv').open('w',newline='') as stream:
			writer=csv.writer(stream)
			writer.writerow(['sT','eT','dT_seconds'])
			for a,b,step in previous:
				writer.writerow([a.isoformat(),b.isoformat(),step])
				print('Previous LASCO request:',a,b,'dT=',step)
		# LASCO acquisition coverage comes from request filenames, NOT exposures.
		lasco_covered=[any(abs((t.datetime-q).total_seconds())<=1 for q in requested) or
			any(a<=t.datetime<=b and 0<step<=float(dT)+1 for a,b,step in previous) for t in targets]
		database={}
		SDO_instrument=SDO_instrument.upper()
		if SDO_instrument=='AIA':
			SDO_product=int(SDO_product)
			if SDO_product not in dh.AIA_WAVES:
				raise ValueError('Select a supported AIA wavelength')
		elif SDO_instrument!='HMI' or SDO_product not in ('continuum','magnetogram'):
			raise ValueError('Choose AIA/wavelength or HMI/continuum or HMI/magnetogram')
		SDO_label=f'{SDO_instrument} {SDO_product}'
		products=[('LASCO',LASCO,img_folder),(SDO_instrument,SDO_product,SDO_fold)]
		with (root/'download_plan.csv').open('w',newline='') as stream:
			plan=csv.writer(stream)
			plan.writerow(['instrument','product','start','end','reason','action'])
			for instrument,key,folder in products:
				cadence=dT if instrument=='LASCO' else SDO_dT
				product_targets=list(oh._target_times(sT,eT,cadence))
				if instrument!='LASCO' and product_targets[-1]<end:
					product_targets.append(end)
				window_lo=start-(reference_lookback if instrument=='LASCO' else SDO_dT/2)*u.s
				window_hi=end+(0 if instrument=='LASCO' else SDO_dT/2)*u.s
				cached=scan_image_cache(folder,instrument,key,root,window_lo,window_hi)
				times=[r[0] for r in cached]
				missing_name=f'AIA{key}' if instrument=='AIA' else instrument
				missing_log=Path(folder).expanduser().resolve()/f'missing_{missing_name}.txt'
				known_missing=set(missing_log.read_text().splitlines()) if missing_log.is_file() else set()
				attempted=[]
				if instrument=='LASCO' and requested:
					covered=lasco_covered
				else:
					# Independently check each SDO product at the new requested cadence.
					# Existing FITS also cover LASCO if no JP2 request history exists.
					tolerance=min(max_image_dt if instrument=='LASCO' else SDO_dT/2,float(cadence)/2)
					covered=[any((instrument!='LASCO' or start<=q<=end) and
						abs((q-t).to_value(u.s))<=tolerance for q in times) for t in product_targets]
				missing=[]
				for i,yes in enumerate(covered):
					if yes:
						continue
					if instrument in ('AIA','HMI') and product_targets[i].isot in known_missing:
						print(f'{instrument}/{key}: skipping previously unavailable {product_targets[i].isot}')
						plan.writerow([instrument,key,product_targets[i].isot,product_targets[i].isot,'previous unsuccessful search','skip'])
					else:
						missing.append(i)
				groups=np.split(missing,np.flatnonzero(np.diff(missing)>1)+1) if missing else []
				for group in groups:
					lo,hi=product_targets[int(group[0])],product_targets[int(group[-1])]
					# Include nearby SDO exposures around missing requested slots.
					if instrument in ('AIA','HMI'):
						padding=cadence/2
						lo=lo-padding*u.s
						hi=hi+padding*u.s
					elif hi==lo:
						if lo<end:
							hi=min(end,lo+1*u.s)
						else:
							lo=max(start,hi-1*u.s)
					reason='new request range/finer cadence' if instrument=='LASCO' else 'SDO coverage missing at requested cadence'
					plan.writerow([instrument,key,lo.isot,hi.isot,reason,'download' if download else 'download=False'])
					stream.flush()
					print(f'Missing {instrument}/{key}: {lo.isot} -- {hi.isot}')
					if not download:
						continue
					if instrument=='LASCO':
						dm.get_LASCO_imgs(lo.isot,hi.isot,dT,LASCO,outdir=folder,need_calib=need_LASCO_calib)
						continue
					attempted.extend(int(i) for i in group)
					try:
						if 'instruments' in inspect.signature(dm.get_SDO_imgs).parameters:
							dm.get_SDO_imgs(lo.isot,hi.isot,cadence,instruments=[instrument],
								waves=[[key]],outdir=folder,nproc=1)
						else:
							dm.get_SDO_imgs(lo.isot,hi.isot,cadence,instrument=instrument,
								wave=float(key) if instrument=='AIA' else key,outdir=folder)
					except Exception as exc:
						print(f'{instrument}/{key}: download unsuccessful; continuing: {exc}')
				if download and missing:
					cached=scan_image_cache(folder,instrument,key,root,window_lo,window_hi)
					times=[r[0] for r in cached]
				if attempted:
					# Record only slots still missing AFTER a real download attempt.
					for i in attempted:
						if not any(abs((q-product_targets[i]).to_value(u.s))<=tolerance for q in times):
							known_missing.add(product_targets[i].isot)
					missing_log.parent.mkdir(parents=True,exist_ok=True)
					missing_log.write_text(''.join(t+'\n' for t in sorted(known_missing)))
				if download and instrument=='LASCO' and diff_type=='running_diff' and not first_frame_intensity:
					pre_start=(start-reference_lookback*u.s).datetime
					pre_handled=any(pre_start<=q<start.datetime for q in requested)
					if not pre_handled and not any(start-reference_lookback*u.s<=q<start for q in times):
						lo,hi=start-reference_lookback*u.s,start-1e-6*u.s
						plan.writerow([instrument,key,lo.isot,hi.isot,'predecessor','download'])
						stream.flush()
						try:
							dm.get_LASCO_imgs(lo.isot,hi.isot,dT,LASCO,outdir=folder,need_calib=need_LASCO_calib)
						except (OSError,RuntimeError,ValueError) as exc:
							print('No pre-sT image retrieved; intensity fallback remains available:',exc)
						cached=scan_image_cache(folder,instrument,key,root,window_lo,window_hi)
				if not missing:
					print(f'{instrument}/{key}: covered; no repeat event request')
				database[instrument if instrument!='AIA' else f'AIA{key}']=cached
		if not np.isfinite(PFSS_dT) or PFSS_dT<=0:
			raise ValueError('PFSS_dT must be positive seconds')
		model_root=Path(pfss_dir).expanduser().resolve()
		model_files=('metadata.json','model_3d.npz','field_lines.npz','boundary.fits')
		# UTC-anchored epochs make snapshots reusable across overlapping runs.
		step=float(PFSS_dT)
		first=np.floor(start.unix/step)*step
		last=np.ceil(end.unix/step)*step
		AR_selected={}
		pfss_models=[]
		for seconds in np.arange(first,last+step/2,step):
			epoch=Time(seconds,format='unix')
			model_folder=model_root/epoch.strftime('%Y%m%d_%H%M%S')
			ready=all((model_folder/name).is_file() for name in model_files)
			if ready:
				metadata=json.loads((model_folder/'metadata.json').read_text())
				ready=all(metadata.get(key)==value for key,value in pfss_kwargs.items())
			if not ready:
				if not download:
					raise FileNotFoundError(f'Missing PFSS snapshot {model_folder}; set download=True')
				print('Generating PFSS snapshot:',epoch.isot)
				so.build_pfss(epoch.isot,epoch.isot,model_folder,**pfss_kwargs)
			else:
				print('Reusing PFSS snapshot:',model_folder)
			pfss_models.append((epoch.isot,str(model_folder)))
			if focus_AR:
				# Same AR rectangle, shared cache (PFSS/<snapshot>/AR_field_lines_rooted.npz) and selection as the flow frames.
				sel=ar_rooted_lines(model_folder,AR_lon,AR_lat,AR_time,ar_half_lon,ar_half_lat,AR_seed_spacing_deg,AR_r_seed,
					NOAA_number,bool(trace_longARrooted_loops),n_closed,n_AR_open)
				out=root/'AR_pfss'/model_folder.name;out.mkdir(parents=True,exist_ok=True)
				np.savez_compressed(out/'AR_selected.npz',**{k:sel[k] for k in ('lon','lat','r','seg','pol')},time=sel['time'].isot,rss=sel['rss'])
				AR_selected[str(model_folder)]=str(out/'AR_selected.npz')
		with (root/'PFSS_snapshots.csv').open('w',newline='') as stream:
			writer=csv.writer(stream)
			writer.writerow(['model_time','folder'])
			writer.writerows(pfss_models)
		lasco=[r for r in database['LASCO'] if start-reference_lookback*u.s<=r[0]<=end]
		sdo=database[SDO_instrument if SDO_instrument!='AIA' else f'AIA{SDO_product}']
		selected=[]
		for target in oh._target_times(sT,eT,dT):
			row=choose_image([r for r in lasco if start<=r[0]<=end],target,max_image_dt,True)
			if row is not None and row not in selected:
				selected.append(row)
		selected.sort(key=lambda row:row[0].mjd)
		if not selected:
			raise ValueError('No LASCO images found near the requested time bins')
		before=[r for r in lasco if r[0]<start]
		predecessor=None
		if before:
			latest=max(r[0].mjd for r in before)
			near=[r for r in before if (latest-r[0].mjd)*86400<=max_image_dt]
			predecessor=min(near,key=lambda r:(route_rank(r[2]),-r[0].mjd))
		base_reference=None
		if diff_type=='base_diff':
			base=Time(base_diff_DT)
			if base<start:
				lo,hi=base-reference_lookback*u.s,base
			elif base>end:
				lo,hi=base,base+reference_lookback*u.s
			else:
				lo,hi=start,end
			refs=scan_image_cache(img_folder,'LASCO',LASCO,root,lo,hi)
			refs=[r for r in refs if lo<=r[0]<=hi]
			if not refs and download:
				print('Downloading base reference within:',lo.isot,'--',hi.isot)
				dm.get_LASCO_imgs(lo.isot,hi.isot,dT,LASCO,outdir=img_folder,need_calib=need_LASCO_calib)
				refs=scan_image_cache(img_folder,'LASCO',LASCO,root,lo,hi)
				refs=[r for r in refs if lo<=r[0]<=hi]
			if not refs:
				raise FileNotFoundError(f'No base reference within {lo.isot} -- {hi.isot}; save an eligible image or enable download=True')
			# Before sT: latest <= base. After eT: earliest >= base. Inside: nearest.
			# Same product family as most frames (a mixed pair cannot be differenced), then route, then time.
			ranks=[r[2] for r in selected]
			main=max(set(ranks),key=ranks.count) if ranks else None
			base_reference=min(refs,key=lambda r:(r[2]!=main,route_rank(r[2]),abs((r[0]-base).to_value(u.s))))
			print('Using base reference:',base_reference[0].isot,base_reference[1])
		frames=[]
		with (root/'image_sources.csv').open('w',newline='') as stream:
			writer=csv.writer(stream)
			writer.writerow(['datetime','instrument','source','path','reference','status'])
			jobs=[]
			for i,row in enumerate(selected):
				reference=base_reference if diff_type=='base_diff' else predecessor
				if i==0 and first_frame_intensity:
					reference=None
				jobs.append((row,reference,min(sdo,key=lambda r:abs((r[0]-row[0]).to_value(u.s))) if sdo else None))
				predecessor=row
			settings={key:globals()[key] for key in ('root','LASCO','SDO_label','SDO_instrument',
				'SDO_product','npix','dpi','pfss_models','focus_AR','n_global_closed','AR_selected','diff_type','closed_field_color','AR_closedB_col','AR_openB_col')}
			jobs=[(job,settings) for job in jobs]
			if nproc==1:
				_init_frame_worker({})
				outputs=map(_safe_overlay_frame,jobs)
				for png,records in outputs:
					writer.writerows(records)
					if png is not None:
						frames.append(png)
			else:
				with get_context('spawn').Pool(nproc,initializer=_init_frame_worker,initargs=({},)) as pool:
					for png,records in pool.imap(_safe_overlay_frame,jobs,chunksize=1):
						writer.writerows(records)
						stream.flush()
						if png is not None:
							frames.append(png)
		goes_dir=Path(goes_folder).expanduser().resolve();goes_dir.mkdir(parents=True,exist_ok=True)
		found=covering_goes(goes_dir,sT,eT) if goes_pickle is None else None  # any cached file spanning sT-eT
		if goes_pickle is None and found is None:
			tag=dt.fromisoformat(sT).strftime('%Y%m%d_%H%M%S')+'-'+dt.fromisoformat(eT).strftime('%Y%m%d_%H%M%S')
			goes_pickle=goes_dir/f'GOES_data_deriv_{tag}.p'
			if not goes_pickle.is_file():
				# The frozen acquisition module uses its existing cache convention.
				legacy_tag=dt.fromisoformat(sT).isoformat().replace('-','').replace(':','')+'-'+dt.fromisoformat(eT).isoformat().replace('-','').replace(':','')
				legacy=goes_dir/f'GOES_data_deriv_{legacy_tag}.p'
				if not legacy.is_file():
					dm.plot_goes_derivative(sT,eT,savedata=True,outdir=str(goes_dir))
				shutil.copy2(legacy,goes_pickle)
				pdf=legacy.with_suffix('.pdf')
				if pdf.is_file():
					shutil.copy2(pdf,goes_pickle.with_suffix('.pdf'))
		if found is not None:goes=found
		else:
			with Path(goes_pickle).open('rb') as fil:
				g=pickle.load(fil)
			goes=pd.DataFrame({'flux':g['Flux']},index=pd.to_datetime(g['Time'],utc=True))
		if nproc==1 or not frames:
			paths=save_lightcurve_frames(frames,goes,lat_csv,sT,eT,Path(final_image_folder).expanduser(),dpi=dpi)
		else:
			settings={'_frame_goes':goes,'lat_csv':lat_csv,'sT':sT,'eT':eT,
				'final_image_folder':final_image_folder,'dpi':dpi}
			with get_context('spawn').Pool(nproc,initializer=_init_frame_worker,initargs=({},)) as pool:
				paths=[p for group in pool.imap(_render_lightcurve_frame,((frame,settings) for frame in frames),chunksize=1) for p in group]
		print(f'Saved {len(paths)} PNG frames in {Path(final_image_folder).expanduser().resolve()}')

		# MP4 of this run's frames, in time order (450 ms per frame), saved beside them.
		import os,shutil,subprocess,tempfile
		frames_png=sorted(str(p) for p in paths if str(p).lower().endswith('.png'))
		if not frames_png:
			print('No PNG frames: MP4 not written')
		elif shutil.which('ffmpeg') is None:
			print('ffmpeg not found on PATH: MP4 not written')
		else:
			folder=Path(frames_png[0]).resolve().parent
			mp4=folder/f'{folder.name}.mp4'
			with tempfile.TemporaryDirectory() as tmp:
				for i,p in enumerate(frames_png):
					os.symlink(Path(p).resolve(),Path(tmp)/f'{i:05d}.png')
				subprocess.run(['ffmpeg','-y','-v','error','-framerate','20/9','-i',str(Path(tmp)/'%05d.png'),
					'-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white','-c:v','libx264','-crf','15',
					'-pix_fmt','yuv420p','-movflags','+faststart',str(mp4)],check=True)
			print(f'Saved MP4 of {len(frames_png)} frames: {mp4}')
	finally:
		os.chdir(cwd)
