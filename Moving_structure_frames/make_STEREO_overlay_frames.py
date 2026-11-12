"""STEREO COR1/COR2 overlay frames: observer-projected PFSS plus GOES/LAT panels.

Single combined task script and function module. Edit the INPUT block, then run:
	python make_STEREO_overlay_frames.py
Importing this file only defines the INPUT values and functions; nothing runs.
Requires Multiwb_datamine, datamine_calib_helper, solar_overlay and
solar_overlay_helper on the import path, including get_STEREO_imgs,
_stereo_matches and _is_preview_image.
"""
from pathlib import Path
from datetime import datetime as dt, timedelta
from functools import lru_cache
from multiprocess import get_context
import csv,json,copy,sys,warnings,re,urllib.request
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
from sunpy.coordinates import frames,get_horizons_coord
import sunpy.map
from sunpy.visualization import colormaps as _sunpy_cmaps  # Register standard EUVI tables
from sunpy.net import Fido,attrs as a
import sunpy.timeseries as ts
import Multiwb_datamine as dm
import datamine_calib_helper as dh
import solar_overlay as so
import solar_overlay_helper as oh
from pfss_projection import ar_rooted_lines  # shared AR selection with the flow frames

#################### INPUT #################################################
basedir='/path/to/event_folder'  # Event folder: all relative paths below live inside it
sT='YYYY-MM-DDThh:mm:ss'  # Event start, UTC
eT='YYYY-MM-DDThh:mm:ss'  # Event end, UTC
dT=600                             # COR images/output cadence, seconds
spacecraft='A'                     # 'A' or 'B'
instrument='COR2'                  # 'COR1' or 'COR2'
EUVI_needed=True                   # Overlay ONE non-differenced EUVI channel
EUVI_wave=195                      # 171,195,284 or 304 Angstrom
EUVI_dT=7200                       # EUVI cache/download cadence, seconds
EUVI_radius=1.3                    # Visible EUVI radius in solar radii
EUVI_BW=False                     # True: display EUVI in grayscale; False: standard channel colors
polar=0                            # COR1 single-polarizer FITS: 0,120,240
img_folder='./STEREO'               # STEREO database ROOT (files in STEREO_<sc>/<instrument>), given to get_STEREO_imgs
outdir='./STEREO_LAT_frames'
final_image_folder='./STEREO_LAT_frames/pfss_Cor2bd_E195_LC'
need_GOES=True
user_datacsv='LAT_fluxLC.csv'  # Same Fermi-LAT file used by LASCO script
goes_pickle=None                  # Optional LASCO GOES derivative .p (Time,Flux); None: STEREO GOES cache/Fido
goes_folder='./GOES'  # GOES cache: read if present, downloads saved here. None: <outdir>/GOES

diff_type='base_diff'            # 'running_diff', 'base_diff', 'intensity'
first_frame_intensity=False
base_diff_DT='YYYY-MM-DDThh:mm:ss'  # Quiet base-difference reference, UTC, at or before sT
reference_search=3600              # Bounded reference-search window, seconds
download=True                     # Fill missing image bins / PFSS snapshots
helioviewer=True                   # Save/use separate original JP2 previews
need_pfss=True
pfss_dir='./PFSS'  # Reuse the same existing 3D PFSS database

### PFSS parameters ####
PFSS_dT=6.                         # PFSS snapshot cadence, HOURS
save_PFSS_maps=True     # PFSS maps at the sT-eT midpoint in basedir (PFSS_*.png), on the disk image or a blank Stonyhurst-grid disk: True/'both', 'full' (full Sun only), 'AR' (AR only), False (skip, no redraw)
rss= 2.5
nrho=35
r_seed=1.05
n_lon=36
n_lat=18

### Active-region PFSS tracing ####
focus_AR=True                    # True: add AR-seeded traces
NOAA_number=None  # Set the NOAA active-region number (e.g. 11402) when focus_AR is on
SRS_file=None                      # Optional local NOAA SRS text file for offline use
SRS_folder='./NOAA_SRS'            # NOAA SRS reports: read here, downloads saved here (shared with LASCO and flow codes)
AR_lookup_max_age=36               # Maximum age (hours) for the NOAA region summary location
AR_lon=None                        # Optional manual Stonyhurst longitude (deg, west + / east -)
AR_lat=None                        # Optional manual Stonyhurst latitude (deg, north + / south -)
AR_time=sT                         # UTC epoch for NOAA lookup and/or manual AR_lon/AR_lat
ar_half_lon=12.          # AR rectangle half-width in longitude, deg (Stonyhurst, about the NOAA centre)
ar_half_lat=12.          # AR rectangle half-width in latitude, deg
AR_seed_spacing_deg=1.   # Seed spacing over the AR rectangle, deg (+-12 deg at 1 deg = 25 x 25 seeds)
trace_longARrooted_loops=1 # 1: loops with one footpoint in the rectangle (large AR loops kept); 0: both footpoints inside
AR_loop_reach_deg=50.   # Closed AR loops need both ends within this distance (deg on the Sun) of the NOAA centre: keeps AR-quiet Sun and AR-AR connectors, drops longer PFSS loops (e.g. to the polar caps); '' = no limit
AR_r_seed=1.01           # AR trace start radius, Rsun
n_closed=30              # Max AR closed loops drawn (tallest first with trace_longARrooted_loops=1); np.nan: all
n_AR_open=5              # Max open AR field lines drawn (magenta), spread evenly; np.nan: all
AR_closedB_col='lime'
AR_openB_col='magenta'

observer_location=None            # None: Spacecraft location longitude w.r.t Earth-facing solar disk longitude, else JPL Horizons.
									# observer_location='W36.785'       # Manual Stonyhurst longitude: W positive
observer_latitude=None             # Degrees; None retains header/Horizons value
observer_distance_au=None          # AU; None retains header/Horizons value
nproc=2                           # Rendering workers; avoid high memory use
npix=1024
dpi=90
n_global_closed=20       # Global closed PFSS loops drawn
closed_color='white'
###########################################################################

flux_csv=Path(user_datacsv) if user_datacsv else None  # inside basedir
                                      # UTC datetime,flux,flux_error (+ optional UL/errors)
pfss_kwargs={'rss':rss,'nrho':nrho,'r_seed':r_seed,'n_lon':n_lon,'n_lat':n_lat}
open_positive_color='red'
open_negative_color='cyan'

def stereo_cache(folder,spacecraft,instrument,polar=0,wave=None):
	'''Read actual image metadata recursively; retain FITS and JP2 separately.'''
	rows=[]
	for path in sorted(Path(folder).expanduser().resolve().rglob('*')):
		if not path.is_file() or not path.name.lower().endswith(('.fits','.fit','.fts','.fits.gz','.fit.gz','.fts.gz','.jp2','.j2k')):
			continue
		try:
			m=dh.load_map(path)
			preview=oh._is_preview_image(m,path)
			# JP2 preview polarizers are display products, not COR1 single-angle FITS.
			if preview:
				obs=str(m.observatory).upper().replace('_','').replace('-','').replace(' ','')
				if obs not in ('STEREO'+spacecraft,'STEREO'+('AHEAD' if spacecraft=='A' else 'BEHIND')) or str(m.detector).upper()!=instrument:
					continue
			elif not dh._stereo_matches(m,spacecraft,instrument,wave,polar):
				continue
			if instrument=='EUVI' and not np.isclose(m.wavelength.to_value(u.angstrom),wave):
				continue
			rows.append((m.date,str(path),preview))
		except Exception as exc:
			warnings.warn(f'Skipping cached image {path}: {exc}')
	return sorted(rows,key=lambda r:(r[0].mjd,r[2],r[1]))

def _stereo_rank(row):
	"""0 = Level-1 (calibrated MSB) FITS, 1 = Helioviewer, 2 = Level-0.5 FITS."""
	if row[2]:
		return 1
	from astropy.io import fits
	try:
		h=fits.getheader(row[1])
		unit=str(h.get('BUNIT','')).replace(' ','').lower()
		return 0 if unit in ('msb','bsun','b_sun','meansolarbrightness') else 2
	except (OSError,ValueError):
		return 2

def stereo_observer(m,spacecraft,location=None,latitude=None,distance_au=None):
	'''Return Stonyhurst observer geometry: W positive, E negative; no pixel rotation.'''
	meta=m.meta
	if all(k in meta for k in ('hgln_obs','hglt_obs','dsun_obs')):
		lon,lat,dist=(float(meta[k]) for k in ('hgln_obs','hglt_obs','dsun_obs'))
		source='image header'
	elif location is not None and latitude is not None and distance_au is not None:
		lon,lat,dist=0.,float(latitude),(float(distance_au)*u.au).to_value(u.m)
		source='manual observer'
	else:
		obs=get_horizons_coord('STEREO-'+spacecraft,m.date).transform_to(
			frames.HeliographicStonyhurst(obstime=m.date))
		lon,lat,dist=obs.lon.to_value(u.deg),obs.lat.to_value(u.deg),obs.radius.to_value(u.m)
		source='JPL Horizons'
	if location is not None:
		if isinstance(location,str):
			value=location.strip().upper()
			lon=float(value[1:])*(1 if value[0]=='W' else -1) if value[:1] in ('W','E') else float(value)
		else:
			lon=float(location)
		source+='; manual longitude override'
	if latitude is not None:
		lat=float(latitude)
	if distance_au is not None:
		dist=(float(distance_au)*u.au).to_value(u.m)
	if not np.all(np.isfinite([lon,lat,dist])) or not -90<=lat<=90 or dist<=0:
		raise ValueError('Invalid observer longitude, latitude or distance.')
	return dict(hgln_obs=lon,hglt_obs=lat,dsun_obs=dist,source=source)

def _observer_label(observer):
	"""Compact spacecraft location: W/E whole degrees and Sun distance in AU."""
	lon=float(observer['hgln_obs'])
	dist=float((observer['dsun_obs']*u.m).to_value(u.au))
	return f"{'W' if lon>=0 else 'E'}{abs(lon):.0f}, {dist:.2f}AU"

@lru_cache(maxsize=2)
def _field_lines(folder):
	return oh._load_pfss(folder)

@lru_cache(maxsize=3)
def _ar_field_lines(path):
	'''Load traced AR lines for a single saved PFSS snapshot.'''
	with np.load(path,allow_pickle=False) as saved:
		out={k:saved[k].copy() for k in ('lon','lat','r','seg','pol')}
		out['time']=Time(str(saved['time'].item()))
		out['rss']=float(saved['rss'])
	return out

def find_NOAA_location(number,requested,cache_dir,local_file=None,max_age=36):
	"""Return Stonyhurst lon, lat, validity epoch and NOAA SRS provenance."""
	if number is None or int(number)!=number or int(number)<=0:
		raise ValueError('Set NOAA_number to the NOAA active-region number')
	number=int(number)
	target=dt.fromisoformat(requested.replace('Z','+00:00'))
	if target.tzinfo is not None:
		target=pd.Timestamp(target).tz_convert('UTC').tz_localize(None).to_pydatetime()
	cache=Path(cache_dir)
	cache.mkdir(parents=True,exist_ok=True)
	reports=[];failures=[]
	for offset in ([0] if local_file is not None else [-1,0,1]):
		day=target+timedelta(days=offset)
		url=f'https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_region_summaries/{day:%Y/%m/%Y%m%d}SRS.txt'
		path=Path(local_file) if local_file is not None else cache/f'{day:%Y%m%d}_000000_SRS.txt'
		try:
			if not path.is_file():
				if local_file is not None:raise FileNotFoundError(path)
				with urllib.request.urlopen(url,timeout=30) as response:
					text=response.read().decode('utf-8',errors='replace')
				if 'Solar Region Summary' not in text and 'SOLAR REGION SUMMARY' not in text:
					raise ValueError('Response is not a NOAA SRS report')
				path.write_text(text)
			text=path.read_text()
			issued=re.search(r':Issued:\s*(\d{4}\s+\w{3}\s+\d{1,2}\s+\d{4})\s+UTC',text,re.I)
			if issued:issue=dt.strptime(issued[1],'%Y %b %d %H%M')
			else:
				issued=re.search(r'Issued at (\d{4})Z on (\d{1,2} \w{3} \d{4})',text,re.I)
				if issued is None:raise ValueError('Missing issue date in SRS header')
				issue=dt.strptime(issued[2]+' '+issued[1],'%d %b %Y %H%M')
			valid=re.search(r'(?:Locations|Positions) Valid at\s+(\d{2})/(\d{2})(\d{2})Z',text,re.I)
			if valid is None:raise ValueError('Missing location-validity epoch in SRS header')
			dd,hh,mm=map(int,valid.groups())
			if not (0<=hh<=24 and 0<=mm<60 and (hh<24 or mm==0)):
				raise ValueError('Invalid SRS validity time')
			epochs=[]
			for shift in (-2,-1,0,1):
				candidate=issue.replace(hour=0,minute=0,second=0,microsecond=0)+timedelta(days=shift)
				if candidate.day==dd:epochs.append(candidate+timedelta(hours=hh,minutes=mm))
			if not epochs:raise ValueError('Cannot resolve SRS validity date')
			epoch=min(epochs,key=lambda t:abs((t-issue).total_seconds()))
			disk=re.split(r'^\s*II\.',text,flags=re.M)[0]
			for match in re.finditer(r'^\s*(\d{4,5})\s+([NS])(\d{2})([EW])(\d{2})\b',disk,re.M):
				n=int(match[1])
				if n!=number and not (n<10000 and number>=10000 and n==number%10000):continue
				latitude=int(match[3])*(1 if match[2]=='N' else -1)
				longitude=int(match[5])*(1 if match[4]=='W' else -1)
				age=abs((epoch-target).total_seconds())/3600
				if age<=max_age:
					reports.append(dict(noaa=number,lon=longitude,lat=latitude,epoch=epoch.isoformat(),issue=issue.isoformat(),offset_hours=age,file=str(path.resolve()),source=str(path.resolve()) if local_file else url))
		except (OSError,ValueError) as exc:
			failures.append(f'{path.name}: {exc}')
	if not reports:
		raise ValueError(f'NOAA {number}: no disk location within {max_age} h of {requested}. Check the number/date or supply SRS_file. '+'; '.join(failures))
	best=min(reports,key=lambda row:row['offset_hours'])
	(cache/'AR_location.json').write_text(json.dumps(best,indent=2))
	print(f'NOAA {number}: lon={best["lon"]:+g}, lat={best["lat"]:+g}; valid {best["epoch"]} UT')
	return best['lon'],best['lat'],best['epoch'],best


def _draw_stereo_pfss(ax,tmap,observer,model,cfg):
	"""Project the saved 3-D Carrington field into this STEREO exposure."""
	obs=tmap.observer_coordinate.transform_to(frames.HeliographicStonyhurst(obstime=tmap.date))
	dlon=(obs.lon.to_value(u.deg)-observer['hgln_obs']+180)%360-180
	dlat=obs.lat.to_value(u.deg)-observer['hglt_obs']
	distance=obs.radius.to_value(u.m)
	if abs(dlon)>.1 or abs(dlat)>.1 or not np.isclose(distance,observer['dsun_obs'],rtol=1e-5,atol=0):
		raise ValueError('STEREO observer WCS mismatch: PFSS would be misaligned')
	# The helper transforms Carrington points into the target HPC WCS, including
	# the observer's longitude, latitude and distance. No longitude offset or
	# image-plane rotation is applied to the saved field.
	oh._draw_pfss(ax,tmap,_field_lines(model),N_Bloop=cfg['n_global_closed'],
		colors={0:cfg['closed_color'],1:cfg['open_positive_color'],-1:cfg['open_negative_color']})
	if cfg.get('ar_model') is not None:
		oh._draw_pfss(ax,tmap,_ar_field_lines(cfg['ar_model']),N_Bloop=np.nan,  # already selected and capped
			colors={0:cfg['AR_closedB_col'],1:cfg['AR_openB_col'],-1:cfg['AR_openB_col']})


def _stereo_frame(task):
	'''One independent frame; never subtract native FITS from JP2 display pixels.'''
	row,reference,observer,ref_observer,model,cfg=task
	t,path,preview=row
	try:
		plt.switch_backend('Agg')
		m=dh.load_map(path)
		meta=copy.deepcopy(m.meta)
		meta.update({k:observer[k] for k in ('hgln_obs','hglt_obs','dsun_obs')})
		if preview:
			dh._hv_tags(meta,path)  # Same tags as every JP2 -> FITS route.
		m=sunpy.map.Map(np.asarray(m.data,float),meta)
		if reference is not None:
			ref=dh.load_map(reference[1])
			if preview!=reference[2]:
				raise ValueError('Mixed FITS/JP2 difference pair')
			if not preview and (str(m.unit)!=str(ref.unit) or m.meta.get('polar')!=ref.meta.get('polar')):
				raise ValueError('Incompatible FITS units or polarizer angles')
			if not preview and str(m.unit).lower() in ('dn','adu','ct','count','counts'):
				if not np.isclose(m.exposure_time.to_value(u.s),ref.exposure_time.to_value(u.s)):
					raise ValueError('Raw count exposures differ; supply consistently prepared FITS')
			refmeta=copy.deepcopy(ref.meta)
			refmeta.update({k:ref_observer[k] for k in ('hgln_obs','hglt_obs','dsun_obs')})
			ref=sunpy.map.Map(np.asarray(ref.data,float),refmeta)
			meta.pop('blank',None)
			meta.update({'diff_ref':ref.date.isot,'diff_typ':cfg['diff_type'],
				'diff_dt':float((m.date-ref.date).to_value(u.s))})
			m=sunpy.map.Map(m.data-oh._regrid(ref,m),meta)
			folder=Path(cfg['outdir'])/'difference_FITS'/cfg['spacecraft']/cfg['instrument']/cfg['diff_type']/('JP2' if preview else 'FITS')
		else:
			folder=Path(cfg['outdir'])/'intensity_FITS'/cfg['spacecraft']/cfg['instrument']/('JP2' if preview else 'FITS')
		folder.mkdir(parents=True,exist_ok=True)
		m.save(folder/f'{cfg["instrument"]}_{t.strftime("%Y%m%d_%H%M%S")}.fits',overwrite=True)
		tmap=oh._target_map(m,cfg['npix'])
		coords=sunpy.map.all_coordinates_from_map(tmap)
		rho=np.hypot(coords.Tx.to_value(u.arcsec),coords.Ty.to_value(u.arcsec))/tmap.rsun_obs.to_value(u.arcsec)
		data=np.asarray(tmap.data,float).copy()
		data[(rho<cfg['r_inner'])|(rho>cfg['r_outer'])]=np.nan
		valid=data[np.isfinite(data)]
		if not valid.size:
			raise ValueError('No valid pixels in coronagraph field of view')
		if reference is not None:
			limit=max(float(np.percentile(abs(valid),99)),1e-30)
			lo,hi=-limit,limit
		else:
			lo,hi=np.percentile(valid,[1,99.5])
			if hi<=lo:hi=lo+1
		fig=Figure(figsize=(8,8),dpi=cfg['dpi'])
		ax=fig.add_subplot(projection=tmap)
		ax.set_facecolor('black')
		ax.imshow(data,origin='lower',cmap='gray',vmin=lo,vmax=hi)
		euvi_note=''
		if cfg.get('euvi_row') is not None:
			try:
				euvi=cfg['euvi_row'];emap=dh.load_map(euvi[1])
				emeta=copy.deepcopy(emap.meta)
				emeta.update({k:cfg['euvi_observer'][k] for k in ('hgln_obs','hglt_obs','dsun_obs')})
				emap=sunpy.map.Map(np.asarray(emap.data,float),emeta)
				e=oh._regrid(emap,tmap)
				mask=np.isfinite(e)&(rho<=cfg['EUVI_radius'])
				if not np.any(mask):raise ValueError('No projected EUVI pixels inside disk mask')
				cmap,norm=dh._image_display_settings(emap,'EUVI',cfg['EUVI_wave'],euvi[2])
				if cfg['EUVI_BW']:cmap='gray'
				# ImageNormalize(clip=True) discards masks and paints outside EUVI white.
				norm.clip=False
				ax.imshow(np.ma.masked_where(~mask,e),origin='lower',cmap=cmap,norm=norm,
					alpha=mask.astype(float))
				euvi_note=f" + EUVI {cfg['EUVI_wave']} Å"
				ax.text(.02,.94,f"EUVI: {euvi[0].strftime('%H:%M:%S UT')}"+(' (JP2)' if euvi[2] else ''),
					transform=ax.transAxes,va='top',fontsize=10,color='white',
					bbox=dict(facecolor='black',alpha=.6,pad=2))
			except Exception as exc:
				warnings.warn(f'EUVI overlay omitted for {t.isot}: {exc}')
				with (Path(cfg['outdir'])/'EUVI_overlay_errors.txt').open('a') as stream:
					stream.write(f'{t.isot}: {exc}\n')
		if model is not None:_draw_stereo_pfss(ax,tmap,observer,model,cfg)
		ax.set_xlabel('Solar X (arcsec)',fontsize=16)
		ax.set_ylabel('Solar Y (arcsec)',fontsize=16)
		mode=cfg['diff_type'].replace('_',' ') if reference is not None else 'intensity'
		ax.set_title(f'STEREO-{cfg["spacecraft"]} {cfg["instrument"]}: {mode}'+euvi_note,fontsize=17)
		ax.text(.02,.98,t.strftime('%Y-%m-%d %H:%M:%S UT'),transform=ax.transAxes,
			va='top',color='white',bbox=dict(facecolor='black',alpha=.6,pad=3))
		ax.text(.98,.98,f'STEREO-{cfg["spacecraft"]}: '+_observer_label(observer),
			transform=ax.transAxes,va='top',ha='right',color='white',fontsize=9,
			bbox=dict(facecolor='black',alpha=.6,pad=2))
		if preview:
			ax.text(.02,.02,'JP2 display product',transform=ax.transAxes,color='white',fontsize=10)
		fig.tight_layout()
		png=Path(cfg['outdir'])/'overlay_images'/cfg['spacecraft']/cfg['instrument']/cfg['diff_type']/f'overlay_{t.strftime("%Y%m%d_%H%M%S")}.png'
		png.parent.mkdir(parents=True,exist_ok=True)
		fig.savefig(png,dpi=cfg['dpi'])
		fig.clear()
		return str(png),[t.isot,path,'JP2 preview' if preview else 'FITS',reference[1] if reference else '',observer['source'],'saved']
	except Exception as exc:
		print(f'Frame {t.isot} skipped: {exc}')
		return None,[t.isot,path,'JP2 preview' if preview else 'FITS',reference[1] if reference else '',observer.get('source',''),str(exc)]

def make_STEREO_frames(sT,eT,dT,spacecraft='A',instrument='COR2',img_folder='./STEREO',
	outdir='./STEREO_frames',final_image_folder=None,diff_type='running_diff',
	first_frame_intensity=True,base_diff_DT=None,reference_search=3600,polar=0,
	download=True,helioviewer=True,need_pfss=True,pfss_dir='./PFSS',PFSS_dT=6.,
	pfss_kwargs=None,observer_location=None,observer_latitude=None,observer_distance_au=None,
	nproc=2,npix=1024,dpi=90,n_global_closed=20,closed_color='yellow',
	open_positive_color='red',open_negative_color='cyan',need_GOES=True,flux_csv=None,
	EUVI_needed=False,EUVI_wave=195,EUVI_dT=7200,EUVI_radius=1.3,EUVI_BW=False,
	focus_AR=False,AR_lon=None,AR_lat=None,AR_time=None,ar_half_lon=12.,
	ar_half_lat=12.,AR_seed_spacing_deg=1.,trace_longARrooted_loops=1,AR_r_seed=1.01,AR_loop_reach_deg=None,
	n_closed=30,n_AR_open=5,AR_closedB_col='lime',AR_openB_col='magenta',goes_pickle=None,goes_folder=None,SRS_folder=None,
		NOAA_number=None,SRS_file=None,AR_lookup_max_age=36):
	'''COR1/COR2 intensity/difference frames with PFSS in the spacecraft viewpoint.

	PFSS_dT is hours. dT/reference_search are seconds. location=None uses headers,
	then Horizons; W37/E37 overrides Stonyhurst longitude. No image pixels rotate
	to fake a different observer. COR1 polar selects one raw polarizer angle.
	Neither COR1 nor COR2 is newly SECCHI_PREP-calibrated by this routine.
	The GOES/LAT lower panel follows Earth reception times, with no light-time shift.
	EUVI_needed adds one intensity EUVI layer, in standard colors or grayscale
	(EUVI_BW=True), for either FITS or JP2. EUVI_dT controls downloads; overlay uses the closest
	available EUVI timestamp without a time cutoff, independent of COR cadence.
	Missing EUVI leaves the COR/PFSS frame intact and is recorded in EUVI_sources.csv.
	For AR tracing, NOAA_number retrieves Stonyhurst AR location from NOAA SRS,
	or AR_lon/AR_lat can be given manually at AR_time (west/north positive).
	The fixed Carrington seeds are projected from each STEREO
	image observer (header/Horizons); they are NOT manually longitude-shifted.
	AR trace files are saved in outdir/AR_pfss, not the shared LASCO PFSS cache.
	flux_csv reads the same LAT flux/error table as the LASCO frame script.
	Return saved PNG paths; source/skipped-frame and observer CSV logs are in outdir.
	'''
	spacecraft=str(spacecraft).upper();instrument=str(instrument).upper()
	if spacecraft not in ('A','B') or instrument not in ('COR1','COR2'):
		raise ValueError('Choose spacecraft A/B and instrument COR1/COR2.')
	if diff_type not in ('running_diff','base_diff','intensity'):
		raise ValueError('diff_type must be running_diff, base_diff or intensity.')
	if int(nproc)!=nproc or nproc<1 or not np.isfinite(PFSS_dT) or PFSS_dT<=0:
		raise ValueError('nproc must be a positive integer; PFSS_dT positive hours.')
	if not np.isfinite(reference_search) or reference_search<=0:
		raise ValueError('reference_search must be positive seconds.')
	if EUVI_needed and (EUVI_wave not in (171,195,284,304) or not np.isfinite(EUVI_dT) or EUVI_dT<=0 or not 0<EUVI_radius<=1.7):
		raise ValueError('EUVI: choose wave 171/195/284/304, positive cadence and radius <=1.7 Rsun.')
	root=Path(outdir).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
	resolved_ar=None
	if focus_AR:
		if not need_pfss:raise ValueError('focus_AR=True requires need_pfss=True')
		if min(ar_half_lon,ar_half_lat,AR_seed_spacing_deg)<=0 or AR_r_seed<=1:
			raise ValueError('AR rectangle and seed spacing must be positive, AR_r_seed >1 Rsun')
		AR_time=Time(AR_time or sT).isot
		if NOAA_number is not None:
			AR_lon,AR_lat,AR_time,resolved_ar=find_NOAA_location(int(NOAA_number),AR_time,Path(SRS_folder).expanduser().resolve() if SRS_folder else root/'NOAA_SRS',SRS_file,AR_lookup_max_age)
		elif AR_lon is None or AR_lat is None:
			raise ValueError('When focus_AR=True, set NOAA_number or provide manual AR_lon and AR_lat')
		if not np.all(np.isfinite([AR_lon,AR_lat])) or not -180<=AR_lon<=180 or not -90<=AR_lat<=90:
			raise ValueError('Invalid resolved/manual AR coordinates')
	if need_GOES and flux_csv is not None:
		flux_csv=Path(flux_csv).expanduser()
		if not flux_csv.is_absolute():flux_csv=Path.cwd()/flux_csv
		if not flux_csv.is_file():raise FileNotFoundError(f'LAT flux CSV missing: {flux_csv}')
	if need_GOES and goes_pickle is not None:
		goes_pickle=Path(goes_pickle).expanduser()
		if not goes_pickle.is_absolute():goes_pickle=Path.cwd()/goes_pickle
		if not goes_pickle.is_file():raise FileNotFoundError(f'GOES pickle missing: {goes_pickle}')
	targets=dh._target_times(sT,eT,dT);start,end=Time(sT),Time(eT)
	img_folder=Path(img_folder).expanduser().resolve();img_folder.mkdir(parents=True,exist_ok=True)
	rows=stereo_cache(img_folder,spacecraft,instrument,polar)
	# Download only cadence bins not already represented, grouping consecutive gaps.
	missing=[i for i,t in enumerate(targets) if not any(abs((r[0]-t).to_value(u.s))<=dT/2 for r in rows)]
	groups=[]
	for i in missing:
		if groups and i==groups[-1][-1]+1:groups[-1].append(i)
		else:groups.append([i])
	if download:
		for group in groups:
			lo=max(start,targets[group[0]]-dT/2*u.s);hi=min(end,targets[group[-1]]+dT/2*u.s)
			if hi<=lo:continue
			try:
				dm.get_STEREO_imgs(lo.isot,hi.isot,dT,spacecraft=spacecraft,
					instruments=[instrument],waves=[None],outdir=str(img_folder),
					nproc=1,polar=polar,helioviewer=helioviewer)
			except Exception as exc:warnings.warn(f'STEREO download failed: {exc}')
		rows=stereo_cache(img_folder,spacecraft,instrument,polar)
	selected=[];uncovered=[]
	for t in targets:
		near=[r for r in rows if start<=r[0]<=end and abs((r[0]-t).to_value(u.s))<=dT/2]
		if near:
			# Route: Level-1 -> Helioviewer -> Level-0.5, then nearest in time.
			row=min(near,key=lambda r:(_stereo_rank(r),abs((r[0]-t).to_value(u.s))))
			if row not in selected:selected.append(row)
		else:uncovered.append([t.isot,'','','','','skipped: no exposure near requested bin'])
	selected.sort(key=lambda r:r[0].mjd)
	if not selected:raise FileNotFoundError('No usable COR images inside the requested interval.')
	base=Time(base_diff_DT) if diff_type=='base_diff' and base_diff_DT is not None else start
	lo,hi=(start-reference_search*u.s,start-1e-6*u.s) if diff_type=='running_diff' else (
		(base-reference_search*u.s,base) if base<start else (base,base+reference_search*u.s) if base>end else (start,end))
	refs=[r for r in rows if lo<=r[0]<=hi]
	if diff_type!='intensity' and not refs and download and (diff_type=='base_diff' or not first_frame_intensity):
		try:
			dm.get_STEREO_imgs(lo.isot,hi.isot,dT,spacecraft=spacecraft,
				instruments=[instrument],waves=[None],outdir=str(img_folder),
				nproc=1,polar=polar,helioviewer=helioviewer)
		except Exception as exc:warnings.warn(f'Reference retrieval failed: {exc}')
		rows=stereo_cache(img_folder,spacecraft,instrument,polar)
		refs=[r for r in rows if lo<=r[0]<=hi]
	models=[];ar_models={}
	if need_pfss:
		step=float(PFSS_dT)*3600;kwargs=pfss_kwargs or {}
		for epoch_s in np.arange(np.floor(start.unix/step)*step,np.ceil(end.unix/step)*step+step/2,step):
			epoch=Time(epoch_s,format='unix');folder=Path(pfss_dir).expanduser().resolve()/epoch.strftime('%Y%m%d_%H%M%S')
			ready=all((folder/name).is_file() for name in ('metadata.json','field_lines.npz','model_3d.npz','boundary.fits'))
			if ready:
				meta=json.loads((folder/'metadata.json').read_text())
				ready=all(meta.get(k)==v for k,v in kwargs.items())
			if not ready:
				if not download:raise FileNotFoundError(f'Missing matching PFSS model: {folder}')
				so.build_pfss(epoch.isot,epoch.isot,folder,**kwargs)
			models.append((epoch,str(folder)))
			if focus_AR:
				# Same AR rectangle, shared cache (PFSS/<snapshot>/AR_field_lines_rooted.npz) and selection as the flow frames.
				sel=ar_rooted_lines(folder,AR_lon,AR_lat,AR_time,ar_half_lon,ar_half_lat,AR_seed_spacing_deg,AR_r_seed,
					NOAA_number or 0,bool(trace_longARrooted_loops),n_closed,n_AR_open,
					reach_deg=None if AR_loop_reach_deg in ('',None) else float(AR_loop_reach_deg))
				out=root/'AR_pfss'/folder.name;out.mkdir(parents=True,exist_ok=True)
				np.savez_compressed(out/'AR_selected.npz',**{k:sel[k] for k in ('lon','lat','r','seg','pol')},time=sel['time'].isot,rss=sel['rss'])
				ar_models[str(folder)]=str(out/'AR_selected.npz')
	if focus_AR:
		with (root/'AR_trace_sources.csv').open('w',newline='') as stream:
			writer=csv.writer(stream);writer.writerow(['PFSS_model','AR_trace_file','AR_lon_deg','AR_lat_deg','AR_epoch','NOAA_number'])
			for model,path in ar_models.items():writer.writerow([model,path,AR_lon,AR_lat,Time(AR_time).isot,NOAA_number])
		meta={'AR_lon_deg':AR_lon,'AR_lat_deg':AR_lat,'AR_epoch':Time(AR_time).isot,'NOAA_number':NOAA_number,'SRS_file':str(SRS_file) if SRS_file is not None else None,'AR_lookup_max_age_hr':AR_lookup_max_age}
		if resolved_ar is not None:meta.update(resolved_ar)
		(root/'AR_location_used.json').write_text(json.dumps(meta,indent=2))
	cfg=dict(spacecraft=spacecraft,instrument=instrument,outdir=str(root),diff_type=diff_type,
		npix=npix,dpi=dpi,n_global_closed=n_global_closed,closed_color=closed_color,
		open_positive_color=open_positive_color,open_negative_color=open_negative_color,
		r_inner=1.4 if instrument=='COR1' else 2.5,r_outer=4. if instrument=='COR1' else 15.,
		AR_closedB_col=AR_closedB_col,AR_openB_col=AR_openB_col)
	euvi_rows=[];euvi_log=[]
	if EUVI_needed:
		euvi_rows=stereo_cache(img_folder,spacecraft,'EUVI',wave=EUVI_wave)
		euvi_targets=dh._target_times(sT,eT,EUVI_dT)
		missing=[i for i,t in enumerate(euvi_targets) if not any(
			abs((r[0]-t).to_value(u.s))<=EUVI_dT/2 for r in euvi_rows)]
		groups=[]
		for index in missing:
			if groups and index==groups[-1][-1]+1:groups[-1].append(index)
			else:groups.append([index])
		if download:
			for group in groups:
				lo=max(start,euvi_targets[group[0]]-EUVI_dT/2*u.s)
				hi=min(end,euvi_targets[group[-1]]+EUVI_dT/2*u.s)
				if hi<=lo:continue
				try:
					dm.get_STEREO_imgs(lo.isot,hi.isot,EUVI_dT,spacecraft=spacecraft,
						instruments=['EUVI'],waves=[[EUVI_wave]],outdir=str(img_folder),
						nproc=1,helioviewer=helioviewer)
				except Exception as exc:warnings.warn(f'EUVI download failed: {exc}')
			euvi_rows=stereo_cache(img_folder,spacecraft,'EUVI',wave=EUVI_wave)
	cfg.update(EUVI_needed=EUVI_needed,EUVI_wave=EUVI_wave,EUVI_radius=EUVI_radius,EUVI_BW=EUVI_BW)
	jobs=[];records=list(uncovered);positions={}
	for i,row in enumerate(selected):
		reference=None
		if diff_type!='intensity' and not (i==0 and first_frame_intensity):
			candidates=refs if diff_type=='base_diff' or i==0 else [selected[i-1]]
			candidates=[r for r in candidates if r[2]==row[2] and r[0]!=row[0]]
			if not candidates:
				if i!=0 or diff_type!='running_diff':
					records.append([row[0].isot,row[1],'JP2' if row[2] else 'FITS','','','skipped: no same-format reference']);continue
				warnings.warn('No pre-sT same-format image: first frame will be intensity.')
			else:reference=min(candidates,key=lambda r:abs((r[0]-(base if diff_type=='base_diff' else row[0])).to_value(u.s)))
		try:
			for entry in (row,reference):
				if entry is not None and entry[1] not in positions:
					positions[entry[1]]=stereo_observer(dh.load_map(entry[1]),spacecraft,
						observer_location,observer_latitude,observer_distance_au)
			model=min(models,key=lambda x:abs((x[0]-row[0]).to_value(u.s)))[1] if models else None
			frame_cfg=dict(cfg)
			if focus_AR:frame_cfg['ar_model']=ar_models[model]
			if EUVI_needed:
				euvi=min(euvi_rows,key=lambda r:abs((r[0]-row[0]).to_value(u.s))) if euvi_rows else None
				if euvi is not None:
					try:
						if euvi[1] not in positions:
							positions[euvi[1]]=stereo_observer(dh.load_map(euvi[1]),spacecraft,
								observer_location,observer_latitude,observer_distance_au)
						frame_cfg.update(euvi_row=euvi,euvi_observer=positions[euvi[1]])
						euvi_log.append([row[0].isot,euvi[0].isot,euvi[1],'JP2' if euvi[2] else 'FITS','selected'])
					except Exception as exc:euvi_log.append([row[0].isot,euvi[0].isot,euvi[1],'',str(exc)])
				else:euvi_log.append([row[0].isot,'','','','missing: render COR/PFSS only'])
			print(f'{row[0].isot}  STEREO-{spacecraft}: {_observer_label(positions[row[1]])}',flush=True)
			jobs.append((row,reference,positions[row[1]],positions[reference[1]] if reference else None,model,frame_cfg))
		except Exception as exc:records.append([row[0].isot,row[1],'','','',f'skipped observer: {exc}'])
	if EUVI_needed:
		with (root/'EUVI_sources.csv').open('w',newline='') as stream:
			writer=csv.writer(stream);writer.writerow(['COR_datetime','EUVI_datetime','path','format','status']);writer.writerows(euvi_log)
	with (root/'observer_positions.csv').open('w',newline='') as stream:
		writer=csv.writer(stream);writer.writerow(['file','hgln_obs_deg','hglt_obs_deg','dsun_obs_m','source'])
		for path,p in positions.items():writer.writerow([path,p['hgln_obs'],p['hglt_obs'],p['dsun_obs'],p['source']])
	if nproc==1:outputs=list(map(_stereo_frame,jobs))
	else:
		if not hasattr(sys.modules['__main__'],'__spec__'):sys.modules['__main__'].__spec__=None
		with get_context('spawn').Pool(min(nproc,max(1,len(jobs))),maxtasksperchild=8) as pool:
			outputs=list(pool.imap(_stereo_frame,jobs,chunksize=1))
	pngs=[]
	for png,record in outputs:
		records.append(record)
		if png is not None:pngs.append(png)
	with (root/'image_sources.csv').open('w',newline='') as stream:
		writer=csv.writer(stream);writer.writerow(['datetime','image','format','reference','observer_source','status']);writer.writerows(records)
	with (root/'skipped_frames.txt').open('w') as stream:
		for record in records:
			if record[-1]!='saved':stream.write(record[0]+' '+str(record[-1])+'\n')
	if not need_GOES or not pngs:return pngs
	goes_file=(Path(goes_folder).expanduser().resolve() if goes_folder else root/'GOES')/f'GOES_{start.strftime("%Y%m%d_%H%M%S")}_{end.strftime("%Y%m%d_%H%M%S")}.p'
	goes_file.parent.mkdir(parents=True,exist_ok=True)
	try:
		if goes_pickle is not None:
			g=pd.read_pickle(goes_pickle)
			if isinstance(g,dict) and 'Time' in g and 'Flux' in g:
				goes=pd.DataFrame({'flux':g['Flux']},index=pd.to_datetime(g['Time'],utc=True))
			elif isinstance(g,pd.DataFrame) and 'flux' in g.columns:goes=g[['flux']]
			else:raise ValueError('GOES pickle requires Time/Flux dict or flux DataFrame')
		elif goes_file.is_file():goes=pd.read_pickle(goes_file)
		elif (found:=covering_goes(goes_file.parent,start.isot,end.isot)) is not None:goes=found  # any cached file spanning the window
		else:
			res=Fido.search(a.Time(sT,eT),a.Instrument('XRS'),a.Resolution('flx1s'))
			table=res['xrs'];sat=max(int(v) for v in table['SatelliteNumber'])
			table=table[np.asarray(table['SatelliteNumber'],int)==sat]
			files=Fido.fetch(table,path=str(goes_file.parent/'{file}'))
			if not files:raise FileNotFoundError('No GOES files downloaded')
			lc=ts.TimeSeries(list(files),source='XRS',concatenate=True).truncate(sT,eT)
			goes=lc.to_dataframe()[['xrsb']].rename(columns={'xrsb':'flux'})
			goes['flux']*=lc.units['xrsb'].to(u.W/u.m**2)
			goes.to_pickle(goes_file)
		final=Path(final_image_folder) if final_image_folder else root/'PNGs'/spacecraft/instrument/diff_type
		return save_lightcurve_frames(pngs,goes,flux_csv,sT,eT,final,dpi=dpi)
	except Exception as exc:
		(root/'lightcurve_error.txt').write_text(str(exc)+'\n')
		if flux_csv is not None:
			warnings.warn(f'GOES/LAT panel preparation failed ({exc}); attempting LAT-only panels')
			try:
				blank=pd.DataFrame({'flux':pd.Series(dtype=float)},index=pd.DatetimeIndex([],tz='UTC'))
				final=Path(final_image_folder) if final_image_folder else root/'PNGs'/spacecraft/instrument/diff_type
				return save_lightcurve_frames(pngs,blank,flux_csv,sT,eT,final,dpi=dpi)
			except Exception as lat_exc:warnings.warn(f'LAT-only panels failed: {lat_exc}')
		warnings.warn(f'Light-curve panels failed; upper-panel PNGs remain available: {exc}')
		return pngs

# Same light-curve renderer as the finalized LASCO frame script.
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
	lat=None
	if flux_csv is not None:
		lat=pd.read_csv(flux_csv)
		lat['datetime']=pd.to_datetime(lat['datetime'],utc=True)
		lat=lat.loc[lat['datetime'].between(start,end)].sort_values('datetime')
		if 'is_upper_limit' not in lat:
			lat['is_upper_limit']=False
		lat['is_upper_limit']=lat['is_upper_limit'].astype(str).str.lower().isin(['true','1'])
		if lat.empty:raise ValueError('No LAT measurements inside the requested interval')
		if not np.all(np.isfinite(lat['flux'])&(lat['flux']>0)):
			raise ValueError('CSV flux must be finite and positive for the log axis')
	if goes.empty and lat is None:raise ValueError('No GOES or LAT data in the event interval')
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
			gax=ax.twinx() if lat is not None and not goes.empty else ax
			if not goes.empty:
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

if __name__=='__main__':
	_unset=[k for k in ('basedir','sT','eT','base_diff_DT') if k in globals() and ('YYYY' in str(globals()[k]) or str(globals()[k]).startswith('/path/to'))]
	if _unset:raise ValueError('Set these in the INPUT block first: '+', '.join(_unset))
	if save_PFSS_maps:  # AR + full-Sun PFSS maps at the sT-eT midpoint, saved in basedir
		import solar_overlay as _so;from astropy.time import Time as _T
		try:_so.pfss_context_maps((_T(sT)+(_T(eT)-_T(sT))/2).isot,'STEREO-'+spacecraft,'EUVI',EUVI_wave,img_folder=img_folder,basedir=basedir,pfss_dir=pfss_dir,rss=rss,nrho=nrho,r_seed=r_seed,n_lon=n_lon,n_lat=n_lat,PFSS_dT=PFSS_dT,SRS_folder=SRS_folder,NOAA_number=NOAA_number if focus_AR else None,ar_half_lon=ar_half_lon,ar_half_lat=ar_half_lat,AR_seed_spacing_deg=AR_seed_spacing_deg,trace_longARrooted_loops=bool(trace_longARrooted_loops),AR_r_seed=AR_r_seed,n_closed=n_closed,n_AR_open=None if n_AR_open in ('',None) or n_AR_open!=n_AR_open else int(n_AR_open),n_global_closed=n_global_closed,AR_closedB_col=AR_closedB_col,AR_openB_col=AR_openB_col,maps='both' if save_PFSS_maps is True else save_PFSS_maps,AR_loop_reach_deg=None if AR_loop_reach_deg=='' else float(AR_loop_reach_deg))
		except Exception as exc:print('PFSS maps not saved:',exc)
	import os
	cwd=os.getcwd()
	os.chdir(basedir)  # all relative paths are inside the event folder
	try:
		# Call through the module imported by its file name, so the spawn-pool worker
		# _stereo_frame is pickled by reference rather than from __main__.
		import importlib
		sys.path.insert(0,str(Path(__file__).resolve().parent))
		module=importlib.import_module(Path(__file__).stem)
		settings={key:globals()[key] for key in (
			'sT','eT','dT','spacecraft','instrument','polar','img_folder','outdir',
			'final_image_folder','diff_type','first_frame_intensity','base_diff_DT',
			'reference_search','download','helioviewer','need_pfss','pfss_dir','PFSS_dT',
			'pfss_kwargs','observer_location','observer_latitude','observer_distance_au',
			'nproc','npix','dpi','n_global_closed','closed_color','open_positive_color',
			'open_negative_color','need_GOES','flux_csv','goes_pickle','goes_folder','EUVI_needed','EUVI_wave','EUVI_dT','EUVI_radius','EUVI_BW',
			'focus_AR','NOAA_number','SRS_file','SRS_folder','AR_lookup_max_age','AR_lon','AR_lat','AR_time','ar_half_lon','ar_half_lat',
			'AR_seed_spacing_deg','trace_longARrooted_loops','AR_loop_reach_deg','AR_r_seed','n_closed','n_AR_open','AR_closedB_col','AR_openB_col')}

		paths=module.make_STEREO_frames(**settings)
		print('Saved',len(paths),'STEREO PNG frames')

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
