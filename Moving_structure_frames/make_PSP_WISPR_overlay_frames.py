"""PSP/WISPR PFSS shell frames with GOES and Fermi-LAT lower panels.

Single combined task script and function module. Edit the INPUT block, then run:
	python make_PSP_WISPR_overlay_frames.py
Importing this file only defines the INPUT values and functions; nothing runs.

Purpose
-------
Download or reuse Parker Solar Probe / WISPR FITS files, select one frame per
requested time bin, and render blank shell frames in the WISPR viewpoint with
projected PFSS field lines. No solar image overlay is used. The background is a
black canvas with a spherical-shell wireframe and PFSS lines. A lower panel can
be added with GOES and Fermi-LAT light curves, following the STEREO frame code.

Notes
-----
* This script assumes you already use the Solar_data_mining helper modules
  solar_overlay.py and solar_overlay_helper.py for PFSS creation/loading.
* The code uses WISPR headers (or Horizons fallback) for the PSP observer
  geometry; the PFSS remains in physical Carrington coordinates and is
  projected into the WISPR FoV.
* The user wording "Whisper" is interpreted here as WISPR, the PSP white-light
  imager.
"""
from pathlib import Path
from functools import lru_cache
from multiprocess import get_context
import csv,json,copy,sys,warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import astropy.units as u
from astropy.time import Time
from astropy.constants import R_sun
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames,get_horizons_coord
import sunpy.map
from sunpy.net import Fido,attrs as a
import sunpy.timeseries as ts
import solar_overlay as so
import solar_overlay_helper as oh

#################### INPUT #################################################
basedir='/path/to/event_folder'  # Event folder: every relative path below is inside it; run from anywhere
sT='2025-05-30T22:30:00'
eT='2025-05-31T17:00:00'
dT=600                              # WISPR output cadence in seconds
detector='INNER'                    # 'INNER' or 'OUTER'
img_folder='./PSP_WISPR'            # WISPR cache/download root
outdir='./PSP_WISPR_LAT_frames/WISPR'
final_image_folder='./PSP_WISPR_LAT_frames/pfss_wispr_LC'
download=True                       # Download missing WISPR files via Fido/VSO
need_pfss=True
pfss_dir='./PFSS'                   # Shared PFSS snapshots (same UTC epochs as the LASCO/STEREO codes)
### PFSS parameters ####
PFSS_dT=6.                          # PFSS snapshot cadence in HOURS
rss=2.5
nrho=35
r_seed=1.05
n_lon=36
n_lat=18
### Shell plotting ####
shell_r=2.5                         # Rsun spherical shell radius to draw
shell_grid_dlon=20                  # deg between shell meridians
shell_grid_dlat=15                  # deg between shell parallels
shell_color='gray'
shell_alpha=0.55
shell_lw=0.6
observer_location=None              # None: use WISPR header, else JPL Horizons; manual e.g. 'W37'
observer_latitude=None              # deg; None retains header/Horizons
observer_distance_au=None           # AU; None retains header/Horizons
nproc=2                             # rendering workers
npix=1024
dpi=90
N_closed=100                        # maximum visible closed PFSS loops
closed_color='white'
open_positive_color='red'
open_negative_color='cyan'
### Light curves ####
need_GOES=True
flux_csv='LAT_fluxLC.csv'           # Fermi-LAT CSV inside basedir, same format as LASCO/STEREO script
goes_pickle=None                    # optional cached GOES pickle
goes_folder='./GOES'                # Shared GOES cache: any pickle spanning sT-eT is reused; downloads saved here
flux_label='LAT >100 MeV'
flux_ylabel=r'LAT flux (ph cm$^{-2}$ s$^{-1}$)'
###########################################################################

pfss_kwargs={'rss':rss,'nrho':nrho,'r_seed':r_seed,'n_lon':n_lon,'n_lat':n_lat}

#################### HELPERS ###############################################
def _target_times(sT,eT,dT):
	start,end=Time(sT),Time(eT)
	if end<start:raise ValueError('eT must be >= sT')
	seconds=(end-start).to_value(u.s)
	n=int(np.floor(seconds/float(dT)))+1
	return [start+i*float(dT)*u.s for i in range(max(n,1))]


def format_we(lon_deg):
	lon=float(lon_deg)
	return ('W' if lon>=0 else 'E')+str(int(round(abs(lon))))


def wispr_cache(folder,detector='INNER'):
	"""Read actual WISPR FITS metadata recursively."""
	rows=[]
	want=str(detector).upper()
	for path in sorted(Path(folder).expanduser().resolve().rglob('*')):
		if not path.is_file() or not path.name.lower().endswith(('.fits','.fit','.fts','.fits.gz','.fit.gz','.fts.gz')):
			continue
		try:
			m=sunpy.map.Map(path)
			obs=str(getattr(m,'observatory','')).upper().replace(' ','')
			det=str(getattr(m,'detector','')).upper().strip()
			if 'PARKER' not in obs and 'PSP' not in obs:
				continue
			if 'WISPR' not in str(getattr(m,'instrument','')).upper():
				continue
			if want and det!=want:
				continue
			rows.append((m.date,str(path),det))
		except Exception as exc:
			warnings.warn(f'Skipping cached image {path}: {exc}')
	return sorted(rows,key=lambda r:(r[0].mjd,r[1]))


def get_PSP_WISPR_imgs(sT,eT,dT,detector='INNER',outdir='./PSP_WISPR'):
	"""Query VSO for PSP/WISPR and fetch the returned FITS files."""
	root=Path(outdir).expanduser().resolve()
	root.mkdir(parents=True,exist_ok=True)
	try:
		query=Fido.search(a.Time(sT,eT),a.Source('PSP'),a.Instrument('WISPR'))
	except Exception:
		query=Fido.search(a.Time(sT,eT),a.Instrument('WISPR'))
	if not query.file_num:
		raise FileNotFoundError('No PSP/WISPR entries returned by Fido in the requested interval.')
	files=Fido.fetch(query,path=str(root/'{file}'),overwrite=False)  # existing files are kept, not refetched
	if not files:
		raise FileNotFoundError('Fido returned no downloaded WISPR files.')
	rows=wispr_cache(root,detector)
	if not rows:
		raise FileNotFoundError(f'No WISPR-{detector} FITS files found after download.')
	return rows


def psp_observer(m,location=None,latitude=None,distance_au=None):
	"""Return PSP Stonyhurst observer geometry."""
	meta=m.meta
	if all(k in meta for k in ('hgln_obs','hglt_obs','dsun_obs')):
		lon,lat,dist=(float(meta[k]) for k in ('hgln_obs','hglt_obs','dsun_obs'))
		source='image header'
	elif location is not None and latitude is not None and distance_au is not None:
		lon,lat,dist=0.,float(latitude),(float(distance_au)*u.au).to_value(u.m)
		source='manual observer'
	else:
		obs=get_horizons_coord('Parker Solar Probe',m.date).transform_to(
			frames.HeliographicStonyhurst(obstime=m.date))
		lon,lat,dist=obs.lon.to_value(u.deg),obs.lat.to_value(u.deg),obs.radius.to_value(u.m)
		source='JPL Horizons'
	if location is not None:
		if isinstance(location,str):
			value=location.strip().upper()
			lon=float(value[1:])*(1 if value[:1]=='W' else -1) if value[:1] in ('W','E') else float(value)
		else:
			lon=float(location)
		source+='; manual longitude override'
	if latitude is not None:lat=float(latitude)
	if distance_au is not None:dist=(float(distance_au)*u.au).to_value(u.m)
	if not np.all(np.isfinite([lon,lat,dist])) or not -90<=lat<=90 or dist<=0:
		raise ValueError('Invalid PSP observer longitude, latitude or distance.')
	return dict(hgln_obs=lon,hglt_obs=lat,dsun_obs=dist,source=source)


@lru_cache(maxsize=2)
def _field_lines(folder):
	return oh._load_pfss(folder)


def _plot_segments(ax,tmap,coord,color,lw=0.7,alpha=0.6):
	"""Plot transformed coordinates robustly, splitting finite segments."""
	try:
		px,py=tmap.world_to_pixel(coord)
		x=np.asarray(px.value if hasattr(px,'value') else px,float)
		y=np.asarray(py.value if hasattr(py,'value') else py,float)
	except Exception:
		return
	good=np.isfinite(x)&np.isfinite(y)
	if not np.any(good):
		return
	idx=np.where(good)[0]
	cuts=np.where(np.diff(idx)>1)[0]
	start=0
	for stop in list(cuts+1)+[len(idx)]:
		seg=idx[start:stop]
		if len(seg)>=2:ax.plot(x[seg],y[seg],color=color,lw=lw,alpha=alpha)
		start=stop


def draw_spherical_shell(ax,tmap,r_shell=2.5,dlon=20,dlat=15,color='gray',alpha=0.55,lw=0.6):
	"""Draw a wireframe spherical shell in the current WISPR viewpoint."""
	obs=tmap.observer_coordinate
	obstime=tmap.date
	frame=frames.HeliographicCarrington(observer=obs,obstime=obstime)
	lats=np.linspace(-89.5,89.5,720)
	lons=np.linspace(0,360,1441)
	for lon in np.arange(0,360,dlon):
		coord=SkyCoord(np.full_like(lats,lon)*u.deg,lats*u.deg,
			np.full_like(lats,r_shell)*R_sun,frame=frame)
		_plot_segments(ax,tmap,coord,color,lw,alpha)
	for lat in np.arange(-75,90,dlat):
		coord=SkyCoord(lons*u.deg,np.full_like(lons,lat)*u.deg,
			np.full_like(lons,r_shell)*R_sun,frame=frame)
		_plot_segments(ax,tmap,coord,color,lw,alpha)
	try:
		rsun=np.arcsin((r_shell*R_sun/tmap.observer_coordinate.radius).decompose().value)*u.rad
		ang=np.linspace(0,2*np.pi,720)
		coord=SkyCoord((rsun*np.cos(ang)).to(u.arcsec),(rsun*np.sin(ang)).to(u.arcsec),
			frame=frames.Helioprojective(observer=obs,obstime=obstime))
		_plot_segments(ax,tmap,coord,color,max(1.2*lw,lw),min(alpha+0.15,1))
	except Exception:
		pass


def _wispr_shell_frame(task):
	row,observer,model,cfg=task
	t,path,det=row
	try:
		plt.switch_backend('Agg')
		m=sunpy.map.Map(path)
		meta=copy.deepcopy(m.meta)
		meta.update({k:observer[k] for k in ('hgln_obs','hglt_obs','dsun_obs')})
		m=sunpy.map.Map(np.asarray(m.data,float),meta)
		try:
			tmap=m.resample(u.Quantity([cfg['npix'],cfg['npix']],u.pixel))
		except Exception:
			tmap=m
		obs=tmap.observer_coordinate.transform_to(frames.HeliographicStonyhurst(obstime=tmap.date))
		dlon=(obs.lon.to_value(u.deg)-observer['hgln_obs']+180)%360-180
		if abs(dlon)>0.25 or abs(obs.lat.to_value(u.deg)-observer['hglt_obs'])>0.25:
			raise ValueError('PSP observer WCS mismatch: PFSS would be misaligned')
		fig=Figure(figsize=(8,8),dpi=cfg['dpi'])
		ax=fig.add_subplot(projection=tmap)
		ax.set_facecolor('black')
		blank=np.full(np.asarray(tmap.data).shape,np.nan)
		ax.imshow(blank,origin='lower',cmap='gray',vmin=0,vmax=1)
		draw_spherical_shell(ax,tmap,cfg['shell_r'],cfg['shell_grid_dlon'],cfg['shell_grid_dlat'],
			cfg['shell_color'],cfg['shell_alpha'],cfg['shell_lw'])
		if model is not None:
			oh._draw_pfss(ax,tmap,_field_lines(model),N_Bloop=cfg['N_closed'],
				colors={0:cfg['closed_color'],1:cfg['open_positive_color'],-1:cfg['open_negative_color']})
		label=f"PSP: {format_we(observer['hgln_obs'])}, {(observer['dsun_obs']*u.m).to_value(u.au):.2f}AU"
		print(f"{t.isot}  {label}")
		ax.set_xlabel('Solar X (arcsec)',fontsize=16)
		ax.set_ylabel('Solar Y (arcsec)',fontsize=16)
		ax.set_title(f'PSP/WISPR-{det[0]} PFSS shell  r={cfg["shell_r"]:g} $R_\\odot$',fontsize=17)
		ax.text(.02,.98,t.strftime('%Y-%m-%d %H:%M:%S UT'),transform=ax.transAxes,
			va='top',color='white',bbox=dict(facecolor='black',alpha=.6,pad=3))
		ax.text(.98,.98,label,transform=ax.transAxes,va='top',ha='right',color='white',fontsize=10,
			bbox=dict(facecolor='black',alpha=.6,pad=2))
		fig.tight_layout()
		png=Path(cfg['outdir'])/'overlay_images'/cfg['detector']/f'overlay_{t.strftime("%Y%m%d_%H%M%S")}.png'
		png.parent.mkdir(parents=True,exist_ok=True)
		fig.savefig(png,dpi=cfg['dpi'])
		fig.clear()
		return str(png),[t.isot,path,det,observer['source'],'saved']
	except Exception as exc:
		print(f'Frame {t.isot} skipped: {exc}')
		return None,[t.isot,path,det,observer.get('source',''),str(exc)]


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
	"""Combine overlay PNGs with GOES and optional UTC LAT flux/error CSV."""
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
		stamp=pd.Timestamp(Path(frame).stem.removeprefix('overlay_').replace('_',' '),tz='UTC')
		stamp=pd.to_datetime(stamp,format='%Y%m%d %H%M%S',utc=True)
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
			fig.savefig(path,dpi=dpi)
			written.append(str(path))
		finally:
			plt.close(fig)
	return written

#################### DRIVER ################################################
def make_PSP_WISPR_overlay_frames(sT,eT,dT,detector='INNER',img_folder='./PSP_WISPR',
	outdir='./PSP_WISPR_LAT_frames/WISPR',final_image_folder=None,download=True,
	need_pfss=True,pfss_dir='./PFSS',PFSS_dT=6.,pfss_kwargs=None,shell_r=2.5,
	shell_grid_dlon=20,shell_grid_dlat=15,shell_color='gray',shell_alpha=0.55,
	shell_lw=0.6,observer_location=None,observer_latitude=None,observer_distance_au=None,
	nproc=2,npix=1024,dpi=90,N_closed=100,closed_color='white',open_positive_color='red',
	open_negative_color='cyan',need_GOES=True,flux_csv=None,goes_pickle=None,goes_folder=None,
	flux_label='LAT >100 MeV',flux_ylabel=r'LAT flux (ph cm$^{-2}$ s$^{-1}$)'):
	"""Render PSP/WISPR PFSS shell frames, optionally with GOES and LAT lower panels."""
	detector=str(detector).upper()
	if detector not in ('INNER','OUTER'):
		raise ValueError("detector must be 'INNER' or 'OUTER'.")
	if int(nproc)!=nproc or nproc<1 or not np.isfinite(PFSS_dT) or PFSS_dT<=0:
		raise ValueError('nproc must be a positive integer; PFSS_dT positive hours.')
	if not np.isfinite(shell_r) or shell_r<=1:
		raise ValueError('shell_r must be > 1 Rsun.')
	root=Path(outdir).expanduser().resolve()
	root.mkdir(parents=True,exist_ok=True)
	img_folder=Path(img_folder).expanduser().resolve()
	img_folder.mkdir(parents=True,exist_ok=True)
	pfss_dir=Path(pfss_dir).expanduser().resolve()
	pfss_dir.mkdir(parents=True,exist_ok=True)
	if need_GOES and flux_csv is not None:
		flux_csv=Path(flux_csv).expanduser()
		if not flux_csv.is_absolute():flux_csv=Path.cwd()/flux_csv
		if not flux_csv.is_file():raise FileNotFoundError(f'LAT flux CSV missing: {flux_csv}')
	if need_GOES and goes_pickle is not None:
		goes_pickle=Path(goes_pickle).expanduser()
		if not goes_pickle.is_absolute():goes_pickle=Path.cwd()/goes_pickle
		if not goes_pickle.is_file():raise FileNotFoundError(f'GOES pickle missing: {goes_pickle}')
	kwargs=dict(pfss_kwargs or {})
	if not kwargs:kwargs=dict(rss=2.5,nrho=35,r_seed=1.05,n_lon=36,n_lat=18)
	targets=_target_times(sT,eT,dT)
	start,end=Time(sT),Time(eT)
	rows=wispr_cache(img_folder,detector)
	# Download only cadence bins with no cached exposure within dT/2 (consecutive gaps
	# fetched together), so files already in img_folder are never requested again.
	missing=[i for i,t in enumerate(targets) if not any(abs((r[0]-t).to_value(u.s))<=dT/2 for r in rows)]
	if download and missing:
		groups=[]
		for i in missing:
			if groups and i==groups[-1][-1]+1:groups[-1].append(i)
			else:groups.append([i])
		print(f'PSP/WISPR: {len(missing)}/{len(targets)} bins not cached; requesting {len(groups)} window(s)',flush=True)
		for g in groups:
			lo,hi=targets[g[0]]-dT/2*u.s,targets[g[-1]]+dT/2*u.s
			try:get_PSP_WISPR_imgs(lo.isot,hi.isot,dT,detector=detector,outdir=img_folder)
			except Exception as exc:warnings.warn(f'PSP/WISPR download failed for {lo.isot}--{hi.isot}: {exc}')
		rows=wispr_cache(img_folder,detector)
	else:print(f'PSP/WISPR: all {len(targets)} bins cached; no download',flush=True)
	selected=[]
	records=[]
	for t in targets:
		near=[r for r in rows if start<=r[0]<=end and abs((r[0]-t).to_value(u.s))<=dT/2]
		if near:
			row=min(near,key=lambda r:abs((r[0]-t).to_value(u.s)))
			if row not in selected:selected.append(row)
		else:
			records.append([t.isot,'','', '', 'skipped: no WISPR exposure near requested bin'])
	selected.sort(key=lambda r:r[0].mjd)
	if not selected:raise FileNotFoundError('No usable WISPR images inside the requested interval.')
	positions={}
	for _,path,_ in selected:
		m=sunpy.map.Map(path)
		positions[path]=psp_observer(m,observer_location,observer_latitude,observer_distance_au)
	with (root/'observer_positions.csv').open('w',newline='') as stream:
		writer=csv.writer(stream)
		writer.writerow(['file','hgln_obs_deg','hglt_obs_deg','dsun_obs_m','source'])
		for path,p in positions.items():writer.writerow([path,p['hgln_obs'],p['hglt_obs'],p['dsun_obs'],p['source']])
	models=[]
	if need_pfss:
		# UTC-anchored epochs (00, 06, ... UT for PFSS_dT=6), bracketing the interval: the
		# same snapshots as the LASCO/STEREO codes, so a shared pfss_dir is reused.
		step=PFSS_dT*3600.
		for es in np.arange(np.floor(targets[0].unix/step)*step,np.ceil(targets[-1].unix/step)*step+step/2,step):
			epoch=Time(es,format='unix')
			folder=pfss_dir/epoch.strftime('%Y%m%d_%H%M%S')
			ready=all((folder/name).is_file() for name in ('metadata.json','field_lines.npz','model_3d.npz','boundary.fits'))
			if ready:
				meta=json.loads((folder/'metadata.json').read_text())
				ready=all(meta.get(k)==v for k,v in kwargs.items())
			if not ready:
				if not download:raise FileNotFoundError(f'Missing matching PFSS model: {folder}')
				so.build_pfss(epoch.isot,epoch.isot,folder,**kwargs)
			models.append((epoch,str(folder)))
	def nearest_model(t):
		if not models:return None
		return min(models,key=lambda item:abs((item[0]-t).to_value(u.s)))[1]
	cfg=dict(outdir=str(root),detector=detector,npix=npix,dpi=dpi,N_closed=N_closed,
		closed_color=closed_color,open_positive_color=open_positive_color,
		open_negative_color=open_negative_color,shell_r=shell_r,
		shell_grid_dlon=shell_grid_dlon,shell_grid_dlat=shell_grid_dlat,
		shell_color=shell_color,shell_alpha=shell_alpha,shell_lw=shell_lw)
	jobs=[(row,positions[row[1]],nearest_model(row[0]),cfg) for row in selected]
	if nproc==1:outputs=list(map(_wispr_shell_frame,jobs))
	else:
		if not hasattr(sys.modules['__main__'],'__spec__'):sys.modules['__main__'].__spec__=None
		with get_context('spawn').Pool(min(nproc,max(1,len(jobs))),maxtasksperchild=8) as pool:
			outputs=list(pool.imap(_wispr_shell_frame,jobs,chunksize=1))
	pngs=[]
	for png,record in outputs:
		records.append(record)
		if png is not None:pngs.append(png)
	with (root/'image_sources.csv').open('w',newline='') as stream:
		writer=csv.writer(stream)
		writer.writerow(['datetime','image','detector','observer_source','status'])
		writer.writerows(records)
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
			elif isinstance(g,pd.DataFrame) and 'flux' in g.columns:
				goes=g[['flux']]
			else:
				raise ValueError('GOES pickle requires Time/Flux dict or flux DataFrame')
		elif goes_file.is_file():
			goes=pd.read_pickle(goes_file)
		elif (found:=covering_goes(goes_file.parent,start.isot,end.isot)) is not None:
			goes=found  # any cached file spanning the window
		else:
			res=Fido.search(a.Time(sT,eT),a.Instrument('XRS'),a.Resolution('flx1s'))
			table=res['xrs']
			sat=max(int(v) for v in table['SatelliteNumber'])
			table=table[np.asarray(table['SatelliteNumber'],int)==sat]
			files=Fido.fetch(table,path=str(goes_file.parent/'{file}'))
			if not files:raise FileNotFoundError('No GOES files downloaded')
			lc=ts.TimeSeries(list(files),source='XRS',concatenate=True).truncate(sT,eT)
			goes=lc.to_dataframe()[['xrsb']].rename(columns={'xrsb':'flux'})
			goes['flux']*=lc.units['xrsb'].to(u.W/u.m**2)
			goes.to_pickle(goes_file)
		final=Path(final_image_folder) if final_image_folder else root/'PNGs'/detector
		return save_lightcurve_frames(pngs,goes,flux_csv,sT,eT,final,dpi=dpi,
			flux_label=flux_label,flux_ylabel=flux_ylabel)
	except Exception as exc:
		(root/'lightcurve_error.txt').write_text(str(exc)+'\n')
		if flux_csv is not None:
			warnings.warn(f'GOES/LAT panel preparation failed ({exc}); attempting LAT-only panels')
			try:
				blank=pd.DataFrame({'flux':pd.Series(dtype=float)},index=pd.DatetimeIndex([],tz='UTC'))
				final=Path(final_image_folder) if final_image_folder else root/'PNGs'/detector
				return save_lightcurve_frames(pngs,blank,flux_csv,sT,eT,final,dpi=dpi,
					flux_label=flux_label,flux_ylabel=flux_ylabel)
			except Exception as lat_exc:warnings.warn(f'LAT-only panels failed: {lat_exc}')
		warnings.warn(f'Light-curve panels failed; upper-panel PNGs remain available: {exc}')
		return pngs


if __name__=='__main__':
	import os
	cwd=os.getcwd()
	os.chdir(basedir)  # all relative paths are inside the event folder
	try:
		import importlib
		sys.path.insert(0,str(Path(__file__).resolve().parent))
		module=importlib.import_module(Path(__file__).stem)
		settings={key:globals()[key] for key in (
			'sT','eT','dT','detector','img_folder','outdir','final_image_folder','download','need_pfss','pfss_dir',
			'PFSS_dT','pfss_kwargs','shell_r','shell_grid_dlon','shell_grid_dlat','shell_color',
			'shell_alpha','shell_lw','observer_location','observer_latitude','observer_distance_au',
			'nproc','npix','dpi','N_closed','closed_color','open_positive_color','open_negative_color',
			'need_GOES','flux_csv','goes_pickle','goes_folder','flux_label','flux_ylabel')}
		paths=module.make_PSP_WISPR_overlay_frames(**settings)
		print('Saved',len(paths),'PSP/WISPR PNG frames')
	finally:
		os.chdir(cwd)
