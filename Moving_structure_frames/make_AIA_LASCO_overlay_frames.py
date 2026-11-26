"""SOHO/LASCO C2 frames with an SDO/AIA full-disk image inside the occulter, a full-disk AIA panel
and a STIX + GOES panel, plus an MP4 per (panel a, panel b) display pair.

Each frame, left (a): LASCO image with the AIA image nearest in time filling the occulter;
right (b): the AIA full disk (+-disk_rsun). Each panel is chosen separately (panels) as running
difference, base difference (base_time) or intensity (log LASCO, asinh AIA); bottom (c, lc_panel=1): STIX quick-look
bands (stix_bands; Multiwb_datamine.get_STIX_LC, already shifted to Earth) and GOES XRS-B
(goes_cache.resolve_goes_pickle) over the whole window, with a dotted line at the frame time
shifted from SOHO to Earth light arrival (seconds, kept for consistency with the SolO maker).
Each (panel a, panel b) pair in panels makes one movie folder.
Images are aligned by helioprojective angle like make_SolO_overlay_frames; AIA angles are
scaled by the ratio of solar radii so the SDO limb sits on the SOHO limb (SOHO at L1 sees the
Sun about 1% larger). Axes are Solar X/Y in solar radii, north up, Sun-centred.
LASCO FITS are read from lasco_dir (Multiwb_datamine.get_LASCO_imgs, need_calib=False: DN/s,
outside the FoV NaN); AIA images are the JSOC synoptic
files (1k, 2-min cadence, level 1.5) nearest each LASCO time (get_aia_synoptic), divided by EXPTIME.
The MP4 is written by track_coronagraph_flows.write_mp4 (450 ms per frame).
Dependencies: numpy, matplotlib, astropy, sunpy, scipy, pandas, FFmpeg; Solar_data_mining on
PYTHONPATH. Importing this file defines the INPUT values and functions only; nothing runs.
Importable: fits_times, get_aia_synoptic, load_cor, load_disk, to_grid, make_frames (extent_rsun,
hgs_grid and light_curves come from make_SolO_overlay_frames).
"""
import os,glob,warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.constants import c as c_light
from astropy.visualization import ImageNormalize,AsinhStretch
import sunpy.map
from sunpy.map.header_helper import make_fitswcs_header
from sunpy.coordinates import get_earth
from scipy.ndimage import map_coordinates
from make_SolO_overlay_frames import extent_rsun,hgs_grid,light_curves
from track_coronagraph_flows import write_mp4

# Parameter guide
# basedir    : event folder; every relative path below is inside it.
# outdir     : frame folder stem; frames and MP4 go to <outdir>_<panels tag>/ (see panels).
# lasco_dir  : LASCO C2 (or C3) FITS folder, e.g. LASCO/C2/maps from get_LASCO_imgs.
# aia_dir    : folder for the JSOC AIA synoptic FITS (1k, level 1.5); missing ones are downloaded.
# aia_wave   : AIA wavelength [A], e.g. 94, 131, 171, 193, 211, 304, 335.
# sT,eT      : UTC window of LASCO images; with an 'rdiff' panel the first image is only the reference.
# panels     : list of (panel a LASCO, panel b AIA) display pairs, one movie each; each entry is
#              'rdiff' (running difference), 'bdiff' (base difference) or 'intensity'.
#              Output: <outdir>_<a> if a==b, else <outdir>_<a>-<b>.
# base_time  : UTC of the base image for 'bdiff' (nearest LASCO image; AIA matched to it);
#              None = first LASCO image in sT-eT.
# npix       : panel (a) grid size; half-width is the farthest LASCO pixel.
# diff_clip  : percentile setting the grey limits (|difference|, or intensity 100-p..p for log).
# disk_rsun  : half-width of the full-disk AIA panel (b), solar radii.
# lc_panel   : 1 adds the STIX + GOES panel; 0 omits it.
# stix_dir   : STIX quick-look folder (get_STIX_LC; CSV for sT-eT reused).
# stix_bands : STIX bands, from '4-10 keV','10-15 keV','15-25 keV','25-50 keV','50-84 keV'.
# goes_folder: GOES pickle cache (plot_goes_derivative); downloaded if no file covers sT-eT.
########### INPUT ###########################################################
basedir='/path/to/event_folder'
outdir='LASCO_AIA_frames/C2_AIA94'
lasco_dir='LASCO/C2/maps'
aia_dir='SDO/AIA/94_synoptic'
aia_wave=94
sT='2025-05-30T07:00:00'
eT='2025-05-31T00:00:00'
panels=[('rdiff','rdiff'),('intensity','intensity')]
base_time=None
npix=1000
diff_clip=99.
disk_rsun=1.3
lc_panel=1
stix_dir='SolO/STIX'
stix_bands=['4-10 keV','15-25 keV','25-50 keV']
goes_folder='GOES'
#############################################################################


def fits_times(folder,sT,eT):
	"""[(Time, path)] of FITS in folder (recursive) with DATE-OBS inside [sT,eT], time-sorted."""
	t0,t1=Time(sT),Time(eT);out=[]
	for p in glob.glob(os.path.join(folder,'**','*.fits'),recursive=True):
		with fits.open(p) as hl:
			d=next((h.header.get('DATE-OBS') for h in hl if 'DATE-OBS' in h.header),None)
		if d and t0<=Time(d)<=t1:out.append((Time(d),p))
	return sorted(out,key=lambda x:x[0].mjd)

def get_aia_synoptic(times,wave,folder):
	"""Download the JSOC AIA synoptic file (1k, level 1.5, 2-min cadence) nearest each time into
	folder, skipping files already there; returns the local paths (None where unavailable)."""
	import urllib.request
	Path(folder).mkdir(parents=True,exist_ok=True);out=[]
	for t in times:
		t2=Time(round(t.unix/120)*120,format='unix').datetime
		name=f'AIA{t2:%Y%m%d_%H%M}_{int(wave):04d}.fits';p=Path(folder)/name
		if not p.is_file():
			try:urllib.request.urlretrieve(f'https://jsoc1.stanford.edu/data/aia/synoptic/{t2:%Y/%m/%d}/H{t2:%H}00/{name}',p)
			except OSError:p=None
		out.append(str(p) if p else None)
	return out

def load_cor(path):
	"""Coronagraph Map (LASCO, COR1/COR2); DN -> DN/s by EXPTIME; non-positive pixels -> NaN."""
	m=sunpy.map.Map(path);d=np.asarray(m.data,float)
	if str(m.meta.get('bunit','')).strip().upper()=='DN':d=d/m.exposure_time.to_value(u.s)
	d[~(d>0)]=np.nan
	return sunpy.map.Map(d,m.meta)

def load_disk(path):
	"""EUV disk Map (AIA, EUVI) in DN/s (data / EXPTIME)."""
	m=sunpy.map.Map(path)
	return sunpy.map.Map(np.asarray(m.data,float)/m.exposure_time.to_value(u.s),m.meta)

def to_grid(m,hdr,fac=1.):
	"""Sample Map m on header hdr by helioprojective angle, angles multiplied by fac; NaN outside m."""
	g=sunpy.map.Map(np.zeros((hdr['naxis2'],hdr['naxis1'])),hdr)
	yy,xx=np.indices(g.data.shape);c=g.pixel_to_world(xx*u.pix,yy*u.pix)
	sx,sy=m.world_to_pixel(SkyCoord(c.Tx*fac,c.Ty*fac,frame=m.coordinate_frame))
	return map_coordinates(np.asarray(m.data,float),[sy.value,sx.value],order=1,mode='constant',cval=np.nan)

def _grid(m,half,n):
	"""n x n north-up header centred on the Sun from m's observer, half-width half [arcsec]."""
	return make_fitswcs_header((n,n),SkyCoord(0*u.arcsec,0*u.arcsec,frame=m.coordinate_frame),scale=[2*half/n]*2*u.arcsec/u.pix,rotation_angle=0*u.deg)

LABEL={'rdiff':'running diff.','bdiff':'base diff.'}

def make_frames(lfiles,afiles,outdir,a_mode='rdiff',b_mode='rdiff',base=0,npix=1000,diff_clip=99.,disk_rsun=1.3,lc=None,lc_window=None,inner_r=1.25,occ_rsun=None):
	"""Write one PNG per LASCO image (after the first if a panel is 'rdiff'); returns the PNG paths.
	lfiles: time-sorted LASCO FITS; afiles: matching AIA FITS or None (same length).
	a_mode (LASCO panel a) and b_mode (AIA panel b): 'rdiff' = minus the previous image,
	'bdiff' = minus the base image lfiles[base] (AIA: afiles[base]), 'intensity' = log LASCO /
	asinh AIA. The AIA inside the occulter is always intensity. lc: (stix, goes) from
	light_curves for panel (c) over lc_window. inner_r: radius [R_sun] of the disk image shown
	inside the occulter; occ_rsun: if set, coronagraph pixels inside this radius [R_sun]
	(occulter, diffraction ring) are blanked. Instrument names, colormap and titles come from the FITS headers, so
	this also serves STEREO COR2 + EUVI (make_STEREO_EUVI_overlay_frames)."""
	for m in (a_mode,b_mode):
		if m not in ('rdiff','bdiff','intensity'):raise ValueError(f"panel mode {m!r}: use 'rdiff', 'bdiff' or 'intensity'")
	Path(outdir).mkdir(parents=True,exist_ok=True)
	i0=1 if 'rdiff' in (a_mode,b_mode) else 0
	L0,ab=load_cor(lfiles[base]),afiles[base]
	prev,ap=(load_cor(lfiles[0]),afiles[0]) if i0 else (None,None)
	def aia(path,k):
		m=load_disk(path);return m.superpixel([k,k]*u.pix,func=np.nanmean) if k>1 else m
	frames=[]
	for lf,af in zip(lfiles[i0:],afiles[i0:]):
		cur=load_cor(lf)
		c=sunpy.map.all_coordinates_from_map(cur)
		hdr=_grid(cur,float(np.hypot(c.Tx,c.Ty).to_value(u.arcsec)[np.isfinite(cur.data)].max())*1.02,npix)
		lr={'rdiff':prev,'bdiff':L0}.get(a_mode)
		img=to_grid(cur,hdr)-(to_grid(lr,hdr) if lr is not None else 0)
		q=(np.arange(npix)+1-hdr['crpix1'])*hdr['cdelt1']/cur.rsun_obs.to_value(u.arcsec);R=np.hypot(*np.meshgrid(q,q))
		if occ_rsun:img[R<occ_rsun]=np.nan
		inner=disk=ar=None
		if af:
			a=load_disk(af);k=max(1,int(a.data.shape[0]//1024));a=a.superpixel([k,k]*u.pix,func=np.nanmean) if k>1 else a
			fac=a.rsun_obs.to_value(u.arcsec)/cur.rsun_obs.to_value(u.arcsec)
			dh=_grid(a,disk_rsun*a.rsun_obs.to_value(u.arcsec),800)
			dd=to_grid(a,dh);ar={'rdiff':ap,'bdiff':ab}.get(b_mode)
			if b_mode!='intensity':dd=dd-to_grid(aia(ar,k),dh) if ar else np.full_like(dd,np.nan)
			kk=max(1,int(np.floor(hdr['cdelt1']/a.scale.axis1.to_value(u.arcsec/u.pix))))
			inner=to_grid(a.superpixel([kk,kk]*u.pix,func=np.nanmean) if kk>1 else a,hdr,fac)
			inner[R>inner_r]=np.nan  # round edge inside the disk imager field
			disk=sunpy.map.Map(dd,dh)
		frames.append((cur.date,lr.date if lr is not None else None,sunpy.map.Map(img,hdr),inner,disk,af,ar,cur))
		prev,ap=cur,af
	allv=np.concatenate([f[2].data[np.isfinite(f[2].data)] for f in frames])
	if a_mode=='intensity':ln=dict(norm=LogNorm(*np.percentile(allv[allv>0],[100-diff_clip,diff_clip]),clip=False))
	else:lim=np.percentile(np.abs(allv),diff_clip);ln=dict(vmin=-lim,vmax=lim)
	first=next((f for f in frames if f[3] is not None),None)
	if first is None:raise ValueError('No AIA image for any LASCO frame.')
	v=first[3][np.isfinite(first[3])&(first[3]>0)]  # AIA intensity stretch from the first frame
	an=ImageNormalize(vmin=np.percentile(v,1),vmax=np.percentile(v,99.9),stretch=AsinhStretch(0.02),clip=False)
	if b_mode!='intensity':
		dv=[f[4].data[np.isfinite(f[4].data)] for f in frames if f[4] is not None]
		dlim=np.percentile(np.abs(np.concatenate(dv)),diff_clip)
	clear=lambda name:plt.get_cmap(name).with_extremes(bad=(0,0,0,0))
	fs=dict(label=18,tick=15,title=16,legend=13,sup=18)
	def style(a):
		a.set_xlabel(r'Solar X ($R_\odot$)',fontsize=fs['label']);a.set_ylabel(r'Solar Y ($R_\odot$)',fontsize=fs['label'])
		a.tick_params(labelsize=fs['tick']);a.set_aspect('equal')
	paths=[];dcm,dobs='gray',''
	for t,tp,img,inner,disk,af,ar,cur in frames:
		rs=cur.rsun_obs.to_value(u.arcsec)
		det=f'LASCO {cur.detector}' if 'LASCO' in str(cur.instrument).upper() else str(cur.detector)
		dm=sunpy.map.Map(af) if af else None;aw=f'{dm.detector} {int(dm.wavelength.to_value(u.angstrom))} Å' if dm else 'disk'
		if dm is not None:dcm=dm.plot_settings['cmap'];dobs=str(dm.observatory)
		fig=plt.figure(figsize=(15,11 if lc else 7.8))
		gs=fig.add_gridspec(2 if lc else 1,2,height_ratios=[4,1.6] if lc else [1],hspace=.32,wspace=.22)
		ax=fig.add_subplot(gs[0,0]);ax.set_facecolor('black');E=extent_rsun(img.meta,rs)
		if inner is not None:ax.imshow(np.ma.masked_invalid(inner),origin='lower',extent=E,cmap=clear(dcm),norm=an)
		ax.imshow(np.ma.masked_invalid(img.data),origin='lower',extent=E,cmap=clear('gray'),**ln)
		ax.add_patch(plt.Circle((0,0),1,fill=False,color='w',lw=.6,alpha=.6));style(ax)
		ax.set_title((f'(a) {det} {t.isot[11:19]} UT' if a_mode=='intensity' else f'(a) {det} {LABEL[a_mode]} {t.isot[11:16]}−{tp.isot[11:16]} UT')+f'\n+ {aw} inside the occulter',fontsize=fs['title'])
		bx=fig.add_subplot(gs[0,1]);bx.set_facecolor('black')
		if disk is not None:
			ra=disk.rsun_obs.to_value(u.arcsec);Z=extent_rsun(disk.meta,ra)
			if b_mode!='intensity':
				bx.imshow(np.ma.masked_invalid(disk.data),origin='lower',extent=Z,cmap=clear('gray'),vmin=-dlim,vmax=dlim)
				title=f'(b) {aw} {LABEL[b_mode]} {sunpy.map.Map(af).date.isot[11:16]}−{sunpy.map.Map(ar).date.isot[11:16] if ar else "none"} UT'
			else:
				bx.imshow(np.ma.masked_invalid(disk.data),origin='lower',extent=Z,cmap=clear(dcm),norm=an)
				title=f'(b) {aw} {sunpy.map.Map(af).date.isot[11:19]} UT'
			hgs_grid(bx,disk.coordinate_frame,ra);bx.set_xlim(Z[:2]);bx.set_ylim(Z[2:])
			bx.set_title(title,fontsize=fs['title'])
		else:bx.set_title('(b) no AIA image',fontsize=fs['title'])
		style(bx)
		fig.suptitle((f'{cur.observatory} {det} + {aw}' if dobs==str(cur.observatory) else f'{cur.observatory}/{det} + {dobs}/{aw}').replace('STEREO A','STEREO-A')+f'  ·  {t.isot[:10]}',fontsize=fs['sup'])
		if lc:  # STIX (left) + GOES (right) for the whole window; line at this frame (Earth arrival)
			stix,goes=lc;lx=fig.add_subplot(gs[1,:]);rx=lx.twinx()
			for b,col in zip(stix.columns,['#2a78d6','#1baf7a','#6250d6','#e87ba4','#008300']):
				lx.plot(stix.index,stix[b].where(stix[b]>0),color=col,lw=1.3,label=f'STIX {b}')
			rx.plot(goes.index,goes.flux.where(goes.flux>0),color='orange',lw=1.5,label='GOES 1–8 Å')
			lx.axvline((t+(get_earth(t).radius-cur.dsun)/c_light).to_datetime(),color='k',ls=':',lw=1.4)
			lx.set_yscale('log');rx.set_yscale('log')
			lx.set_xlim(pd.Timestamp(lc_window[0]),pd.Timestamp(lc_window[1]))
			lx.set_ylabel('STIX (counts s$^{-1}$)',fontsize=fs['label']);rx.set_ylabel('GOES (W m$^{-2}$)',fontsize=fs['label'])
			lx.set_xlabel(f'Time (UT, Earth light-arrival) on {lc_window[0][:10]}',fontsize=fs['label'])
			for a in (lx,rx):a.tick_params(labelsize=fs['tick'])
			lx.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'));lx.grid(alpha=.2)
			lx.text(.01,.95,'(c)',transform=lx.transAxes,va='top',fontsize=fs['title'])
			h1,l1=lx.get_legend_handles_labels();h2,l2=rx.get_legend_handles_labels()
			lx.legend(h1+h2,l1+l2,loc='upper center',bbox_to_anchor=(.5,1.22),fontsize=fs['legend'],ncol=len(l1)+1,frameon=False)
		p=os.path.join(outdir,f'{Path(outdir).name}_{t.strftime("%Y%m%d_%H%M%S")}.png')
		fig.savefig(p,dpi=110,bbox_inches='tight');plt.close(fig)
		paths.append(p)
	return paths

if __name__=='__main__':
	cwd=os.getcwd()
	os.chdir(basedir)
	try:
		warnings.filterwarnings('ignore')
		lrows=fits_times(lasco_dir,sT,eT);lfiles=[p for t,p in lrows]
		afiles=get_aia_synoptic([t for t,p in lrows],aia_wave,aia_dir)
		print(f'{len(lfiles)} LASCO images; {sum(a is not None for a in afiles)} with a synoptic AIA {aia_wave} image (<=1 min)')
		lc=light_curves(sT,eT,stix_dir,stix_bands,goes_folder) if lc_panel else None
		base=0 if base_time is None else int(np.argmin([abs((t-Time(base_time)).to_value(u.s)) for t,p in lrows]))
		for pa,pb in panels:
			out=f'{outdir}_{pa}'+('' if pa==pb else f'-{pb}')
			paths=make_frames(lfiles,afiles,out,pa,pb,base,npix,diff_clip,disk_rsun,lc,(sT,eT))
			mp4=Path(out)/f'{Path(out).name}.mp4'
			write_mp4(paths,mp4);print(f'{len(paths)} frames, MP4: {mp4}')
	finally:
		os.chdir(cwd)
