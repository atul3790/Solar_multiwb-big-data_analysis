"""Solar Orbiter overlay frames: Metis running difference with EUI inside the occulter, a magnified EUI panel and STIX/GOES light curves, plus a movie.

Each frame, left: Metis VL total-brightness running difference (image minus the
previous Metis image) outside the Metis occulter, and the EUI Full Sun Imager
intensity image nearest in time filling the region inside it; right: that EUI
image magnified (+-zoom_rsun) to show the source region, as intensity or, with
eui_rdiff=1, as a running difference over the same image pair as Metis.
Lower panel (lc_panel=1): STIX quick-look bands (stix_bands; Multiwb_datamine.get_STIX_LC)
and GOES XRS-B (goes_cache.resolve_goes_pickle) for the whole window, all in Earth
light-arrival UT; a dotted line marks each frame, its Metis time shifted from Solar
Orbiter to Earth by (Earth-Sun - SolO-Sun distance)/c. Both are
reprojected onto one north-up helioprojective grid seen from Solar Orbiter at the
Metis time, so the coronagraph structures can be traced back to their source region.
Metis L2 FITS are read from metis_dir (get_Metis_imgs saves them); missing EUI L2
files are downloaded from ESA SOAR with datamine_calib_helper._soar_query/_soar_fetch.
The movie is written with track_coronagraph_flows.write_mp4 (ffmpeg with libx264).
Run: python make_SolO_overlay_frames.py (edit only the INPUT block).
Importing this file defines the INPUT values and functions only; nothing runs.
Images are aligned by helioprojective angle (same spacecraft), like
motion_inputs._build_cache: a 3-D reprojection between exposure times would
blank every off-disk pixel.
Axes are Solar X/Y in solar radii (north-up grids centred on the Sun).
Importable: metis_files, load_metis, get_eui, frame_grid, to_grid, extent_rsun, hgs_grid,
light_curves, make_frames.
"""
import os,glob,warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize,AsinhStretch
import sunpy.map
from sunpy.map.header_helper import make_fitswcs_header
from sunpy.visualization import colormaps as _sunpy_cmaps  # registers 'solar orbiterfsi174'
from scipy.ndimage import map_coordinates
import datamine_calib_helper as dh
import matplotlib.dates as mdates
import pandas as pd
import pickle
from astropy.constants import c as c_light
from sunpy.coordinates import get_earth
from track_coronagraph_flows import write_mp4

# Parameter guide
# basedir    : event folder; every relative path below is inside it.
# outdir     : frame folder (the movie is saved in it).
# metis_dir  : Metis L2 FITS folder (e.g. from Multiwb_datamine.get_Metis_imgs).
# eui_dir    : EUI FITS folder; nearest missing files are downloaded here.
# sT,eT      : UTC window of Metis images; the first image only serves as reference.
# eui_product: SOAR descriptor, 'eui-fsi174-image' or 'eui-fsi304-image'.
# eui_maxdt  : largest allowed Metis-EUI time difference [s]; else no EUI in the frame.
# npix       : output grid size (square); half-width is the Metis outer field edge.
# diff_clip  : percentile of |difference| over all frames that sets the grey limits.
# zoom_rsun  : half-width of the magnified EUI panel, solar radii.
# eui_rdiff  : 0 = EUI intensity in the magnified panel; 1 = EUI running difference there
#              (this EUI image minus the one matched to the previous Metis image, same
#              pair as Metis). Frames then go to <outdir>_EUIrdiff, so both sets are kept.
# lc_panel   : 1 adds the STIX + GOES light-curve panel below the images; 0 omits it.
# stix_dir   : STIX quick-look folder (Multiwb_datamine.get_STIX_LC; CSV for sT-eT reused).
# stix_bands : STIX bands to draw, from '4-10 keV', '10-15 keV', '15-25 keV', '25-50 keV',
#              '50-84 keV' (4-10 keV is the band of GOES XRS-B 1-8 A). 1-min means.
# goes_folder: GOES pickle cache (plot_goes_derivative); downloaded if no file covers sT-eT.
########### INPUT ###########################################################
basedir='/path/to/event_folder'  # Event folder: all relative paths below live inside it
outdir='SolO/Metis_EUI_rdiff_frames'
metis_dir='SolO/Metis/metis_tb'
eui_dir='SolO/EUI/fsi174'
sT='YYYY-MM-DDThh:mm:ss'  # Event start, UTC
eT='YYYY-MM-DDThh:mm:ss'  # Event end, UTC
eui_product='eui-fsi174-image'
eui_maxdt=900
npix=1000
diff_clip=99.
zoom_rsun=1.6
eui_rdiff=0
lc_panel=1
stix_dir='SolO/STIX'
stix_bands=['4-10 keV','15-25 keV','25-50 keV']
goes_folder='GOES'
#############################################################################


def load_metis(path):
	"""Metis image (primary HDU) as a Map; zero/negative pixels (occulter, outside FoV) -> NaN."""
	with fits.open(path) as h:
		d=h[0].data.astype(float);hd=h[0].header
	d[~(d>0)]=np.nan
	return sunpy.map.Map(d,hd)

def metis_files(folder,sT,eT):
	"""Metis FITS in folder with DATE-OBS inside [sT,eT], time-sorted."""
	t0,t1=Time(sT),Time(eT)
	got=[(dh._hdr_time(p),p) for p in glob.glob(os.path.join(folder,'*.fits'))]
	return [p for t,p in sorted(got,key=lambda x:x[0].mjd) if t0<=t<=t1]

def get_eui(times,folder,product,maxdt):
	"""Nearest SOAR L2 EUI file to each time (None if none within maxdt s); downloads missing ones."""
	rows=dh._soar_query('v_sc_data_item','EUI',product,'L2',min(times)-maxdt*u.s,max(times)+maxdt*u.s)
	if not rows:return [None]*len(times)
	tr=Time([r['begin_time'] for r in rows])
	out=[]
	for t in times:
		k=int(np.argmin(np.abs((tr-t).to_value(u.s))))
		out.append(dh._soar_fetch(rows[k],folder) if abs((tr[k]-t).to_value(u.s))<=maxdt else None)
	return out

def frame_grid(m,npix):
	"""North-up helioprojective header centred on the Sun, seen from m's observer at m's time,
	with half-width equal to the farthest valid Metis pixel."""
	c=sunpy.map.all_coordinates_from_map(m)
	r=np.hypot(c.Tx,c.Ty).to_value(u.arcsec)[np.isfinite(m.data)]
	half=float(r.max())*1.02
	ref=SkyCoord(0*u.arcsec,0*u.arcsec,frame=m.coordinate_frame)
	return make_fitswcs_header((npix,npix),ref,scale=[2*half/npix]*2*u.arcsec/u.pix,rotation_angle=0*u.deg)

def to_grid(m,hdr):
	"""Sample Map m on the grid of header hdr by helioprojective angle (no 3-D transform); NaN outside m."""
	g=sunpy.map.Map(np.zeros((hdr['naxis2'],hdr['naxis1'])),hdr)
	yy,xx=np.indices(g.data.shape)
	c=g.pixel_to_world(xx*u.pix,yy*u.pix)
	sx,sy=m.world_to_pixel(SkyCoord(c.Tx,c.Ty,frame=m.coordinate_frame))
	return map_coordinates(np.asarray(m.data,float),[sy.value,sx.value],order=1,mode='constant',cval=np.nan)

def extent_rsun(meta,rsun_arcsec):
	"""imshow extent (x0,x1,y0,y1) in solar radii of a north-up grid centred on the Sun (CRVAL=0)."""
	e=lambda k:[((v-meta[f'crpix{k}'])*meta[f'cdelt{k}'])/rsun_arcsec for v in (.5,meta[f'naxis{k}']+.5)]
	return e(1)+e(2)

def hgs_grid(ax,frame,rsun_arcsec,step=15):
	"""Stonyhurst grid every step degrees (visible hemisphere) in solar radii on ax."""
	for lon in np.arange(-180,180,step):
		for lo,la in ((np.full(181,lon),np.linspace(-90,90,181)),(np.linspace(-180,180,361),np.full(361,lon/2))) if abs(lon)<90 else ((np.full(181,lon),np.linspace(-90,90,181)),):
			c=SkyCoord(lo*u.deg,la*u.deg,frame='heliographic_stonyhurst',obstime=frame.obstime).transform_to(frame)
			v=c.is_visible()
			ax.plot(np.where(v,c.Tx.to_value(u.arcsec),np.nan)/rsun_arcsec,np.where(v,c.Ty.to_value(u.arcsec),np.nan)/rsun_arcsec,color='w',lw=.5,alpha=.3)

def light_curves(sT,eT,stix_dir,stix_bands,goes_folder):
	"""(STIX 1-min means of stix_bands, GOES XRS-B) DataFrames for sT-eT, both in Earth light-arrival UT."""
	from Multiwb_datamine import get_STIX_LC
	from goes_cache import resolve_goes_pickle
	tag=lambda t:Time(t).strftime('%Y%m%dT%H%M%S')
	csv=Path(stix_dir)/f'STIX_QL_{tag(sT)}-{tag(eT)}.csv'
	stix=pd.read_csv(csv) if csv.is_file() else get_STIX_LC(sT,eT,outdir=stix_dir)[0]
	stix['datetime']=pd.to_datetime(stix['datetime'],utc=True,format='mixed')
	bad=[b for b in stix_bands if b not in stix]
	if bad:raise ValueError(f'Unknown STIX band(s) {bad}; use {[k for k in stix if "keV" in k]}.')
	stix=stix.set_index('datetime')[list(stix_bands)].resample('1min').mean()
	with open(resolve_goes_pickle(goes_folder,sT,eT),'rb') as f:g=pickle.load(f)
	goes=pd.DataFrame(dict(flux=np.asarray(g['Flux'],float)),index=pd.to_datetime(g['Time'],utc=True))
	return stix,goes

def make_frames(mfiles,efiles,outdir,npix=1000,diff_clip=99.,zoom_rsun=1.6,eui_rdiff=False,lc=None,lc_window=None):
	"""Write one PNG per Metis image after the first; returns the PNG paths.
	mfiles: time-sorted Metis FITS; efiles: matching EUI FITS or None (same length).
	Left: Metis running difference outside the occulter with EUI inside. Right: EUI
	magnified to +-zoom_rsun solar radii, to see the source region; eui_rdiff=True shows
	there the EUI running difference over the same pair as Metis (grey, +-diff_clip limits).
	lc: (stix, goes) from light_curves for a lower panel over lc_window=(sT,eT); None omits it."""
	Path(outdir).mkdir(parents=True,exist_ok=True)
	frames=[];ref=None  # ref: first magnified EUI intensity, sets the EUI stretch
	prev=load_metis(mfiles[0])
	for mf,ef,ep in zip(mfiles[1:],efiles[1:],efiles[:-1]):
		cur=load_metis(mf)
		hdr=frame_grid(cur,npix)
		diff=sunpy.map.Map(to_grid(cur,hdr)-to_grid(prev,hdr),hdr)
		eui=zoom=None
		if ef:
			e=sunpy.map.Map(ef)
			half=zoom_rsun*cur.rsun_obs.to_value(u.arcsec)
			zh=make_fitswcs_header((700,700),SkyCoord(0*u.arcsec,0*u.arcsec,frame=cur.coordinate_frame),scale=[2*half/700]*2*u.arcsec/u.pix,rotation_angle=0*u.deg)
			zd=to_grid(e,zh)
			if ref is None:ref=zd.copy()
			if eui_rdiff:zd=zd-to_grid(sunpy.map.Map(ep),zh) if ep else np.full_like(zd,np.nan)
			zoom=sunpy.map.Map(zd,zh)
			k=int(max(1,np.floor(hdr['cdelt1']/e.scale.axis1.to_value(u.arcsec/u.pix))))
			if k>1:e=e.superpixel([k,k]*u.pix,func=np.nanmean)
			eui=sunpy.map.Map(to_grid(e,hdr),hdr)
		frames.append((cur.date,prev.date,diff,eui,zoom,ef,ep))
		prev=cur
	lim=np.nanpercentile(np.abs(np.concatenate([f[2].data[np.isfinite(f[2].data)] for f in frames])),diff_clip)
	en=None
	if eui_rdiff:
		zv=[f[4].data[np.isfinite(f[4].data)] for f in frames if f[4] is not None]
		zlim=np.nanpercentile(np.abs(np.concatenate(zv)),diff_clip) if zv else 1.
	clear=lambda name:plt.get_cmap(name).with_extremes(bad=(0,0,0,0))
	fs=dict(label=18,tick=15,title=16,legend=13,sup=18)  # paper-size fonts
	def style(a,ylabel=True):
		a.set_xlabel(r'Solar X ($R_\odot$)',fontsize=fs['label'])
		if ylabel:a.set_ylabel(r'Solar Y ($R_\odot$)',fontsize=fs['label'])
		a.tick_params(labelsize=fs['tick']);a.set_aspect('equal')
	paths=[]
	for t,tp,diff,eui,zoom,ef,ep in frames:
		rs=diff.rsun_obs.to_value(u.arcsec)
		fig=plt.figure(figsize=(15,11 if lc else 7.8))
		gs=fig.add_gridspec(2 if lc else 1,2,height_ratios=[4,1.6] if lc else [1],hspace=.32,wspace=.22)
		ax=fig.add_subplot(gs[0,0]);E=extent_rsun(diff.meta,rs)
		ax.set_facecolor('black')
		if ref is not None and en is None:
			v=ref[np.isfinite(ref)&(ref>0)]
			en=ImageNormalize(vmin=np.percentile(v,2),vmax=np.percentile(v,99.9),stretch=AsinhStretch(0.01),clip=False)  # clip=True would paint masked pixels
		if eui is not None:ax.imshow(np.ma.masked_invalid(eui.data),origin='lower',extent=E,cmap=clear('solar orbiterfsi174'),norm=en)
		ax.imshow(np.ma.masked_invalid(diff.data),origin='lower',extent=E,cmap=clear('gray'),vmin=-lim,vmax=lim)
		ax.add_patch(plt.Circle((0,0),1,fill=False,color='w',lw=.6,alpha=.6));style(ax)
		ax.set_title(f'(a) Metis tB running diff. {t.isot[11:16]}−{tp.isot[11:16]} UT\n+ EUI inside the occulter',fontsize=fs['title'])
		bx=fig.add_subplot(gs[0,1]);bx.set_facecolor('black')
		if zoom is not None:
			Z=extent_rsun(zoom.meta,rs);band=f'EUI {eui_product.split("fsi")[1][:3]} Å'
			if eui_rdiff:
				bx.imshow(np.ma.masked_invalid(zoom.data),origin='lower',extent=Z,cmap=clear('gray'),vmin=-zlim,vmax=zlim)
				title=f'(b) {band} running diff. {sunpy.map.Map(ef).date.isot[11:16]}−{sunpy.map.Map(ep).date.isot[11:16] if ep else "none"} UT'
			else:
				bx.imshow(np.ma.masked_invalid(zoom.data),origin='lower',extent=Z,cmap=clear('solar orbiterfsi174'),norm=en)
				title=f'(b) {band} {sunpy.map.Map(ef).date.isot[11:19]} UT'
			hgs_grid(bx,zoom.coordinate_frame,rs)
			bx.set_xlim(Z[:2]);bx.set_ylim(Z[2:])
			bx.set_title(title,fontsize=fs['title'])
		else:bx.set_title('(b) no EUI image within eui_maxdt',fontsize=fs['title'])
		style(bx)
		o=diff.observer_coordinate
		fig.suptitle(f'Solar Orbiter {t.isot[:10]}  ·  Stonyhurst lon {o.lon.to_value(u.deg):+.0f}°, lat {o.lat.to_value(u.deg):+.0f}°, '
			f'{diff.dsun.to_value(u.AU):.2f} AU',fontsize=fs['sup'])
		if lc:  # STIX (left axis) + GOES (right axis) for the whole window; line at this frame
			stix,goes=lc;lx=fig.add_subplot(gs[1,:]);rx=lx.twinx()
			for b,col in zip(stix.columns,['#2a78d6','#1baf7a','#6250d6','#e87ba4','#008300']):
				lx.plot(stix.index,stix[b].where(stix[b]>0),color=col,lw=1.3,label=f'STIX {b}')
			rx.plot(goes.index,goes.flux.where(goes.flux>0),color='orange',lw=1.5,label='GOES 1–8 Å')
			te=(t+(get_earth(t).radius-diff.dsun)/c_light).to_datetime()  # Metis time -> Earth light-arrival
			lx.axvline(te,color='k',ls=':',lw=1.4)
			lx.set_yscale('log');rx.set_yscale('log')
			lx.set_xlim(pd.Timestamp(lc_window[0]),pd.Timestamp(lc_window[1]))
			lx.set_ylabel('STIX (counts s$^{-1}$)',fontsize=fs['label']);rx.set_ylabel('GOES (W m$^{-2}$)',fontsize=fs['label'])
			lx.set_xlabel(f'Time (UT, Earth light-arrival) on {lc_window[0][:10]}',fontsize=fs['label'])
			for a in (lx,rx):a.tick_params(labelsize=fs['tick'])
			lx.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'));lx.grid(alpha=.2)
			lx.text(.01,.95,'(c)',transform=lx.transAxes,va='top',fontsize=fs['title'])
			h1,l1=lx.get_legend_handles_labels();h2,l2=rx.get_legend_handles_labels()
			lx.legend(h1+h2,l1+l2,loc='upper center',bbox_to_anchor=(.5,1.22),fontsize=fs['legend'],ncol=len(l1)+1,frameon=False)
		p=os.path.join(outdir,f'Metis_EUI_rdiff_{t.strftime("%Y%m%d_%H%M%S")}.png')
		fig.savefig(p,dpi=110,bbox_inches='tight');plt.close(fig)
		paths.append(p)
	return paths

if __name__=='__main__':
	cwd=os.getcwd()
	os.chdir(basedir)
	try:
		warnings.filterwarnings('ignore')
		mfiles=metis_files(metis_dir,sT,eT)
		print(f'{len(mfiles)} Metis images in {sT} -- {eT}')
		efiles=get_eui([dh._hdr_time(p) for p in mfiles],eui_dir,eui_product,eui_maxdt)
		print(f'{sum(e is not None for e in efiles)} frames have an EUI image within {eui_maxdt} s')
		out=outdir+('_EUIrdiff' if eui_rdiff else '')
		lc=light_curves(sT,eT,stix_dir,stix_bands,goes_folder) if lc_panel else None
		paths=make_frames(mfiles,efiles,out,npix,diff_clip,zoom_rsun,bool(eui_rdiff),lc,(sT,eT))
		print(f'{len(paths)} frames in {out}')
		mp4=Path(out)/('Metis_tB_rdiff_EUI'+eui_product.split('fsi')[1][:3]+('rdiff' if eui_rdiff else '')+'.mp4')
		write_mp4(paths,mp4);print(f'Saved MP4 of {len(paths)} frames: {mp4}')
	finally:
		os.chdir(cwd)
