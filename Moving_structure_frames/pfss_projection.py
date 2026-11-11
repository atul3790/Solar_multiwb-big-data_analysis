"""PFSS projection for the motion frames.

Drawing functions come from solar_overlay_helper (Solar_data_mining-main), so
fixes there apply here too. This file adds only the active-region selectors.
Radial continuation beyond the source surface is display only.
"""
import numpy as np
import astropy.units as u
from astropy.constants import R_sun
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from solar_overlay_helper import _quiet, _nan_sep, _pfss_to_fov, _closed_loop_limit, _draw_pfss  # noqa: F401


def select_ar_closed(lines,tmap,epoch,lon,lat,half_lon,half_lat,half_box,max_closed,rooted='both'):
	"""Closed loops of the NOAA region.

	rooted='both' (default): both footpoints inside the Stonyhurst bounds and the
	whole trace inside the HPC box (half_box arcsec). rooted='one': at least one
	footpoint inside the bounds, no box (half_box may be None), so large loops
	rooted in the AR are kept; when more than max_closed qualify, the tallest
	are kept first.
	"""
	hgs=frames.HeliographicStonyhurst(obstime=epoch)
	hpc=frames.Helioprojective(observer='earth',obstime=epoch)
	c=SkyCoord(lines['lon']*u.deg,lines['lat']*u.deg,lines['r']*R_sun,
		frame=frames.HeliographicCarrington(observer='earth',obstime=epoch))
	s=c.transform_to(hgs);p=c.transform_to(hpc)
	center=SkyCoord(lon*u.deg,lat*u.deg,R_sun,frame=hgs).transform_to(hpc)
	ok=(np.abs((s.lon.deg-lon+180)%360-180)<=half_lon)&(np.abs(s.lat.deg-lat)<=half_lat)
	box=np.ones(ok.shape,bool) if half_box is None else (abs((p.Tx-center.Tx).arcsec)<=half_box)&(abs((p.Ty-center.Ty).arcsec)<=half_box)
	ids=[]
	for k in np.unique(lines['seg'][lines['pol']==0]):
		i=np.flatnonzero(lines['seg']==k);ends=ok[i[[0,-1]]]
		if (ends.any() if rooted=='one' else ends.all()) and box[i].all():ids.append((k,lines['r'][i].max()))
	if rooted=='one':ids.sort(key=lambda kh:-kh[1])
	return np.isin(lines['seg'],[k for k,_ in ids[:max_closed]])


def select_ar_open(lines,epoch,lon,lat,half_lon,half_lat,half_box):
	"""Select open traces by their lower endpoint, independent of trace direction.

	half_box None skips the HPC box test.
	"""
	ends=[]
	ids=np.unique(lines['seg'][lines['pol']!=0])
	for k in ids:
		i=np.flatnonzero(lines['seg']==k)
		pair=i[[0,-1]]
		ends.append(pair[np.argmin(lines['r'][pair])])
	if not ends:return np.zeros(lines['seg'].shape,dtype=bool)
	hgs=frames.HeliographicStonyhurst(obstime=epoch)
	hpc=frames.Helioprojective(observer='earth',obstime=epoch)
	c=SkyCoord(lines['lon'][ends]*u.deg,lines['lat'][ends]*u.deg,lines['r'][ends]*R_sun,
		frame=frames.HeliographicCarrington(observer='earth',obstime=epoch))
	s=c.transform_to(hgs);p=c.transform_to(hpc)
	center=SkyCoord(lon*u.deg,lat*u.deg,R_sun,frame=hgs).transform_to(hpc)
	ok=(np.abs((s.lon.deg-lon+180)%360-180)<=half_lon)&(np.abs(s.lat.deg-lat)<=half_lat)
	if half_box is not None:ok&=(abs((p.Tx-center.Tx).arcsec)<=half_box)&(abs((p.Ty-center.Ty).arcsec)<=half_box)
	return np.isin(lines['seg'],ids[ok])


RSUN_1AU_ARCSEC=float(np.degrees(np.arctan(695700./1.495978707e8))*3600)  # 959.6 arcsec


def ar_rooted_lines(snapshot,lon,lat,reported,half_lon,half_lat,spacing_deg,r_seed,noaa,
	long_loops=True,n_closed=None,n_open=None):
	"""AR field lines for one PFSS snapshot, selected as in the flow frames.

	snapshot: PFSS snapshot folder (metadata.json, model_3d.npz). lon, lat:
	Stonyhurst AR position (deg) at time `reported`; it is carried to the
	snapshot epoch at fixed Carrington position. Seeds every spacing_deg over
	a patch covering +-half_lon x +-half_lat (deg); all traced lines are cached
	as snapshot/AR_field_lines_rooted.npz (shared with the flow frames, reused
	only for the same rectangle, spacing and NOAA number). Selection: closed
	loops with one footpoint in the rectangle (long_loops=True, tallest first)
	or both; open lines whose lower end lies in it. At most n_closed loops and
	n_open open lines (spread evenly); None keeps all.
	Returns a line dict (lon, lat, r, seg, pol, rss, time) for drawing.
	"""
	import json
	from pathlib import Path
	from astropy.time import Time
	from pfss_fixed_seeds import trace_ar_lines
	snapshot=Path(snapshot);keys=('lon','lat','r','seg','pol')
	meta=json.loads((snapshot/'metadata.json').read_text());epoch=Time(meta['model_time'])
	c=SkyCoord(lon*u.deg,lat*u.deg,1*u.solRad,frame=frames.HeliographicStonyhurst(obstime=Time(reported)))
	c=c.transform_to(frames.HeliographicCarrington(observer='earth',obstime=Time(reported)))
	c=SkyCoord(c.lon,c.lat,c.radius,frame=frames.HeliographicCarrington(observer='earth',obstime=epoch))
	c=c.transform_to(frames.HeliographicStonyhurst(obstime=epoch))
	rect=np.array([half_lon,half_lat,spacing_deg],float);path=snapshot/'AR_field_lines_rooted.npz';lines=None
	if path.is_file():
		with np.load(path) as z:
			if 'ar_rect' in z and np.allclose(z['ar_rect'],rect) and ('noaa' not in z or int(z['noaa'])==int(noaa)):
				lines={k:z[k].copy() for k in keys}|{'rss':float(z['rss'])}
	if lines is None:
		half=RSUN_1AU_ARCSEC*np.sin(np.radians(max(half_lon,half_lat)));step=RSUN_1AU_ARCSEC*np.radians(spacing_deg)
		t=trace_ar_lines(snapshot,c.lon.deg,c.lat.deg,epoch,half,None,None,r_seed,step)
		lines={k:t[k] for k in keys}|{'rss':float(meta['rss'])}
		np.savez_compressed(path,**lines,ar_rect=rect,noaa=int(noaa),traced_source=str((snapshot/'field_lines.npz').resolve()),
			halfwidth_arcsec=half,spacing_arcsec=step,r_seed=r_seed)
		print(f'Saved traced AR field lines: {path}',flush=True)
	closed=select_ar_closed(lines,None,epoch,c.lon.deg,c.lat.deg,half_lon,half_lat,None,
		len(lines['seg']) if n_closed is None or not np.isfinite(n_closed) else int(n_closed),'one' if long_loops else 'both')
	opened=select_ar_open(lines,epoch,c.lon.deg,c.lat.deg,half_lon,half_lat,None)
	ids=np.unique(lines['seg'][opened])
	if n_open is not None and np.isfinite(n_open) and len(ids)>n_open:
		opened&=np.isin(lines['seg'],ids[np.linspace(0,len(ids),int(n_open),endpoint=False,dtype=int)])
	keep=closed|opened
	print(f'NOAA {noaa} at {epoch.isot}: {len(np.unique(lines["seg"][opened]))} open, {len(np.unique(lines["seg"][closed]))} closed AR traces drawn',flush=True)
	return {k:lines[k][keep] for k in keys}|{'rss':lines['rss'],'time':epoch}
