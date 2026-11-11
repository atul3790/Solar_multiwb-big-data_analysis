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
