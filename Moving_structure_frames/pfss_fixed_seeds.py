"""Retrace a fixed set of Carrington seed positions through saved PFSS boundaries."""
import json
from pathlib import Path

import numpy as np
from datamine_calib_helper import load_map


def select_trace_seeds(lines,seed_radius):
	"""Return Carrington lon, lat and radius at the saved vertex nearest seed_radius."""
	seeds=[]
	for seg in np.unique(lines['seg']):
		idx=np.flatnonzero(lines['seg']==seg)
		i=idx[np.argmin(abs(lines['r'][idx]-seed_radius))]
		seeds.append([lines[k][i] for k in ('lon','lat','r')])
	return np.asarray(seeds,dtype=float).reshape(-1,3)


def retrace_pfss(pfss_file,seed_groups):
	"""Solve once from the saved prepared boundary and trace all seed groups together."""
	import astropy.units as u
	from astropy.coordinates import SkyCoord
	import pfsspy
	from pfsspy import tracing
	import sunpy.map

	folder=Path(pfss_file).parent
	boundary=folder/'boundary.fits'
	if not boundary.is_file():
		raise FileNotFoundError(f'Fixed PFSS seeds require the saved prepared boundary: {boundary}')
	meta=json.loads((folder/'metadata.json').read_text())
	rss=float(meta['rss'])
	keys=('lon','lat','r','seg','pol')
	groups=[{k:[] for k in keys} for _ in seed_groups]
	seeds=np.concatenate(seed_groups)
	if len(seeds):
		if not np.isfinite(seeds).all() or np.any((seeds[:,2]<=1)|(seeds[:,2]>=rss)):
			raise ValueError(f'PFSS seeds must be finite and inside 1 < r < {rss:g} Rsun.')
		model=pfsspy.pfss(pfsspy.Input(load_map(boundary),int(meta['nrho']),rss))  # load_map writes RSUN_REF if absent
		coords=SkyCoord(seeds[:,0]*u.deg,seeds[:,1]*u.deg,seeds[:,2]*u.solRad,
			frame=model.coordinate_frame)
		try:
			tracer=tracing.FortranTracer()
		except (ImportError,RuntimeError):
			tracer=tracing.PythonTracer()
		group_ids=np.repeat(np.arange(len(groups)),[len(group) for group in seed_groups])
		for i,line in enumerate(tracer.trace(coords,model)):
			c=line.coords
			if len(c)<2:
				continue
			layer=groups[group_ids[i]]
			for key,value in zip(keys,(c.lon.deg,c.lat.deg,c.radius.to_value(u.solRad),
				np.full(len(c),i),np.full(len(c),int(line.polarity)))):
				layer[key].append(value)
	return tuple({k:np.concatenate(v) if v else np.array([]) for k,v in layer.items()}|{'rss':rss}
		for layer in groups)


def trace_ar_lines(directory,lon,lat,epoch,halfwidth,nlon,nlat,radius,spacing=40.):
	"""Dense AR field lines through a saved PFSS snapshot (model_3d.npz).

	directory: snapshot folder from build_pfss; lon/lat: Stonyhurst AR centre at
	epoch [deg]; halfwidth: patch half-size [arcsec, at 1 AU]; nlon/nlat: seed
	counts (None: 2*halfwidth/spacing+1); radius: start radius [Rsun].
	Seeds lie on the solar surface around the AR, so they stay valid when the
	region is beyond Earth's limb. Integrates the saved field; the PFSS model is
	neither changed nor recomputed. Copied from make_STEREO_overlay_frames.
	Returns lines in the solar_overlay_helper format (lon, lat, r, seg, pol).
	"""
	from scipy.interpolate import RegularGridInterpolator
	from scipy.integrate import solve_ivp
	import astropy.units as u
	from astropy.constants import R_sun
	from astropy.coordinates import SkyCoord
	from astropy.time import Time
	from sunpy.coordinates import frames
	folder=Path(directory)
	meta=json.loads((folder/'metadata.json').read_text())
	with np.load(folder/'model_3d.npz') as f:
		phi=np.asarray(f['longitude_deg'],float)
		sine=np.asarray(f['sin_latitude'],float)
		r=np.asarray(f['radius_rsun'],float)
		field=np.stack([f['Bphi'],f['Btheta'],f['Br']],axis=-1)
	if field.shape[:3]!=(len(phi),len(sine),len(r)):
		raise ValueError('Saved PFSS field dimensions do not match its coordinate axes')
	if not (-180<=lon<=180 and -90<=lat<=90 and np.isfinite(halfwidth) and halfwidth>0 and r[0]<radius<r[-1]):
		raise ValueError('Invalid AR coordinates, patch width or starting radius')
	if not np.isfinite(spacing) or spacing<=0:
		raise ValueError('AR seed spacing must be positive arcseconds')
	nlon=int(np.ceil(2*halfwidth/spacing))+1 if nlon is None else nlon
	nlat=int(np.ceil(2*halfwidth/spacing))+1 if nlat is None else nlat
	if any(int(n)!=n or n<2 for n in (nlon,nlat)):
		raise ValueError('AR seed counts must be integers >=2')
	nlon,nlat=int(nlon),int(nlat)
	order=np.argsort(phi%360)
	phi=(phi%360)[order]
	field=field[order]
	# PFSS grids can include both endpoints of the same periodic meridian.
	phi,unique=np.unique(np.round(phi,10),return_index=True)
	field=field[unique]
	phi=np.r_[phi[-1]-360,phi,phi[0]+360]
	field=np.concatenate([field[-1:],field,field[:1]],axis=0)
	interpolate=RegularGridInterpolator((np.deg2rad(phi),sine,r),field,
		bounds_error=False,fill_value=None)
	# Build an AR-centred patch on the solar surface; its seeds remain valid
	# when the region is beyond Earth's limb but visible from STEREO.
	# Angular size uses the solar apparent radius at 1 AU, independent of the
	# current spacecraft distance. Convert fixed Stonyhurst seeds to Carrington.
	unit_rsun=np.arctan2(R_sun.to_value(u.m),u.au.to(u.m))*180/np.pi*3600
	dx,dy=np.meshgrid(np.linspace(-halfwidth,halfwidth,nlon),np.linspace(-halfwidth,halfwidth,nlat))
	x,y=dx.ravel()/unit_rsun,dy.ravel()/unit_rsun
	rho2=x*x+y*y
	if np.any(rho2>=1):
		raise ValueError('AR patch corners exceed one solar-radius projection; reduce AR_halfwidth_arcsec')
	lon0,lat0=np.deg2rad([lon,lat])
	center=np.array([np.cos(lat0)*np.cos(lon0),np.cos(lat0)*np.sin(lon0),np.sin(lat0)])
	east=np.array([-np.sin(lon0),np.cos(lon0),0.])
	north=np.array([-np.sin(lat0)*np.cos(lon0),-np.sin(lat0)*np.sin(lon0),np.cos(lat0)])
	xyz=np.sqrt(1-rho2)[:,None]*center+x[:,None]*east+y[:,None]*north
	seed_hgs_lon=np.rad2deg(np.arctan2(xyz[:,1],xyz[:,0]))
	seed_hgs_lat=np.rad2deg(np.arcsin(np.clip(xyz[:,2],-1,1)))
	patch=SkyCoord(seed_hgs_lon*u.deg,seed_hgs_lat*u.deg,R_sun,
		frame=frames.HeliographicStonyhurst(obstime=Time(epoch)))
	patch=patch.transform_to(frames.HeliographicCarrington(obstime=Time(epoch),observer='earth'))
	seed_lon=patch.lon.to_value(u.rad)
	seed_sine=np.sin(patch.lat.to_value(u.rad))
	valid=np.isfinite(seed_lon)&np.isfinite(seed_sine)
	seed_lon,seed_sine=seed_lon[valid],seed_sine[valid]
	if not len(seed_lon):
		raise ValueError('AR patch has no finite photospheric seed positions')
	print(f'AR grid: {nlon} x {nlat}; {len(seed_lon)} photospheric seeds; half-width {halfwidth:g} arcsec')
	def direction(length,q):
		p,s,rr=q
		s=np.clip(s,-0.999999,0.999999)
		b=np.asarray(interpolate([[p%(2*np.pi),s,np.clip(rr,r[0],r[-1])]]))[0]
		norm=np.linalg.norm(b)
		if not np.isfinite(norm) or norm<1e-20:
			return np.zeros(3)
		c=np.sqrt(1-s*s)
		return np.array([b[0]/(rr*c),-c*b[1]/rr,b[2]])/norm
	def surface(length,q):
		return (q[2]-r[0])*(r[-1]-q[2])
	surface.terminal=True
	surface.direction=-1
	result={k:[] for k in ('lon','lat','r','seg','pol')}
	for i,(p,s) in enumerate(zip(seed_lon,seed_sine)):
		branches=[]
		for sign in (-1,1):
			sol=solve_ivp(lambda l,q:sign*direction(l,q),(0,100),[p,s,radius],
				events=surface,max_step=0.025,rtol=1e-5,atol=1e-7)
			branches.append(sol.y.T)
		# Discard paths that failed to reach either physical boundary.
		if any(min(abs(b[-1,2]-r[0]),abs(b[-1,2]-r[-1]))>1e-3 for b in branches):
			continue
		q=np.concatenate([branches[0][::-1],branches[1][1:]])
		open_line=np.any(q[:,2]>=r[-1]-1e-3)
		pol=int(np.sign(direction(0,[p,s,radius])[2])) if open_line else 0
		for k,v in [('lon',np.rad2deg(q[:,0])%360),('lat',np.rad2deg(np.arcsin(np.clip(q[:,1],-1,1)))),
			('r',q[:,2]),('seg',np.full(len(q),i)),('pol',np.full(len(q),pol))]:
			result[k].append(v)
	if not result['lon']:
		raise RuntimeError('No AR traces reached the PFSS boundaries')
	result={k:np.concatenate(v) for k,v in result.items()}
	result.update(time=Time(meta['model_time']),rss=float(meta['rss']))
	return result
