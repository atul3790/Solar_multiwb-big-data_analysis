"""Optional co-observer disk underlay in the coronagraph pixel grid."""
import numpy as np
from scipy.ndimage import map_coordinates
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import propagate_with_solar_surface
from matplotlib.colors import Normalize, PowerNorm

def disk_layer(target,source,max_minutes=60.,resolution=160):
	"""Return a masked high-resolution disk, pixel extent, colormap and norm."""
	if source is None or abs((target.date-source.date).to_value(u.min))>max_minutes:
		return None
	identity=(str(target.observatory)+' '+str(target.instrument)).upper()
	src=(str(source.observatory)+' '+str(source.instrument)+' '+str(source.detector)).upper()
	if 'LASCO' in identity and not any(k in src for k in ('AIA','HMI')):
		raise ValueError('LASCO disk underlay requires AIA or HMI')
	if any(k in identity for k in ('STEREO','SECCHI','COR1','COR2')):
		if 'EUVI' not in src:raise ValueError('STEREO coronagraph disk underlay requires EUVI')
		a,b=target.observer_coordinate,source.observer_coordinate.transform_to(target.observer_coordinate.frame)
		if a.separation(b)>5*u.deg:raise ValueError('EUVI must be from the matching STEREO viewpoint')
	hpc=target.coordinate_frame
	center=hpc
	c=SkyCoord(0*u.arcsec,0*u.arcsec,frame=center)
	px,py=target.world_to_pixel(c)
	rx,ry=[abs((target.rsun_obs/s).to_value(u.pix)) for s in target.scale]
	x0,x1=px.value-rx,px.value+rx;y0,y1=py.value-ry,py.value+ry
	f=(np.arange(resolution)+.5)/resolution
	x,y=np.meshgrid(x0+(x1-x0)*f,y0+(y1-y0)*f)
	disk=((x-px.value)/rx)**2+((y-py.value)/ry)**2<=.995**2
	sample=target.pixel_to_world(x[disk]*u.pix,y[disk]*u.pix).make_3d()
	with propagate_with_solar_surface():
		w=sample.transform_to(source.coordinate_frame)
	sx,sy=source.world_to_pixel(SkyCoord(w))
	arr=np.full(x.shape,np.nan)
	values=map_coordinates(np.asarray(source.data,float),[sy.value,sx.value],order=1,mode='constant',cval=np.nan)
	visible=w.distance.to_value(u.m)<=np.sqrt(source.dsun.to_value(u.m)**2-source.rsun_meters.to_value(u.m)**2)
	arr[disk]=np.where(visible,values,np.nan)
	good=arr[np.isfinite(arr)]
	if not good.size:return None
	if 'HMI' in src and ('gauss' in str(source.unit).lower() or str(source.unit)=='G'):
		return np.ma.masked_invalid(arr),(x0,x1,y0,y1),'gray',Normalize(-250,250,clip=True)
	lo,hi=np.percentile(good,[1,99.5])
	return np.ma.masked_invalid(arr),(x0,x1,y0,y1),source.plot_settings.get('cmap','gray'),PowerNorm(.5,vmin=lo,vmax=max(hi,lo+1e-12),clip=True)

def disk_inventory(folder,instrument,product,target):
	"""Index local FITS disk images by observation time and requested product."""
	from pathlib import Path
	from astropy.io import fits
	from astropy.time import Time
	if folder is None:return []
	root=Path(folder).expanduser()
	if not root.is_dir():return []
	instrument=instrument.upper()
	target_name=str(target.observatory).upper().replace('_','').replace('-','').replace(' ','')
	rows=[]
	for p in sorted(root.rglob('*')):
		if not p.is_file() or not p.name.lower().endswith(('.fits','.fits.gz','.fit','.fts','.fits.fz')):continue
		try:
			with fits.open(p,lazy_load_hdus=True) as hdus:
				for hdu in hdus:
					h=hdu.header
					if int(h.get('ZNAXIS',h.get('NAXIS',0)))<2:continue
					label=' '.join(str(h.get(k,'')) for k in ('INSTRUME','DETECTOR')).upper()
					if instrument not in label:continue
					if instrument in ('AIA','EUVI') and int(float(h.get('WAVELNTH',-1)))!=int(product):continue
					if instrument=='EUVI':
						name=str(h.get('OBSRVTRY',h.get('TELESCOP',''))).upper().replace('_','').replace('-','').replace(' ','')
						normalize=lambda n:n.replace('AHEAD','A').replace('BEHIND','B')
						if normalize(name)!=normalize(target_name):continue
					if instrument=='HMI':
						desc=' '.join(str(h.get(k,'')) for k in ('CONTENT','BTYPE','BUNIT','SERIES'))+' '+p.name
						desc=desc.lower()
						# Full words first; short series codes last ('_HV_' names contain 'v_').
						kind=('continuum' if 'continuum' in desc else
							'dopplergram' if 'doppler' in desc else
							'magnetogram' if 'magnet' in desc or 'gauss' in desc or h.get('BUNIT')=='G' else
							'continuum' if 'ic_' in desc else 'dopplergram' if 'v_' in desc else
							'magnetogram' if 'm_' in desc else None)
						if kind!=str(product).lower():continue
					stamp=h.get('DATE-OBS') or h.get('DATE_OBS')
					if stamp:rows.append((Time(stamp),p))
					break
		except (OSError,ValueError,TypeError):continue
	return rows

def nearest_disk(tmap,inventory,max_minutes,load_map):
	"""Return the closest usable reprojected disk and its source path."""
	for t,path in sorted(inventory,key=lambda row:abs((row[0]-tmap.date).to_value(u.min))):
		if abs((t-tmap.date).to_value(u.min))>max_minutes:break
		try:
			disk=disk_layer(tmap,load_map(path),max_minutes)
			if disk is not None:return disk,path
		except (OSError,ValueError,TypeError):continue
	return None,None
