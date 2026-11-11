"""Sample previously estimated intensity flow near current bright/dark edges.

Flow is source->target (dy,dx), in image pixels. Quality is an internal
consistency mask, not a calibrated measurement uncertainty.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

def sample_edge_motion(edge_points,dx,dy,quality,x_rsun,y_rsun,dt_seconds,
	spacing=12.,radius=3.,min_support=5,min_displacement=.1):
	"""Place local median vectors on target-frame edges and return a row list.

	edge_points: Nx2 array of target (x,y) pixel positions.
	dx/dy: intensity-flow displacements at SOURCE pixels; they are advected to
	TARGET positions before searching for support within radius pixels.
	x_rsun/y_rsun: common image-plane coordinates. Speeds use R_sun=695700km
	and the local coordinate Jacobian, divided by actual pair duration.
	Returned dx/dy are unamplified pixels; vx/vy and speed are projected km/s.
	"""
	if dt_seconds<=0:raise ValueError('Positive pair duration required.')
	y,x=np.indices(dx.shape);q=quality&np.isfinite(dx)&np.isfinite(dy)
	tx=x[q]+dx[q];ty=y[q]+dy[q];vx=dx[q];vy=dy[q]
	inside=(tx>=0)&(tx<dx.shape[1]-1)&(ty>=0)&(ty<dx.shape[0]-1)
	tx,ty,vx,vy=tx[inside],ty[inside],vx[inside],vy[inside]
	if not len(tx):return []
	tree=cKDTree(np.column_stack([tx,ty]));used=[];rows=[]
	x_dy,x_dx=np.gradient(x_rsun);y_dy,y_dx=np.gradient(y_rsun)
	for px,py in np.asarray(edge_points):
		if not np.isfinite(px+py):continue
		if any((px-a)**2+(py-b)**2<spacing**2 for a,b in used):continue
		idx=tree.query_ball_point([px,py],radius)
		if len(idx)<min_support:continue
		mx,my=float(np.median(vx[idx])),float(np.median(vy[idx]));mag=np.hypot(mx,my)
		spread=float(np.median(np.hypot(vx[idx]-mx,vy[idx]-my)))
		if mag<min_displacement or spread>max(.3,.5*mag):continue
		ix,iy=int(round(px)),int(round(py))
		if not 0<=ix<dx.shape[1] or not 0<=iy<dx.shape[0]:continue
		vx_kms=(mx*x_dx[iy,ix]+my*x_dy[iy,ix])*695700/dt_seconds
		vy_kms=(mx*y_dx[iy,ix]+my*y_dy[iy,ix])*695700/dt_seconds
		X,Y=float(x_rsun[iy,ix]),float(y_rsun[iy,ix]);radius_rsun=np.hypot(X,Y)
		rows.append(dict(x_pixel=float(px),y_pixel=float(py),dx_pixel=mx,dy_pixel=my,
			vx_projected_kms=float(vx_kms),vy_projected_kms=float(vy_kms),
			speed_projected_kms=float(np.hypot(vx_kms,vy_kms)),support_pixels=len(idx),
			displacement_scatter_pixel=spread,x_rsun=X,y_rsun=Y,
			v_radial_projected_kms=float((vx_kms*X+vy_kms*Y)/radius_rsun) if radius_rsun else np.nan,
			clock_angle_deg=float(np.degrees(np.arctan2(X,Y))%360)))
		used.append((px,py))
	return rows

def contour_pixels(mask):
	"""Return ordered boundary pixels without requiring a new segmentation."""
	edge=mask&~ndi.binary_erosion(mask)
	y,x=np.nonzero(edge)
	return np.column_stack([x,y])

def sample_centroid_motion(labels,dx,dy,quality,x_rsun,y_rsun,dt_seconds,
	min_area,min_support,min_displacement,*,intensity,sign=1):
	"""Draw one signed-difference-weighted mean flow at the weighted region center."""
	weights=np.where(np.isfinite(intensity),np.maximum(sign*intensity,0.),0.)
	y,x=np.nonzero(quality&np.isfinite(dx)&np.isfinite(dy))
	u,v=dx[y,x],dy[y,x]
	tx,ty=np.rint(x+u).astype(int),np.rint(y+v).astype(int)
	ok=(tx>=0)&(ty>=0)&(tx<labels.shape[1])&(ty<labels.shape[0])
	tx,ty,u,v=tx[ok],ty[ok],u[ok],v[ok]
	target=labels[ty,tx];w=weights[ty,tx]
	xd_y,xd_x=np.gradient(x_rsun);yd_y,yd_x=np.gradient(y_rsun)
	for label in np.unique(target):
		if not label:continue
		yy,xx=np.nonzero(labels==label);match=(target==label)&(w>0)
		count=int(match.sum())
		if len(xx)<min_area or count<min_support:continue
		wf=w[match];du,dv=u[match],v[match]
		mx,my=np.average(du,weights=wf),np.average(dv,weights=wf)
		mag=np.hypot(mx,my)
		scatter=np.median(np.hypot(du-mx,dv-my))
		if mag<=0 or mag<min_displacement or scatter>max(.3,.5*mag):continue
		wc=weights[yy,xx]
		px,py=np.average(xx,weights=wc),np.average(yy,weights=wc)
		j,i=ty[match],tx[match]
		vx=np.average(du*xd_x[j,i]+dv*xd_y[j,i],weights=wf)*695700/dt_seconds
		vy=np.average(du*yd_x[j,i]+dv*yd_y[j,i],weights=wf)*695700/dt_seconds
		yield dict(component=int(label),x_pixel=float(px),y_pixel=float(py),dx_pixel=float(mx),dy_pixel=float(my),
			speed_projected_kms=float(np.hypot(vx,vy)),vx_projected_kms=float(vx),vy_projected_kms=float(vy),
			support_pixels=count,displacement_scatter_pixel=float(scatter))
