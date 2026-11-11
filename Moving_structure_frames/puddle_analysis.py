"""Full-field signed detection and conservative transport diagnostics."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

def usable_grid(x,y,valid,spoke_pa,halfwidth,buffer_px):
	"""Clock angle from solar north, clockwise on the displayed image."""
	pa=np.degrees(np.arctan2(x,y))
	clear=np.abs((pa-spoke_pa+180)%360-180)>halfwidth
	return ndi.binary_erosion(valid&clear,iterations=buffer_px) if buffer_px else valid&clear

def detect_signed(data,valid,sigma,min_area,floor,bright_contrast=8.,bright_area=35,grid=None,min_span=0.,flow_support=None,weak_factor=.55):
	"""Mask-normalized smoothing and robust global signed threshold."""
	v=np.where(valid,data,0)
	s=ndi.gaussian_filter(v,1.5)/np.maximum(ndi.gaussian_filter(valid.astype(float),1.5),1e-6)
	c=np.median(s[valid]); threshold=max(floor,sigma*1.4826*np.median(abs(s[valid]-c)))
	out=[]
	for sign in (1,-1):
		lab,_=ndi.label((sign*(s-c)>threshold)&valid,np.ones((3,3)))
		sizes=np.bincount(lab.ravel()); keep=sizes>=min_area;keep[0]=False
		if sign==1:
			for k in np.flatnonzero(keep):
				core=lab==k
				if sizes[k]<bright_area:
					keep[k]=False;continue
				dist=ndi.distance_transform_edt(~core)
				ring=(dist>=3)&(dist<=9)&valid&(lab==0)
				keep[k]=ring.sum()>=20 and np.median(s[core])-np.median(s[ring])>=bright_contrast
		out.append(ndi.label(keep[lab],np.ones((3,3)))[0])
	if flow_support is not None:
		# Keep the original bright components intact; test only additional patches.
		bright=out[0]
		exclusion=ndi.binary_dilation(bright>0)
		weak=(s-c>weak_factor*threshold)&valid&~exclusion&flow_support
		lab,n=ndi.label(weak,np.ones((3,3)))
		next_id=int(bright.max())
		for k in range(1,n+1):
			core=lab==k
			if core.sum()<min_area:continue
			dist=ndi.distance_transform_edt(~core)
			ring=(dist>=3)&(dist<=9)&valid&(lab==0)&(bright==0)
			if ring.sum()<20 or np.median(s[core])-np.median(s[ring])<weak_factor*bright_contrast:continue
			next_id+=1;bright[core]=next_id
	if grid is not None and min_span>0:
		x,y=grid
		for i,lab in enumerate(out):
			new=np.zeros_like(lab);n=0
			for k in np.unique(lab[lab>0]):
				core=lab==k
				xy=np.column_stack((x[core],y[core]))
				_,_,axes=np.linalg.svd(xy-xy.mean(axis=0),full_matrices=False)
				if np.ptp(xy@axes[0])<min_span:continue
				n+=1;new[core]=n
			out[i]=new
	return out,threshold

def outward_support(x,y,dx,dy,q,radius_rsun=1.5,min_votes=5,min_cos=.65,min_step=.5):
	"""Vote near destinations of quality-gated, outward projected flow samples."""
	yy,xx=np.indices(q.shape)
	ok=q&np.isfinite(dx)&np.isfinite(dy)
	i,j=yy[ok],xx[ok]
	tx=j+dx[ok];ty=i+dy[ok]
	x1=ndi.map_coordinates(x,[ty,tx],order=1,mode='constant',cval=np.nan)
	y1=ndi.map_coordinates(y,[ty,tx],order=1,mode='constant',cval=np.nan)
	x0=x[ok];y0=y[ok];u=x1-x0;v=y1-y0
	r=np.hypot(x0,y0);speed=np.hypot(u,v)
	good=(u*x0+v*y0)>min_cos*r*speed
	good&=np.hypot(dx[ok],dy[ok])>=min_step
	votes=np.zeros(q.shape,dtype=float)
	np.add.at(votes,(np.rint(ty[good]).astype(int),np.rint(tx[good]).astype(int)),1)
	scale=np.median(np.hypot(np.diff(x,axis=1),np.diff(y,axis=1)))
	n=int(np.ceil(radius_rsun/scale));a=np.arange(-n,n+1)
	kernel=(a[:,None]**2+a[None,:]**2)*scale**2<=radius_rsun**2
	return ndi.convolve(votes,kernel.astype(float),mode='constant')>=min_votes

def association(previous,current,dx,dy,quality,radius=2.):
	"""Advect previous masks with accepted source flow. Uncovered pixels are unknown."""
	y,x=np.indices(dx.shape);q=quality&np.isfinite(dx)&np.isfinite(dy)
	pts=np.column_stack(((x+dx)[q],(y+dy)[q]))
	if previous is None or not len(pts):return np.zeros(current.shape,bool),[]
	dist,ix=cKDTree(pts).query(np.column_stack((x.ravel(),y.ravel())))
	known=dist.reshape(current.shape)<=radius
	old=previous[q][ix].reshape(current.shape)
	new=np.zeros(current.shape,bool); rows=[]
	for k in range(1,int(current.max())+1):
		m=current==k; covered=m&known
		coverage=covered.sum()/m.sum()
		ids,counts=np.unique(old[covered&(old>0)],return_counts=True)
		overlap=counts.sum()/max(covered.sum(),1)
		status='unknown' if coverage<.5 else ('advected_overlap' if overlap>=.2 else 'new_candidate')
		if status=='new_candidate':new|=covered&(old==0)
		rows.append(dict(component=k,pixels=int(m.sum()),coverage=float(coverage),overlap=float(overlap),previous_components=';'.join(map(str,ids)),status=status))
	return new,rows
