"""Compare the TV-L1 flow of two motion caches built with different flow_input.

Typical use: cache_a built with flow_input='intensity', cache_b a copy
recomputed with motion_inputs/combined_motion_inputs.recompute_flow(copy,
'base_difference'). Both caches must share the grid and frame tags.

Per frame, and per zone (cor2_zone, hi_zone) for COR2+HI1 caches ('all' for
single-instrument caches), the table holds for each cache:
  quality_frac   fraction of valid pixels with quality=True (forward-backward
                 error < 1.5 px, as in motion_inputs.flow);
  speed_med/p90  median and 90th percentile flow speed on quality pixels, in
                 px/frame and Rsun/h;
and, for both together, angle_diff_med_deg (median angle between the two flow
fields where both have quality) and hours_since_base (frame time minus base
time; HI1 sum time minus HI1 base time in the hi_zone). If regions.csv files of
both rendered runs are given, per-frame region counts, mean coverage and status
counts are added.

Importable: compare_flow_inputs, zone_stats. Saves under outdir:
flow_input_comparison.csv and flow_input_comparison.png. Inputs are not changed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from astropy.time import Time

########### INPUT ###########################################################
basedir='/path/to/event_folder/LASCO_LAT_frames'  # Folder holding motion_cache/; outputs are saved under it
outdir='flow_input_comparison'               # inside basedir
cache_a='motion_cache/<intensity cache>'       # intensity flow (relative to basedir)
cache_b='motion_cache/<base-difference copy>'  # base-difference flow
regions_a=''                                  # regions.csv of the cache_a render; '' skips
regions_b=''                                  # regions.csv of the cache_b render; '' skips
#############################################################################

def _scale(cache):
	"""Grid pixel size (Rsun) from coordinates.npz."""
	with np.load(Path(cache)/'coordinates.npz') as c:x,y=c['x_rsun'],c['y_rsun']
	return float(np.nanmedian(np.hypot(np.diff(x,axis=1),np.diff(y,axis=1))))

def zone_stats(d,zone,px_rsun,dt_h):
	"""Quality fraction and speed stats in one zone mask of a frame dict d."""
	v=zone&d['valid'];q=d['quality']&v;s=np.hypot(d['dx'],d['dy'])[q]
	r=dict(quality_frac=q.sum()/max(v.sum(),1),n_quality=int(q.sum()))
	for k,f in (('med',50),('p90',90)):
		r[f'speed_{k}_px']=float(np.percentile(s,f)) if s.size else np.nan
		r[f'speed_{k}_rsun_h']=r[f'speed_{k}_px']*px_rsun/dt_h if dt_h>0 else np.nan
	return r,q

def _regions(path):
	"""Per-frame region count, mean coverage and status counts from regions.csv."""
	r=pd.read_csv(path)
	g=r.groupby('datetime').agg(n_regions=('component','size'),coverage_mean=('coverage','mean'))
	st=r.pivot_table(index='datetime',columns='status',values='component',aggfunc='size',fill_value=0).add_prefix('status_')
	return g.join(st)

def compare_flow_inputs(cache_a,cache_b,outdir,regions_a=None,regions_b=None):
	"""Compare the flow of cache_a and cache_b; return the DataFrame.

	Inputs: two cache folders on the same grid; outdir for outputs; optional
	regions.csv paths of both renders. Saves outdir/flow_input_comparison.csv
	and .png.
	"""
	cache_a,cache_b,outdir=Path(cache_a),Path(cache_b),Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
	ta=json.loads((cache_a/'times.json').read_text());tb=json.loads((cache_b/'times.json').read_text())
	px=_scale(cache_a);rows=[]
	for tag in [t for t in ta if t in tb]:
		it=ta[tag]
		with np.load(cache_a/(tag+'.npz')) as z:a={k:z[k] for k in z.files}
		with np.load(cache_b/(tag+'.npz')) as z:b={k:z[k] for k in z.files}
		dt_h=(Time(it['datetime'])-Time(it['previous_time'])).sec/3600
		zones={n:a[n] for n in ('cor2_zone','hi_zone') if n in a} or {'all':np.ones(a['valid'].shape,bool)}
		for name,zone in zones.items():
			sa,qa=zone_stats(a,zone,px,dt_h);sb,qb=zone_stats(b,zone,px,dt_h);both=qa&qb
			ang=np.angle(np.exp(1j*(np.arctan2(a['dy'],a['dx'])-np.arctan2(b['dy'],b['dx'])))[both])
			if name=='hi_zone' and it.get('hi_time') and it.get('hi_base_time'):since=(Time(it['hi_time'])-Time(it['hi_base_time'])).sec/3600
			else:since=(Time(it['datetime'])-Time(it['base_time'])).sec/3600
			rows.append(dict(datetime=it['datetime'],zone=name,hours_since_base=since,
				**{k+'_a':v for k,v in sa.items()},**{k+'_b':v for k,v in sb.items()},
				n_both=int(both.sum()),angle_diff_med_deg=float(np.degrees(np.median(np.abs(ang)))) if ang.size else np.nan))
	df=pd.DataFrame(rows)
	if regions_a and regions_b and Path(regions_a).is_file() and Path(regions_b).is_file():
		reg=_regions(regions_a).add_suffix('_a').join(_regions(regions_b).add_suffix('_b'),how='outer').fillna(0)
		df=df.merge(reg,left_on='datetime',right_index=True,how='left')
	df.to_csv(outdir/'flow_input_comparison.csv',index=False)
	_plot(df,outdir/'flow_input_comparison.png',cache_a.name,cache_b.name)
	return df

def _plot(df,path,la,lb):
	"""Quality fraction, median speed and angle difference versus time since base."""
	import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
	fig,ax=plt.subplots(3,1,figsize=(8,9),sharex=True)
	for name,g in df.groupby('zone'):
		for suf,ls,lab in (('_a','-',la),('_b','--',lb)):
			ax[0].plot(g.hours_since_base,g['quality_frac'+suf],ls,marker='.',label=f'{name} {lab}')
			ax[1].plot(g.hours_since_base,g['speed_med_rsun_h'+suf],ls,marker='.',label=f'{name} {lab}')
		ax[2].plot(g.hours_since_base,g.angle_diff_med_deg,marker='.',label=name)
	ax[0].set_ylabel('quality fraction');ax[1].set_ylabel('median speed (Rsun/h)');ax[2].set_ylabel('median angle diff (deg)')
	ax[2].set_xlabel('hours since base');[a.legend(fontsize=7) for a in ax];fig.tight_layout();fig.savefig(path,dpi=110);plt.close(fig)

if __name__=='__main__':
	import os
	cwd=os.getcwd()
	os.chdir(basedir)
	try:
		df=compare_flow_inputs(cache_a,cache_b,outdir,regions_a or None,regions_b or None)
		print(df.describe().T[['mean','50%']].to_string())
	finally:
		os.chdir(cwd)
