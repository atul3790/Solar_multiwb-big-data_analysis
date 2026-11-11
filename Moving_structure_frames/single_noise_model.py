"""Noise normalisation for single-coronagraph motion caches (LASCO C2/C3, COR1/COR2).

The same image processing as the COR2+HI1 combined cache, for caches written by
motion_inputs.make_motion_cache (one instrument, no zones):
  'none'   differences in input image units (as built; the original launchers).
  'global' differences divided by one robust noise scale (median over frames of
           1.4826*MAD of the running difference), so thresholds are in sigma.
  'ring'   'global', then each difference (base, running) divided by its own
           noise-versus-radius profile**strength (time median of ring MAD).
The state is kept in noise_scales.json in the cache, so a change is reversible
and repeating the same model does nothing. Detection and flow are unaffected
apart from the units of the difference images.

Importable: apply_noise_model, noise_scale, ring_noise_profile, robust_sigma.
"""
from pathlib import Path
import json,warnings
import numpy as np

DIFFS=('running_difference','base_difference')
STATE='noise_scales.json'

def robust_sigma(values):
	values=values[np.isfinite(values)]
	if not values.size:return np.nan
	return 1.4826*float(np.median(np.abs(values-np.median(values))))

def _tags(out):return list(json.loads((Path(out)/'times.json').read_text()))

def _radius(out):
	with np.load(Path(out)/'coordinates.npz') as c:return np.hypot(c['x_rsun'],c['y_rsun'])

def noise_scale(out,key='running_difference'):
	"""Median over frames of the robust sigma of `key` in the valid area."""
	vals=[]
	for tag in _tags(out):
		with np.load(Path(out)/(tag+'.npz')) as z:vals.append(robust_sigma(z[key][z['valid']]))
	s=float(np.nanmedian(vals))
	if not np.isfinite(s) or s<=0:raise ValueError(f'No usable noise scale in {out}.')
	return s

def ring_noise_profile(out,step=1.,key='running_difference'):
	"""Ring centres (Rsun) and the time median of each ring's robust sigma.

	A time median, so a CME filling a ring in some frames does not raise that
	ring's noise. Rings with fewer than 200 valid pixels are skipped.
	"""
	R=_radius(out);edges=np.arange(0,np.ceil(R.max()/step)*step+step,step);centres=(edges[:-1]+edges[1:])/2
	rows=[]
	for tag in _tags(out):
		with np.load(Path(out)/(tag+'.npz')) as z:
			d=z[key];v=z['valid']
			rows.append([robust_sigma(d[s]) if (s:=v&(R>=a)&(R<b)).sum()>=200 else np.nan for a,b in zip(edges[:-1],edges[1:])])
	with warnings.catch_warnings():
		warnings.simplefilter('ignore',RuntimeWarning)
		sig=np.nanmedian(np.array(rows,dtype=float),axis=0)
	ok=np.isfinite(sig)&(sig>0)
	return centres[ok].tolist(),sig[ok].tolist()

def _noise_map(R,valid,profile):
	r,sig=profile;noise=np.ones(R.shape)
	if len(r):noise[valid]=np.interp(R[valid],r,sig)
	return noise

def _rewrite(out,func):
	for tag in _tags(out):
		p=Path(out)/(tag+'.npz')
		with np.load(p) as z:d={k:z[k] for k in z.files}
		for name in DIFFS:d[name]=func(name,d[name],d['valid']).astype('float32')
		np.savez_compressed(p,**d)

def apply_noise_model(out,noise_model='ring',step=1.,strength=.7):
	"""Set the cache's normalisation in place: 'none', 'global' or 'ring'.

	Inputs: out cache folder; step ring width (Rsun); strength 0-1 power on the
	ring profile (1 = full local-noise units). Returns out. Saves the state and
	profiles in out/noise_scales.json.
	"""
	if noise_model not in ('none','global','ring'):raise ValueError("noise_model must be 'none', 'global' or 'ring'.")
	out=Path(out);f=out/STATE
	info=json.loads(f.read_text()) if f.is_file() else dict(noise_model='none')
	cur=info['noise_model']
	if cur==noise_model and (cur!='ring' or (info.get('ring_step_rsun')==step and info.get('ring_strength')==strength)):
		print(f'Noise model already {noise_model}: {out}',flush=True);return out
	R=_radius(out)
	if cur=='ring':  # back to global units
		prof,pw=info['ring_profiles'],float(info['ring_strength'])
		_rewrite(out,lambda n,d,v:d*_noise_map(R,v,prof[n])**pw)
		for k in ('ring_profiles','ring_step_rsun','ring_strength'):info.pop(k,None)
		cur=info['noise_model']='global'
	if cur=='global' and noise_model=='none':
		s=info['scale'];_rewrite(out,lambda n,d,v:d*s);cur=info['noise_model']='none'
	if cur=='none' and noise_model!='none':
		s=noise_scale(out);_rewrite(out,lambda n,d,v:d/s)
		info.update(scale=s,scale_note='input units per sigma (running difference)');cur=info['noise_model']='global'
		print(f'Noise scale: {s:.4g} input units per sigma',flush=True)
	if noise_model=='ring':
		prof={n:ring_noise_profile(out,step,n) for n in DIFFS}
		_rewrite(out,lambda n,d,v:d/_noise_map(R,v,prof[n])**float(strength))
		info.update(noise_model='ring',ring_step_rsun=step,ring_strength=strength,ring_profiles=prof)
		for n,(r,sig) in prof.items():
			if r:print(f'Ring noise {n}: {sig[0]:.3g} at {r[0]:.1f} Rsun -> {sig[-1]:.3g} at {r[-1]:.1f} Rsun (sigma units)',flush=True)
	f.write_text(json.dumps(info,indent=2))
	print(f'Noise model set to {noise_model}: {out}',flush=True)
	return out
