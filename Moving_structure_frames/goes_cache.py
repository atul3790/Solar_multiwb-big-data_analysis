"""Select a trusted local GOES pickle, or populate the cache on demand."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

def covering_pickle(folder,start,end):
	"""Prefer the smallest excess coverage, using actual timestamps."""
	best=None
	for path in sorted(Path(folder).rglob('*.p')):
		try:
			with path.open('rb') as stream:data=pickle.load(stream)
			t=pd.to_datetime(data['Time'],utc=True,errors='coerce')
			f=np.asarray(data['Flux'],dtype=float)
			if f.ndim!=1 or len(t)!=len(f):continue
			t=t[(~pd.isna(t))&np.isfinite(f)&(f>0)]
			if len(t)<2:continue
			lo,hi=t.min(),t.max()
		except Exception:
			continue  # Unreadable or unrelated cache entry
		tolerance=pd.Timedelta(seconds=2)  # GOES 1-s samples need not hit exact endpoints
		if lo>start+tolerance or hi<end-tolerance:continue
		score=abs(lo-start)+abs(hi-end)
		if best is None or score<best[0]:best=score,path
	return best[1] if best else None

def resolve_goes_pickle(folder,sT,eT,download=True):
	"""Return a covering pickle; download through Multiwb_datamine if needed."""
	folder=Path(folder)
	start,end=pd.to_datetime([sT,eT],utc=True)
	if start>=end:raise ValueError('GOES requires sT before eT.')
	path=covering_pickle(folder,start,end)
	if path is None and download:
		from Multiwb_datamine import plot_goes_derivative
		folder.mkdir(parents=True,exist_ok=True)
		pad=pd.Timedelta(seconds=60)
		lo,hi=[t.tz_localize(None).isoformat() for t in (start-pad,end+pad)]
		plot_goes_derivative(lo,hi,savedata=True,outdir=str(folder))
		path=covering_pickle(folder,start,end)
	if path is None:raise FileNotFoundError(f'No GOES pickle with data spanning {sT}–{eT} UTC in {folder}.')
	return path
