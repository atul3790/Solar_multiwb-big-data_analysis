"""STEREO COR2 frames with the EUVI disk inside the occulter, a full-disk EUVI panel and a
STIX + GOES panel, plus an MP4 per (panel a, panel b) display pair.

Same layout and display options as make_AIA_LASCO_overlay_frames, whose make_frames does the
drawing: (a) COR2 with EUVI (intensity) filling the occulter, (b) the EUVI full disk, (c) STIX
quick-look bands and GOES XRS-B over the whole window in Earth light-arrival UT, with a dotted
line at each frame time shifted from STEREO to Earth by (Earth-Sun - STEREO-Sun distance)/c.
Each of (a) and (b) is 'rdiff' (running difference), 'bdiff' (base difference at base_time) or
'intensity'. COR2 and EUVI share the observer, so they are aligned by helioprojective angle.
COR2: total-brightness FITS in <stereo_root>/STEREO_<sc>/COR2/maps (Multiwb_datamine.
get_STEREO_imgs); only the full-resolution images are used (256-px beacon-type images are
dropped so running differences do not mix resolutions); DN are divided by EXPTIME.
EUVI: get_STEREO_imgs fills <stereo_root>/STEREO_<sc>/EUVI/<wave>/ at euvi_dT cadence (cached
files reused); the image nearest each COR2 time within euvi_maxdt is used, in DN/s.
Dependencies: numpy, matplotlib, astropy, sunpy, scipy, pandas, FFmpeg; Solar_data_mining on
PYTHONPATH. Importing this file defines the INPUT values and functions only; nothing runs.
Importable: cor_files, nearest (make_frames and fits_times come from make_AIA_LASCO_overlay_frames).
"""
import os,warnings
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from make_AIA_LASCO_overlay_frames import fits_times,make_frames
from make_SolO_overlay_frames import light_curves
from track_coronagraph_flows import write_mp4

# Parameter guide
# basedir    : event folder; every relative path below is inside it.
# outdir     : frame folder stem; frames and MP4 go to <outdir>_<a> (a==b) or <outdir>_<a>-<b>.
# stereo_root: STEREO folder (holds STEREO_<sc>/COR2/maps and STEREO_<sc>/EUVI/<wave>).
# spacecraft : 'A' or 'B'.
# euvi_wave  : EUVI wavelength [A]: 171, 195, 284 or 304.
# euvi_dT    : EUVI download cadence [s] for get_STEREO_imgs.
# euvi_maxdt : largest allowed COR2-EUVI time difference [s]; else no EUVI in that frame.
# sT,eT      : UTC window of COR2 images; with an 'rdiff' panel the first image is only the reference.
# panels     : list of (panel a COR2, panel b EUVI) display pairs, one movie each; each entry is
#              'rdiff' (running difference), 'bdiff' (base difference) or 'intensity'.
# base_time  : UTC of the base image for 'bdiff' (nearest COR2 image; EUVI matched to it);
#              None = first COR2 image in sT-eT.
# npix       : panel (a) grid size; half-width is the farthest COR2 pixel.
# diff_clip  : percentile setting the grey limits (|difference|, or intensity 100-p..p for log).
# disk_rsun  : half-width of the full-disk EUVI panel (b), solar radii.
# inner_r    : radius [R_sun] of the EUVI image shown inside the COR2 occulter (EUVI field ~1.6).
# occ_rsun   : COR2 pixels inside this radius [R_sun] are blanked (occulter pedestal and
#              diffraction ring in the FITS; science field starts at ~2.5).
# lc_panel   : 1 adds the STIX + GOES panel; 0 omits it.
# stix_dir   : STIX quick-look folder (get_STIX_LC; CSV for sT-eT reused).
# stix_bands : STIX bands, from '4-10 keV','10-15 keV','15-25 keV','25-50 keV','50-84 keV'.
# goes_folder: GOES pickle cache (plot_goes_derivative); downloaded if no file covers sT-eT.
########### INPUT ###########################################################
basedir='/path/to/event_folder'
outdir='STEREO_EUVI_frames/COR2_EUVI171'
stereo_root='STEREO'
spacecraft='A'
euvi_wave=171
euvi_dT=900
euvi_maxdt=600
sT='2025-05-30T07:00:00'
eT='2025-05-31T00:00:00'
panels=[('rdiff','intensity'),('rdiff','rdiff')]
base_time=None
npix=1000
diff_clip=99.
disk_rsun=1.3
inner_r=1.55
occ_rsun=2.5
lc_panel=1
stix_dir='SolO/STIX'
stix_bands=['4-10 keV','15-25 keV','25-50 keV']
goes_folder='GOES'
#############################################################################


def cor_files(folder,sT,eT):
	"""[(Time, path)] of the full-resolution coronagraph FITS in folder inside [sT,eT], time-sorted."""
	rows=fits_times(folder,sT,eT)
	size=lambda p:max(int(h.header.get('NAXIS1',0)) for h in fits.open(p))
	n=max(size(p) for t,p in rows)
	return [(t,p) for t,p in rows if size(p)==n]

def nearest(times,rows,maxdt):
	"""Path in rows=[(Time,path)] nearest to each time, or None beyond maxdt s."""
	if not rows:return [None]*len(times)
	tr=Time([r[0] for r in rows]);out=[]
	for t in times:
		k=int(np.argmin(np.abs((tr-t).to_value(u.s))))
		out.append(rows[k][1] if abs((tr[k]-t).to_value(u.s))<=maxdt else None)
	return out

if __name__=='__main__':
	cwd=os.getcwd()
	os.chdir(basedir)
	try:
		warnings.filterwarnings('ignore')
		from Multiwb_datamine import get_STEREO_imgs
		sc=f'{stereo_root}/STEREO_{spacecraft.upper()}'
		get_STEREO_imgs(sT,eT,euvi_dT,spacecraft,['EUVI'],[[euvi_wave]],outdir=stereo_root,helioviewer=False)
		crows=cor_files(f'{sc}/COR2/maps',sT,eT);cfiles=[p for t,p in crows]
		efiles=nearest([t for t,p in crows],fits_times(f'{sc}/EUVI/{euvi_wave}',Time(sT)-euvi_maxdt*u.s,Time(eT)+euvi_maxdt*u.s),euvi_maxdt)
		print(f'{len(cfiles)} COR2 images; {sum(e is not None for e in efiles)} with EUVI {euvi_wave} within {euvi_maxdt} s')
		lc=light_curves(sT,eT,stix_dir,stix_bands,goes_folder) if lc_panel else None
		base=0 if base_time is None else int(np.argmin([abs((t-Time(base_time)).to_value(u.s)) for t,p in crows]))
		for pa,pb in panels:
			out=f'{outdir}_{pa}'+('' if pa==pb else f'-{pb}')
			paths=make_frames(cfiles,efiles,out,pa,pb,base,npix,diff_clip,disk_rsun,lc,(sT,eT),inner_r,occ_rsun)
			mp4=Path(out)/f'{Path(out).name}.mp4'
			write_mp4(paths,mp4);print(f'{len(paths)} frames, MP4: {mp4}')
	finally:
		os.chdir(cwd)
