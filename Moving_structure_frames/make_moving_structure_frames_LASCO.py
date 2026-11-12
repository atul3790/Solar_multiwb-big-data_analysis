"""LASCO C2/C3 moving-structure frames: contours, edge and centroid flow arrows, PFSS, disk and LAT/GOES light curves.

Detects bright and dark structures in quiet-base and running difference images
(in noise-sigma units with a ring noise profile), tracks their apparent motion
with TV-L1 optical flow, and draws them with projected PFSS field lines, an
optional disk image and the LAT/GOES light curves. Saves PNG frames, an MP4,
edge/centroid vector CSVs and a reusable motion cache (reuse_cache redraws only).
Edit only the INPUT block. Uses motion_frame_pipeline_V1.make_frames_V1.
"""
from pathlib import Path
from motion_frame_pipeline_V1 import make_frames_V1

#################### INPUT PARAMETER GUIDE ####################
# Edit only the INPUT block below. Switches use 1 (on) or 0 (off).
#
# Project, event and observations
# basedir: Absolute project root. Relative image, cache, light-curve and output paths below
#     are resolved here, regardless of where Python starts.
# sT: Start of the requested UTC event window, as YYYY-MM-DDThh:mm:ss.
# eT: End of the requested UTC event window. Use a time later than sT.
# LASCO_choice: 1 = C2; 2 = C3.
# LASCO_imgfolder: Blank uses LASCO/C2 or LASCO/C3. Otherwise enter the folder searched for intensity FITS and used for
#     missing-image retrieval.
# need_LASCO_calib: Request the preferred calibration retrieval route for missing LASCO
#     files. Does not recalibrate files already selected from disk.
# SDO_fold: Root of the SDO image database. Set blank to omit the solar disk overlay.
# SDO_instrument_choice: 1 = HMI; 2 = AIA.
# SDO_product: For HMI, choose magnetogram, continuum or dopplergram; for AIA, specify a
#     wavelength in angstroms, e.g. 193.
# download: 1 allows filling missing image bins and building missing scheduled PFSS
#     snapshots. GOES retrieval is controlled separately.
# helioviewer: Allow the downloader's Helioviewer fallback where supported. Motion analysis
#     still requires usable intensity FITS with WCS.
#
# Sampling and frame pairing
# dT: Target coronagraph sampling in seconds. Actual exposure times are retained; 600
#     requests one sample per 10 minutes.
# SDO_dT: Target cadence in seconds for filling gaps in the SDO cache. It does not set the
#     coronagraph frame rate.
# PFSS_dT: Spacing between UTC-anchored PFSS snapshots, in hours. blank uses the saved model
#     nearest the event midpoint.
# disk_max_minutes: Maximum separation between a frame and its nearest usable disk image, in
#     minutes. The frame renders without the disk if none is close enough.
# base_diff_DT: Quiet reference UTC, at or before sT. The selected exposure is subtracted
#     from every event image in the base-difference panel.
# max_image_dt: Largest allowed timing error in seconds when matching an exposure to a
#     cadence target or quiet reference. A target with no match is skipped; a missing quiet
#     reference prevents the run.
# reference_lookback: How far back in seconds from the first selected exposure to search for
#     its running-difference predecessor. A predecessor is required by this pipeline.
# max_gap_seconds: Largest actual exposure-to-exposure separation for estimating flow, in
#     seconds. Longer pairs can still yield a difference image but no flow arrows.
#
# PFSS model and cache
# pfss_needed: 1 loads or builds PFSS and plots field lines. 0 bypasses both steps.
# pfss_dir: Directory containing an existing PFSS model or its timestamped snapshots. New
#     snapshots are saved here.
# gong_cache: Local cache of GONG photospheric boundary maps used when building PFSS.
# rss: PFSS source-surface radius in solar radii, measured from Sun centre.
# nrho: Number of radial cells in a new PFSS model. Higher values increase radial resolution
#     and cost.
# r_seed: Radius in solar radii at which the global PFSS traces are seeded.
# n_lon: Number of longitude samples in the global PFSS seed grid.
# n_lat: Number of latitude samples in the global PFSS seed grid.
#
# NOAA active region
# focus_AR: 1 highlights cached closed traces selected near the NOAA region. 0
#     displays the global field without AR highlighting.
# NOAA_number: Full NOAA active-region number, for example 14100.
# AR_time: UTC epoch for selecting the NOAA region report; blank uses event start.
# SRS_file: Optional local NOAA Solar Region Summary file. blank lets the code locate the
#     report.
# AR_lookup_max_age: Largest allowed time offset, in hours, between the NOAA report and
#     AR_time.
# ar_half_lon, ar_half_lat: The AR rectangle, half-widths in degrees about the NOAA
#     centre (Stonyhurst); 12 means ±12°. Field lines are seeded over it, and a
#     closed loop is an AR loop if at least one footpoint lies inside it, so large
#     loops rooted in the AR are kept however far they reach (tallest first, up to
#     n_closed). Open AR lines: lower end inside the rectangle.
# AR_seed_spacing_deg: Seed spacing over the AR rectangle, degrees; ±12° at 1° gives
#     25 x 25 seeds. AR lines are cached per PFSS snapshot as AR_field_lines_rooted.npz
#     and retraced when the rectangle or spacing changes.
# trace_longARrooted_loops: 1 keeps closed loops with at least one footpoint in the
#     rectangle (large AR-rooted loops, tallest first); 0 needs both footpoints inside.
# AR_r_seed: Start radius of the AR traces in solar radii (low, for small closed loops).
#
# PFSS line display
# n_open: Maximum number of global open PFSS traces drawn. blank shows all; lower it to
#     reduce clutter.
# n_global_closed: Number of global closed PFSS traces drawn outside the highlighted set.
# n_closed: Maximum number of closed traces highlighted for the selected active region.
# closed_field_color: Matplotlib colour for ordinary global closed PFSS traces.
# AR_closedB_col: Matplotlib colour for selected closed active-region traces.
# AR_openB_col: Matplotlib colour for open active-region traces when the AR cache supplies
#     them.
#
# Image grid and instrument masks
# npix: Side length of the square analysis grid in pixels. Changes the physical scale of
#     pixel-based detection thresholds and arrows.
# r_inner: Inner usable radius from Sun centre in solar radii; masks the occulter and inner
#     edge. blank uses the detector default.
# r_outer: Outer usable radius from Sun centre in solar radii. blank uses the detector
#     default. Pixels beyond it are excluded.
# spoke_pa: Centre angle of the coronagraph support spoke in degrees: image top is 0°, right
#     is 90°, bottom is 180°.
# spoke_halfwidth: Angular half-width of the excluded spoke sector in degrees. 0 disables
#     the spoke mask.
# spoke_arrow_buffer_deg: Extra angular margin on either side of an enabled spoke mask, in
#     degrees, where arrows are withheld.
# mask_buffer_px: Pixel margin at the valid-image boundary to suppress edge artefacts.
#
# V1 image processing
# noise_model: 'ring' (default) divides base and running differences by one robust
#     noise scale and then by their own noise-versus-radius profile**ring_strength,
#     so faint outer structures are detected like bright inner ones. 'global' uses
#     the single scale only; 'none' keeps input image units (original behaviour).
# ring_step_rsun: Ring width (Rsun) for the noise profile.
# ring_strength: 1 = full local-noise units; 0 = global. 0.7 keeps bright inner features.
# show_velocity_vectors: 1 draws the edge and centroid velocity arrows and speed colour
#     bar; 0 shows contours only (same layout). Flow and vector CSVs are still saved.
# reuse_cache: blank builds a new V1 motion cache. 'latest' (newest V1 cache in
#     outdir/motion_cache) or a cache folder redraws only: no downloads, alignment
#     or flow; that cache's noise_model is switched (reversibly). Fast for tuning
#     detection and display.
# diff_panel: 'both' (base and running, as before), 'base' or 'running'.
# display_stretch: 'asinh' (faint structures enhanced, bright fronts not saturated) or 'linear'.
# display_soft: asinh is linear within +/- this many sigma, logarithmic beyond.
# display_limit: Grey-scale +/- limit in sigma: a number, 'median' (of per-frame
#     99th percentiles), 'frame' (per frame) or None (largest 99th percentile).
# display_smooth_px: Gaussian smoothing (grid pixels) of the displayed image only; 0 = off.
# Display settings never change detection.
#
# Feature detection (with noise_model 'ring' or 'global', floor and contrast are in sigma)
# detect_sigma: Multiplier on the robust spread of difference intensities (scaled median
#     absolute deviation). Larger values reject weaker features.
# detect_floor: Minimum absolute detection threshold in analysis-image intensity units. The
#     code takes the larger of this floor and the robust spread threshold.
# detect_min_area: Minimum size of a detected component, in analysis pixels.
# region_span_rsun: Minimum component span along its major axis, in solar radii. Helps
#     reject small features even when pixel resolution changes.
# bright_contrast: Minimum median bright-core intensity minus the median in its surrounding
#     ring, in analysis-image units. Raise it to reject low-contrast bright patches.
# bright_min_area: Minimum area for bright components, in analysis pixels.
# weak_bright_factor: Fraction of the normal threshold and contrast used for weak bright
#     candidates near supported flow. Smaller values admit fainter candidates.
# flow_support_radius_rsun: Neighborhood radius in solar radii for considering weak bright
#     candidates near supported flow.
#
# Motion arrows
# arrow_min_area: Minimum component area in analysis pixels to receive motion arrows.
# edge_spacing: Minimum separation in analysis pixels between plotted edge arrows.
# support_radius: Neighborhood radius in pixels for collecting valid flow samples near an
#     edge. The separate min_displacement setting controls the minimum shift.
# min_support: Minimum number of valid flow samples needed to draw an edge arrow.
#	e.g: min_support=5; Around a candidate point on the edge, at least 5 nearby pixels 
#   must have usable optical-flow estimates. With fewer than 5, the code skips that arrow.
# min_displacement: Minimum measured displacement in analysis pixels. The estimated shift
#    between the two exposures must be at least 0.1 pixel on the analysis grid.
#    Optical flow can estimate shifts smaller than one pixel. 
#    Raising this setting hides arrows for very small shifts.
# arrow_length_pixels: Fixed drawn contour arrow length in analysis pixels. Direction comes from
#     flow; speed is encoded by colour. Every arrow that passes those checks is drawn 14 
#     analysis pixels long. Change this to make arrows easier to see. The arrowhead shows 
#     direction, and the colour shows estimated speed.
# centroid_arrow_length_pixels: Drawn length at each accepted region centroid.
#     Bright arrows are solid; dark arrows are dotted. Set 0 to disable them.
#     Uses the same speed colours as the contour arrows.
# speed_bins: Increasing colour-band boundaries in km/s for projected pattern speed; fixed
#     across panels and frames.
#
# Light curves
# user_datacsv: LAT CSV with datetime and flux; optional columns specify flux errors, time
#     errors and upper limits.
# goes_folder: Directory searched for trusted GOES pickle files covering the event and used
#     to save fetched data.
# goes_download: 1 fetches GOES through Multiwb_datamine on a cache miss. 0 requires
#     a suitable cached file or explicit override.
# goes_pickle: Explicit trusted GOES pickle with Time and Flux entries. Takes precedence
#     over CSV and automatic lookup. blank leaves it unset.
# goes_csv: Explicit GOES CSV with datetime and flux columns, used when goes_pickle is blank.
#     If both are blank, the code searches goes_folder.
#
# Plot and output
# bright_border: Matplotlib colour for bright component contours.
# dark_border: Matplotlib colour for dark component contours.
# new_border: Matplotlib colour for contours marking new candidates.
# border_width: Contour stroke width in Matplotlib points. Controls appearance only.
# dpi: Output PNG dots per inch. Controls export resolution, independent of the analysis
#     grid.
# outdir: Run output root, including the generated motion cache. A relative path is resolved
#     under basedir.
# output_tag: Optional suffix for the automatically named output folder, e.g. "trial2".

#################### INPUT ####################

#### Project, event and observations ####
basedir='/path/to/event_folder'  # Event folder: all relative paths below live inside it
sT='YYYY-MM-DDThh:mm:ss'  # Event start, UTC
eT='YYYY-MM-DDThh:mm:ss'  # Event end, UTC
LASCO_choice=2 # LASCO_choice: 1 = C2; 2 = C3.
LASCO_imgfolder='LASCO/C3' # Blank follows LASCO_choice: LASCO/C3 for choice 2.
need_LASCO_calib=True
SDO_fold='SDO'
SDO_instrument_choice=1
SDO_product='magnetogram'
download=1
helioviewer=1

#### Sampling and frame pairing ####
dT=600
SDO_dT=7200
PFSS_dT=6.
save_PFSS_maps=True     # True: save AR + full-Sun PFSS maps (on the disk image, or a blank Stonyhurst-grid disk) at the sT-eT midpoint in basedir as PFSS_*.png; False: skip (no redraw)
disk_max_minutes=120.
base_diff_DT='YYYY-MM-DDThh:mm:ss'  # Quiet base-difference reference, UTC, at or before sT
max_image_dt=600
reference_lookback=3600
max_gap_seconds=1800

#### PFSS model and cache ####
pfss_needed=1
pfss_dir='PFSS'
gong_cache='solar_data/gong'
rss=2.5
nrho=35
r_seed=1.05
n_lon=36
n_lat=18

#### NOAA active region ####
focus_AR=1
NOAA_number=None  # Set the NOAA active-region number (e.g. 11402) when focus_AR is on
AR_time=sT # Blank uses sT.
SRS_file='NOAA_SRS'  # NOAA SRS reports folder shared by all launchers and overlay makers; downloads saved here
AR_lookup_max_age=36
ar_half_lon=12.          # AR rectangle half-width in longitude, deg
ar_half_lat=12.          # AR rectangle half-width in latitude, deg
AR_seed_spacing_deg=1.   # Seed spacing over the AR rectangle, deg
trace_longARrooted_loops=1 # 1: loops with one footpoint in the rectangle (large AR loops kept); 0: both footpoints inside
n_AR_open=5              # Max open AR field lines drawn (magenta), spread evenly; '' draws all
AR_r_seed=1.01 # AR trace start radius, Rsun

#### PFSS line display ####
n_open=30
n_global_closed=20
n_closed=30
closed_field_color='white'
pfss_linewidth=1.5 # Width in points for all PFSS lines.
AR_closedB_col='lime'
AR_openB_col='magenta'

#### Image grid and instrument masks ####
npix=512
r_inner=''
r_outer=28
spoke_pa=0. # C3 pylon centre (0 = north, 90 = west); measure it for your date, e.g. on a Helioviewer C3 image
spoke_halfwidth=0. # Half-width of the masked pylon sector, deg (about 6 for C3 once spoke_pa is set); 0 = no mask
spoke_arrow_buffer_deg=5. # Suppress arrows within ±11° total.
mask_buffer_px=5

#### V1 image processing ####
noise_model='ring'      # 'ring', 'global' or 'none' (input image units, as the original launcher)
ring_step_rsun=1.       # Ring width for the noise profile, Rsun
ring_strength=.7        # 1 = full local-noise units; 0 = global
reuse_cache=''          # '' new cache; 'latest' or a V1 cache folder to redraw only
diff_panel='both'       # 'both', 'base' or 'running'
display_stretch='asinh' # 'asinh' or 'linear'
display_soft=1.         # asinh linear range, sigma
display_limit=40        # Grey-scale +/- limit in sigma, or 'median', 'frame', None
display_smooth_px=1.    # Display-only Gaussian smoothing, grid pixels; 0 = off

#### Feature detection (sigma units with noise_model 'ring'/'global') ####
detect_sigma=2.         # original launcher: 2.5
detect_floor=1.         # original: 4 (input image units)
detect_min_area=18
region_span_rsun=2.6
base_region_span_rsun=2. # Minimum projected span (Rsun) for base difference.
bright_contrast=1.      # original: 8 (input image units)
detect_smooth_px=1.5    # Gaussian smoothing (grid px) before thresholding; 4 tested on COR2+HI1 for faint diffuse structures
bright_contrast_snr=0   # 1: bright_contrast is a unitless local SNR (region 95th pct - ring mean)/ring std (~2 faint, ~3 clear)
region_min_width_rsun='' # Min breadth (minor axis of the region's fitted box), Rsun; '' = no limit (COR2+HI1: 0.8)
region_max_aspect=''    # Max length/breadth of the box; drops thin lines; '' = no limit (COR2+HI1: 6)
bright_min_area=35
weak_bright_factor=.45
flow_support_radius_rsun=1.5

#### Motion arrows ####
arrow_min_area=100
edge_spacing=24.
support_radius=3.
min_support=5
min_displacement=.1
#### Plot labels and sizes (points) ####
xlabel=r'Solar X ($R_\odot$)'
ylabel=r'Solar Y ($R_\odot$)'
xticks=None # Solar radii; None keeps instrument defaults.
yticks=None # Solar radii; None keeps instrument defaults.
xlabel_size=18
ylabel_size=18
xtick_size=15
ytick_size=15
title=None # None keeps instrument, timestamp and cadence.
title_size=18
subtitle_size=15
base_title='Quiet-base difference'
running_title='Running difference'
panel_title_size=18
legend=True
legend_loc='lower left'
legend_size=xtick_size
colorbar_label='Projected Speed |v| (km/s)'
colorbar_label_size=18
colorbar_tick_size=15
legend_bright_label='Bright region'
legend_dark_label='Dark region'

#### Arrow lengths ####
arrow_length_pixels=20. # Contour arrows.
centroid_arrow_length_pixels=46. # Longer centroid arrows.
speed_bins='0,50,100,200,300,400,500,600,800' # speed bins needed for colorbar in km/s
show_velocity_vectors=1 # 1: draw edge and centroid velocity arrows and the speed colour bar; 0: contours only (vectors still saved in the CSVs)

#### Light curves ####
user_datacsv='LAT_fluxLC.csv'
goes_folder='GOES'
goes_download=1
goes_pickle=''
goes_csv=''

#### Plot and output ####
bright_border='#ffff00'
dark_border='#ff9300'
new_border='white'
border_width=1.65
dpi=90
outdir='LASCO_LAT_frames'
output_tag=''  # Optional suffix for the output folder name

################################################

if __name__=='__main__':
	_unset=[k for k in ('basedir','sT','eT','base_diff_DT') if k in globals() and ('YYYY' in str(globals()[k]) or str(globals()[k]).startswith('/path/to'))]
	if _unset:raise ValueError('Set these in the INPUT block first: '+', '.join(_unset))
	if save_PFSS_maps:  # AR + full-Sun PFSS maps at the sT-eT midpoint, saved in basedir
		import solar_overlay as _so;from astropy.time import Time as _T
		try:_so.pfss_context_maps((_T(sT)+(_T(eT)-_T(sT))/2).isot,'SDO',{1:'HMI',2:'AIA'}.get(SDO_instrument_choice) if SDO_fold else None,SDO_product,img_folder=SDO_fold or None,basedir=basedir,pfss_dir=pfss_dir,rss=rss,nrho=nrho,r_seed=r_seed,n_lon=n_lon,n_lat=n_lat,PFSS_dT=PFSS_dT or 6.,gong_cache=gong_cache,SRS_folder=SRS_file,closed_field_color=closed_field_color,NOAA_number=NOAA_number if focus_AR else None,ar_half_lon=ar_half_lon,ar_half_lat=ar_half_lat,AR_seed_spacing_deg=AR_seed_spacing_deg,trace_longARrooted_loops=bool(trace_longARrooted_loops),AR_r_seed=AR_r_seed,n_closed=n_closed,n_AR_open=None if n_AR_open in ('',None) or n_AR_open!=n_AR_open else int(n_AR_open),n_global_closed=n_global_closed,AR_closedB_col=AR_closedB_col,AR_openB_col=AR_openB_col)
		except Exception as exc:print('PFSS maps not saved:',exc)
	plot_style=dict(
		xlabel=xlabel,
		ylabel=ylabel,
		xticks=xticks,
		yticks=yticks,
		xlabel_size=xlabel_size,
		ylabel_size=ylabel_size,
		xtick_size=xtick_size,
		ytick_size=ytick_size,
		title=title,
		title_size=title_size,
		subtitle_size=subtitle_size,
		base_title=base_title,
		running_title=running_title,
		panel_title_size=panel_title_size,
		legend=legend,
		legend_loc=legend_loc,
		legend_size=legend_size,
		colorbar_label=colorbar_label,
		colorbar_label_size=colorbar_label_size,
		colorbar_tick_size=colorbar_tick_size,
		legend_labels=(legend_bright_label,legend_dark_label))
	LASCO={1:'C2',2:'C3'}.get(LASCO_choice)
	if LASCO is None:raise ValueError('Invalid LASCO_choice.')
	SDO_instrument={1:'HMI',2:'AIA'}.get(SDO_instrument_choice)
	if SDO_instrument is None:raise ValueError('Invalid SDO_instrument_choice.')
	LASCO_imgfolder=LASCO_imgfolder or str(Path('LASCO')/LASCO)
	final_image_folder=Path(outdir)/(f'pfss_las{LASCO}_motion_{SDO_instrument}_LC'+(f'_{output_tag}' if output_tag else ''))
	speed_bins=[float(value) for value in speed_bins.split(',')]
	AR_time=AR_time or sT
	SRS_file=None if SRS_file=='' else SRS_file
	n_open=None if n_open=='' else n_open
	r_inner=None if r_inner=='' else r_inner
	r_outer=None if r_outer=='' else r_outer
	goes_pickle=None if goes_pickle=='' else goes_pickle
	goes_csv=None if goes_csv=='' else goes_csv
	blank=lambda v:None if v=='' else v
	make_frames_V1('LASCO',
		sT=sT, eT=eT, dT=dT, base_diff_DT=base_diff_DT,
		max_image_dt=max_image_dt, reference_lookback=reference_lookback, max_gap_seconds=max_gap_seconds, input_folder=basedir,
		LASCO=LASCO, LASCO_imgfolder=LASCO_imgfolder, need_LASCO_calib=need_LASCO_calib, SDO_fold=SDO_fold,
		SDO_instrument=SDO_instrument, SDO_product=SDO_product, SDO_dT=SDO_dT, disk_max_minutes=disk_max_minutes,
		download=download, helioviewer=helioviewer, pfss_needed=pfss_needed, pfss_dir=pfss_dir,
		gong_cache=gong_cache, PFSS_dT=PFSS_dT, rss=rss, nrho=nrho,
		r_seed=r_seed, n_lon=n_lon, n_lat=n_lat, focus_AR=focus_AR,
		NOAA_number=NOAA_number, AR_time=AR_time, SRS_file=SRS_file, AR_lookup_max_age=AR_lookup_max_age,
		AR_seed_spacing_deg=AR_seed_spacing_deg, trace_longARrooted_loops=trace_longARrooted_loops, n_AR_open=None if n_AR_open=='' else int(n_AR_open), AR_r_seed=AR_r_seed,
		AR_closedB_col=AR_closedB_col, AR_openB_col=AR_openB_col, closed_field_color=closed_field_color, pfss_linewidth=pfss_linewidth, n_open=n_open,
		n_global_closed=n_global_closed, n_closed=n_closed, ar_half_lon=ar_half_lon, ar_half_lat=ar_half_lat,
		npix=npix, r_inner=r_inner, r_outer=r_outer,
		spoke_pa=spoke_pa, spoke_halfwidth=spoke_halfwidth, spoke_arrow_buffer_deg=spoke_arrow_buffer_deg, mask_buffer_px=mask_buffer_px,
		detect_sigma=detect_sigma, detect_floor=detect_floor, detect_min_area=detect_min_area, region_span_rsun=region_span_rsun, base_region_span_rsun=base_region_span_rsun,
		weak_bright_factor=weak_bright_factor, flow_support_radius_rsun=flow_support_radius_rsun, bright_contrast=bright_contrast,detect_smooth_px=detect_smooth_px,bright_contrast_snr=bool(bright_contrast_snr),region_min_width_rsun=float(region_min_width_rsun or 0.),region_max_aspect=None if region_max_aspect=='' else float(region_max_aspect), bright_min_area=bright_min_area,
		arrow_min_area=arrow_min_area, edge_spacing=edge_spacing, support_radius=support_radius, min_support=min_support,
		min_displacement=min_displacement, user_datacsv=user_datacsv, goes_folder=goes_folder, goes_download=goes_download,
		goes_pickle=goes_pickle, goes_csv=goes_csv, bright_border=bright_border, dark_border=dark_border,
		new_border=new_border, border_width=border_width, arrow_length_pixels=arrow_length_pixels,
		centroid_arrow_length_pixels=centroid_arrow_length_pixels, speed_bins=speed_bins, plot_style=plot_style,
		dpi=dpi, outdir=outdir, final_image_folder=final_image_folder,
		noise_model=noise_model, ring_step_rsun=ring_step_rsun, ring_strength=ring_strength,
		reuse_cache=blank(reuse_cache), diff_panel=diff_panel, display_stretch=display_stretch,
		display_soft=display_soft, display_limit=display_limit, display_smooth_px=display_smooth_px,
		show_vectors=bool(show_velocity_vectors))
