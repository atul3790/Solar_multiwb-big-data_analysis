"""Solar Orbiter Metis moving-structure frames: contours, edge and centroid flow arrows, PFSS, EUI disk and LAT/GOES light curves.

Same analysis as the STEREO launcher, from Solar Orbiter: detects bright and dark
structures in quiet-base and running difference Metis images (noise-sigma units,
ring noise profile), tracks their apparent motion with TV-L1 optical flow, and
draws them with PFSS field lines projected from Solar Orbiter's position, an EUI
disk image from the same viewpoint and the LAT/GOES light curves. Saves PNG
frames, an MP4, edge/centroid vector CSVs and a reusable motion cache.
Images: Metis Level 2 (MSB) and EUI Level 2 from ESA SOAR through
Multiwb_datamine.get_Metis_imgs / get_SolO_imgs; bins without public Level 2
get low-latency LL02 quick-looks (a separate product family, never mixed with L2).
Edit only the INPUT block. Uses motion_frame_pipeline_V1.make_frames_V1('SOLO').
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
# metis_product: 1 = VL total brightness (tB); 2 = VL polarized brightness (pB);
#     3 = UV H I Lyman-alpha. Sets the detector (VLD or UVD) and the product folder.
# metis_fold: Metis product folder <root>/<product>; the flow uses its maps/ subfolder
#     (primary-HDU FITS, NaN outside the field) and downloads go to
#     <root>/<product>/{raw,raw_LL02,maps} (get_Metis_imgs). Product folder names:
#     vl-tb, vl-pb, uv.
# EUI_fold: EUI product folder <root>/<product>, e.g. 'SolO/EUI/fsi174'. Blank omits the
#     disk overlay.
# EUI_wave: EUI Full Sun Imager channel in angstroms: 174 or 304.
# download: 1 allows filling missing image bins and building missing scheduled PFSS
#     snapshots. GOES retrieval is controlled separately.
#     Metis science data are public only at Level 2, released with a long delay
#     (complete to mid-2025 as of Nov 2026); bins without L2 get low-latency LL02.
#
# Sampling and frame pairing
# dT: Target coronagraph sampling in seconds. Actual exposure times are retained; 600
#     requests one sample per 10 minutes.
# EUI_dT: Target cadence in seconds for filling gaps in the EUI cache. It does not set the
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
# SRS_file: NOAA Solar Region Summary reports: a folder (the shared NOAA_SRS, used by all
#     launchers and overlay makers) or one report file. It is read first; missing reports
#     within AR_lookup_max_age are downloaded into that folder. Blank downloads into the
#     run's output folder.
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
# AR_loop_reach_deg: Closed AR loops are drawn only if both ends lie within this distance
#     (degrees on the Sun) of the NOAA centre. Keeps AR-quiet Sun and AR-AR connectors and
#     drops longer PFSS loops (e.g. to the polar caps). '' = no limit. Selection only:
#     cached traces are reused.
# AR_r_seed: Start radius of the AR traces in solar radii (low, for small closed loops).
#
# PFSS line display
# n_open: Maximum number of global open PFSS traces drawn. blank shows all; lower it to
#     reduce clutter.
# n_global_closed: Number of global closed PFSS traces drawn outside the highlighted set.
# n_closed: Maximum number of closed traces highlighted for the selected active region.
# n_AR_open: Maximum number of open AR field lines drawn (AR_openB_col), spread evenly over
#     those traced. '' draws all.
# closed_field_color: Matplotlib colour for ordinary global closed PFSS traces.
# AR_closedB_col: Matplotlib colour for selected closed active-region traces.
# AR_openB_col: Matplotlib colour for open active-region traces when the AR cache supplies
#     them.
# pfss_linewidth: Line width in points for all PFSS lines (global and AR).
#
# Image grid and instrument masks
# npix: Side length of the square analysis grid in pixels. Changes the physical scale of
#     pixel-based detection thresholds and arrows.
# r_inner: Inner usable radius from Sun centre in solar radii; masks the occulter and inner
#     edge. blank uses the Metis header field (INN_FOV, +5%); the field in solar radii
#     changes with Solar Orbiter's distance (5-11 Rsun at 0.85 AU, 3-6 Rsun at 0.3 AU).
# r_outer: Outer usable radius from Sun centre in solar radii. blank uses OUT_FOV -3%.
# spoke_pa: Centre angle of an excluded sector in degrees: image top is 0°, right is 90°,
#     bottom is 180°. Metis has no support spoke; keep spoke_halfwidth=0.
# spoke_halfwidth: Angular half-width of the excluded sector in degrees. 0 disables it.
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
# flow_input: images entering TV-L1 flow. 'intensity' (default, as the original
#     launcher) or 'base_difference' (frame - base, same base for both frames of a
#     pair; removes the static instrument level, radial gradient and F-corona).
#     A reused cache must have the same flow_input (recompute_flow on a copy).
# flow_bin: TV-L1 runs on flow_bin x flow_bin pixel blocks, then is scaled back to the
#     grid; detection and contours keep the full grid. 'auto' sets it from the grid
#     pixel size and the frame spacing so a 1500 km/s structure moves ~10 flow
#     pixels per frame (C2 at npix=512: 6; C3: 1; COR2 at 15 min: 3). An integer
#     overrides it; 1 = grid pixels (the earlier behaviour).
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
# base_region_span_rsun: Minimum projected span (Rsun) of a region in the base-difference
#     panel; region_span_rsun applies to the running-difference panel. None uses
#     region_span_rsun.
# bright_contrast: Minimum median bright-core intensity minus the median in its surrounding
#     ring, in analysis-image units. Raise it to reject low-contrast bright patches.
# detect_smooth_px: Gaussian smoothing (grid pixels) of the difference image before
#     thresholding. 1.5 keeps compact features; about 4 brings out faint diffuse structures
#     (then lower detect_floor).
# bright_contrast_snr: 1: bright_contrast is a unitless local SNR, (region 95th percentile -
#     ring mean) / ring std (about 2 faint, 3 clear). 0: bright_contrast is the median
#     core-minus-ring difference in image (sigma) units.
# bright_min_area: Minimum area for bright components, in analysis pixels.
# region_min_width_rsun: Minimum breadth (minor axis of the region's fitted box), Rsun. '' =
#     no limit (COR2+HI1 uses 0.8).
# region_max_aspect: Maximum length/breadth of the region's fitted box; drops thin lines
#     such as HI1 artefacts. '' = no limit (COR2+HI1 uses 6).
# weak_bright_factor: Fraction of the normal threshold and contrast used for weak bright
#     candidates near supported flow. Smaller values admit fainter candidates.
# flow_support_radius_rsun: Neighbourhood radius in solar radii for considering weak bright
#     candidates near supported flow.
#
# Motion arrows
# arrow_min_area: Minimum component area in analysis pixels to receive motion arrows.
# edge_spacing: Minimum separation in analysis pixels between plotted edge arrows.
# support_radius: Neighbourhood radius in pixels for collecting valid flow samples near an
#     edge.
# min_support: Minimum number of valid flow samples needed to draw an edge arrow.
# min_displacement: Minimum measured displacement in analysis pixels over the actual
#     exposure pair.
# centroid_arrow_length_pixels: Drawn length at a region centroid; 0 disables these arrows.
#     Bright centroid arrows are solid and dark centroid arrows are dotted.
# arrow_length_pixels: Fixed drawn contour arrow length in analysis pixels. Direction comes from
#     flow; speed is encoded by colour.
# colorbar_style: 'continuous' colour bar over speed_limits, or 'discrete' bands from
#     speed_bins. One colour axis for all panels and frames.
# speed_limits: continuous only. 'auto' = 0 to the fastest edge or centroid vector in
#     sT-eT, rounded up (1221 -> 1500 km/s); or 'vmin,vmax' in km/s.
# speed_bins: discrete only. 'auto' = bands 0,50,100,200,300,400,500,600,800,1000,1200,
#     1500,2000,... up to the fastest vector; or increasing km/s boundaries.
#
# Plot labels, titles and sizes (points)
# xlabel: x-axis label of the image panels (Matplotlib mathtext allowed).
# ylabel: y-axis label of the image panels.
# xticks: x tick values in solar radii, e.g. [-20,0,20]. None picks round values inside the
#     field (LASCO C2 -5..5, C3 -20..20).
# yticks: y tick values in solar radii; None as for xticks.
# xlabel_size: Font size (points) of the x-axis label.
# ylabel_size: Font size (points) of the y-axis label.
# xtick_size: Font size (points) of the x tick labels.
# ytick_size: Font size (points) of the y tick labels.
# title: Figure title. None writes the instrument, frame time and pair duration.
# title_size: Font size (points) of the figure title.
# subtitle_size: Font size (points) of the two explanatory lines under the title.
# base_title: Title of the base-difference panel.
# running_title: Title of the running-difference panel.
# panel_title_size: Font size (points) of the panel titles.
# legend: True draws the bright/dark region legend in the first panel.
# legend_loc: Matplotlib legend location, e.g. 'lower left'.
# legend_size: Font size (points) of the legend text.
# colorbar_label: Label of the speed colour bar.
# colorbar_label_size: Font size (points) of the colour-bar label.
# colorbar_tick_size: Font size (points) of the colour-bar tick labels.
# legend_bright_label: Legend text for bright-region contours.
# legend_dark_label: Legend text for dark-region contours.
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
# STIX_folder: Solar Orbiter STIX quick-look light curve folder (Multiwb_datamine.get_STIX_LC;
#     a CSV for sT-eT there is reused). Overplotted in red on its own axis. Blank = no STIX.
# stix_band: STIX band drawn: '4-10 keV' (matches GOES XRS-B 1-8 A), '10-15 keV',
#     '15-25 keV', '25-50 keV' or '50-84 keV'. Times are shifted to Earth light-arrival.
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
metis_product=1
metis_fold='SolO/Metis/vl-tb'
EUI_fold='SolO/EUI/fsi174'
EUI_wave=174
download=1

#### Sampling and frame pairing ####
dT=1800
EUI_dT=1800
PFSS_dT=6.
disk_max_minutes=30.
base_diff_DT='YYYY-MM-DDThh:mm:ss'  # Quiet base-difference reference, UTC, at or before sT
max_image_dt=900
reference_lookback=3600
max_gap_seconds=3600

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
AR_time=sT
SRS_file='NOAA_SRS'  # NOAA SRS reports folder shared by all launchers and overlay makers; downloads saved here
AR_lookup_max_age=36
ar_half_lon=12.          # AR rectangle half-width in longitude, deg
ar_half_lat=12.          # AR rectangle half-width in latitude, deg
AR_seed_spacing_deg=1.   # Seed spacing over the AR rectangle, deg
trace_longARrooted_loops=1 # 1: loops with one footpoint in the rectangle (large AR loops kept); 0: both footpoints inside
AR_loop_reach_deg=50.   # Closed AR loops need both ends within this distance (deg on the Sun) of the NOAA centre; '' = no limit
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
r_outer=''
spoke_pa=0.
spoke_halfwidth=0. # Metis has no support spoke
spoke_arrow_buffer_deg=5.
mask_buffer_px=5

#### V1 image processing ####
noise_model='ring'      # 'ring', 'global' or 'none' (input image units, as the original launcher)
ring_step_rsun=1.       # Ring width for the noise profile, Rsun
ring_strength=.7        # 1 = full local-noise units; 0 = global
reuse_cache=''          # '' new cache; 'latest' or a V1 cache folder to redraw only
flow_input='base_difference'  # TV-L1 input: 'intensity' or 'base_difference'; intensity gave <30 km/s on 2025-05-30 (radial gradient dominates), base_difference ~50-350 km/s
flow_bin='auto'         # TV-L1 pixel blocks: 'auto' (from pixel size and cadence) or an integer, 1 = grid pixels
diff_panel='both'       # 'both', 'base' or 'running'
display_stretch='asinh' # 'asinh' or 'linear'
display_soft=1.         # asinh linear range, sigma
display_limit=40        # Grey-scale +/- limit in sigma, or 'median', 'frame', None
display_smooth_px=1.    # Display-only Gaussian smoothing, grid pixels; 0 = off

#### Feature detection (sigma units with noise_model 'ring'/'global') ####
detect_sigma=2.         # original launcher: 2.5
detect_floor=1.         # original: 4 (input image units)
detect_min_area=18
region_span_rsun=1.2    # Metis field is ~5.5 Rsun wide at 0.85 AU (COR2: 2.6)
base_region_span_rsun=1. # Minimum projected span (Rsun) for base difference.
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
arrow_length_pixels=20.
centroid_arrow_length_pixels=46.
colorbar_style='continuous' # Speed colour bar: 'continuous' or 'discrete'
speed_limits='auto'      # continuous: 'auto' (0 to the fastest vector in sT-eT, rounded up) or 'vmin,vmax' in km/s, e.g. '0,1000'
speed_bins='auto'        # discrete: 'auto' (bands 0,50,100,200,...,5000 up to the fastest vector) or km/s boundaries, e.g. '0,50,100,200,300,400,500,600,800'
show_velocity_vectors=1 # 1: draw edge and centroid velocity arrows and the speed colour bar; 0: contours only (vectors still saved in the CSVs)

#### Light curves ####
user_datacsv='LAT_fluxLC.csv'
goes_folder='GOES'
goes_download=1
goes_pickle=''
goes_csv=''
STIX_folder='SolO/STIX'
stix_band='4-10 keV'

#### Plot and output ####
bright_border='#ffff00'
dark_border='#ff9300'
new_border='white'
border_width=1.65
dpi=90
outdir='SolO_LAT_frames'
output_tag=''  # Optional suffix for the output folder name

################################################

if __name__=='__main__':
	_unset=[k for k in ('basedir','sT','eT','base_diff_DT') if k in globals() and ('YYYY' in str(globals()[k]) or str(globals()[k]).startswith('/path/to'))]
	if _unset:raise ValueError('Set these in the INPUT block first: '+', '.join(_unset))
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
	instrument,product={1:('VLD','vl-tb'),2:('VLD','vl-pb'),3:('UVD','uv')}.get(metis_product,(None,None))
	if instrument is None:raise ValueError('metis_product must be 1, 2 or 3.')
	if Path(metis_fold).name!=product:print(f'Note: metis_fold {metis_fold} is not named {product}; downloads use that folder name as the product.')
	final_image_folder=Path(outdir)/(f'pfss_solo_Metis-{product}_motion_EUI{EUI_wave}_LC'+(f'_{output_tag}' if output_tag else ''))
	speed_bins='auto' if str(speed_bins).strip().lower()=='auto' else [float(value) for value in speed_bins.split(',')]
	speed_limits='auto' if str(speed_limits).strip().lower()=='auto' else [float(v) for v in speed_limits.split(',')]
	AR_time=AR_time or sT
	SRS_file=None if SRS_file=='' else SRS_file
	n_open=None if n_open=='' else n_open
	r_inner=None if r_inner=='' else r_inner
	r_outer=None if r_outer=='' else r_outer
	goes_pickle=None if goes_pickle=='' else goes_pickle
	goes_csv=None if goes_csv=='' else goes_csv
	blank=lambda v:None if v=='' else v
	make_frames_V1('SOLO',
		sT=sT, eT=eT, dT=dT, base_diff_DT=base_diff_DT,
		max_image_dt=max_image_dt, reference_lookback=reference_lookback, max_gap_seconds=max_gap_seconds, input_folder=basedir,
		spacecraft='SOLO', instrument=instrument, STEREO_imgfolder=str(Path(metis_fold)/'maps'), polar=None,
		EUVI_fold=EUI_fold or None, EUVI_wave=EUI_wave, EUVI_dT=EUI_dT, disk_max_minutes=disk_max_minutes,
		download=download, helioviewer=0, pfss_needed=pfss_needed, pfss_dir=pfss_dir,
		gong_cache=gong_cache, PFSS_dT=PFSS_dT, rss=rss, nrho=nrho,
		r_seed=r_seed, n_lon=n_lon, n_lat=n_lat, focus_AR=focus_AR,
		NOAA_number=NOAA_number, AR_time=AR_time, SRS_file=SRS_file, AR_lookup_max_age=AR_lookup_max_age,
		AR_seed_spacing_deg=AR_seed_spacing_deg, trace_longARrooted_loops=trace_longARrooted_loops, AR_loop_reach_deg=None if AR_loop_reach_deg=='' else float(AR_loop_reach_deg), n_AR_open=None if n_AR_open=='' else int(n_AR_open), AR_r_seed=AR_r_seed,
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
		centroid_arrow_length_pixels=centroid_arrow_length_pixels, speed_bins=speed_bins, colorbar_style=colorbar_style, speed_limits=speed_limits, plot_style=plot_style,
		dpi=dpi, outdir=outdir, final_image_folder=final_image_folder,
		noise_model=noise_model, ring_step_rsun=ring_step_rsun, ring_strength=ring_strength,
		reuse_cache=blank(reuse_cache), flow_input=flow_input, flow_bin=flow_bin, diff_panel=diff_panel, display_stretch=display_stretch,
		display_soft=display_soft, display_limit=display_limit, display_smooth_px=display_smooth_px,
		show_vectors=bool(show_velocity_vectors),
		STIX_folder=STIX_folder or None, stix_band=stix_band)
