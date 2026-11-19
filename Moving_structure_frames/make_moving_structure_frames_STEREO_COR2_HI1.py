"""STEREO COR2 + HI1 moving structure frames (one difference panel + LAT/GOES).

Same products as make_moving_structure_frames_STEREO.py (contours, edge and
centroid velocity arrows, PFSS, EUVI disk, light curves, CSVs, MP4), but the
flow field is COR2 joined with HI1 on one helioprojective grid, and only one
difference panel is drawn (diff_panel='base' or 'running'). diff_panel='both' builds the
cache once and saves base and running frames in two folders (output_tag + '_bdiff' / '_rdiff').

HI1 comes from Multiwb_datamine.get_STEREO_imgs(instruments=['HI1']):
Level-1 (local UKSSDC files) -> Helioviewer (no HI source) -> Level-0.5 (SSC).
COR2 uses one product family, the first of Level-1, Helioviewer, Level-0.5 covering
more than half of the slots; helioviewer=0 excludes Helioviewer.
Each instrument is differenced against itself and divided by its own noise
scale, so detect_floor and bright_contrast below are in noise (sigma) units.
"""
from pathlib import Path
from combined_frame_pipeline import make_combined_frames

#################### INPUT PARAMETER GUIDE ####################
# Edit only the INPUT block below. Switches use 1 (on) or 0 (off). Detection
# thresholds are in noise (sigma) units: 1 = one robust sigma (local sigma with
# noise_model='ring').
#
# Project, event and observations
# basedir: Absolute project root. Relative image, cache, light-curve and output paths below
#     are resolved here, regardless of where Python starts.
# sT: Start of the requested UTC event window, as YYYY-MM-DDThh:mm:ss.
# eT: End of the requested UTC event window. Use a time later than sT.
# diff_panel: 'running' or 'base': the one difference image analysed and drawn. 'both' runs
#     both from one motion cache and saves them in separate folders, the output_tag
#     getting '_bdiff' (base) and '_rdiff' (running).
# spacecraft_choice: 1 = STEREO-A; 2 = STEREO-B (B ended in 2014).
# STEREO_root: Database root given to get_STEREO_imgs; COR2 lives in <root>/STEREO_<sc>/COR2
#     and HI1 in <root>/STEREO_<sc>/HI1. Shared with the STEREO overlay maker (its
#     img_folder).
# EUVI_fold: Root of the EUVI image database. Set blank to omit the disk overlay.
# EUVI_wave: EUVI channel in angstroms: 171, 195, 284 or 304.
# download: 1 allows filling missing image bins and building missing scheduled PFSS
#     snapshots. GOES retrieval is controlled separately.
# helioviewer: 1 allows Helioviewer COR2 images. Order: Level-1, then Helioviewer, then
#     Level-0.5; the first that covers more than half of the cadence slots is used (sources
#     are never mixed). 0 never fetches, converts or uses Helioviewer images, even if they
#     are already on disk. HI1 has no Helioviewer source.
#
# Sampling and frame pairing
# dT: Target COR2 sampling in seconds (one output frame per slot). Actual exposure times are
#     kept for differencing and speeds.
# EUVI_dT: Target cadence in seconds for filling gaps in the EUVI cache. It does not set the
#     coronagraph frame rate.
# PFSS_dT: Spacing between UTC-anchored PFSS snapshots, in hours. blank uses the saved model
#     nearest the event midpoint.
# save_PFSS_maps: PFSS context maps at the sT-eT midpoint, saved in basedir as PFSS_*.png,
#     on the disk image (nearest cached one first) or a blank Stonyhurst-grid disk with the
#     AR rectangle. True or 'both', 'full' (full Sun only), 'AR' (AR only), or False (skip,
#     e.g. on reruns).
# disk_max_minutes: Maximum separation between a frame and its nearest usable disk image, in
#     minutes. The frame renders without the disk if none is close enough.
# base_diff_DT: Quiet reference UTC, at or before sT. The selected exposure is subtracted
#     from every event image in the base-difference panel.
# max_image_dt: Largest allowed offset (s) between a COR2 slot and the exposure used for it.
# reference_lookback: How far back (s) from the first selected COR2 exposure to look for its
#     predecessor (running difference).
# max_gap_seconds: COR2 pairs longer than this (s) give differences but no flow or arrows.
#
# HI1
# hi_max_image_dt: Largest offset (s) between a COR2 slot and the HI1 sum used with it. HI1
#     sums every 40 min, so 1500 s fills every 10-min slot; one HI1 pair can repeat over
#     several COR2 slots.
# hi_lookback: Largest gap (s) to the previous HI1 sum (HI1 running difference).
# hi_max_gap: HI1 pairs longer than this (s) give differences but no HI1 flow. HI1
#     displacements are rescaled from the HI1 pair to the COR2 pair duration.
# hi_star_filter_px: Median filter (native HI1 pixels) that removes stars before
#     differencing; 0 or 1 disables it.
# hi_edge_px: Grid pixels trimmed from the HI1 field edge.
# seam_rsun: COR2/HI1 boundary radius (Rsun); blank = r_outer. COR2 inside, HI1 outside (a
#     grey dotted line marks it on the frames).
#
# Noise model and redraw
# noise_model: 'ring' divides each instrument's differences by its noise-versus-radius
#     profile (median over all frames of the robust sigma in rings of ring_step_rsun) to the
#     power ring_strength, so faint outer HI1 structures are detected as readily as bright
#     inner ones. 'global' uses one noise scale per instrument (HI1 noise then varies ~80x
#     across the field).
# ring_step_rsun: Ring width (Rsun) for the noise profile.
# ring_strength: Power on the ring profile: 1 = full local-noise units, 0 = global. 0.7
#     keeps inner-HI1 bright features and the faint outer arcs.
# reuse_cache: Blank builds a new motion cache (download, alignment, flow). 'latest' (newest
#     <A|B>_COR2_HI1_* cache in outdir/motion_cache) or a cache folder only redraws frames
#     and switches that cache's noise_model (reversibly). Fast for trying display or
#     detection settings.
# flow_input: Images entering TV-L1 flow. 'intensity' (default) or 'base_difference': COR2
#     frame - COR2 base, and for HI1 the cleaned sum - the cleaned base sum (HI1 pixels lose
#     flow quality when there is no HI1 base sum). A reused cache must have the same
#     flow_input (recompute_flow on a copy). There is no flow_bin here: TV-L1 runs on the
#     grid pixels.
# hi_flow_clean: 1: HI1 flow from sums with the F-corona (10th percentile over the sums) and
#     the star field (star-aligned median of +-3 sums) removed. Raw HI1 flow tracks the
#     sunward star drift (~1 px per 40 min). 0: flow on raw HI1 sums.
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
# n_AR_open: Maximum number of open AR field lines drawn (AR_openB_col), spread evenly over
#     those traced. '' draws all.
# AR_r_seed: Start radius of the AR traces in solar radii (low, for small closed loops).
#
# PFSS line display
# n_open: Maximum number of global open PFSS traces drawn. blank shows all; lower it to
#     reduce clutter.
# n_global_closed: Number of global closed PFSS traces drawn outside the highlighted set.
# n_closed: Maximum number of closed traces highlighted for the selected active region.
# closed_field_color: Matplotlib colour for ordinary global closed PFSS traces.
# pfss_linewidth: Line width in points for all PFSS lines (global and AR).
# AR_closedB_col: Matplotlib colour for selected closed active-region traces.
# AR_openB_col: Matplotlib colour for open active-region traces when the AR cache supplies
#     them.
#
# Combined grid and COR2 mask
# npix: Pixels along the longer side of the combined grid (~105 x 80 Rsun for STEREO-A in
#     2025, so 1024 gives ~0.1 Rsun per pixel).
# r_inner: Inner COR2 radius (Rsun); blank uses 2.5.
# r_outer: Outer COR2 radius (Rsun); blank uses 15.
# spoke_pa: Centre angle of a COR2 support spoke in degrees (image top 0, right 90). Applied
#     to the COR2 zone only.
# spoke_halfwidth: Half-width (deg) of the excluded COR2 spoke sector; 0 disables it (COR2-A
#     shows no pylon).
# spoke_arrow_buffer_deg: Extra margin (deg) either side of an enabled COR2 spoke mask where
#     arrows are withheld.
# mask_buffer_px: Pixel margin at the valid-image boundary to suppress edge artefacts.
#
# Feature detection (sigma units)
# detect_sigma: Multiplier on the robust spread (scaled MAD) of the difference values, in
#     noise (sigma) units. Larger values reject weaker features.
# detect_floor: Minimum absolute detection threshold, in sigma. The larger of this and
#     detect_sigma x spread is used.
# detect_min_area: Minimum size of a detected component, in analysis pixels.
# region_span_rsun: Minimum component span along its major axis, in solar radii. Helps
#     reject small features even when pixel resolution changes.
# base_region_span_rsun: Minimum projected span (Rsun) of a region in the base-difference
#     panel; region_span_rsun applies to the running-difference panel. None uses
#     region_span_rsun.
# detect_smooth_px: Gaussian smoothing (grid pixels) of the difference image before
#     thresholding. 1.5 keeps compact features; about 4 brings out faint diffuse structures
#     (then lower detect_floor).
# bright_contrast: Minimum bright-core contrast: with bright_contrast_snr=1 a unitless local
#     SNR (about 2 faint, 3 clear); with 0 the median core-minus-ring difference in sigma.
# bright_contrast_snr: 1: bright_contrast is a unitless local SNR, (region 95th percentile -
#     ring mean) / ring std (about 2 faint, 3 clear). 0: bright_contrast is the median
#     core-minus-ring difference in image (sigma) units.
# region_min_width_rsun: Minimum breadth (minor axis of the region's fitted box), Rsun. '' =
#     no limit (COR2+HI1 uses 0.8).
# region_max_aspect: Maximum length/breadth of the region's fitted box; drops thin lines
#     such as HI1 artefacts. '' = no limit (COR2+HI1 uses 6).
# bright_min_area: Minimum area for bright components, in analysis pixels.
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
#
# Plot labels and sizes (points)
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
# Arrow lengths
# arrow_length_pixels: Fixed drawn contour arrow length in analysis pixels. Direction comes from
#     flow; speed is encoded by colour.
# centroid_arrow_length_pixels: Drawn length at a region centroid; 0 disables these arrows.
#     Bright centroid arrows are solid and dark centroid arrows are dotted.
# colorbar_style: 'continuous' colour bar over speed_limits, or 'discrete' bands from
#     speed_bins. One colour axis for all panels and frames.
# speed_limits: continuous only. 'auto' = 0 to the fastest edge or centroid vector in
#     sT-eT, rounded up (1221 -> 1500 km/s); or 'vmin,vmax' in km/s.
# speed_bins: discrete only. 'auto' = bands 0,50,100,200,300,400,500,600,800,1000,1200,
#     1500,2000,... up to the fastest vector; or increasing km/s boundaries.
# show_velocity_vectors: 1 draws the edge and centroid velocity arrows and speed colour
#     bar; 0 shows contours only (same layout). Flow and vector CSVs are still saved.
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
# display_stretch: 'asinh' (faint structures enhanced, bright fronts not saturated) or 'linear'.
# display_soft: asinh is linear within +/- this many sigma, logarithmic beyond.
# display_limit: Grey-scale +/- limit in sigma: a number, 'median' (of per-frame
#     99th percentiles), 'frame' (per frame) or None (largest 99th percentile).
# display_smooth_px: Gaussian smoothing (grid pixels) of the displayed image only; 0 = off.
#     Detection and arrows always use the unsmoothed differences.
# outdir: Run output root, including the generated motion cache. A relative path is resolved
#     under basedir.
# output_tag: Optional suffix for the output folder, which is named
#     pfss_stereo<sc>_COR2_HI1_motion_<diff_panel>_EUVI<wave>_LC_<output_tag>. With
#     diff_panel='both': ..._motion_base_..._<output_tag>_bdiff and
#     ..._motion_running_..._<output_tag>_rdiff.

#################### INPUT ####################

#### Project, event and observations ####
basedir='/path/to/event_folder'  # Event folder: all relative paths below live inside it
sT='YYYY-MM-DDThh:mm:ss'  # Event start, UTC
eT='YYYY-MM-DDThh:mm:ss'  # Event end, UTC
diff_panel='base'                 # 'running', 'base' or 'both' (two folders)
spacecraft_choice=1
STEREO_root='STEREO'
EUVI_fold='STEREO/STEREO_A/EUVI'
EUVI_wave=195
download=1
helioviewer=1

#### Sampling and frame pairing ####
dT=600
EUVI_dT=7200
PFSS_dT=6.
save_PFSS_maps=True     # PFSS maps at the sT-eT midpoint in basedir (PFSS_*.png), on the disk image or a blank Stonyhurst-grid disk: True/'both', 'full' (full Sun only), 'AR' (AR only), False (skip, no redraw)
disk_max_minutes=120.
base_diff_DT='YYYY-MM-DDThh:mm:ss'  # Quiet base-difference reference, UTC, at or before sT
max_image_dt=600
reference_lookback=3600
max_gap_seconds=1800

#### HI1 ####
hi_max_image_dt=1500
hi_lookback=3600
hi_max_gap=5400
hi_star_filter_px=5
hi_edge_px=8
seam_rsun=''

#### Noise model and redraw ####
noise_model='ring'      # 'ring' (local noise units) or 'global'
ring_step_rsun=3.       # Ring width for the noise profile, Rsun
ring_strength=.7        # 1 = full local-noise units; 0 = global. 0.7 keeps inner-HI1 bright
                        # features and the faint outer arcs (tested on 2025-05-31)
reuse_cache=''          # '' new cache; 'latest' or a cache folder to redraw only
flow_input='intensity'  # TV-L1 input: 'intensity' or 'base_difference'
hi_flow_clean=1         # 1: HI1 flow from images with F-corona and star field removed
                        # (raw HI1 flow tracks the sunward star drift, ~1 px / 40 min)

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
AR_loop_reach_deg=50.   # Closed AR loops need both ends within this distance (deg on the Sun) of the NOAA centre: keeps AR-quiet Sun and AR-AR connectors, drops longer PFSS loops (e.g. to the polar caps); '' = no limit
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

#### Combined grid and COR2 mask ####
npix=1024
r_inner=''
r_outer=14
spoke_pa=217.
spoke_halfwidth=6.
spoke_arrow_buffer_deg=5.
mask_buffer_px=5

#### Feature detection (sigma units) ####
detect_sigma=2.         # was 2.5; tuned with ring_strength=0.7
detect_floor=.4         # was 1; detect_smooth_px=4 lowers the noise level, so the floor must drop
detect_min_area=18
region_span_rsun=2.6
base_region_span_rsun=2. # Minimum projected span (Rsun) for base difference.
detect_smooth_px=4.     # Gaussian smoothing (grid px) before thresholding; 1.5 = old value; 4 brings out faint diffuse structures
bright_contrast=2.      # bright_contrast_snr=1: unitless local SNR (region 95th pct - ring mean)/ring std; ~2 faint, ~3 clear
bright_contrast_snr=1   # 1: bright_contrast is the local SNR above; 0: old median difference (noise units)
region_min_width_rsun=.8 # Min breadth (minor axis of the region's fitted box), Rsun; '' = no limit
region_max_aspect=6.    # Max length/breadth of the box; drops thin lines such as HI1 artifacts; '' = no limit
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
xticks=None # Solar radii; None keeps defaults.
yticks=None # Solar radii; None keeps defaults.
xlabel_size=18
ylabel_size=18
xtick_size=15
ytick_size=15
title=None # None keeps instrument, timestamps and cadence.
title_size=18
subtitle_size=15
base_title='Quiet-base difference (COR2 + HI1)'
running_title='Running difference (COR2 + HI1)'
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

#### Plot and output ####
bright_border='#ffff00'
dark_border='#ff9300'
new_border='white'
border_width=1.65
dpi=90
# Display only; detection and arrows always use the unsmoothed, linear differences.
display_stretch='asinh'  # 'asinh' (faint structures enhanced, bright fronts not saturated) or 'linear'
display_soft=1.          # asinh: linear within +/- this many sigma, logarithmic beyond
display_limit=40         # Grey-scale +/- limit in sigma: a number, 'median' (of per-frame 99th
                         # percentiles), 'frame' (each frame its own) or None (largest frame).
                         # For display_stretch='linear', 'median' or 5-16 are good choices.
display_smooth_px=1.5    # Gaussian smoothing (grid pixels) of the displayed image; 0 = off
outdir='STEREO_LAT_frames'
output_tag=''  # Optional suffix for the output folder name

################################################

if __name__=='__main__':
	_unset=[k for k in ('basedir','sT','eT','base_diff_DT') if k in globals() and ('YYYY' in str(globals()[k]) or str(globals()[k]).startswith('/path/to'))]
	if _unset:raise ValueError('Set these in the INPUT block first: '+', '.join(_unset))
	if save_PFSS_maps:  # AR + full-Sun PFSS maps at the sT-eT midpoint, saved in basedir
		import solar_overlay as _so;from astropy.time import Time as _T
		try:_so.pfss_context_maps((_T(sT)+(_T(eT)-_T(sT))/2).isot,'STEREO-'+{1:'A',2:'B'}.get(spacecraft_choice,'A'),'EUVI' if EUVI_fold else None,EUVI_wave,img_folder=STEREO_root,basedir=basedir,pfss_dir=pfss_dir,rss=rss,nrho=nrho,r_seed=r_seed,n_lon=n_lon,n_lat=n_lat,PFSS_dT=PFSS_dT or 6.,gong_cache=gong_cache,SRS_folder=SRS_file,closed_field_color=closed_field_color,NOAA_number=NOAA_number if focus_AR else None,ar_half_lon=ar_half_lon,ar_half_lat=ar_half_lat,AR_seed_spacing_deg=AR_seed_spacing_deg,trace_longARrooted_loops=bool(trace_longARrooted_loops),AR_r_seed=AR_r_seed,n_closed=n_closed,n_AR_open=None if n_AR_open in ('',None) or n_AR_open!=n_AR_open else int(n_AR_open),n_global_closed=n_global_closed,AR_closedB_col=AR_closedB_col,AR_openB_col=AR_openB_col,maps='both' if save_PFSS_maps is True else save_PFSS_maps,max_image_dt=float(disk_max_minutes or 60)*60,AR_loop_reach_deg=None if AR_loop_reach_deg=='' else float(AR_loop_reach_deg))
		except Exception as exc:print('PFSS maps not saved:',exc)
	plot_style=dict(xlabel=xlabel,ylabel=ylabel,xticks=xticks,yticks=yticks,
		xlabel_size=xlabel_size,ylabel_size=ylabel_size,xtick_size=xtick_size,ytick_size=ytick_size,
		title=title,title_size=title_size,subtitle_size=subtitle_size,base_title=base_title,
		running_title=running_title,panel_title_size=panel_title_size,legend=legend,legend_loc=legend_loc,
		legend_size=legend_size,colorbar_label=colorbar_label,colorbar_label_size=colorbar_label_size,
		colorbar_tick_size=colorbar_tick_size,legend_labels=(legend_bright_label,legend_dark_label))
	spacecraft={1:'A',2:'B'}.get(spacecraft_choice)
	if spacecraft is None:raise ValueError('Invalid spacecraft_choice.')
	name=lambda p,tag:Path(outdir)/(f'pfss_stereo{spacecraft}_COR2_HI1_motion_{p}_EUVI{EUVI_wave}_LC'+(f'_{tag}' if tag else ''))
	if str(diff_panel).lower()=='both':  # one folder per panel: <output_tag>_bdiff, <output_tag>_rdiff
		final_image_folder={p:name(p,(f'{output_tag}_' if output_tag else '')+s) for p,s in (('base','bdiff'),('running','rdiff'))}
	else:final_image_folder=name(diff_panel,output_tag)
	blank=lambda v:None if v=='' else v
	make_combined_frames(diff_panel=diff_panel,STEREO_root=STEREO_root,spacecraft=spacecraft,
		sT=sT,eT=eT,dT=dT,base_diff_DT=base_diff_DT,max_image_dt=max_image_dt,
		reference_lookback=reference_lookback,max_gap_seconds=max_gap_seconds,input_folder=basedir,
		download=download,helioviewer=helioviewer,
		hi_max_image_dt=hi_max_image_dt,hi_lookback=hi_lookback,hi_max_gap=hi_max_gap,
		hi_star_filter_px=hi_star_filter_px,hi_edge_px=hi_edge_px,seam_rsun=blank(seam_rsun),
		npix=npix,r_inner=blank(r_inner),r_outer=blank(r_outer),spoke_pa=spoke_pa,
		spoke_halfwidth=spoke_halfwidth,spoke_arrow_buffer_deg=spoke_arrow_buffer_deg,
		EUVI_fold=blank(EUVI_fold),EUVI_wave=EUVI_wave,EUVI_dT=EUVI_dT,disk_max_minutes=disk_max_minutes,
		pfss_needed=pfss_needed,pfss_dir=pfss_dir,gong_cache=gong_cache,PFSS_dT=PFSS_dT,
		rss=rss,nrho=nrho,r_seed=r_seed,n_lon=n_lon,n_lat=n_lat,
		noise_model=noise_model,ring_step_rsun=ring_step_rsun,ring_strength=ring_strength,hi_flow_clean=bool(hi_flow_clean),reuse_cache=blank(reuse_cache),flow_input=flow_input,
		user_datacsv=user_datacsv,goes_folder=goes_folder,goes_download=goes_download,
		goes_pickle=blank(goes_pickle),goes_csv=blank(goes_csv),outdir=outdir,final_image_folder=final_image_folder,
		# Settings passed straight to track_coronagraph_flows.render_frames
		focus_AR=focus_AR,NOAA_number=NOAA_number,AR_time=AR_time or sT,SRS_file=blank(SRS_file),
		AR_lookup_max_age=AR_lookup_max_age,ar_seed_spacing_deg=AR_seed_spacing_deg,ar_long_loops=bool(trace_longARrooted_loops),ar_loop_reach_deg=None if AR_loop_reach_deg=='' else float(AR_loop_reach_deg),n_AR_open=None if n_AR_open=='' else int(n_AR_open),AR_r_seed=AR_r_seed,
		AR_closedB_col=AR_closedB_col,AR_openB_col=AR_openB_col,closed_field_color=closed_field_color,
		pfss_linewidth=pfss_linewidth,plot_style=plot_style,n_open=blank(n_open),
		n_global_closed=n_global_closed,n_closed=n_closed,ar_half_lon=ar_half_lon,ar_half_lat=ar_half_lat,
		ar_half_box_arcsec=None,mask_buffer_px=mask_buffer_px,
		detect_sigma=detect_sigma,detect_floor=detect_floor,detect_min_area=detect_min_area,
		region_span_rsun=region_span_rsun,base_region_span_rsun=base_region_span_rsun,
		weak_bright_factor=weak_bright_factor,flow_support_radius_rsun=flow_support_radius_rsun,
		bright_contrast=bright_contrast,detect_smooth_px=detect_smooth_px,bright_contrast_snr=bool(bright_contrast_snr),region_min_width_rsun=float(region_min_width_rsun or 0.),region_max_aspect=None if region_max_aspect=='' else float(region_max_aspect),bright_min_area=bright_min_area,arrow_min_area=arrow_min_area,
		edge_spacing=edge_spacing,support_radius=support_radius,min_support=min_support,
		min_displacement=min_displacement,bright_border=bright_border,dark_border=dark_border,
		new_border=new_border,border_width=border_width,arrow_length_pixels=arrow_length_pixels,
		centroid_arrow_length_pixels=centroid_arrow_length_pixels,
		speed_bins='auto' if str(speed_bins).strip().lower()=='auto' else [float(v) for v in speed_bins.split(',')],colorbar_style=colorbar_style,speed_limits='auto' if str(speed_limits).strip().lower()=='auto' else [float(v) for v in speed_limits.split(',')],dpi=dpi,display_limit=display_limit,
		display_stretch=display_stretch,display_soft=display_soft,display_smooth_px=display_smooth_px,show_vectors=bool(show_velocity_vectors))
