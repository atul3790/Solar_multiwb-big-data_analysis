"""STEREO COR2 + HI1 moving structure frames (one difference panel + LAT/GOES).

Same products as make_moving_structure_frames_STEREO.py (contours, edge and
centroid velocity arrows, PFSS, EUVI disk, light curves, CSVs, MP4), but the
flow field is COR2 joined with HI1 on one helioprojective grid, and only one
difference panel is drawn (diff_panel='base' or 'running').

HI1 comes from Multiwb_datamine.get_STEREO_imgs(instruments=['HI1']):
Level-1 (local UKSSDC files) -> Helioviewer (no HI source) -> Level-0.5 (SSC).
Each instrument is differenced against itself and divided by its own noise
scale, so detect_floor and bright_contrast below are in noise (sigma) units.
"""
from pathlib import Path
from combined_frame_pipeline import make_combined_frames

#################### INPUT PARAMETER GUIDE ####################
# Edit only the INPUT block below. Switches use 1 (on) or 0 (off).
# Parameters shared with make_moving_structure_frames_STEREO.py mean the same
# there; only the new or changed ones are described here.
#
# diff_panel: 'running' or 'base'. The one difference image analysed and drawn.
# STEREO_root: Database root given to get_STEREO_imgs; COR2 lives in
#     <root>/STEREO_<sc>/COR2, HI1 in <root>/STEREO_<sc>/HI1. Shared with the
#     STEREO overlay maker (its img_folder).
# spacecraft_choice: 1 = A; 2 = B (B ended 2014).
#
# HI1
# hi_max_image_dt: Largest offset (s) between a COR2 slot and the HI1 sum used
#     with it. HI1 sums every 40 min, so 1500 s fills every 10-min slot.
# hi_lookback: Largest gap (s) to the previous HI1 sum (running difference).
# hi_max_gap: HI1 pairs longer than this (s) give differences but no HI1 flow.
# hi_star_filter_px: Median filter in native HI1 pixels that removes stars
#     before differencing; 0 or 1 disables it.
# hi_edge_px: Grid pixels trimmed from the HI1 field edge.
# seam_rsun: COR2/HI1 boundary radius (Rsun). blank = r_outer. COR2 inside,
#     HI1 outside (a grey dotted line marks it on the frames).
#
# Grid and COR2 mask
# npix: Pixels along the longer side of the combined grid (~105 x 80 Rsun
#     for STEREO-A in 2025, so 1024 gives ~0.1 Rsun per pixel).
# r_inner / r_outer: COR2 radial limits (Rsun); blank uses 2.5 / 15.
# spoke_pa / spoke_halfwidth / spoke_arrow_buffer_deg: COR2 support spoke,
#     applied to the COR2 zone only.
#
# Noise model and redraw
# noise_model: 'ring' divides each instrument's differences by its noise-versus-
#     radius profile (median over all frames of the robust sigma in rings of
#     ring_step_rsun), so every pixel is in local-noise units: faint outer HI1
#     structures are detected as readily as bright inner ones. 'global' uses one
#     noise scale per instrument (HI1 noise then varies ~80x across the field).
# ar_half_lon, ar_half_lat: AR rectangle (deg about the NOAA centre); seeds every
#     AR_seed_spacing_deg over it (cached as AR_field_lines_rooted.npz, retraced when
#     these change). trace_longARrooted_loops: 1 keeps closed loops with one footpoint
#     in the rectangle (large AR-rooted loops, tallest first); 0 needs both inside.
# show_velocity_vectors: 1 draws the edge and centroid velocity arrows and speed colour
#     bar; 0 shows contours only (same layout). Flow and vector CSVs are still saved.
# reuse_cache: blank builds a new motion cache (download, alignment, flow).
#     'latest' or a motion_cache folder skips all that and only redraws frames
#     (and switches that cache's noise_model, reversibly). Fast for trying
#     display or detection settings.
#
# Feature detection (noise units: 1 = one robust sigma; local sigma with
# noise_model='ring')
# detect_floor: Minimum absolute detection threshold, in sigma.
# bright_contrast: Minimum bright core minus surrounding ring, in sigma.
# Other detection, arrow, PFSS, AR, light-curve and plot settings: as in the
# STEREO flow launcher.

#################### INPUT ####################

#### Project, event and observations ####
basedir='/path/to/event_folder'  # Event folder: all relative paths below live inside it
sT='YYYY-MM-DDThh:mm:ss'  # Event start, UTC
eT='YYYY-MM-DDThh:mm:ss'  # Event end, UTC
diff_panel='base'                 # 'running' or 'base'
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
save_PFSS_maps=True     # True: save AR + full-Sun PFSS maps (on the disk image, or a blank Stonyhurst-grid disk) at the sT-eT midpoint in basedir as PFSS_*.png; False: skip (no redraw)
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
speed_bins='0,50,100,200,300,400,500,600,800'
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
		try:_so.pfss_context_maps((_T(sT)+(_T(eT)-_T(sT))/2).isot,'STEREO','EUVI' if EUVI_fold else None,EUVI_wave,img_folder=STEREO_root,spacecraft={1:'A',2:'B'}.get(spacecraft_choice,'A'),basedir=basedir,pfss_dir=pfss_dir,rss=rss,nrho=nrho,r_seed=r_seed,n_lon=n_lon,n_lat=n_lat,PFSS_dT=PFSS_dT or 6.,gong_cache=gong_cache,SRS_folder=SRS_file,closed_field_color=closed_field_color,NOAA_number=NOAA_number if focus_AR else None,ar_half_lon=ar_half_lon,ar_half_lat=ar_half_lat,AR_seed_spacing_deg=AR_seed_spacing_deg,trace_longARrooted_loops=bool(trace_longARrooted_loops),AR_r_seed=AR_r_seed,n_closed=n_closed,n_AR_open=None if n_AR_open in ('',None) or n_AR_open!=n_AR_open else int(n_AR_open),n_global_closed=n_global_closed,AR_closedB_col=AR_closedB_col,AR_openB_col=AR_openB_col)
		except Exception as exc:print('PFSS maps not saved:',exc)
	plot_style=dict(xlabel=xlabel,ylabel=ylabel,xticks=xticks,yticks=yticks,
		xlabel_size=xlabel_size,ylabel_size=ylabel_size,xtick_size=xtick_size,ytick_size=ytick_size,
		title=title,title_size=title_size,subtitle_size=subtitle_size,base_title=base_title,
		running_title=running_title,panel_title_size=panel_title_size,legend=legend,legend_loc=legend_loc,
		legend_size=legend_size,colorbar_label=colorbar_label,colorbar_label_size=colorbar_label_size,
		colorbar_tick_size=colorbar_tick_size,legend_labels=(legend_bright_label,legend_dark_label))
	spacecraft={1:'A',2:'B'}.get(spacecraft_choice)
	if spacecraft is None:raise ValueError('Invalid spacecraft_choice.')
	final_image_folder=Path(outdir)/(f'pfss_stereo{spacecraft}_COR2_HI1_motion_{diff_panel}_EUVI{EUVI_wave}_LC'+(f'_{output_tag}' if output_tag else ''))
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
		noise_model=noise_model,ring_step_rsun=ring_step_rsun,ring_strength=ring_strength,hi_flow_clean=bool(hi_flow_clean),reuse_cache=blank(reuse_cache),
		user_datacsv=user_datacsv,goes_folder=goes_folder,goes_download=goes_download,
		goes_pickle=blank(goes_pickle),goes_csv=blank(goes_csv),outdir=outdir,final_image_folder=final_image_folder,
		# Settings passed straight to track_coronagraph_flows.render_frames
		focus_AR=focus_AR,NOAA_number=NOAA_number,AR_time=AR_time or sT,SRS_file=blank(SRS_file),
		AR_lookup_max_age=AR_lookup_max_age,ar_seed_spacing_deg=AR_seed_spacing_deg,ar_long_loops=bool(trace_longARrooted_loops),n_AR_open=None if n_AR_open=='' else int(n_AR_open),AR_r_seed=AR_r_seed,
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
		speed_bins=[float(v) for v in speed_bins.split(',')],dpi=dpi,display_limit=display_limit,
		display_stretch=display_stretch,display_soft=display_soft,display_smooth_px=display_smooth_px,show_vectors=bool(show_velocity_vectors))
