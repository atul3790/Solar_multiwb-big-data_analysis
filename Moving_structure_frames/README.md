# Moving_structure_frames

Frame and movie makers for following coronal mass ejections and other moving structures through white-light coronagraph and heliospheric-imager data, with projected PFSS magnetic field lines and Fermi-LAT / GOES light curves in a lower panel.

Two families of runnable scripts share one set of modules:

- **Moving-structure (flow) frames** detect connected bright and dark structures in base-difference and running-difference images, measure their in-plane drift with TV-L1 optical flow, and draw contour and centroid drift arrows. The result is a diagnostic movie of how structures form, flow out and merge.
- **Overlay frames** come in two kinds:
  - PFSS overlays: difference images from LASCO, STEREO or PSP/WISPR with PFSS field lines projected from that spacecraft's viewpoint, plus LAT/GOES light curves.
  - Coronagraph + EUV overlays: a coronagraph with an EUV image filling its occulter, beside a full-disk or magnified EUV panel. Below them is a STIX + GOES panel covering the whole window, with a line marking each frame's time. The pairs are Solar Orbiter Metis + EUI, SOHO/LASCO + SDO/AIA and STEREO COR2 + EUVI.

The drift arrows are an image-plane proxy for pattern speed. Projection effects and detection uncertainties remain, and they are not plasma velocities.

Image download and preparation come from the companion repository **[Solar_data_mining](https://github.com/atul3790/Solar_data_mining)**: `Multiwb_datamine.py`, `datamine_calib_helper.py`, `solar_overlay.py` and `solar_overlay_helper.py`. That repository must be on your `PYTHONPATH`.

## Contents

- [Installation](#installation)
- [Event folder layout](#event-folder-layout)
- [Runnable scripts](#runnable-scripts)
- [How a flow run works](#how-a-flow-run-works)
- [Modules](#modules)
- [Coronagraph + EUV overlay makers: functions](#coronagraph--euv-overlay-makers-functions)
- [Key settings](#key-settings)
- [Outputs](#outputs)
- [Example](#example)
- [Notes and limitations](#notes-and-limitations)

## Installation

```bash
git clone https://github.com/atul3790/Solar_data_mining.git
git clone https://github.com/atul3790/Solar_multiwb-big-data_analysis.git
export PYTHONPATH=$PWD/Solar_data_mining:$PWD/Solar_multiwb-big-data_analysis/Moving_structure_frames:$PYTHONPATH
python -m pip install -r Solar_multiwb-big-data_analysis/Moving_structure_frames/requirements.txt
```

Set `basedir` in each launcher to your own event folder; the published scripts carry the placeholder `/path/to/event_folder`.

The scripts are modules, not an installed package. Keep every `.py` file of this folder together, because the launchers import their sibling modules by name. MP4 writing needs the `ffmpeg` executable on your `PATH`.

## Event folder layout

Each event study uses one `basedir`. Every relative path in the launchers' INPUT blocks resolves inside it, and the overlay makers change into it first, so any script can be run from anywhere.

```
<basedir>/                       one folder per event
  LAT_fluxLC.csv                 Fermi-LAT light curve (user supplied)
  GOES/                          shared GOES cache: any pickle spanning the window is reused
  NOAA_SRS/                      NOAA Solar Region Summaries + AR_location.json
  PFSS/<YYYYMMDD_HHMMSS>/        PFSS snapshots at UTC multiples of PFSS_dT (00, 06, ... UT)
  solar_data/gong/               GONG synoptic maps used to build PFSS
  LASCO/C3/  SDO/HMI/...         image databases (raw/, maps/, maps_HV/)
  LASCO/C2/CDAW/                 CDAW C2 PNGs (use_CDAW_pngs=1 only)
  STEREO/STEREO_A/{COR2,HI1,EUVI/195}/
  PSP_WISPR/
  LASCO_LAT_frames/              frames, MP4s, CSVs and motion_cache/ for LASCO runs
  STEREO_LAT_frames/             same for STEREO runs
  SolO/Metis/<product>/{raw,raw_LL02,maps}/  Solar Orbiter Metis (vl-tb, vl-pb, uv); the flow reads maps/
  SolO/EUI/<product>/raw/        Solar Orbiter EUI Level 2 (fsi174, fsi304)
  SolO/STIX/                     STIX quick-look light-curve CSVs (raw/ holds the daily SOAR files)
  SolO_LAT_frames/               same for Solar Orbiter Metis runs
  SolO/Metis_EUI_frames_<metis_mode>-<eui_mode>/  make_SolO_overlay_frames output
  SDO/AIA/<wave>_synoptic/       JSOC AIA synoptic FITS (1k, 2-min) fetched by make_AIA_LASCO_overlay_frames
  LASCO_AIA_frames/<stem>_<a>[-<b>]/  make_AIA_LASCO_overlay_frames output
  STEREO_EUVI_frames/<stem>_<a>[-<b>]/  make_STEREO_EUVI_overlay_frames output
  PSP_WISPR_LAT_frames/
```

GOES, NOAA SRS, PFSS and GONG files are downloaded once per event and reused by every script. Image folders hold only the files they need, and missing cadence bins are downloaded on demand. STEREO settings may name the STEREO root or an instrument folder inside it, because downloads always go to `<root>/STEREO_<sc>/<instrument>`.

## Runnable scripts

Edit only the INPUT block of a script, then run it with `python script.py` or `%run script.py`. Each INPUT block is preceded by a commented guide that explains every setting.

| Script | What it makes |
|---|---|
| `make_moving_structure_frames_LASCO.py` | LASCO C2 or C3 flow frames: base and running difference panels, bright/dark contours, edge and centroid drift arrows, PFSS, an SDO disk image (HMI or AIA) and LAT/GOES curves. C2 can also run on the CDAW PNGs (`use_CDAW_pngs=1`). |
| `make_moving_structure_frames_STEREO.py` | Same for STEREO COR1 or COR2, with an EUVI disk image and PFSS projected from STEREO's viewpoint |
| `make_moving_structure_frames_SolO.py` | Same for Solar Orbiter Metis (VL total or polarized brightness, or UV Lyman-alpha), with an EUI disk image and PFSS projected from Solar Orbiter's viewpoint. The light-curve panel adds the STIX 4–10 keV quick-look curve in red (own axis, 1-min means, Earth light-arrival time) next to GOES (`STIX_folder`, `stix_band`; `Multiwb_datamine.get_STIX_LC`). Metis and EUI Level 2 come from ESA SOAR; bins without public Level 2 use low-latency LL02 quick-looks (a separate family, never mixed). |
| `make_moving_structure_frames_STEREO_COR2_HI1.py` | COR2 and HI1 joined on one helioprojective grid from STEREO-A, following structures out to about 90 R☉. Draws one panel (`diff_panel='base'` or `'running'`); `'both'` builds the cache once and saves the base and running frames in two folders (`<output_tag>_bdiff`, `<output_tag>_rdiff`). |
| `flow_input_comparison.py` | Compares the flow of two caches built with different `flow_input` (quality fraction, speed, angle difference, time since base); CSV and PNG |
| `make_LASCO_overlay_frames.py` | LASCO difference images with HMI/AIA disk, global and AR-focused PFSS lines, and LAT/GOES panels |
| `make_STEREO_overlay_frames.py` | STEREO COR1/COR2 difference images with EUVI disk and observer-projected PFSS, plus LAT/GOES panels |
| `make_PSP_WISPR_overlay_frames.py` | PSP/WISPR field of view with a PFSS spherical-shell wireframe and projected field lines, plus LAT/GOES panels |
| `make_SolO_overlay_frames.py` | Solar Orbiter Metis tB with the EUI image filling the occulter, a magnified EUI panel and a STIX (`stix_bands`) + GOES panel over the whole window with a line at each frame, all in Earth light-arrival UT. Axes in solar radii; Metis and EUI aligned by helioprojective angle. Metis/EUI/STIX from ESA SOAR. `metis_mode` (panel a) and `eui_mode` (panel b) are each `'rdiff'` (running difference), `'bdiff'` (base difference at `base_time`) or `'intensity'`; `zoom_box=[x0,x1,y0,y1]` (R☉) sets panel (b) to a chosen box, e.g. around active regions, and `zoom_track=1` moves that box with solar differential rotation. |
| `make_AIA_LASCO_overlay_frames.py` | SOHO/LASCO C2 (or C3) frames with an SDO/AIA full-disk image inside the occulter, a full-disk AIA panel and the same STIX + GOES panel. `panels` lists (panel a, panel b) display pairs, one movie each; each panel is `'rdiff'` (running difference), `'bdiff'` (base difference at `base_time`) or `'intensity'` (folders `<outdir>_<a>` or `<outdir>_<a>-<b>`). AIA images are JSOC synoptic files (1k, 2-min cadence) fetched for each LASCO time; AIA angles are scaled by the solar-radius ratio so the SDO limb lies on the SOHO limb. |
| `make_STEREO_EUVI_overlay_frames.py` | The same frames for STEREO COR2 with EUVI (`euvi_wave`) inside the occulter and as the full-disk panel, using `make_AIA_LASCO_overlay_frames.make_frames` (names, colormap and titles from the FITS headers). Only full-resolution COR2 images are used, blanked inside `occ_rsun` (occulter and diffraction ring); EUVI is fetched with `get_STEREO_imgs` and matched within `euvi_maxdt`. Same `panels` and `base_time` options. |

## How a flow run works

```
launcher INPUT
  └─ motion_frame_pipeline_V1.make_frames_V1      (LASCO / STEREO / SolO launchers)
     combined_frame_pipeline.make_combined_frames (COR2 + HI1 launcher)
       1. prepare_sources       fill missing image bins: Level-1 → Helioviewer → Level-0.5
                                (Helioviewer skipped with helioviewer=0)
       2. motion_inputs         choose one product family, align on one grid,
          combined_motion_inputs   base/running differences, TV-L1 flow → motion_cache/
          cdaw_inputs           use_CDAW_pngs=1 (C2): replaces steps 1–2 with the CDAW PNGs
       3. single_noise_model    differences → noise-sigma units (ring noise profile)
          combined_motion_inputs   (same for COR2 + HI1; HI1 flow from star-cleaned images)
       4. prepare_sources       EUVI/EUI/SDO disk images and PFSS snapshots
       5. track_coronagraph_flows.render_frames
            puddle_analysis     signed bright/dark detection
            edge_motion         flow sampled along contours and at centroids
            pfss_projection / pfss_fixed_seeds   AR and global field lines
            solar_disk_layer    disk underlay;  noaa_region   AR position
          → PNG frames, MP4, CSVs
```

Steps 1–3 build the motion cache, which is slow (downloads, alignment and flow). With `reuse_cache='latest'` or a cache path, only step 5 runs again. This is the quick way to change detection thresholds, display, AR lines, the speed colour bar or arrows. `flow_input` and `flow_bin` are part of the cache: to change them, copy the cache and run `recompute_flow` on the copy.

## Modules

### Pipelines

| Module | Role | Main functions |
|---|---|---|
| `motion_frame_pipeline_V1.py` | One LASCO, STEREO or Solar Orbiter Metis event from download to frames, with sigma-unit noise model, asinh display, redraw mode and `flow_input` | `make_frames_V1` |
| `combined_frame_pipeline.py` | COR2 and HI1 event: COR2 route, HI1 through `get_STEREO_imgs`, combined cache (with `flow_input`), render | `make_combined_frames` |
| `motion_frame_pipeline.py` | Earlier pipeline with differences in input image units. No launcher uses it now; kept for reference. | `make_frames` |

### Building the motion cache

| Module | Role | Main functions |
|---|---|---|
| `motion_inputs.py` | Inventories local coronagraph FITS (filtered by observatory for STEREO-A/B and Solar Orbiter) and groups them into product families (Level-1, Helioviewer, Level-0.5; `BTYPE` keeps Metis tB and pB apart). Picks one family covering the event, aligns exposures on one angular grid, writes base/running differences and TV-L1 flow on intensity or base differences (`flow_input`, recorded in `noise_scales.json`), optionally on pixel blocks (`flow_bin`). `recompute_flow` rewrites `dx`, `dy`, `quality` of an existing cache for the other input from the images listed in `times.json`. | `inventory`, `select_family`, `make_motion_cache`, `flow`, `recompute_flow`, `auto_flow_bin` |
| `combined_motion_inputs.py` | COR2 and HI1 on one helioprojective grid, each differenced against itself and scaled by its own noise. HI1 flow is computed after removing the F-corona and the drifting star field. `flow_input='base_difference'` uses frame − base in COR2 and cleaned sum − cleaned base sum in HI1 (recorded in `combined_scales.json`). | `make_combined_cache`, `combined_grid`, `apply_noise_model`, `hi_clean_flow`, `star_drift`, `recompute_flow` |
| `cdaw_inputs.py` | `use_CDAW_pngs` (LASCO C2): downloads the CDAW running-difference (`*_lasc2rdf.png`) and intensity-composite (`*_lasc2aia193.png`) images and writes a motion cache: regions from the running differences, flow and base differences from the intensity images. Geometry from the white 1 R☉ limb circle; display units. | `make_cdaw_cache`, `download_cdaw`, `cdaw_geometry` |
| `flow_input_comparison.py` | Per-frame (and per-zone) flow comparison of two caches, plus region counts from both renders' `regions.csv` | `compare_flow_inputs`, `zone_stats` |
| `single_noise_model.py` | Puts single-instrument differences in noise-sigma units, optionally divided by a radial noise profile. Reversible, with the state kept in the cache. | `apply_noise_model`, `noise_scale`, `ring_noise_profile` |
| `prepare_sources.py` | Fills missing image bins through Solar_data_mining. Converts Helioviewer JP2s to FITS where science FITS are missing. Builds UTC-anchored PFSS snapshots. Solar Orbiter Metis/EUI through `get_Metis_imgs`/`get_SolO_imgs` (no Helioviewer route). | `fill_coronagraph`, `fill_disk`, `ensure_hv_coronagraph`, `ensure_hv_disk`, `prepare_pfss`, `stereo_root`, `solo_root` |
| `goes_cache.py` | Finds a cached GOES pickle covering the window, or downloads one | `resolve_goes_pickle`, `covering_pickle` |

### Rendering

| Module | Role | Main functions |
|---|---|---|
| `track_coronagraph_flows.py` | Draws each frame: difference panel(s), contours, edge and centroid drift arrows coloured by speed, PFSS and AR lines, disk underlay, LAT/GOES curves. One speed colour axis for all frames (continuous or discrete, automatic from the fastest vector or user-set). Axes in solar radii; LASCO ticks are symmetric values inside the field. Writes the CSVs and MP4. | `render_frames`, `write_mp4`, `speed_colour_axis`, `auto_speed_bins` |
| `puddle_analysis.py` | Signed full-field detection of bright and dark regions above a robust noise threshold, plus transport diagnostics | `detect_signed`, `usable_grid`, `outward_support`, `association` |
| `edge_motion.py` | Samples the flow field along bright/dark contours and at region centroids | `sample_edge_motion`, `sample_centroid_motion`, `contour_pixels` |
| `pfss_projection.py` | Selects AR closed loops and open lines. Loops can be rooted in the AR rectangle by one footpoint (long loops kept) or by both. `ar_rooted_lines` does tracing, caching and selection in one call for the overlay makers. | `select_ar_closed`, `select_ar_open`, `ar_rooted_lines` |
| `pfss_fixed_seeds.py` | Retraces one fixed set of Carrington seeds through each PFSS snapshot, so lines evolve smoothly between snapshots. Also traces AR seed patches. | `retrace_pfss`, `trace_ar_lines`, `select_trace_seeds` |
| `solar_disk_layer.py` | Nearest EUVI, EUI or SDO image, reprojected into the coronagraph grid as a disk underlay | `disk_layer`, `nearest_disk`, `disk_inventory` |
| `noaa_region.py` | Looks up a NOAA active region's position in the daily SRS reports, downloading them if missing | `find_NOAA_location` |

The three PFSS overlay makers (`make_LASCO_overlay_frames`, `make_STEREO_overlay_frames`, `make_PSP_WISPR_overlay_frames`) are self-contained scripts. They hold their own helpers (image cache scans, observer geometry, AR tracing, light-curve panels) and use `solar_overlay` and `solar_overlay_helper` from Solar_data_mining for PFSS and image compositing. The three coronagraph + EUV makers are importable modules; they are described in the next section.

## Coronagraph + EUV overlay makers: functions

These three scripts share one design:

- Every image is resampled onto a north-up helioprojective grid seen from the coronagraph's spacecraft, aligned by helioprojective angle (no 3-D reprojection between exposure times, which would blank off-disk pixels). Axes are Solar X/Y in solar radii.
- Panels (a) and (b) can each be a running difference (`'rdiff'`), a base difference at `base_time` (`'bdiff'`) or `'intensity'`.
- Grey limits are set by the `diff_clip` percentile of |difference| over all frames, so the contrast is the same in every frame.
- The STIX + GOES panel is in Earth light-arrival UT. Each frame's time is shifted from the spacecraft to Earth by light travel time before its line is drawn.

Importing any of them only defines the INPUT values and functions, so the functions can be used in your own scripts.

### `make_SolO_overlay_frames.py` (Solar Orbiter Metis + EUI)

| Function | Use |
|---|---|
| `metis_files(folder,sT,eT)` | Metis FITS in `folder` whose DATE-OBS lies in [sT, eT], time-sorted. |
| `load_metis(path)` | Metis image as a SunPy Map; non-positive pixels (occulter, outside the field) become NaN. |
| `get_eui(times,folder,product,maxdt)` | The SOAR Level 2 EUI file nearest each time (`product` e.g. `'eui-fsi174-image'`), downloaded into `folder` if missing; `None` where none is within `maxdt` s. |
| `frame_grid(m,npix)` | `npix`² north-up header centred on the Sun, seen from Map `m`'s observer, half-width = the farthest valid Metis pixel. |
| `to_grid(m,hdr)` | Samples Map `m` on header `hdr` by helioprojective angle; NaN outside `m`. |
| `extent_rsun(meta,rsun_arcsec)` | `imshow` extent (x0, x1, y0, y1) in solar radii for a north-up header. |
| `hgs_grid(ax,frame,rsun_arcsec,step=15)` | Draws a Stonyhurst grid (visible hemisphere) in solar radii on `ax`. |
| `light_curves(sT,eT,stix_dir,stix_bands,goes_folder)` | `(stix, goes)` DataFrames in Earth light-arrival UT: STIX 1-min means of `stix_bands` (reuses `STIX_QL_<sT>-<eT>.csv`, else `Multiwb_datamine.get_STIX_LC`) and GOES XRS-B (`goes_cache.resolve_goes_pickle`). |
| `make_frames(mfiles,efiles,outdir,npix=1000,`<br>`diff_clip=99.,zoom_rsun=1.6,eui_mode='intensity',`<br>`lc=None,lc_window=None,zoom_box=None,`<br>`zoom_track=False,metis_mode='rdiff',base=0)` | Writes one PNG per Metis image and returns the paths. (a) Metis with EUI intensity inside the occulter; (b) EUI magnified to ±`zoom_rsun` or to `zoom_box=[x0,x1,y0,y1]` (R☉), which with `zoom_track=True` follows solar differential rotation from the first frame; (c) the `lc` light curves over `lc_window`. `base` is the index of the base image for `'bdiff'`. |

Run as a script, it writes `<outdir>_<metis_mode>-<eui_mode>/Metis_EUI_*.png` and an MP4 (name ends in `_zoombox` or `_zoomtrack` when a box is used).

### `make_AIA_LASCO_overlay_frames.py` (SOHO/LASCO + SDO/AIA)

| Function | Use |
|---|---|
| `fits_times(folder,sT,eT)` | `[(Time, path)]` of the FITS in `folder` (recursive) with DATE-OBS in [sT, eT], time-sorted. |
| `get_aia_synoptic(times,wave,folder)` | Downloads the JSOC AIA synoptic file (1k, level 1.5, 2-min cadence) nearest each time into `folder` and returns the paths. |
| `load_cor(path)` | Coronagraph Map (LASCO, COR1/COR2) in DN/s; non-positive pixels become NaN. |
| `load_disk(path)` | EUV disk Map (AIA, EUVI) in DN/s. |
| `to_grid(m,hdr,fac=1.)` | Like the SolO `to_grid`, with angles multiplied by `fac`. AIA angles are scaled by the solar-radius ratio so the SDO limb lies on the SOHO limb. |
| `make_frames(lfiles,afiles,outdir,a_mode='rdiff',`<br>`b_mode='rdiff',base=0,npix=1000,diff_clip=99.,`<br>`disk_rsun=1.3,lc=None,lc_window=None,`<br>`inner_r=1.25,occ_rsun=None)` | Writes one PNG per coronagraph image and returns the paths. (a) coronagraph in `a_mode` with EUV intensity inside the occulter; (b) full-disk EUV to ±`disk_rsun` in `b_mode`; (c) the light curves. `inner_r` (R☉) is the radius of the EUV image shown inside the occulter; `occ_rsun` blanks the coronagraph inside that radius (occulter and diffraction ring). Instrument names, colormap and titles come from the FITS headers, so the same function serves STEREO. |

As a script, `panels=[(a,b),...]` makes one movie per pair, in `<outdir>_<a>` (when a = b) or `<outdir>_<a>-<b>`.

### `make_STEREO_EUVI_overlay_frames.py` (STEREO COR2 + EUVI)

| Function | Use |
|---|---|
| `cor_files(folder,sT,eT)` | `[(Time, path)]` of the full-resolution COR2 FITS in [sT, eT] (the largest frame size in the folder); binned frames are skipped. |
| `nearest(times,rows,maxdt)` | The path in `rows=[(Time,path)]` nearest each time, or `None` beyond `maxdt` s. |

The script fetches EUVI (`euvi_wave`, every `euvi_dT` s) with `Multiwb_datamine.get_STEREO_imgs`, matches it to each COR2 image within `euvi_maxdt`, and draws the frames with `make_AIA_LASCO_overlay_frames.make_frames` (`inner_r`, `occ_rsun`; same `panels` and `base_time`).

### Using the functions directly

```python
import make_SolO_overlay_frames as S
mf=S.metis_files('SolO/Metis/metis_tb','2025-05-30T11:00','2025-05-30T12:30')
ef=S.get_eui([S.dh._hdr_time(p) for p in mf],'SolO/EUI/fsi174','eui-fsi174-image',900)
lc=S.light_curves('2025-05-30T07:00','2025-05-31T00:00','SolO/STIX',['4-10 keV','15-25 keV'],'GOES')
S.make_frames(mf,ef,'my_frames',eui_mode='rdiff',lc=lc,lc_window=('2025-05-30T07:00','2025-05-31T00:00'),
	zoom_box=[0.45,0.84,-0.76,0.13],zoom_track=True)
```

Run this from the event folder (or `os.chdir(basedir)` first), since the folder paths are relative.

## Key settings

These are the less obvious ones; the guide above each INPUT block documents every setting.

| Setting | Meaning |
|---|---|
| `basedir` | The event folder. Every other path is relative to it. |
| `sT`, `eT`, `dT`, `base_diff_DT` | Event window (UTC), output cadence in seconds, and the quiet reference time for base differences |
| `download` | 1 fills missing images and PFSS snapshots. 0 uses only what is on disk. |
| `need_LASCO_calib`, `helioviewer` | Image source for the flow runs (one source per run, never mixed). Order: Level-1, then Helioviewer and Level-0.5 (Helioviewer first with `need_LASCO_calib=True` and for STEREO; Level-0.5 first with `need_LASCO_calib=False`). The first source with event frames, a predecessor and a base that covers more than half of the cadence slots is used; otherwise the one covering most slots. `helioviewer=0` never fetches, converts or uses Helioviewer images, even if they are on disk. Helioviewer images are 8-bit display products (255 levels, bright parts clipped), so science FITS give better region finding. |
| `use_CDAW_pngs` | LASCO C2 only. 1 builds the cache from the CDAW PNGs (saved in `<LASCO_imgfolder>/CDAW`) instead of FITS: regions from CDAW's running differences, flow and base differences from its intensity composites. Values are display units (the running differences have ~11 grey levels), so speeds are approximate; `npix` is fixed at the PNG grid (512). |
| `reuse_cache` | `''` builds a new motion cache. `'latest'` or a cache path redraws only. |
| `noise_model`, `ring_strength`, `ring_step_rsun` | Differences in sigma units (`'ring'`, `'global'`) or input units (`'none'`, LASCO/STEREO only). With `'ring'`, faint outer structures are detected as readily as bright inner ones. |
| `detect_sigma`, `detect_floor`, `bright_contrast` | Detection threshold, in sigma when a noise model is used |
| `detect_smooth_px`, `bright_contrast_snr` | Smoothing (grid pixels) before thresholding. With `bright_contrast_snr=1`, `bright_contrast` is a unitless local SNR: (region 95th percentile − ring mean) / ring std. COR2 + HI1 uses 4 px and SNR ≥ 2, which brings out faint diffuse structures. |
| `region_min_width_rsun`, `region_max_aspect` | Minimum breadth and maximum length/breadth of each region's fitted box. Drops thin lines such as HI1 artifacts (COR2 + HI1: 0.8 Rsun, 6). `''` = no limit. |
| `display_stretch`, `display_limit`, `display_smooth_px` | Display only (asinh stretch, grey-scale limit in sigma, smoothing). Detection is unaffected. |
| `diff_panel` | `'both'`, `'base'` or `'running'`. LASCO/STEREO: `'both'` draws both panels in one frame. COR2 + HI1: one panel per frame; `'both'` renders base and running from the same cache into separate folders, `output_tag` + `_bdiff` / `_rdiff`. |
| `ar_half_lon`, `ar_half_lat`, `AR_seed_spacing_deg` | AR rectangle (degrees about the NOAA centre) and seed spacing for AR field lines. Same settings in the flow launchers and the LASCO/STEREO overlay makers, which share one traced-line cache per PFSS snapshot (`PFSS/<snapshot>/AR_field_lines_rooted.npz`). |
| `trace_longARrooted_loops` | 1 keeps closed loops with one footpoint in the AR rectangle, so long AR loops are shown. 0 needs both footpoints inside. |
| `AR_loop_reach_deg` | Closed AR loops are drawn only if both ends lie within this distance (degrees on the Sun) of the NOAA centre; default 50. Keeps AR–quiet-Sun and AR–AR connectors and drops longer PFSS loops (e.g. to the polar caps, where PFSS is least reliable). `''` = no limit. Selection only: cached traces are reused. |
| `n_AR_open`, `n_closed`, `n_open` | Maximum magenta AR open lines, AR closed loops and global open lines drawn. Drawing only; cached lines are kept. |
| `pfss_linewidth`, `xlabel`, `ylabel`, `xticks`, `yticks`, `xlabel_size`, `ylabel_size`, `xtick_size`, `ytick_size` | PFSS line width and axis labels/sizes. Same in the flow launchers and the LASCO/STEREO overlay makers. Flow-frame ticks (None) are round values inside the field (LASCO C2 −5…5, C3 −20…20 R☉). Overlay axes are in solar radii, symmetric about Sun centre and rounded out to the next multiple of 10 (else 5) R☉ within 3 R☉, with minor ticks every 5 R☉. |
| `show_velocity_vectors` | 0 draws contours only, with the same layout. Vectors are still saved in the CSVs. |
| `save_PFSS_maps` | PFSS maps at the sT–eT midpoint in `basedir` (`PFSS_*.png`), on the chosen disk image (nearest cached one first) or a blank Stonyhurst-grid disk with the AR rectangle (`solar_overlay.pfss_context_maps`). `True`/`'both'`, `'full'` (full Sun only), `'AR'` (AR only) or `False` (skip, e.g. on reruns). In all six frame makers. |
| `colorbar_style`, `speed_limits`, `speed_bins` | Speed colour bar, one axis for all frames. `'continuous'`: limits from `speed_limits`, `'auto'` = 0 to the fastest drawn vector (edge or centroid) in sT–eT rounded up to half a decade step (1221 → 1500 km/s), or `'vmin,vmax'`. `'discrete'`: bands from `speed_bins`, `'auto'` = 0, 50, 100, 200, …, 1500, 2000, … km/s up to the fastest vector, or a list of km/s boundaries. Recorded in `parameters.json` (`colorbar_style`, `speed_limits_kms`, `speed_bins_kms`, `fastest_vector_kms`). |
| `flow_input` | Flow launchers. Images entering TV-L1: `'intensity'` (default, identical to earlier versions) or `'base_difference'` (frame − base for both frames of a pair, so the static instrument level, radial gradient and F-corona drop out). Detection is unchanged. A reused cache must have the same `flow_input`; to switch, copy the cache folder and run `recompute_flow(copy, flow_input)`. |
| `flow_bin` | LASCO/STEREO launchers. TV-L1 on `flow_bin`×`flow_bin` pixel blocks, displacements scaled back to the grid; detection and contours keep the full grid. `'auto'` (default) picks it from the grid pixel size and frame spacing so a 1500 km/s structure moves about 10 flow pixels per frame (`motion_inputs.auto_flow_bin`): C2 at `npix=512` → 6 (a fast CME moves ~60 grid pixels per 12-min frame there, beyond what TV-L1 follows), C3 → 1, single COR2 at 15 min → 3. An integer overrides it; 1 = grid pixels, the earlier behaviour. Recorded in `noise_scales.json` (when >1) and `parameters.json`. |
| `hi_flow_clean` | COR2 + HI1: compute HI1 flow on F-corona- and star-removed images. Without it the star field drifting with the spacecraft pulls arrows sunward. |

## Outputs

Each flow run writes to `<outdir>/<auto-named folder><output_tag>/`:

- `edge_arrows_<time>.png`: one frame per cadence slot
- `<instrument>_edge_velocity.mp4`
- `edge_vectors.csv`, `centroid_vectors.csv`: positions, pixel displacements, projected velocities (km/s), radial component, edge type
- `regions.csv`, `pfss_sources.csv`, `disk_sources.csv`, `parameters.json` (every render setting, plus `flow_input`, `flow_bin`, `colorbar_style`, `speed_limits_kms` / `speed_bins_kms` and `fastest_vector_kms`)
- `structures_<time>_masks.npz`: detection masks

The motion cache goes in `<outdir>/motion_cache/<name>/`. It holds the aligned differences, flow, `times.json` and the noise-model and flow state (`noise_scales.json` or `combined_scales.json`: `flow_input`, `flow_bin`), and can be reused for redraws. Caches are named `V1_<LASCO|craft>_<detector>_*`; CDAW caches are `V1_LASCO_C2_CDAW_*` (state `source='CDAW PNG'`), and `reuse_cache='latest'` picks a CDAW cache only when `use_CDAW_pngs=1`. The PFSS overlay makers write their frames and MP4 to `final_image_folder`, with intermediate images and logs in `outdir`. The coronagraph + EUV makers write PNGs and the MP4 to one folder per panel choice, named from `outdir` plus the panel modes (see their section above).

## Example

Redraw a STEREO COR2 + HI1 event with fewer AR lines and no arrows, without downloading or recomputing flow. In the launcher's INPUT block:

```python
basedir='/data/events/SGRE20250530'
sT='2025-05-30T22:30:00'
eT='2025-05-31T17:00:00'
diff_panel='base'
reuse_cache='latest'
n_AR_open=5
show_velocity_vectors=0
```

Then run:

```bash
python make_moving_structure_frames_STEREO_COR2_HI1.py
```

## Notes and limitations

- **Arrow speeds are nominal.** They are projected, image-plane pattern speeds and are prone to detection and projection effects. Use them to follow structures, not as plasma velocities.
- **STEREO HI1** has no Helioviewer source. Level-1 files are used when present locally (from UKSSDC, which needs a login); otherwise SSC Level-0.5 sums are downloaded.
- **LASCO C3 spoke:** `spoke_pa` masks the C3 support pylon. Check its position angle for each new date; the launcher's comment says to set `spoke_halfwidth=0` for C2.
- **Fast CMEs in LASCO C2:** at `npix=512` (0.025 R☉ per pixel) a 1500 km/s front moves ~60 pixels per 12-min frame. Keep `flow_bin='auto'` (6 for C2) and use `flow_input='base_difference'`: on 2025-05-31 this raised the C2 front speeds from ~40 to ~900 km/s (90th percentile), while intensity input stayed near 100–260 km/s because the steep radial gradient dominates the flow input.
- **Which LASCO source a run used** is printed as a coverage table and saved as `coverage_report.csv` in the cache. If a window shows weak, posterised differences, check whether Helioviewer was chosen (8-bit, bright parts clipped); `helioviewer=0` forces science FITS, at the cost of frames where they are missing.
- **CDAW PNGs (`use_CDAW_pngs=1`)** are C2 only: CDAW has no C3 intensity composite, and `recompute_flow` does not apply to these caches. CDAW's server drops some requests; downloads are retried (then `curl`), files already saved are reused, and a rerun fills remaining gaps. Speeds are approximate (display units).
- **COR2 + HI1 has no `flow_bin`**: its flow runs on the grid pixels (~0.1 R☉ at `npix=1024`).
- **Solar Orbiter Metis** science data are public only at Level 2 (MSB), released with a long delay (complete to mid-2025 as of Nov 2026). L0/L1 and the Metis calibration pipeline are not public; bins without Level 2 use SOAR low-latency LL02 images (preliminary, "not for scientific analysis"), kept as their own family. The Metis field in solar radii scales with Solar Orbiter's distance (about 5–11 R☉ at 0.85 AU), so `r_inner`/`r_outer` default to the header field. There is no Helioviewer route for Solar Orbiter. Use `flow_input='base_difference'` (the SolO launcher default): on 2025-05-30 intensity input gave under 30 km/s for the NW CME, base-difference input 50–350 km/s, still limited by the 30-min Metis cadence. Metis MSB images are stored in the cache in units of 10⁻¹² MSB, because `flow` floors its feature scale at one input unit (LASCO Level-1 MSB input would hit the same floor).
- **PSP/WISPR** data are public only after a delay. The WISPR maker stops with "No usable WISPR images" until they are released.
- **Light curves:** the LAT light curve is supplied by the user (`user_datacsv`). GOES XRS-B is fetched automatically. STIX quick-look curves come from ESA SOAR (`Multiwb_datamine.get_STIX_LC`); they are disk-integrated counts, not background-subtracted, and are shifted to Earth light-arrival time to line up with GOES.
