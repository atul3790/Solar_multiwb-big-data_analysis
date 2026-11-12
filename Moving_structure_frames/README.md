# Moving_structure_frames

Frame and movie makers for following coronal mass ejections and other moving structures through white-light coronagraph and heliospheric-imager data, with projected PFSS magnetic field lines and Fermi-LAT / GOES light curves in a lower panel.

Two families of runnable scripts share one set of modules:

- **Moving-structure (flow) frames** detect connected bright and dark structures in base-difference and running-difference images, measure their in-plane drift with TV-L1 optical flow, and draw contour and centroid drift arrows. The result is a diagnostic movie of how structures form, flow out and merge.
- **Overlay frames** draw difference images from LASCO, STEREO or PSP/WISPR with PFSS field lines projected from that spacecraft's viewpoint, plus the light curves.

The drift arrows are an image-plane proxy for pattern speed. Projection effects and detection uncertainties remain, and they are not plasma velocities.

Image download and preparation come from the companion repository **[Solar_data_mining](https://github.com/atul3790/Solar_data_mining)**: `Multiwb_datamine.py`, `datamine_calib_helper.py`, `solar_overlay.py` and `solar_overlay_helper.py`. That repository must be on your `PYTHONPATH`.

## Contents

- [Installation](#installation)
- [Event folder layout](#event-folder-layout)
- [Runnable scripts](#runnable-scripts)
- [How a flow run works](#how-a-flow-run-works)
- [Modules](#modules)
- [Key settings](#key-settings)
- [Outputs](#outputs)
- [Example](#example)
- [Notes and limitations](#notes-and-limitations)

## Installation

```bash
git clone https://github.com/atul3790/Solar_data_mining.git
git clone -b moving-structure-frames https://github.com/atul3790/Solar_multiwb-big-data_analysis.git
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
  STEREO/STEREO_A/{COR2,HI1,EUVI/195}/
  PSP_WISPR/
  LASCO_LAT_frames/              frames, MP4s, CSVs and motion_cache/ for LASCO runs
  STEREO_LAT_frames/             same for STEREO runs
  PSP_WISPR_LAT_frames/
```

GOES, NOAA SRS, PFSS and GONG files are downloaded once per event and reused by every script. Image folders hold only the files they need, and missing cadence bins are downloaded on demand. STEREO settings may name the STEREO root or an instrument folder inside it, because downloads always go to `<root>/STEREO_<sc>/<instrument>`.

## Runnable scripts

Edit only the INPUT block of a script, then run it with `python script.py` or `%run script.py`. Each INPUT block is preceded by a commented guide that explains every setting.

| Script | What it makes |
|---|---|
| `make_moving_structure_frames_LASCO.py` | LASCO C2 or C3 flow frames: base and running difference panels, bright/dark contours, edge and centroid drift arrows, PFSS, an SDO disk image (HMI or AIA) and LAT/GOES curves |
| `make_moving_structure_frames_STEREO.py` | Same for STEREO COR1 or COR2, with an EUVI disk image and PFSS projected from STEREO's viewpoint |
| `make_moving_structure_frames_STEREO_COR2_HI1.py` | COR2 and HI1 joined on one helioprojective grid from STEREO-A, following structures out to about 90 R☉. Draws one panel (`diff_panel='base'` or `'running'`). |
| `make_LASCO_overlay_frames.py` | LASCO difference images with HMI/AIA disk, global and AR-focused PFSS lines, and LAT/GOES panels |
| `make_STEREO_overlay_frames.py` | STEREO COR1/COR2 difference images with EUVI disk and observer-projected PFSS, plus LAT/GOES panels |
| `make_PSP_WISPR_overlay_frames.py` | PSP/WISPR field of view with a PFSS spherical-shell wireframe and projected field lines, plus LAT/GOES panels |

## How a flow run works

```
launcher INPUT
  └─ motion_frame_pipeline_V1.make_frames_V1      (LASCO / STEREO launchers)
     combined_frame_pipeline.make_combined_frames (COR2 + HI1 launcher)
       1. prepare_sources       fill missing image bins: Level-1 → Helioviewer → Level-0.5
       2. motion_inputs         choose one product family, align on one grid,
          combined_motion_inputs   base/running differences, TV-L1 flow → motion_cache/
       3. single_noise_model    differences → noise-sigma units (ring noise profile)
          combined_motion_inputs   (same for COR2 + HI1; HI1 flow from star-cleaned images)
       4. prepare_sources       EUVI/SDO disk images and PFSS snapshots
       5. track_coronagraph_flows.render_frames
            puddle_analysis     signed bright/dark detection
            edge_motion         flow sampled along contours and at centroids
            pfss_projection / pfss_fixed_seeds   AR and global field lines
            solar_disk_layer    disk underlay;  noaa_region   AR position
          → PNG frames, MP4, CSVs
```

Steps 1–3 build the motion cache, which is slow (downloads, alignment and flow). With `reuse_cache='latest'` or a cache path, only step 5 runs again. This is the quick way to change detection thresholds, display, AR lines, speed bins or arrows.

## Modules

### Pipelines

| Module | Role | Main functions |
|---|---|---|
| `motion_frame_pipeline_V1.py` | One LASCO or STEREO event from download to frames, with sigma-unit noise model, asinh display and redraw mode | `make_frames_V1` |
| `combined_frame_pipeline.py` | COR2 and HI1 event: COR2 route, HI1 through `get_STEREO_imgs`, combined cache, render | `make_combined_frames` |
| `motion_frame_pipeline.py` | Earlier pipeline with differences in input image units. No launcher uses it now; kept for reference. | `make_frames` |

### Building the motion cache

| Module | Role | Main functions |
|---|---|---|
| `motion_inputs.py` | Inventories local coronagraph FITS and groups them into product families (Level-1, Helioviewer, Level-0.5). Picks one family covering the event, aligns exposures on one angular grid, writes base/running differences and TV-L1 flow. | `inventory`, `select_family`, `make_motion_cache`, `flow` |
| `combined_motion_inputs.py` | COR2 and HI1 on one helioprojective grid, each differenced against itself and scaled by its own noise. HI1 flow is computed after removing the F-corona and the drifting star field. | `make_combined_cache`, `combined_grid`, `apply_noise_model`, `hi_clean_flow`, `star_drift` |
| `single_noise_model.py` | Puts single-instrument differences in noise-sigma units, optionally divided by a radial noise profile. Reversible, with the state kept in the cache. | `apply_noise_model`, `noise_scale`, `ring_noise_profile` |
| `prepare_sources.py` | Fills missing image bins through Solar_data_mining. Converts Helioviewer JP2s to FITS where science FITS are missing. Builds UTC-anchored PFSS snapshots. | `fill_coronagraph`, `fill_disk`, `ensure_hv_coronagraph`, `ensure_hv_disk`, `prepare_pfss`, `stereo_root` |
| `goes_cache.py` | Finds a cached GOES pickle covering the window, or downloads one | `resolve_goes_pickle`, `covering_pickle` |

### Rendering

| Module | Role | Main functions |
|---|---|---|
| `track_coronagraph_flows.py` | Draws each frame: difference panel(s), contours, edge and centroid drift arrows coloured by speed, PFSS and AR lines, disk underlay, LAT/GOES curves. Writes the CSVs and MP4. | `render_frames`, `write_mp4` |
| `puddle_analysis.py` | Signed full-field detection of bright and dark regions above a robust noise threshold, plus transport diagnostics | `detect_signed`, `usable_grid`, `outward_support`, `association` |
| `edge_motion.py` | Samples the flow field along bright/dark contours and at region centroids | `sample_edge_motion`, `sample_centroid_motion`, `contour_pixels` |
| `pfss_projection.py` | Selects AR closed loops and open lines. Loops can be rooted in the AR rectangle by one footpoint (long loops kept) or by both. `ar_rooted_lines` does tracing, caching and selection in one call for the overlay makers. | `select_ar_closed`, `select_ar_open`, `ar_rooted_lines` |
| `pfss_fixed_seeds.py` | Retraces one fixed set of Carrington seeds through each PFSS snapshot, so lines evolve smoothly between snapshots. Also traces AR seed patches. | `retrace_pfss`, `trace_ar_lines`, `select_trace_seeds` |
| `solar_disk_layer.py` | Nearest EUVI or SDO image, reprojected into the coronagraph grid as a disk underlay | `disk_layer`, `nearest_disk`, `disk_inventory` |
| `noaa_region.py` | Looks up a NOAA active region's position in the daily SRS reports, downloading them if missing | `find_NOAA_location` |

The three overlay makers are self-contained scripts. They hold their own helpers (image cache scans, observer geometry, AR tracing, light-curve panels) and use `solar_overlay` and `solar_overlay_helper` from Solar_data_mining for PFSS and image compositing.

## Key settings

These are the less obvious ones; the guide above each INPUT block documents every setting.

| Setting | Meaning |
|---|---|
| `basedir` | The event folder. Every other path is relative to it. |
| `sT`, `eT`, `dT`, `base_diff_DT` | Event window (UTC), output cadence in seconds, and the quiet reference time for base differences |
| `download` | 1 fills missing images and PFSS snapshots. 0 uses only what is on disk. |
| `reuse_cache` | `''` builds a new motion cache. `'latest'` or a cache path redraws only. |
| `noise_model`, `ring_strength`, `ring_step_rsun` | Differences in sigma units (`'ring'`, `'global'`) or input units (`'none'`, LASCO/STEREO only). With `'ring'`, faint outer structures are detected as readily as bright inner ones. |
| `detect_sigma`, `detect_floor`, `bright_contrast` | Detection threshold, in sigma when a noise model is used |
| `detect_smooth_px`, `bright_contrast_snr` | Smoothing (grid pixels) before thresholding. With `bright_contrast_snr=1`, `bright_contrast` is a unitless local SNR: (region 95th percentile − ring mean) / ring std. COR2 + HI1 uses 4 px and SNR ≥ 2, which brings out faint diffuse structures. |
| `region_min_width_rsun`, `region_max_aspect` | Minimum breadth and maximum length/breadth of each region's fitted box. Drops thin lines such as HI1 artifacts (COR2 + HI1: 0.8 Rsun, 6). `''` = no limit. |
| `display_stretch`, `display_limit`, `display_smooth_px` | Display only (asinh stretch, grey-scale limit in sigma, smoothing). Detection is unaffected. |
| `diff_panel` | `'both'`, `'base'` or `'running'` (COR2 + HI1: one panel) |
| `ar_half_lon`, `ar_half_lat`, `AR_seed_spacing_deg` | AR rectangle (degrees about the NOAA centre) and seed spacing for AR field lines. Same settings in the flow launchers and the LASCO/STEREO overlay makers, which share one traced-line cache per PFSS snapshot (`PFSS/<snapshot>/AR_field_lines_rooted.npz`). |
| `trace_longARrooted_loops` | 1 keeps closed loops with one footpoint in the AR rectangle, so long AR loops are shown. 0 needs both footpoints inside. |
| `n_AR_open`, `n_closed`, `n_open` | Maximum magenta AR open lines, AR closed loops and global open lines drawn. Drawing only; cached lines are kept. |
| `show_velocity_vectors` | 0 draws contours only, with the same layout. Vectors are still saved in the CSVs. |
| `speed_bins` | Colour bands for projected speed (km/s) |
| `hi_flow_clean` | COR2 + HI1: compute HI1 flow on F-corona- and star-removed images. Without it the star field drifting with the spacecraft pulls arrows sunward. |

## Outputs

Each flow run writes to `<outdir>/<auto-named folder><output_tag>/`:

- `edge_arrows_<time>.png`: one frame per cadence slot
- `<instrument>_edge_velocity.mp4`
- `edge_vectors.csv`, `centroid_vectors.csv`: positions, pixel displacements, projected velocities (km/s), radial component, edge type
- `regions.csv`, `pfss_sources.csv`, `disk_sources.csv`, `parameters.json` (every render setting)
- `structures_<time>_masks.npz`: detection masks

The motion cache goes in `<outdir>/motion_cache/<name>/`. It holds the aligned differences, flow, `times.json` and the noise-model state, and can be reused for redraws. Overlay makers write their frames and MP4 to `final_image_folder`, with intermediate images and logs in `outdir`.

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
- **LASCO C3 spoke:** `spoke_pa` masks the C3 support pylon. Check its position angle for each new date.
- **PSP/WISPR** data are public only after a delay. The WISPR maker stops with "No usable WISPR images" until they are released.
- **Light curves:** the LAT light curve is supplied by the user (`user_datacsv`). GOES XRS-B is fetched automatically.
