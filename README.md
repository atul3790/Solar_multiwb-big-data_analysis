# Solar_multiwb-big-data_analysis

Analysis codes for multi-wavelength solar event studies built on the
[Solar_data_mining](https://github.com/atul3790/Solar_data_mining) download and preparation modules (the base kit: acquisition, overlays, PFSS — set it on `PYTHONPATH` before using anything below).

Each sub-project lives in its own folder on `main`, not on a separate branch. A sub-project is developed on a short-lived feature branch, then merged into `main` under its own folder once it's in a working state.

| Folder | Contents |
|---|---|
| [`Moving_structure_frames/`](Moving_structure_frames) | CME and moving-structure flow movies (LASCO, STEREO COR2 and COR2+HI1) and PFSS/light-curve overlay frames (LASCO, STEREO, PSP/WISPR) |

See the README inside each folder for installation and usage. When a sub-project is ready to accompany a paper, it's extracted into its own standalone public repo via `git subtree split -P <folder> -b <folder>_publish` rather than published from here directly — this repo can stay private with every sub-project's work-in-progress folders together.
