# Changelog

## 0.2.0 — unreleased

### Fixed

- **Output was sheared into a parallelogram whenever the decode window came out
  with an odd width or height.** ffmpeg's `crop` silently rounds odd sizes *and*
  odd offsets down to even on chroma-subsampled formats, so the raw pipe was
  being reshaped at the wrong width and every row slipped a pixel. This affected
  most ROIs; the tests missed it because every ROI in the suite happened to
  produce an even window. Crop is now passed `exact=1`, windows are snapped
  even, and the real frame size is measured rather than assumed — which also
  covers rotation metadata and anamorphic SAR.
### Added

- `vidsr ui` — a Tk desktop window: scrub the video, drag the region, drag the
  timeline to choose the range, shift-drag to mark spans to ignore, pick whether
  the subject is still or moving, and run it with a progress bar. The result,
  the side-by-side comparison and the sharpening variants are shown in the
  window. It builds the equivalent CLI command and shows it to you, so nothing
  about the headless workflow changes. Needs `python3-tk`; prints how to install
  it when missing.
- `set_progress()` so a front end can follow a reconstruction.

## 0.1.0 — 2026-09-07

First public release. Alpha: the pipeline is tested end to end against a
synthetic ground truth, but it has been exercised on few real cameras, and the
CLI may still change.

- `info` / `grid` / `select` / `sr` commands.
- Sub-pixel registration (phase correlation + ECC, translation → euclidean →
  affine → homography), robust frame rejection, trimmed-mean fusion, iterative
  back-projection and Richardson–Lucy deconvolution.
- Self-contained local HTML picker: ROI selection, timeline spans to ignore,
  per-frame include/exclude, outlier auto-flagging.
- Time-span selection from the CLI (`--use` / `--skip`) with an evenly spread
  frame budget.
- Diagnostics: sub-pixel phase coverage, duplicate-frame (encoder skip)
  detection, per-frame alignment report.
