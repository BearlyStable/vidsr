# Changelog

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
