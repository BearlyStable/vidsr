# vidsr

Multi-frame super-resolution for video. Point it at a clip, mark the seconds
worth using, drag a box around the thing you need to read, and it fuses every
usable frame of that box into one sharper image.

**It does not invent detail.** Every output pixel is a weighted measurement of
real sensor samples: frames are aligned to sub-pixel precision, stacked
robustly, then deconvolved with a camera model. There is no generative upscaler
in the pipeline, so the result is something you can defend — a "reimagine this"
model will happily render a *plausible* licence plate that is not the one in the
video.

> **Alpha (0.1.x).** The pipeline is tested end to end against a synthetic
> ground truth, but it has seen few real cameras and the CLI may still change.

## Install

```bash
pip install vidsr
```

You also need **ffmpeg** on `PATH` (`apt install ffmpeg` / `dnf install ffmpeg`).
Everything else is numpy and OpenCV. No GPU, no model weights, and no network
access at any point — your footage never leaves the machine.

<sub>vidsr depends on `opencv-python-headless`. If you already have
`opencv-python` and want to keep it, install with
`pip install --no-deps vidsr numpy` instead.</sub>

## What it works on

Nothing in the algorithm is subject-specific — it aligns and stacks whatever is
in the box. What matters is whether the subject satisfies the method's
assumptions:

| subject | how it does |
|---|---|
| signs, licence plates, badges, stickers, serial numbers, text | **best case** — rigid, flat, high contrast |
| car details, damage, small hardware | works well, same reasons |
| a face that holds still | works; detail improves |
| a face that turns or talks | **poorly** — the motion models are global (translation → euclidean → affine → homography), so a non-rigid subject will not stack and the fusion averages it into mush |
| foliage, water, crowds, anything self-moving | not usable |

`--corners` deskewing additionally assumes the subject is planar.

For faces it is worth being explicit: recovering *detail* is not the same as
establishing *identity*. This tool can sharpen a face it cannot identify.

## The window

```bash
vidsr ui cam.mkv          # or just `vidsr ui` and open a file from there
```

One window that does the whole job: scrub the video, drag a box around the
subject, drag on the timeline to set the range to use, shift-drag to mark spans
to ignore (the two minutes where someone walked through), say whether the
subject is still or moving, and hit Run. A progress bar tracks the
reconstruction and the result appears in the next tab, alongside the
side-by-side comparison and the sharpening variants.

It is a front end for the CLI and nothing more — **Copy CLI command** gives you
the exact equivalent line, so anything you set up in the window can be re-run on
a headless box.

Tk is a separate package on most distributions:

```bash
sudo apt install python3-tk        # Debian / Kali / Ubuntu
sudo dnf install python3-tkinter   # Fedora
```

Without it the UI prints that hint and exits; everything below still works.

## Command line

```bash
vidsr info   cam.mkv                                  # fps, codec, interlacing
vidsr grid   cam.mkv --start 0 --end 20:00            # contact sheet: find the event
vidsr select cam.mkv --use 10:00-15:00 --out work     # build the picker UI
xdg-open work/select.html
vidsr sr     cam.mkv --select selection.json --out out
```

### `select` — the browser-based picker

An alternative to the window that needs no Tk at all, useful over SSH.

`select.html` is a self-contained page (no server, works over `file://`):

- **drag on the image** to draw the ROI; a 6× magnifier follows the cursor.
  The frame you draw on becomes the reference the other frames align to.
- **drag on the timeline** to mark a span to ignore — "someone walked through
  between 12:30 and 14:30" is one drag, not 3000 clicks. Switch the drag mode
  to *keep only this* to throw away everything outside a span instead.
- **click a frame** in the filmstrip to drop just that one; shift-click for a
  range; `space` plays the selection so you can see the pedestrian arrive.
- **auto-flag outliers** scores every frame against the median of the ROI and
  flags the ones that disagree — occlusions, headlight glare, motion smear.
- **download selection.json**, then run the command shown in the panel.

Everything the page can do is also reachable from the command line, so a
headless box is not blocked:

```bash
vidsr sr cam.mkv --roi 812,430,96,34 --use 10:00-15:00 --skip 12:30-14:30 --out out
```

### `sr` — the reconstruction

Reads the selection, aligns, rejects the frames that do not belong, fuses,
back-projects, deconvolves, and writes to `--out`:

| file | what it is |
|---|---|
| `01_reference_crop.png` | the ROI as it appears in one frame |
| `02_baseline_bicubic_x4.png` | naive upscale — the honest comparison |
| `03_fused_x4.png` | the stack, before sharpening |
| `04_backprojected_x4.png` | after iterative back-projection |
| `05_result_x4.png` | final image |
| `compare.png` | all of the above side by side |
| `variants.png` | a grid of deconvolution/contrast settings — pick what reads |
| `review.html` | every frame with its scores; drop more, re-export, re-run |
| `report.json` | per-frame alignment, rejection reasons, all settings |

Look at `variants.png` first. Legibility is a judgement call, and the right
amount of sharpening depends on the subject.

## Two regimes, and why the numbers differ

Multi-frame super-resolution recovers genuine resolution when the subject lands
on **different sub-pixel positions** in different frames — each frame then
samples the scene slightly differently, and the stack holds more information
than any single frame. A car driving past does this beautifully.

A parked car under a rigidly mounted camera does not. What you gain there is:

- **noise averaging** — √N less sensor noise, which is often what makes glyphs
  legible in the first place;
- **compression-artifact averaging** — H.264 quantisation differs frame to
  frame, so stacking cancels much of the blocking and ringing;
- **deconvolution headroom** — with the noise floor knocked down, you can
  deblur far harder before noise explodes.

The tool tells you which regime you are in:

```
kept  123/197 frames   median residual 0.0543   sub-pixel phase coverage 100%
```

**Phase coverage** is the fraction of sub-pixel positions actually sampled. High
means real resolution gain. Low (with a `note:` at the end of the run) means the
scene never moved, and you are getting denoise + deblur — still useful, but
`--scale 8` will not buy more than `--scale 3`.

If it reports frames *identical to their predecessor*, the encoder emitted skip
blocks: those frames are literal copies and carry no new information. Widen the
time range to get genuinely different ones.

Presets set sensible defaults for each regime:

```bash
vidsr sr cam.mkv --select selection.json --preset static   # subject and camera still
vidsr sr cam.mkv --select selection.json --preset moving   # subject crosses the frame
```

## Options worth knowing

| option | why |
|---|---|
| `--use A-B`, `--skip A-B` | time spans, repeatable; `10:00-15:00`, `1:30+45`, `-30` |
| `--max-frames N` | budget (default 300), spread evenly across the selection |
| `--scale` | output magnification (4 is a sane default; 8 rarely adds real detail) |
| `--deint field` | **interlaced cameras**: treat each field as its own time sample, doubling your frames. Check `info` for the field order |
| `--motion` | `auto` walks translation → euclidean → affine. Use `affine` for an approaching subject, `translation` for a still one |
| `--rl`, `--unsharp`, `--clahe` | sharpening and local contrast; compare in `variants.png` |
| `--min-ncc`, `--reject-k` | how eagerly frames are rejected. Loosen if too much is dropped |
| `--corners x1,y1,...` | four corners of an angled flat subject → deskewed output |
| `--bits 16` | 16-bit PNG for further analysis |
| `-v` | per-frame alignment detail |

## Limits

- **There is a hard floor set by how many pixels the subject occupies.**
  Measured on the synthetic clips: a plate ~44 px wide (≈8 px character height)
  comes out cleanly readable from ~90 frames. At ~28 px wide (≈5 px characters)
  the result is clearly sharper than bicubic and the character *positions*
  resolve, but the glyphs stay ambiguous even with 300 frames. That is the
  sampling limit, not a tuning problem — no setting, and no upscaler that does
  not invent detail, gets past it.
- Heavy motion blur in every frame is unrecoverable for the same reason: the
  information is not in the data.
- Interlaced footage decoded without `--deint` fuses two different time samples
  into every frame; that alone can be what is smearing your subject.
- A long `--use` span costs what it costs to decode: sampling 200 frames out of
  five minutes still decodes those five minutes. Narrow the span when you can.
- The ROI is defined on the reference frame. If the subject drifts far, increase
  `--pad` so the alignment window still contains it.

## Development

```bash
git clone https://github.com/BearlyStable/vidsr && cd vidsr
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python tests/run_tests.py           # ~2 min
```

The tests render synthetic clips from a known ground truth — a plate whose text
the test knows — through a simulated lens, sensor noise and a real H.264 encode,
with a pedestrian walking across the subject. They then check the reconstruction
numerically rather than by eye: the result must correlate with the truth better
than a bicubic upscale does, the occluded frames must be rejected while the
clean ones survive, `--use`/`--skip` must decide which seconds are read at all,
and the picker page must be valid.

```bash
.venv/bin/python tests/run_tests.py -k static   # subset
.venv/bin/python tests/run_tests.py --keep      # keep outputs to look at
```

Make your own clips with `python tools/make_test_video.py --help`.

## License

MIT
