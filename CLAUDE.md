# CLAUDE.md

Guidance for Claude Code working in this repo.

## What this is

`vidsr` fuses many video frames of one small region into a single sharper image
(multi-frame super-resolution). Typical use: reading a licence plate, sign or
badge out of security footage.

**The core design promise: no invented detail.** Every output pixel is a
weighted measurement of real sensor samples — aligned, stacked, deconvolved.
Do **not** add a generative/AI upscaler, a diffusion refiner, or a learned
prior, even as an option. The whole value of this tool is that its output is
defensible as evidence; a model that renders a *plausible* plate destroys that
and would be worse than useless in the situations people use this for.

## Setup on a new machine

```bash
git clone git@github.com:BearlyStable/vidsr.git && cd vidsr
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python tests/run_tests.py        # ~90 s, needs ffmpeg
```

Requires `ffmpeg`/`ffprobe` on PATH. Runtime deps are only numpy + OpenCV.
`node` is optional (one test uses `node --check` on the generated HTML picker
and skips without it). For the desktop UI you need `python3-tk` plus `Xvfb`:

```bash
sudo dnf install python3-tkinter xorg-x11-server-Xvfb xdotool   # Fedora
sudo apt install python3-tk xvfb xdotool                        # Debian/Kali
```

With those, `ui/window-drives-a-reconstruction` builds the real window on a
throwaway X server and drives it through Tk's own event system — dragging the
region on the canvas, ignoring a span, pressing Run, waiting for the result to
appear. Without them it skips. `xdotool` is not needed by the tests; it is for
driving the window by hand when you want to *look* at it:

```bash
Xvfb :99 -screen 0 1400x900x24 &
DISPLAY=:99 vidsr ui clip.mkv &
DISPLAY=:99 import -window root shot.png     # then read the screenshot
```

Do that after any layout change. Two bugs got through review and were only
visible in a screenshot: the action bar packed after a tall column fell off the
bottom of the window, and the result canvas measured itself before its tab had
ever been mapped, so the image rendered tiny in a corner. Neither raised an
error anywhere.

## Working agreement

- **Every new feature ships with a test.** No exceptions. Bug fixes get a test
  that fails before the fix.
- **Green tests are trusted.** If the suite passes, the change works — say so
  and move on. Do not re-verify by hand, do not eyeball images "just to be
  sure", do not add defensive re-checks of things a test already covers.
- **If you feel the need to try something manually, that feeling is a missing
  test.** Write the test instead, then trust it. The one-off script you were
  about to run is almost always a better test than a manual look.
- **Commit freely and always push.** The user may step away at any moment and
  wants the remote to be current. Never leave work only in the working tree.
- **Ask before creating a release tag** (that is the one action that publishes
  to PyPI and cannot be undone). Everything else — commits, pushes, branches,
  refactors — proceed without asking.

## Layout

```
src/vidsr.py            the pipeline and CLI (~2100 lines)
src/vidsr_ui.py         optional Tk desktop UI (~700 lines)
tests/run_tests.py      the suite: unit + end-to-end, no pytest needed
tools/make_test_video.py  renders synthetic clips with a known ground truth
.github/workflows/ci.yml       tests on 3.9/3.11/3.13 + build check
.github/workflows/publish.yml  PyPI trusted publishing, on GitHub release
```

`src/vidsr.py` runs top to bottom: utilities → ffprobe/ffmpeg decode →
image helpers → alignment → HR geometry → fusion → back-projection and
deconvolution → post-processing → the HTML picker → commands → CLI.

## Invariants that are easy to break

These were all found the hard way; a unit test guards each one.

- **Warp direction.** Matching `cv2.findTransformECC`, a warp `W` maps
  *reference* coordinates to *moving-frame* coordinates. Resampling therefore
  always uses `WARP_INVERSE_MAP`. `warp_lr_to_hr` and `warp_hr_to_lr` must stay
  exact inverses (`units/warp-round-trip`).
- **The HR grid is pixel-centre aligned**: `hr_transform` offsets by
  `0.5*scale - 0.5`, the same convention `cv2.resize` uses. Indexing corner-to-
  corner instead silently shifts every result by `(scale-1)/2` HR pixels
  against the source it claims to reproduce.
- **Frame rejection gates on NCC, never on ECC's rho.** Rho is barely sensitive
  near convergence (it moves <0.002 where real alignment quality moves 0.05)
  and is undefined when ECC diverges — which happens routinely on small, noisy,
  static patches. Rho is kept as a diagnostic only. When ECC fails entirely,
  fall back to the phase-correlation translation and let NCC judge it.
- **Sharpness weighting is one-sided against the median.** On a static scene
  every frame shows the same thing, so above-median Laplacian energy is *noise*.
  Rewarding it preferentially stacks the noisiest frames.
- **Selection is time-based.** Spans and per-frame exclusions are exported as
  timestamps, not indices, so a selection survives a change of sampling stride.
  Index-based exclusions are a CLI convenience only.
- **The decode window may always shrink to ROI + pad**, because cropping does
  not change frame indexing — only spans, stride, deint and max-frames do.
- **Never assume the size of a decoded frame.** ffmpeg's crop rounds odd sizes
  *and* odd offsets down to even on chroma-subsampled formats, and rotation
  metadata or anamorphic SAR change the size too — all silently. Reshaping the
  raw pipe at the wrong width shears every frame into a parallelogram with no
  error anywhere. `probe_filtered_size` measures what the filter chain really
  emits and that measurement wins; windows are snapped even, and crop is passed
  `exact=1`. If you add a filter, keep the measurement.
- **The UI has no second code path.** `vidsr_ui` builds a `vidsr sr ...`
  argument list and hands it to the real parser, which is what makes the window
  and the headless CLI equivalent. Do not let it call pipeline internals
  directly.
- **Widgets hold no logic.** Geometry and argument building are pure functions
  with their own tests; the window only wires them up. Anything you are tempted
  to compute inside an event handler belongs outside it, where it can be tested
  without a display.

## Domain facts worth keeping in mind

- **Two regimes.** Real resolution gain needs the subject to land on different
  sub-pixel positions across frames (a subject crossing the frame). A still
  subject under a rigid camera gives noise averaging, compression-artifact
  averaging and deconvolution headroom instead — useful, but `--scale 8` buys
  nothing over `--scale 3`. `phase_coverage` in `report.json` says which regime
  you are in.
- **Duplicate frames.** Encoders emit skip-blocks on static scenes; those
  frames are literal copies carrying no information. Detected and dropped.
- **The floor.** ~8 px character height recovers cleanly; ~5 px does not, at any
  frame count. This is a sampling limit, not a tuning problem. Be honest about
  it rather than promising more.

## Tests

Synthetic clips are rendered from a known ground truth (a plate whose text the
test knows) through a simulated lens, sensor noise and a real H.264 encode, then
scored numerically against that truth — the reconstruction must beat a bicubic
upscale of the same clip.

```bash
.venv/bin/python tests/run_tests.py            # all
.venv/bin/python tests/run_tests.py -k static  # subset
.venv/bin/python tests/run_tests.py --fresh    # re-encode the clips
.venv/bin/python tests/run_tests.py --keep     # keep outputs to inspect
VIDSR_TRACE=1 .venv/bin/python tests/run_tests.py   # tracebacks
```

Clips are cached in `.testwork/` (gitignored). Add a test by decorating a
function with `@test("group/name")`; it takes an optional `fresh` argument and
fails by raising. Prefer asserting a *relative* improvement (result vs. the
bicubic baseline of the same clip) over an absolute score — relative assertions
survive a change of ffmpeg encoder, absolute ones do not.

When adding a scenario, put its parameters in `SCENARIOS` and derive the ROI
from `gen.plate_roi_video(...)` rather than hardcoding pixel coordinates; the
fixture knows exactly where it drew the subject.

## Releasing (ask the user first)

1. Bump `version` in `pyproject.toml` **and** `__version__` in `src/vidsr.py`,
   update `CHANGELOG.md`.
2. Commit, push, and confirm CI is green.
3. **Ask the user before this step.** Create a GitHub release with a tag
   (`v0.1.0`); `publish.yml` builds and uploads to PyPI via Trusted Publishing
   (OIDC — no token is stored anywhere).

PyPI's publisher must match: owner `BearlyStable`, repo `vidsr`, workflow
`publish.yml`, environment `pypi`.
