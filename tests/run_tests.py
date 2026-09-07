#!/usr/bin/env python3
"""
Regression tests for vidsr.

Synthetic clips are rendered from a known ground truth, so every claim the tool
makes can be checked numerically instead of by eye: the reconstruction must
correlate with the true plate better than a plain bicubic upscale does, and the
frames a pedestrian walks through must actually get rejected.

    python tests/run_tests.py            # all of them
    python tests/run_tests.py -k static  # just the matching ones
    python tests/run_tests.py --fresh    # re-encode the clips
    python tests/run_tests.py --keep     # keep outputs for a look
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import make_test_video as gen          # noqa: E402
import vidsr as ps                     # noqa: E402

PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src")
           + os.pathsep + os.environ.get("PYTHONPATH", ""))
# run the module, so the tests work against a checkout or an install alike
TOOL = ["-m", "vidsr"]
WORK = os.environ.get("VIDSR_TESTDIR") or os.path.join(ROOT, ".testwork")

TESTS: list[tuple[str, callable]] = []


def test(name):
    def deco(fn):
        TESTS.append((name, fn))
        return fn
    return deco


def eq(got, want, what=""):
    assert got == want, f"{what}: got {got!r}, want {want!r}"


def ge(got, want, what=""):
    assert got >= want, f"{what}: got {got:.4f}, need >= {want:.4f}"


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------

SCENARIOS = {
    # a parked car under a fixed camera: no object motion, only mount shake
    "static": dict(plate="MKX 8317", plate_px=44, dx=0.0, dy=0.0, zoom_rate=0.0,
                   jitter=0.35, noise=7.0, frames=200, occlude="60-95",
                   bitrate="900k", width=960, height=540, fps=25),
    # a rigid mount with no shake at all: the hardest case, no sub-pixel diversity
    "rigid": dict(plate="MKX 8317", plate_px=44, dx=0.0, dy=0.0, zoom_rate=0.0,
                  jitter=0.0, noise=7.0, frames=120, occlude="", bitrate="900k",
                  width=960, height=540, fps=25),
    # a car driving past: real sub-pixel sampling, the classic SR win
    "moving": dict(plate="MKX 8317", plate_px=52, dx=0.9, dy=0.28, zoom_rate=0.0004,
                   jitter=0.25, noise=6.0, frames=110, occlude="", bitrate="1200k",
                   width=960, height=540, fps=25),
}


def clip_path(name: str) -> str:
    return os.path.join(WORK, f"{name}.mkv")


def build_clip(name: str, fresh: bool = False) -> dict:
    p = SCENARIOS[name]
    out = clip_path(name)
    if fresh or not os.path.exists(out):
        os.makedirs(WORK, exist_ok=True)
        cmd = [PY, os.path.join(ROOT, "tools", "make_test_video.py"), "--out", out,
               "--plate", p["plate"], "--plate-px", str(p["plate_px"]),
               "--dx", str(p["dx"]), "--dy", str(p["dy"]),
               "--zoom-rate", str(p["zoom_rate"]), "--jitter", str(p["jitter"]),
               "--noise", str(p["noise"]), "--frames", str(p["frames"]),
               "--bitrate", p["bitrate"], "--fps", str(p["fps"]),
               "--width", str(p["width"]), "--height", str(p["height"]),
               "--occlude", p["occlude"]]
        r = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
        assert r.returncode == 0, f"clip generation failed:\n{r.stdout}\n{r.stderr}"
    return dict(p, path=out, name=name)


def mo_of(p) -> dict:
    return {"dx": p["dx"], "dy": p["dy"], "zoom": p["zoom_rate"]}


def roi_of(p, t=0.0) -> tuple[int, int, int, int]:
    return gen.plate_roi_video(p["width"], p["height"], t, mo_of(p), p["plate_px"])


def run_sr(p, *args, out=None) -> dict:
    out = out or os.path.join(WORK, f"out_{p['name']}")
    shutil.rmtree(out, ignore_errors=True)
    cmd = [PY, *TOOL, "sr", p["path"], "--out", out, *[str(x) for x in args]]
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, f"sr failed:\n{' '.join(cmd)}\n{r.stdout}\n{r.stderr}"
    rep = json.load(open(os.path.join(out, "report.json")))
    rep["_stdout"], rep["_stderr"], rep["_dir"] = r.stdout, r.stderr, out
    return rep


def truth_for(p, report, margin_src=3) -> np.ndarray:
    """The ideal plate image at output scale, with a margin for the shift search."""
    t_index = report["reference_time"] * p["fps"]
    hi = gen.render(p["width"], p["height"], t_index, p["plate"], None,
                    mo_of(p), p["plate_px"])
    S = gen.S
    wx, wy = report["window"][0], report["window"][1]
    rx, ry, rw, rh = report["roi_out"]
    x0, y0 = wx + rx - margin_src, wy + ry - margin_src
    w, h = rw + 2 * margin_src, rh + 2 * margin_src
    crop = hi[int(y0 * S):int((y0 + h) * S), int(x0 * S):int((x0 + w) * S)]
    scale = report["scale"]
    crop = cv2.resize(crop, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def match_score(img_path: str, truth: np.ndarray, sub: int = 4) -> float:
    """Best normalised correlation against the ground truth.

    Both images are upsampled by `sub` with the same kernel before matching, so
    the search resolves 1/sub-pixel offsets. Without that, the reference frame's
    own camera shake leaves a random sub-pixel residual and the score wobbles by
    ~0.01 for reasons that have nothing to do with the reconstruction.
    """
    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    assert im is not None, f"cannot read {img_path}"
    tmpl = im.astype(np.float32) / 255.0
    if tmpl.shape[0] > truth.shape[0] or tmpl.shape[1] > truth.shape[1]:
        tmpl = tmpl[:truth.shape[0], :truth.shape[1]]
    tu = cv2.resize(truth, None, fx=sub, fy=sub, interpolation=cv2.INTER_CUBIC)
    mu = cv2.resize(tmpl, None, fx=sub, fy=sub, interpolation=cv2.INTER_CUBIC)
    return float(cv2.matchTemplate(tu, mu, cv2.TM_CCOEFF_NORMED).max())


def scores(p, report) -> tuple[float, float, float]:
    truth = truth_for(p, report)
    out = report["_dir"]
    tag = f"x{report['scale']:g}"
    base = match_score(os.path.join(out, f"02_baseline_bicubic_{tag}.png"), truth)
    fused = match_score(os.path.join(out, f"03_fused_{tag}.png"), truth)
    final = match_score(os.path.join(out, f"05_result_{tag}.png"), truth)
    return base, fused, final


# ---------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------


@test("units/time-and-span-parsing")
def t_spans():
    eq(ps.parse_time("12.5"), 12.5)
    eq(ps.parse_time("1:02.5"), 62.5)
    eq(ps.parse_time("00:01:02.5"), 62.5)
    eq(ps.parse_span("10:00-15:00", 999), (600.0, 900.0))
    eq(ps.parse_span("10:00+2:00", 999), (600.0, 720.0))
    eq(ps.parse_span("-30", 999), (0.0, 30.0))
    eq(ps.parse_span("2:00-", 900), (120.0, 900.0))
    eq(ps.merge_spans([(0, 5), (4, 8), (20, 22)]), [(0, 8), (20, 22)])
    # "take these five minutes, ignore two of them in the middle"
    eq(ps.subtract_spans([(600, 900)], [(700, 820)]), [(600, 700), (820, 900)])
    eq(ps.subtract_spans([(0, 10)], [(0, 10)]), [])
    eq(ps.subtract_spans([(0, 10)], [(20, 30)]), [(0, 10)])
    eq(ps.spans_total([(0, 10), (20, 25)]), 15.0)
    eq(ps.parse_index_list("3-5,9"), {3, 4, 5, 9})
    eq(ps.parse_rect("10,20,30,40"), (10, 20, 30, 40))


@test("units/warp-round-trip")
def t_warp():
    """LR->HR and HR->LR must be exact inverses, or back-projection diverges."""
    rng = np.random.default_rng(0)
    lr = rng.random((40, 60), np.float32)
    W = np.array([[1, 0, 2.37], [0, 1, -1.15]], np.float32)   # ref -> frame
    T = ps.hr_transform((5, 4, 20, 12), 4.0)
    hr = ps.warp_lr_to_hr(lr, W, T, (80, 48))
    back = ps.warp_hr_to_lr(hr, W, T, (60, 40))
    # the HR grid only covers the ROI (window x 5..25, y 4..16) shifted by W,
    # so score strictly inside that overlap
    m = np.zeros_like(lr, bool)
    m[7:14, 9:21] = True
    r = ps.ncc(lr, back, m)
    assert back[30:, :].max() == 0, "warp wrote outside the covered region"
    ge(r, 0.97, "round-trip correlation")


@test("units/ncc-and-fusion")
def t_fuse():
    rng = np.random.default_rng(1)
    a = rng.random((20, 20), np.float32)
    m = np.ones_like(a, bool)
    ge(ps.ncc(a, a, m), 0.999, "self correlation")
    assert abs(ps.ncc(a, a * 0.5 + 0.2, m) - 1.0) < 1e-3, "ncc must ignore gain/offset"
    # a trimmed mean has to survive a few wildly wrong frames
    truth = np.full((1, 8, 8, 1), 0.5, np.float32)
    stack = np.repeat(truth, 12, axis=0)
    stack[3] = 1.0
    stack[7] = 0.0
    out, cov = ps.fuse(stack, np.ones(12, np.float32), "trimmed", 0.25)
    assert abs(float(out.mean()) - 0.5) < 0.02, f"trimmed mean broke: {out.mean()}"
    eq(float(cov.min()), 12.0, "coverage")


@test("units/selection-json-round-trip")
def t_specjson():
    spec = ps.DecodeSpec(spans=[(1.5, 3.0), (10.0, 12.0)], window=(1, 2, 30, 40),
                         deint="field", step=2, max_frames=99)
    back = ps.DecodeSpec.from_json(json.loads(json.dumps(spec.to_json())))
    eq(back.spans, [(1.5, 3.0), (10.0, 12.0)])
    eq(back.window, (1, 2, 30, 40))
    eq(back.deint, "field")
    eq(back.max_frames, 99)


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------


@test("e2e/odd-sized-window-never-shears")
def t_odd_crop(fresh=False):
    """ffmpeg's crop filter rounds odd sizes down to even on chroma-subsampled
    formats. Reshaping the pipe at the *requested* width then slips every row by
    a pixel and shears the picture into a parallelogram - silently, with no
    error anywhere. Decoded frames must match the same region of a full-frame
    decode no matter what size was asked for."""
    p = build_clip("static", fresh)
    info = ps.probe(p["path"])
    full, _ = ps.read_frames(info, ps.DecodeSpec(spans=[(0.0, 0.2)], max_frames=1))
    ref_frame = full[0]
    worst_mad, worst_shift, checked = 0.0, 0.0, 0
    for win in [(100, 74, 64, 48),      # even, flat wall
                (101, 75, 63, 47),      # odd size and odd offset, flat wall
                (452, 372, 55, 19),     # odd size, over the plate (textured)
                (453, 373, 55, 19),     # odd size and offset, over the plate
                (51, 51, 33, 21)]:      # odd everything
        got, _ = ps.read_frames(info, ps.DecodeSpec(spans=[(0.0, 0.2)], window=win,
                                                    max_frames=1))
        crop = got[0]
        h, w = crop.shape[:2]
        x, y = win[0], win[1]
        assert (w, h) == (win[2], win[3]), f"{win}: got {w}x{h}, asked {win[2]}x{win[3]}"
        ref = ref_frame[y:y + h, x:x + w]
        g_ref, g_got = ps.to_gray32(ref), ps.to_gray32(crop)

        # A shear puts row k off by k pixels, which wrecks both of these.
        mad = float(np.abs(g_ref - g_got).mean())
        dx, dy, _ = ps.phase_shift(ps.prep_for_align(g_ref), ps.prep_for_align(g_got))
        assert mad < 0.02, f"window {win} decoded wrong (mean abs diff {mad:.4f})"
        assert abs(dx) < 0.5 and abs(dy) < 0.5, f"window {win} offset by ({dx:.2f},{dy:.2f})"

        # correlation is only meaningful where there is something to correlate;
        # on a flat wall it measures chroma-resampling noise, not geometry
        if float(g_ref.std()) > 0.02:
            r = ps.ncc(g_ref, g_got, np.ones(g_ref.shape, bool))
            assert r > 0.98, f"window {win} decoded sheared (ncc {r:.3f})"
            checked += 1
        worst_mad = max(worst_mad, mad)
        worst_shift = max(worst_shift, abs(dx), abs(dy))
    assert checked >= 2, "test bug: no textured window exercised"
    print(f"      5 windows incl. odd sizes/offsets: worst diff {worst_mad:.4f}, "
          f"worst shift {worst_shift:.2f}px")


@test("e2e/odd-roi-reconstructs-correctly")
def t_odd_roi(fresh=False):
    """The same failure end to end: an odd-sized ROI must still beat bicubic."""
    p = build_clip("static", fresh)
    x, y, w, h = roi_of(p)
    odd = (x + 1, y + 1, w - 1, h - 1)          # force odd width and height
    assert odd[2] % 2 == 1 and odd[3] % 2 == 1, "test bug: roi is not odd"
    rep = run_sr(p, "--roi", ",".join(map(str, odd)), "--preset", "static",
                 "--scale", 4, "--use", "0-6", "--max-frames", 90, "--ibp", 6,
                 "--no-variants", out=os.path.join(WORK, "out_oddroi"))
    base, fused, final = scores(p, rep)
    print(f"      odd roi {odd}: bicubic {base:.3f} -> result {final:.3f}")
    ge(final, base + 0.008, "odd ROI must reconstruct as well as an even one")


@test("e2e/static-parked-car")
def t_static(fresh=False):
    p = build_clip("static", fresh)
    roi = roi_of(p)
    rep = run_sr(p, "--roi", ",".join(map(str, roi)), "--preset", "static",
                 "--scale", 4, "--use", "0-8", "--max-frames", 120, "--ibp", 8)
    base, fused, final = scores(p, rep)
    print(f"      bicubic {base:.3f} -> fused {fused:.3f} -> result {final:.3f}"
          f"   ({rep['frames_used']}/{rep['frames_decoded']} frames)")
    ge(final, base + 0.010, "result must beat bicubic")
    ge(final, 0.86, "absolute quality")
    ge(rep["frames_used"] / rep["frames_decoded"], 0.4, "kept fraction")


@test("e2e/static-rejects-occlusion")
def t_reject(fresh=False):
    """The pedestrian frames must be thrown out without being told about them."""
    p = build_clip("static", fresh)
    rep = run_sr(p, "--roi", ",".join(map(str, roi_of(p))), "--preset", "static",
                 "--scale", 4, "--use", "0-8", "--ref", 0, "--max-frames", 200,
                 "--ibp", 4, "--no-variants")
    a, b = (int(v) for v in SCENARIOS["static"]["occlude"].split("-"))
    fps, mo, ppx = p["fps"], mo_of(p), p["plate_px"]

    def cover(f):
        """How much of the plate the walker hides in this frame (0 when away)."""
        i = f["t"] * fps
        if not (a <= i <= b):
            return 0.0
        return gen.plate_occlusion(p["width"], p["height"], i, mo, ppx,
                                   (i - a) / float(b - a))

    hidden = [f for f in rep["frames"] if cover(f) > 0.25]   # really covered
    clear = [f for f in rep["frames"] if cover(f) == 0.0]    # walker not on the plate
    assert hidden and clear, "test bug: no frames in one of the groups"
    dropped = sum(1 for f in hidden if not f["used"])
    lost = sum(1 for f in clear if not f["used"])
    print(f"      hidden {dropped}/{len(hidden)} dropped, "
          f"clear frames lost {lost}/{len(clear)}")
    ge(dropped / len(hidden), 0.85, "covered frames must be rejected")
    assert lost / len(clear) < 0.15, f"good frames thrown away: {lost}/{len(clear)}"


@test("e2e/moving-car")
def t_moving(fresh=False):
    p = build_clip("moving", fresh)
    rep = run_sr(p, "--roi", ",".join(map(str, roi_of(p, 0))), "--ref", 0,
                 "--preset", "moving", "--scale", 4, "--use", "0-3",
                 "--max-frames", 75, "--ibp", 10)
    base, fused, final = scores(p, rep)
    print(f"      bicubic {base:.3f} -> fused {fused:.3f} -> result {final:.3f}"
          f"   ({rep['frames_used']}/{rep['frames_decoded']} frames, "
          f"phase cov {rep['phase_coverage']*100:.0f}%)")
    ge(final, base + 0.008, "result must beat bicubic")
    ge(final, 0.88, "absolute quality")


@test("e2e/rigid-mount-no-jitter")
def t_rigid(fresh=False):
    """Zero camera shake: no sub-pixel diversity to exploit. The tool must still
    produce a sane image (noise averaging + deconvolution) and say so honestly."""
    p = build_clip("rigid", fresh)
    rep = run_sr(p, "--roi", ",".join(map(str, roi_of(p))), "--preset", "static",
                 "--scale", 4, "--use", "0-4", "--max-frames", 80, "--ibp", 6,
                 "--no-variants")
    base, fused, final = scores(p, rep)
    print(f"      bicubic {base:.3f} -> fused {fused:.3f} -> result {final:.3f}"
          f"   (phase cov {rep['phase_coverage']*100:.0f}%)")
    ge(final, base, "must not be worse than bicubic")
    ge(final, 0.86, "absolute quality")


@test("e2e/time-spans-are-honoured")
def t_skip(fresh=False):
    """--use / --skip must decide which seconds are read at all."""
    p = build_clip("static", fresh)
    rep = run_sr(p, "--roi", ",".join(map(str, roi_of(p))), "--use", "0-8",
                 "--skip", "2.4-3.8", "--scale", 3, "--ibp", 2, "--no-variants",
                 "--max-frames", 60)
    ts = [f["t"] for f in rep["frames"]]
    # the boundaries themselves are kept: the cut is the open interval
    inside = [t for t in ts if 2.4 < t < 3.8]
    eq(inside, [], "frames from a skipped span")
    assert min(ts) < 2.4 < 3.8 < max(ts), "the span should be cut out of the middle"
    eq([list(map(float, s)) for s in rep["decode"]["spans"]], [[0.0, 2.4], [3.8, 8.0]],
       "decode spans")
    print(f"      decoded {len(ts)} frames, none in 2.4-3.8s")


@test("e2e/frame-budget-spreads-over-selection")
def t_budget(fresh=False):
    p = build_clip("static", fresh)
    rep = run_sr(p, "--roi", ",".join(map(str, roi_of(p))), "--use", "0-8",
                 "--max-frames", 20, "--scale", 2, "--ibp", 1, "--no-variants")
    ts = sorted(f["t"] for f in rep["frames"])
    assert len(ts) <= 20, f"budget ignored: {len(ts)} frames"
    assert ts[-1] - ts[0] > 6.0, f"frames bunched at the front: {ts[0]:.2f}-{ts[-1]:.2f}s"
    print(f"      {len(ts)} frames spread over {ts[-1]-ts[0]:.1f}s")


@test("e2e/cli-info-grid-select")
def t_cli(fresh=False):
    p = build_clip("static", fresh)
    r = subprocess.run([PY, *TOOL, "info", p["path"]], capture_output=True, text=True, env=ENV)
    eq(r.returncode, 0, "info")
    assert "960x540" in r.stdout and "25.0000" in r.stdout, r.stdout

    sheet = os.path.join(WORK, "sheet.png")
    r = subprocess.run([PY, *TOOL, "grid", p["path"], "--start", "0", "--end", "4",
                        "--count", "4", "--cols", "2", "--out", sheet],
                       capture_output=True, text=True, env=ENV)
    eq(r.returncode, 0, f"grid: {r.stderr}")
    assert cv2.imread(sheet) is not None, "contact sheet unreadable"

    work = os.path.join(WORK, "sel")
    shutil.rmtree(work, ignore_errors=True)
    r = subprocess.run([PY, *TOOL, "select", p["path"], "--start", "0", "--dur", "2",
                        "--max-frames", "12", "--out", work],
                       capture_output=True, text=True, env=ENV)
    eq(r.returncode, 0, f"select: {r.stderr}")
    page = os.path.join(work, "select.html")
    html = open(page).read()
    assert "data:image/jpeg;base64," in html, "frames not embedded"
    assert '<script id="payload"' in html and "__DATA__" not in html, "payload not filled"
    data = json.loads(html.split('type="application/json">')[1].split("</script>")[0])
    nf = len(data["frames"])       # budget 12 over 50 frames -> stride 5 -> 10 kept
    assert 8 <= nf <= 12, "embedded frames: %d" % nf
    assert data["decode"]["spans"], "decode spans missing from the page"
    print(f"      select.html {os.path.getsize(page)/1e6:.1f} MB, {len(data['frames'])} frames")


@test("ui/picker-javascript-parses")
def t_js():
    """The picker UI can only be exercised in a browser, so at least prove the
    generated page is syntactically valid JS and carries the data it needs."""
    if not shutil.which("node"):
        print("      skipped (no node)")
        return
    p = build_clip("static")
    work = os.path.join(WORK, "sel_js")
    shutil.rmtree(work, ignore_errors=True)
    r = subprocess.run([PY, *TOOL, "select", p["path"], "--start", "0", "--dur", "1",
                        "--max-frames", "6", "--out", work], capture_output=True, text=True, env=ENV)
    eq(r.returncode, 0, f"select: {r.stderr}")
    html = open(os.path.join(work, "select.html")).read()
    body = html.split("</script>\n<script>")[-1].split("</script>")[0]
    assert "function render()" in body, "main script not found"
    stub = ("var document={getElementById:function(){return null},"
            "createElement:function(){return {getContext:function(){return {}},"
            "addEventListener:function(){},style:{},classList:{}}},"
            "documentElement:{style:{setProperty:function(){}}}};\n")
    js = os.path.join(WORK, "picker_check.js")
    open(js, "w").write(stub + body)
    chk = subprocess.run(["node", "--check", js], capture_output=True, text=True, env=ENV)
    eq(chk.returncode, 0, f"picker JS syntax error:\n{chk.stderr}")
    for hook in ("skip_spans", "exclude_times", "ref_time", "drawTimeline", "keepOf"):
        assert hook in body, f"picker lost {hook}"
    print("      picker JS parses, timeline + span export present")


@test("ui/selection-builds-a-valid-command")
def t_ui_args():
    """The UI has no second code path: it builds a CLI argument list and hands
    it to vidsr's own parser. That contract is what makes the headless and
    windowed workflows equivalent, so test it without needing a display."""
    import vidsr_ui as ui

    sel = ui.Selection(video="cam.mkv", roi=(812, 430, 96, 34), start=600.0,
                       end=900.0, skips=[(750.0, 870.0)], preset="static",
                       scale=4, max_frames=250, out="out")
    args = ui.build_args(sel)
    assert args[:2] == ["sr", "cam.mkv"], args
    for flag, val in (("--roi", "812,430,96,34"), ("--use", "600.000-900.000"),
                      ("--skip", "750.000-870.000"), ("--preset", "static"),
                      ("--scale", "4"), ("--max-frames", "250")):
        assert flag in args and args[args.index(flag) + 1] == val, f"{flag} wrong in {args}"

    # vidsr's own parser must accept it, and read back what the UI meant
    parser = ps.build_parser()
    a = parser.parse_args(args)
    ps.apply_preset(a, a._sr_parser)
    eq(a.roi, "812,430,96,34", "roi")
    eq(a.preset, "static", "preset")
    eq(a.motion, "translation", "preset must reach the reconstruction settings")
    eq(ps.parse_spans(a.use, 1e9), [(600.0, 900.0)], "use span")
    eq(ps.subtract_spans(ps.parse_spans(a.use, 1e9), ps.parse_spans(a.skip, 1e9)),
       [(600.0, 750.0), (870.0, 900.0)], "skip must cut the middle out")
    eq(sel.spans(), [(600.0, 750.0), (870.0, 900.0)], "Selection.spans")

    # and it must refuse a selection that cannot run, rather than build nonsense
    for bad, why in ((ui.Selection(video="c.mkv", start=0, end=5), "no roi"),
                     (ui.Selection(video="c.mkv", roi=(1, 2, 3, 4), start=5, end=5), "empty range"),
                     (ui.Selection(roi=(1, 2, 3, 4), start=0, end=5), "no video")):
        try:
            ui.build_args(bad)
            raise AssertionError(f"accepted a selection with {why}")
        except ValueError:
            pass
    print("      builds and round-trips through the real parser")


@test("ui/coordinate-mapping")
def t_ui_geom():
    """Canvas-to-source mapping decides which pixels get reconstructed. The
    window cannot be driven headlessly, so at least pin the maths."""
    import vidsr_ui as ui

    disp = ui.fit_view(1920, 1080, 780, 440)
    s, ox, oy = disp
    # 780/1920 < 440/1080, so width is the binding constraint here
    assert abs(s - 780 / 1920) < 1e-9, s
    assert ox == 0 and oy >= 0, (ox, oy)          # letterboxed top and bottom

    for pt in [(0, 0), (1919, 1079), (960, 540), (17, 933)]:
        vx, vy = ui.source_to_view(*pt, disp)
        bx, by = ui.view_to_source(vx, vy, disp)
        assert abs(bx - pt[0]) < 0.01 and abs(by - pt[1]) < 0.01, (pt, bx, by)

    # the corners of the displayed image map to the corners of the frame
    x0, y0 = ui.view_to_source(ox, oy, disp)
    assert abs(x0) < 0.01 and abs(y0) < 0.01, (x0, y0)

    # a drag maps to the same rect whichever way it is pulled
    a, b = (100.0, 50.0), (140.0, 80.0)
    eq(ui.rect_from_drag(a, b), (100, 50, 40, 30), "drag rect")
    eq(ui.rect_from_drag(b, a), (100, 50, 40, 30), "reversed drag must match")
    assert ui.rect_from_drag((10, 10), (10, 10))[2] >= 2, "degenerate drag must stay usable"

    # a portrait video letterboxes the other way
    s2, ox2, oy2 = ui.fit_view(1080, 1920, 780, 440)
    assert ox2 > 0 and oy2 == 0 and s2 < 1, (s2, ox2, oy2)

    for dur, width in ((300.0, 800), (7.5, 640)):
        for t in (0.0, dur / 3, dur):
            assert abs(ui.x_to_time(ui.time_to_x(t, dur, width), dur, width) - t) < 1e-6
    eq(ui.x_to_time(-50, 10.0, 100), 0.0, "clamps left")
    eq(ui.x_to_time(9999, 10.0, 100), 10.0, "clamps right")
    print("      round-trips landscape, portrait and timeline mapping")


@test("ui/progress-reaches-one")
def t_progress(fresh=False):
    """The progress bar is only honest if the callback actually advances."""
    p = build_clip("static", fresh)
    seen = []
    ps.set_progress(lambda f, m: seen.append((f, m)))
    try:
        rep = run_sr(p, "--roi", ",".join(map(str, roi_of(p))), "--use", "0-2",
                     "--max-frames", 20, "--scale", 2, "--ibp", 2, "--no-variants",
                     out=os.path.join(WORK, "out_prog"))
    finally:
        ps.set_progress(None)
    # run_sr shells out, so this process sees nothing - the callback is only
    # meaningful in-process, which is how the UI drives it
    eq(seen, [], "a subprocess run must not call our callback")

    import vidsr_ui as ui
    seen.clear()
    sel = ui.Selection(video=p["path"], roi=roi_of(p), start=0.0, end=2.0,
                       preset="static", scale=2, max_frames=20,
                       out=os.path.join(WORK, "out_uirun"))
    rep = ui.run_selection(sel, lambda f, m: seen.append((f, m)))
    assert seen, "no progress reported"
    fracs = [f for f, _ in seen]
    assert fracs == sorted(fracs), f"progress went backwards: {fracs}"
    assert 0.0 <= fracs[0] and fracs[-1] == 1.0, f"progress ended at {fracs[-1]}"
    assert len(seen) >= 5, f"only {len(seen)} updates - the bar would jump"
    assert rep["frames_used"] >= 2, "the UI path must actually reconstruct"
    assert os.path.exists(rep["outputs"]["result"]), "no result written"
    # and the hook must be off again afterwards
    seen.clear()
    ps.progress(0.5, "should go nowhere")
    eq(seen, [], "callback outlived the run")
    print(f"      {len(fracs)} updates, {fracs[0]:.2f} -> {fracs[-1]:.2f}, "
          f"{rep['frames_used']} frames fused")


@test("ui/imports-without-a-display")
def t_ui_headless():
    """Importing the UI must not need Tk or a display: a headless box has to be
    able to run the tests and the CLI."""
    import subprocess as sp
    code = ("import os, sys; os.environ.pop('DISPLAY', None); "
            "sys.path.insert(0, %r); import vidsr_ui; "
            "print('ok', callable(vidsr_ui.build_args), callable(vidsr_ui.run_app))"
            % os.path.join(ROOT, "src"))
    r = sp.run([PY, "-c", code], capture_output=True, text=True, env=ENV)
    eq(r.returncode, 0, f"import failed headless:\n{r.stderr}")
    assert r.stdout.strip() == "ok True True", r.stdout
    print("      imports with no DISPLAY")


@test("e2e/selection-json-drives-sr")
def t_selection(fresh=False):
    """What the UI exports must reproduce end-to-end, spans and all."""
    p = build_clip("static", fresh)
    roi = roi_of(p)
    sel = {
        "version": ps.__version__, "video": p["path"],
        "decode": {"spans": [[0.0, 8.0]], "window": None, "deint": "none",
                   "step": 1, "max_frames": 60},
        "window": [0, 0, p["width"], p["height"]], "roi": list(roi),
        "ref_index": 0, "ref_time": 0.0, "fps": p["fps"],
        "skip_spans": [[2.4, 3.8]],
        "exclude_times": [5.0, 5.04], "exclude": [], "include": [],
    }
    path = os.path.join(WORK, "selection.json")
    json.dump(sel, open(path, "w"), indent=2)
    out = os.path.join(WORK, "out_sel")
    shutil.rmtree(out, ignore_errors=True)
    r = subprocess.run([PY, *TOOL, "sr", "--select", path, "--out", out,
                        "--scale", "3", "--ibp", "2", "--no-variants"],
                       capture_output=True, text=True, env=ENV)
    eq(r.returncode, 0, f"sr --select failed:\n{r.stderr}")
    rep = json.load(open(os.path.join(out, "report.json")))
    eq(rep["roi"], list(roi), "roi from selection")
    assert not [f["t"] for f in rep["frames"] if 2.4 < f["t"] < 3.8], "skip_spans ignored"
    dropped = [f["t"] for f in rep["frames"] if not f["used"]]
    assert any(abs(t - 5.0) < 0.05 for t in dropped), "exclude_times ignored"
    assert os.path.exists(os.path.join(out, "review.html")), "no review page"
    print(f"      {rep['frames_used']}/{rep['frames_decoded']} frames used via selection.json")


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", "--filter", help="only run tests whose name contains this")
    ap.add_argument("--fresh", action="store_true", help="re-encode the test clips")
    ap.add_argument("--keep", action="store_true", help="keep the work directory")
    ap.add_argument("-l", "--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for name, _ in TESTS:
            print(name)
        return 0

    picked = [(n, f) for n, f in TESTS if not a.filter or a.filter in n]
    if not picked:
        print(f"no test matches {a.filter!r}")
        return 2
    os.makedirs(WORK, exist_ok=True)

    fails, t0 = [], time.time()
    for name, fn in picked:
        print(f"  {name} ...", flush=True)
        t1 = time.time()
        try:
            fn(a.fresh) if fn.__code__.co_argcount else fn()
            print(f"      PASS  ({time.time()-t1:.1f}s)")
        except Exception as e:
            fails.append((name, e))
            print(f"      FAIL  {e}")
            if os.environ.get("VIDSR_TRACE"):
                traceback.print_exc()
    dt = time.time() - t0
    print(f"\n{len(picked)-len(fails)}/{len(picked)} passed in {dt:.1f}s")
    for name, e in fails:
        print(f"  FAILED {name}: {e}")
    if not a.keep and not fails:
        for d in os.listdir(WORK):
            if d.startswith("out_") or d == "sel":
                shutil.rmtree(os.path.join(WORK, d), ignore_errors=True)
    else:
        print(f"\nwork dir: {WORK}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
