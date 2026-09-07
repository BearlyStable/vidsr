#!/usr/bin/env python3
"""
vidsr - multi-frame super-resolution for video.

Recovers real detail from a small region of a mostly static scene - a licence
plate, a street sign, a badge, a serial number, a face that holds still - by
fusing many frames of it at sub-pixel offsets. Nothing is invented: every output
pixel is a weighted measurement of actual sensor samples, aligned and
deconvolved. No generative model is involved, so the result stays defensible as
evidence.

The subject only has to be rigid and roughly planar for the frames to stack;
nothing in the pipeline knows or cares what it depicts.

Pipeline:
  decode (ffmpeg) -> pick window/ROI -> sub-pixel align (phase corr + ECC)
  -> outlier rejection -> HR fusion (robust stack) -> iterative back-projection
  -> deconvolution -> post (CLAHE/denoise/rectify)

Commands: info | grid | select | sr
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time as _time
from dataclasses import dataclass, field, asdict
from typing import Iterator, Optional, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("OpenCV missing: pip install 'numpy>=1.24' 'opencv-python-headless>=4.8'")

PROG = "vidsr"
__version__ = "0.1.0"

# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------


def die(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    sys.exit(f"error: {msg}")


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


_PROGRESS = None


def set_progress(cb) -> None:
    """Install a callback(fraction, message) so a UI can follow a run.

    The CLI leaves this unset and just logs; nothing in the pipeline changes
    behaviour depending on whether anyone is listening.
    """
    global _PROGRESS
    _PROGRESS = cb


def progress(frac: float, msg: str) -> None:
    if _PROGRESS is None:
        return
    try:
        _PROGRESS(max(0.0, min(1.0, float(frac))), msg)
    except Exception:
        pass          # a broken UI must never take the reconstruction down


def which_or_die(prog: str) -> str:
    p = shutil.which(prog)
    if not p:
        die(f"{prog} not found on PATH (apt install ffmpeg)")
    return p


def parse_time(s) -> Optional[float]:
    """Accept 12, 12.5, 1:02.5 or 00:01:02.500 and return seconds."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        pass
    die(f"cannot parse time {s!r} (use 12.5, 1:02.5 or 00:01:02.5)")


def fmt_time(t: float) -> str:
    if t is None:
        return "-"
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:06.3f}" if h else f"{m:d}:{s:06.3f}"


def parse_span(s: str, duration: float) -> tuple[float, float]:
    """'10:00-15:00', '600-900', '10:00+2:00', '-30', '2:00-' -> (start, end) seconds."""
    s = str(s).strip()
    if not s:
        die("empty time span")
    if "+" in s:
        a, b = s.split("+", 1)
        st = parse_time(a) if a.strip() else 0.0
        return st, st + parse_time(b)
    # a bare '-' separates start and end; leading '-' means "from the beginning"
    m = re.match(r"^(.*?)-(.*)$", s)
    if not m:
        st = parse_time(s)
        return st, duration
    a, b = m.group(1).strip(), m.group(2).strip()
    st = parse_time(a) if a else 0.0
    en = parse_time(b) if b else duration
    if en <= st:
        die(f"span {s!r}: end must be after start")
    return st, en


def parse_spans(values, duration: float) -> list[tuple[float, float]]:
    """Flatten repeated flags and comma lists into merged, sorted spans."""
    out: list[tuple[float, float]] = []
    for v in (values or []):
        for chunk in str(v).split(","):
            if chunk.strip():
                out.append(parse_span(chunk, duration))
    return merge_spans(out)


def merge_spans(spans) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for st, en in sorted(spans):
        if out and st <= out[-1][1] + 1e-9:
            out[-1] = (out[-1][0], max(out[-1][1], en))
        else:
            out.append((st, en))
    return out


def subtract_spans(base, cuts) -> list[tuple[float, float]]:
    """Remove every cut interval from the base intervals."""
    out = list(base)
    for cs, ce in merge_spans(cuts):
        nxt: list[tuple[float, float]] = []
        for bs, be in out:
            if ce <= bs or cs >= be:
                nxt.append((bs, be))
                continue
            if bs < cs:
                nxt.append((bs, cs))
            if ce < be:
                nxt.append((ce, be))
        out = nxt
    return [(a, b) for a, b in out if b - a > 1e-6]


def spans_total(spans) -> float:
    return float(sum(b - a for a, b in spans))


def fmt_spans(spans) -> str:
    return ", ".join(f"{fmt_time(a)}-{fmt_time(b)}" for a, b in spans) or "-"


def in_spans(t: float, spans) -> bool:
    return any(a <= t <= b for a, b in spans)


def parse_rect(s: str) -> tuple[int, int, int, int]:
    """'x,y,w,h' -> ints."""
    m = re.findall(r"-?\d+", str(s))
    if len(m) != 4:
        die(f"bad rect {s!r}; expected x,y,w,h")
    x, y, w, h = (int(v) for v in m)
    if w <= 0 or h <= 0:
        die(f"bad rect {s!r}: width/height must be positive")
    return x, y, w, h


def parse_index_list(s: Optional[str]) -> set[int]:
    """'3-9,15,20-22' -> {3..9,15,20..22}"""
    out: set[int] = set()
    if not s:
        return out
    for chunk in str(s).replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk[1:]:
            a, b = chunk.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(chunk))
    return out


def clamp_rect(r, W: int, H: int):
    x, y, w, h = r
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


def even_rect(r, W: int, H: int):
    """Snap a crop rect to even offsets and even sizes.

    ffmpeg's crop filter silently rounds odd sizes *down* on chroma-subsampled
    pixel formats (yuv420p and friends): ask for 63x47 and you get 62x46, with
    no warning. Reshaping the raw pipe at the requested width then slips every
    row by a pixel and shears the whole picture into a parallelogram.
    """
    x, y, w, h = clamp_rect(r, W, H)
    x -= x % 2
    y -= y % 2
    w += w % 2
    h += h % 2
    if x + w > W:                      # growing ran off the edge: shrink back
        w = (W - x) - ((W - x) % 2)
    if y + h > H:
        h = (H - y) - ((H - y) % 2)
    return (x, y, max(2, w), max(2, h))


def expand_rect(r, pad_x: int, pad_y: int, W: int, H: int):
    x, y, w, h = r
    return clamp_rect((x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y), W, H)


def even(v: int) -> int:
    return int(v) - (int(v) % 2)


# --------------------------------------------------------------------------
# ffprobe / ffmpeg
# --------------------------------------------------------------------------


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration: float
    codec: str
    nb_frames: Optional[int]
    field_order: str
    pix_fmt: str

    @property
    def interlaced(self) -> bool:
        return self.field_order not in ("progressive", "unknown", "")


def probe(path: str) -> VideoInfo:
    which_or_die("ffprobe")
    if not os.path.exists(path):
        die(f"no such file: {path}")
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,codec_name,nb_frames,field_order,pix_fmt:format=duration",
        "-of", "json", path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        die(f"ffprobe failed: {out.stderr.strip()}")
    data = json.loads(out.stdout)
    if not data.get("streams"):
        die("no video stream found")
    st = data["streams"][0]

    def _rate(v):
        try:
            n, d = str(v).split("/")
            return float(n) / float(d) if float(d) else 0.0
        except Exception:
            return 0.0

    fps = _rate(st.get("avg_frame_rate")) or _rate(st.get("r_frame_rate")) or 25.0
    dur = float(data.get("format", {}).get("duration") or 0.0)
    nb = st.get("nb_frames")
    return VideoInfo(
        path=path,
        width=int(st["width"]),
        height=int(st["height"]),
        fps=fps,
        duration=dur,
        codec=st.get("codec_name", "?"),
        nb_frames=int(nb) if nb and str(nb).isdigit() else None,
        field_order=st.get("field_order", "unknown"),
        pix_fmt=st.get("pix_fmt", "?"),
    )


_CROP_EXACT: Optional[bool] = None


def crop_supports_exact() -> bool:
    """Whether this ffmpeg's crop filter has `exact`.

    Without it, crop snaps odd sizes *and* odd offsets down to even on
    subsampled formats, so the ROI you asked for is not the ROI you get.
    """
    global _CROP_EXACT
    if _CROP_EXACT is None:
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-h", "filter=crop"],
                                 capture_output=True, text=True, timeout=20)
            _CROP_EXACT = " exact " in out.stdout
        except Exception:
            _CROP_EXACT = False
    return _CROP_EXACT


DEINT_FILTERS = {
    "none": None,
    # one output frame per input frame
    "frame": "bwdif=mode=send_frame:parity=auto:deint=all",
    # one output frame per FIELD: doubles the number of time samples, which is
    # exactly what multi-frame SR wants on interlaced cameras
    "field": "bwdif=mode=send_field:parity=auto:deint=all",
}


@dataclass
class DecodeSpec:
    """Everything needed to reproduce the exact same frame sequence twice.

    `spans` are (start, end) second pairs: "use these five minutes, ignore the
    rest" is just a span list, and ignoring a section is span subtraction.
    """
    spans: list = field(default_factory=list)
    window: Optional[tuple[int, int, int, int]] = None  # x,y,w,h in source pixels
    deint: str = "none"
    step: int = 1
    max_frames: Optional[int] = None

    @property
    def total(self) -> float:
        return spans_total(self.spans)

    @property
    def t0(self) -> float:
        return self.spans[0][0] if self.spans else 0.0

    def to_json(self) -> dict:
        return {"spans": [[float(a), float(b)] for a, b in self.spans],
                "window": list(self.window) if self.window else None,
                "deint": self.deint, "step": self.step, "max_frames": self.max_frames}

    @staticmethod
    def from_json(d: dict) -> "DecodeSpec":
        d = dict(d or {})
        w = d.get("window")
        return DecodeSpec(spans=[tuple(x) for x in (d.get("spans") or [])],
                          window=tuple(w) if w else None,
                          deint=d.get("deint", "none"), step=int(d.get("step") or 1),
                          max_frames=d.get("max_frames"))


def ffmpeg_cmd(info: VideoInfo, spec: DecodeSpec, span: tuple[float, float],
               still: bool = False) -> list[str]:
    which_or_die("ffmpeg")
    start, end = span
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", info.path]
    if end and end > start:
        cmd += ["-t", f"{end - start:.6f}"]
    cmd += ["-an", "-sn", "-dn"]
    vf = []
    df = DEINT_FILTERS.get(spec.deint)
    if df:
        vf.append(df)
    if spec.window:
        x, y, w, h = spec.window
        exact = ":exact=1" if crop_supports_exact() else ""
        vf.append(f"crop={w}:{h}:{x}:{y}{exact}")
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if still:
        # one self-describing frame, so its real dimensions can be read back
        cmd += ["-frames:v", "1", "-f", "image2pipe", "-c:v", "bmp", "-"]
    else:
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    return cmd


_SIZE_CACHE: dict = {}


def probe_filtered_size(info: VideoInfo, spec: DecodeSpec) -> tuple[int, int]:
    """The (width, height) the filter chain really emits.

    Never assume it matches the request: crop rounding on subsampled formats,
    rotation metadata and anamorphic SAR all change it, and guessing wrong
    shears every frame silently. One extra decoded frame is cheap insurance.
    """
    key = (info.path, spec.deint, spec.window)
    if key in _SIZE_CACHE:
        return _SIZE_CACHE[key]
    want = (spec.window[2], spec.window[3]) if spec.window else (info.width, info.height)
    span = spec.spans[0] if spec.spans else (0.0, 1.0)
    size = want
    try:
        out = subprocess.run(ffmpeg_cmd(info, spec, span, still=True),
                             capture_output=True, timeout=60).stdout
        img = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
        if img is not None and img.size:
            size = (int(img.shape[1]), int(img.shape[0]))
    except Exception:
        pass
    _SIZE_CACHE[key] = size
    return size


def effective_fps(info: VideoInfo, spec: DecodeSpec) -> float:
    return info.fps * (2.0 if spec.deint == "field" else 1.0)


def plan_stride(info: VideoInfo, spec: DecodeSpec) -> int:
    """Spread the frame budget over the whole selection instead of taking a
    burst from the front - for a static subject, frames far apart in time are
    worth more than consecutive ones."""
    fps = effective_fps(info, spec)
    total = spec.total * fps
    stride = max(1, int(spec.step or 1))
    if spec.max_frames and total / stride > spec.max_frames:
        stride = int(math.ceil(total / spec.max_frames))
    return stride


def decode(info: VideoInfo, spec: DecodeSpec) -> Iterator[tuple[int, float, np.ndarray]]:
    """Stream (index, absolute timestamp, BGR frame) across every span."""
    want = (spec.window[2], spec.window[3]) if spec.window else (info.width, info.height)
    w, h = probe_filtered_size(info, spec)
    if (w, h) != want:
        log(f"note     ffmpeg emits {w}x{h}, not the requested {want[0]}x{want[1]}; "
            f"using the real size (odd crops get rounded on subsampled formats)")
    fps = effective_fps(info, spec)
    nbytes = w * h * 3
    spans = spec.spans or [(0.0, info.duration or 0.0)]
    stride = plan_stride(info, spec)
    kept = 0
    raw = 0
    for span in spans:
        cmd = ffmpeg_cmd(info, spec, span)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                bufsize=nbytes * 2)
        local = 0
        try:
            while True:
                buf = proc.stdout.read(nbytes)
                if not buf or len(buf) < nbytes:
                    break
                if raw % stride == 0:
                    ts = span[0] + (local / fps if fps else 0.0)
                    yield kept, ts, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                    kept += 1
                    if spec.max_frames and kept >= spec.max_frames:
                        raw += 1
                        local += 1
                        break
                raw += 1
                local += 1
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.communicate(timeout=10)
            except Exception:
                proc.kill()
        if spec.max_frames and kept >= spec.max_frames:
            break
    if kept == 0:
        die("ffmpeg produced no frames - check the time range and --deint")


def read_frames(info: VideoInfo, spec: DecodeSpec) -> tuple[list[np.ndarray], list[float]]:
    frames, times = [], []
    for _, ts, f in decode(info, spec):
        frames.append(f.copy())
        times.append(ts)
    return frames, times


def grab_frame(info: VideoInfo, t: float) -> np.ndarray:
    """Single frame at time t (full resolution)."""
    spec = DecodeSpec(spans=[(max(0.0, t), max(0.0, t) + 1.0)], max_frames=1)
    for _, _, f in decode(info, spec):
        return f.copy()
    die("could not grab frame")


# --------------------------------------------------------------------------
# image helpers
# --------------------------------------------------------------------------


def to_gray32(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 3:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        g = bgr
    return g.astype(np.float32) / 255.0


def sharpness(gray32: np.ndarray) -> float:
    """Variance of Laplacian - higher is sharper."""
    return float(cv2.Laplacian(gray32, cv2.CV_32F).var())


def prep_for_align(gray32: np.ndarray, hp_sigma: float = 3.0) -> np.ndarray:
    """High-pass + contrast normalise so ECC ignores exposure/headlight swings."""
    lo = cv2.GaussianBlur(gray32, (0, 0), hp_sigma)
    hp = gray32 - lo
    s = float(hp.std())
    if s < 1e-6:
        return hp
    return hp / s


def eye23() -> np.ndarray:
    return np.eye(2, 3, dtype=np.float32)


def to33(W: np.ndarray) -> np.ndarray:
    if W.shape == (3, 3):
        return W.astype(np.float64)
    M = np.eye(3, dtype=np.float64)
    M[:2, :] = W.astype(np.float64)
    return M


def from33(M: np.ndarray, homography: bool) -> np.ndarray:
    return (M.astype(np.float32) if homography else M[:2, :].astype(np.float32))


MOTION_NAMES = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
    "homography": cv2.MOTION_HOMOGRAPHY,
}


def warp_to_ref(img: np.ndarray, W: np.ndarray, size: tuple[int, int],
                flags: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Resample `img` (moving frame) onto the reference grid of `size`=(w,h).

    Convention (matches cv2.findTransformECC): W maps *reference* coords to
    *moving-frame* coords, so WARP_INVERSE_MAP samples the moving frame at W*p.
    """
    f = flags | cv2.WARP_INVERSE_MAP
    if W.shape == (3, 3):
        return cv2.warpPerspective(img, W, size, flags=f,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.warpAffine(img, W, size, flags=f,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def phase_shift(ref: np.ndarray, mov: np.ndarray) -> tuple[float, float, float]:
    """Sub-pixel translation between two prepped images.

    Returns (dx, dy, response) where (dx,dy) is the offset to put into a warp
    matrix mapping reference coords -> moving coords.
    """
    h, w = ref.shape[:2]
    if min(h, w) < 8:
        return 0.0, 0.0, 0.0
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    a = np.ascontiguousarray(ref, dtype=np.float32)
    b = np.ascontiguousarray(mov, dtype=np.float32)
    try:
        (dx, dy), resp = cv2.phaseCorrelate(a.astype(np.float64), b.astype(np.float64), win.astype(np.float64))
    except cv2.error:
        return 0.0, 0.0, 0.0
    # cv2.phaseCorrelate(src1, src2) reports the shift of src2 relative to src1,
    # i.e. src2(p) ~ src1(p - d). A reference pixel p therefore lives at p + d in
    # the moving frame, which is exactly the translation column of our warp.
    return float(dx), float(dy), float(resp)


def ecc_refine(ref_prep: np.ndarray, mov_prep: np.ndarray, W0: np.ndarray,
               motion: int, mask: Optional[np.ndarray], iters: int,
               eps: float, gauss: int = 5) -> tuple[Optional[np.ndarray], float]:
    W = W0.copy()
    if motion == cv2.MOTION_HOMOGRAPHY and W.shape != (3, 3):
        W = to33(W).astype(np.float32)
    if motion != cv2.MOTION_HOMOGRAPHY and W.shape == (3, 3):
        W = W[:2, :].astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iters), float(eps))
    try:
        rho, W = cv2.findTransformECC(ref_prep, mov_prep, W, motion, crit, mask, gauss)
    except cv2.error:
        return None, -1.0
    if not np.all(np.isfinite(W)):
        return None, -1.0
    return W.astype(np.float32), float(rho)


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Zero-mean normalised cross-correlation inside `mask`: 1 = identical
    structure, 0 = unrelated. Unlike ECC's rho this is defined even when the
    optimiser fails, so it can be trusted as the accept/reject gate."""
    av, bv = a[mask].astype(np.float64), b[mask].astype(np.float64)
    if av.size < 16:
        return 0.0
    av = av - av.mean()
    bv = bv - bv.mean()
    d = math.sqrt(float((av * av).sum()) * float((bv * bv).sum()))
    return float((av * bv).sum() / d) if d > 1e-12 else 0.0


@dataclass
class AlignResult:
    index: int
    time: float
    warp: np.ndarray
    rho: float
    sharp: float
    ncc: float = 0.0
    residual: float = 0.0
    shift: tuple[float, float] = (0.0, 0.0)
    used: bool = True
    note: str = ""


def align_frame(ref_prep: np.ndarray, mov_prep: np.ndarray, mask: Optional[np.ndarray],
                motion: str, init: Optional[np.ndarray], iters: int, eps: float,
                pyramid: bool = True, margin: float = 0.002) -> tuple[Optional[np.ndarray], float]:
    """Estimate the warp mapping reference coords -> moving-frame coords."""
    h, w = ref_prep.shape[:2]

    # --- initial translation guess -------------------------------------
    cands: list[np.ndarray] = []
    if init is not None:
        cands.append(init.astype(np.float32).copy())
    dx, dy, resp = phase_shift(ref_prep, mov_prep)
    if abs(dx) < w and abs(dy) < h:
        W = eye23()
        W[0, 2] = dx
        W[1, 2] = dy
        cands.append(W)
    cands.append(eye23())

    # --- coarse translation on a pyramid, then progressive model upgrade
    stages = (["translation", "euclidean", "affine"] if motion == "auto" else [motion])
    if motion == "homography":
        stages = ["translation", "euclidean", "affine", "homography"]

    best_W, best_rho = None, -1.0
    for W0 in cands:
        W = W0.copy()
        rho = -1.0
        if pyramid and min(h, w) >= 160:
            small_ref = cv2.pyrDown(ref_prep)
            small_mov = cv2.pyrDown(mov_prep)
            small_mask = cv2.pyrDown(mask.astype(np.float32)) if mask is not None else None
            if small_mask is not None:
                small_mask = (small_mask > 0.25).astype(np.uint8)
            Wc = W[:2, :].copy() if W.shape == (3, 3) else W.copy()
            Wc[0, 2] /= 2.0
            Wc[1, 2] /= 2.0
            Wc2, r = ecc_refine(small_ref, small_mov, Wc, cv2.MOTION_TRANSLATION,
                                small_mask, max(30, iters // 3), eps * 4)
            if Wc2 is not None:
                Wc2[0, 2] *= 2.0
                Wc2[1, 2] *= 2.0
                W = Wc2
        for st in stages:
            W2, r = ecc_refine(ref_prep, mov_prep, W, MOTION_NAMES[st], mask, iters, eps)
            if W2 is None:
                break
            # a richer model always fits at least as well; on a static scene the
            # extra freedom just tracks noise, so demand a real improvement
            if rho > -1.0 and r < rho + margin:
                break
            W, rho = W2, r
        if rho > best_rho:
            best_W, best_rho = W, rho
    if best_W is None and len(cands) > 1:
        # ECC diverged on every model (common on small, noisy, static patches).
        # Phase correlation still gives a sound sub-pixel translation; hand that
        # back and let the NCC gate decide whether the frame is usable.
        best_W, best_rho = cands[1], -1.0
    return best_W, best_rho


def build_mask(win_shape: tuple[int, int], roi_in_win: tuple[int, int, int, int],
               grow: float) -> np.ndarray:
    """uint8 mask limiting ECC to the subject and its immediate surroundings."""
    h, w = win_shape
    x, y, rw, rh = roi_in_win
    cx, cy = x + rw / 2.0, y + rh / 2.0
    gw, gh = rw * grow, rh * grow
    x0, y0 = int(max(0, round(cx - gw / 2))), int(max(0, round(cy - gh / 2)))
    x1, y1 = int(min(w, round(cx + gw / 2))), int(min(h, round(cy + gh / 2)))
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


# --------------------------------------------------------------------------
# geometry for the high-resolution grid
# --------------------------------------------------------------------------


def hr_transform(roi_in_win: tuple[int, int, int, int], scale: float) -> np.ndarray:
    """T: window(LR) coords -> output(HR) coords, cropping to the ROI.

    Pixel *centres* are matched, the same convention cv2.resize uses: one LR
    pixel covers an s-by-s block of HR pixels and lands on that block's centre.
    Indexing them corner-to-corner instead would offset the whole result by
    (s-1)/2 HR pixels against the source it claims to reproduce.
    """
    rx, ry = roi_in_win[0], roi_in_win[1]
    off = 0.5 * scale - 0.5
    T = np.eye(3, dtype=np.float64)
    T[0, 0] = T[1, 1] = scale
    T[0, 2] = -scale * rx + off
    T[1, 2] = -scale * ry + off
    return T


def warp_lr_to_hr(img: np.ndarray, W: np.ndarray, T: np.ndarray,
                  hr_size: tuple[int, int], flags: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Resample an LR frame onto the HR reference grid."""
    M = to33(W) @ np.linalg.inv(T)
    homog = W.shape == (3, 3)
    return warp_to_ref(img, from33(M, homog), hr_size, flags)


def warp_hr_to_lr(hr: np.ndarray, W: np.ndarray, T: np.ndarray,
                  lr_size: tuple[int, int], flags: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Project the HR estimate back into one LR frame's pixel grid."""
    M = T @ np.linalg.inv(to33(W))
    homog = W.shape == (3, 3)
    return warp_to_ref(hr, from33(M, homog), lr_size, flags)


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------


def fuse(stack: np.ndarray, weights: np.ndarray, method: str,
         trim: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Combine registered HR frames.

    stack:   (N,h,w,C) float32 with NaN where a frame does not cover a pixel
    weights: (N,) per-frame quality weights
    returns: (fused HxWxC, coverage count HxW)
    """
    N = stack.shape[0]
    valid = ~np.isnan(stack[..., 0])
    coverage = valid.sum(axis=0).astype(np.float32)

    if method == "median":
        out = np.nanmedian(stack, axis=0)
    elif method == "mean":
        w = weights.reshape(N, 1, 1, 1)
        num = np.nansum(np.nan_to_num(stack) * w * valid[..., None], axis=0)
        den = np.nansum(w * valid[..., None], axis=0)
        out = num / np.maximum(den, 1e-6)
    else:  # trimmed mean: reject the extreme tails per pixel, then weighted mean
        lo_p, hi_p = 100.0 * trim / 2.0, 100.0 * (1.0 - trim / 2.0)
        with np.errstate(all="ignore"):
            lo = np.nanpercentile(stack, lo_p, axis=0)
            hi = np.nanpercentile(stack, hi_p, axis=0)
        keep = (stack >= lo[None]) & (stack <= hi[None]) & ~np.isnan(stack)
        w = weights.reshape(N, 1, 1, 1) * keep
        num = np.nansum(np.nan_to_num(stack) * w, axis=0)
        den = w.sum(axis=0)
        out = num / np.maximum(den, 1e-6)
        # pixels that survived no trim window fall back to the plain mean
        empty = den[..., 0] < 1e-6
        if empty.any():
            fb = np.nanmean(stack, axis=0)
            out[empty] = fb[empty]
    out = np.nan_to_num(out, nan=0.0)
    return out.astype(np.float32), coverage


# --------------------------------------------------------------------------
# iterative back-projection + deconvolution
# --------------------------------------------------------------------------


def gauss_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)


def back_project(hr: np.ndarray, obs: list[np.ndarray], warps: list[np.ndarray],
                 T: np.ndarray, weights: np.ndarray, iters: int, psf_sigma: float,
                 lam: float = 0.7, clip_k: float = 3.0,
                 verbose: bool = True) -> np.ndarray:
    """Refine the HR estimate so it reproduces every observed LR frame.

    This is what actually pulls detail out of the stack: the fused image is only
    a starting guess, and each iteration corrects it by the error it makes when
    re-simulating the real frames through the camera model (blur + decimate).
    """
    if iters <= 0 or not obs:
        return hr
    hr_h, hr_w = hr.shape[:2]
    hr_size = (hr_w, hr_h)
    lr_h, lr_w = obs[0].shape[:2]
    lr_size = (lr_w, lr_h)
    X = hr.copy()
    for it in range(iters):
        Xb = gauss_blur(X, psf_sigma)
        acc = np.zeros_like(X)
        wacc = np.zeros((hr_h, hr_w, 1), np.float32)
        for k, (o, W) in enumerate(zip(obs, warps)):
            sim = warp_hr_to_lr(Xb, W, T, lr_size, cv2.INTER_LINEAR)
            cov = warp_hr_to_lr(np.ones((hr_h, hr_w), np.float32), W, T, lr_size,
                                cv2.INTER_NEAREST)
            m = (cov > 0.99).astype(np.float32)
            if m.sum() < 16:
                continue
            r = (o - sim) * m[..., None]
            # Huber-style clipping keeps a stray outlier pixel from smearing
            s = float(np.std(r[m > 0])) or 1e-6
            np.clip(r, -clip_k * s, clip_k * s, out=r)
            R = warp_lr_to_hr(r, W, T, hr_size, cv2.INTER_LINEAR)
            Rm = warp_lr_to_hr(m, W, T, hr_size, cv2.INTER_LINEAR)
            if R.ndim == 2:
                R = R[..., None]
            wk = float(weights[k])
            acc += R * wk
            wacc += (Rm[..., None] if Rm.ndim == 2 else Rm) * wk
        upd = acc / np.maximum(wacc, 1e-6)
        upd = gauss_blur(upd, max(0.5, psf_sigma * 0.8))
        X = np.clip(X + lam * upd, 0.0, 1.0)
        progress(0.62 + 0.22 * (it + 1) / iters, f"back-projection {it + 1}/{iters}")
        if verbose:
            log(f"    back-projection {it + 1}/{iters}  rms={float(np.sqrt((upd ** 2).mean())):.5f}")
    return X


def richardson_lucy(img: np.ndarray, sigma: float, iters: int,
                    eps: float = 1e-6) -> np.ndarray:
    """Gaussian-PSF RL deconvolution; undoes the residual optical/sensor blur."""
    if iters <= 0 or sigma <= 0:
        return img
    obs = np.clip(img, eps, 1.0)
    est = obs.copy()
    for _ in range(iters):
        conv = gauss_blur(est, sigma)
        rel = obs / np.maximum(conv, eps)
        est = est * gauss_blur(rel, sigma)
        np.clip(est, 0.0, 1.0, out=est)
    return est


def unsharp(img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    blur = gauss_blur(img, sigma)
    return np.clip(img + amount * (img - blur), 0.0, 1.0)


# --------------------------------------------------------------------------
# post-processing / output
# --------------------------------------------------------------------------


def apply_clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    u8 = to_u8(img)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    if u8.ndim == 2:
        return cl.apply(u8).astype(np.float32) / 255.0
    lab = cv2.cvtColor(u8, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cl.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0


def denoise(img: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return img
    u8 = to_u8(img)
    if u8.ndim == 2:
        out = cv2.fastNlMeansDenoising(u8, None, strength, 7, 21)
    else:
        out = cv2.fastNlMeansDenoisingColored(u8, None, strength, strength, 7, 21)
    return out.astype(np.float32) / 255.0


def to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def to_u16(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 65535.0 + 0.5, 0, 65535).astype(np.uint16)


def save_img(path: str, img: np.ndarray, bits: int = 8) -> str:
    data = to_u16(img) if bits == 16 else to_u8(img)
    ok = cv2.imwrite(path, data)
    if not ok:
        die(f"failed to write {path}")
    return path


def label_strip(img_u8: np.ndarray, text: str, height: int = 22) -> np.ndarray:
    if img_u8.ndim == 2:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    w = img_u8.shape[1]
    bar = np.full((height, w, 3), 30, np.uint8)
    cv2.putText(bar, text, (6, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([bar, img_u8])


def side_by_side(items: Sequence[tuple[str, np.ndarray]], gap: int = 12) -> np.ndarray:
    tiles = []
    h = max(i[1].shape[0] for i in items)
    for name, im in items:
        u8 = to_u8(im)
        if u8.shape[0] < h:
            pad = np.zeros((h - u8.shape[0], u8.shape[1], 3), np.uint8)
            if u8.ndim == 2:
                u8 = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
            u8 = np.vstack([u8, pad])
        tiles.append(label_strip(u8, name))
    H = max(t.shape[0] for t in tiles)
    out = []
    for t in tiles:
        if t.shape[0] < H:
            t = np.vstack([t, np.zeros((H - t.shape[0], t.shape[1], 3), np.uint8)])
        out.append(t)
        out.append(np.full((H, gap, 3), 20, np.uint8))
    return np.hstack(out[:-1])


# --------------------------------------------------------------------------
# selection UI (self-contained local HTML - no server, nothing leaves the box)
# --------------------------------------------------------------------------


def jpeg_data_uri(bgr: np.ndarray, width: Optional[int] = None, quality: int = 82) -> str:
    img = bgr
    if width and img.shape[1] > width:
        h = max(1, int(round(img.shape[0] * width / img.shape[1])))
        img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        die("jpeg encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


PICKER_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#14161a;--panel:#1c2027;--line:#2c323c;--fg:#e6e9ef;--dim:#98a2b3;
      --acc:#4da3ff;--ok:#3ddc97;--bad:#ff6b6b;--warn:#ffb454}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;
       padding:10px 14px;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:14px;margin:0;font-weight:600;letter-spacing:.02em}
.meta{color:var(--dim);font-size:12px}
.sp{flex:1}
button{background:#252b34;color:var(--fg);border:1px solid var(--line);border-radius:6px;
       padding:6px 11px;font:inherit;font-size:12px;cursor:pointer}
button:hover{border-color:var(--acc)}
button.primary{background:var(--acc);border-color:var(--acc);color:#05121f;font-weight:600}
button.on{background:#0d2b45;border-color:var(--acc);color:var(--acc)}
main{display:flex;gap:14px;padding:14px;align-items:flex-start;flex-wrap:wrap}
#stage{flex:1 1 560px;min-width:340px}
.wrap{position:relative;display:inline-block;line-height:0;
      border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#000}
.wrap img{display:block;max-width:100%;height:auto;user-select:none;-webkit-user-drag:none}
#ov{position:absolute;inset:0;cursor:crosshair;touch-action:none}
aside{flex:0 0 292px;background:var(--panel);border:1px solid var(--line);
      border-radius:8px;padding:12px}
h2{font-size:12px;margin:0 0 8px;color:var(--dim);text-transform:uppercase;
   letter-spacing:.08em;font-weight:600}
aside h2:not(:first-child){margin-top:16px}
.row{display:flex;gap:6px;margin-bottom:6px;align-items:center}
.row label{width:14px;color:var(--dim)}
input[type=number]{width:100%;background:#12151a;color:var(--fg);border:1px solid var(--line);
                   border-radius:4px;padding:4px 6px;font:inherit;font-size:12px}
input[type=range]{width:100%;accent-color:var(--acc)}
#mag{border:1px solid var(--line);border-radius:6px;background:#000;width:100%;height:148px;
     image-rendering:pixelated}
.kv{display:flex;justify-content:space-between;color:var(--dim);font-size:12px;padding:1px 0}
.kv b{color:var(--fg);font-weight:600}
section{padding:0 14px 14px}
.bar{display:flex;gap:10px;align-items:center;margin-bottom:8px;flex-wrap:wrap}
#tl{width:100%;height:74px;display:block;border:1px solid var(--line);border-radius:8px;
    background:#0f1216;cursor:crosshair;touch-action:none}
#spanlist{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.chip{background:#2a1e22;border:1px solid var(--bad);color:#ffc9c9;border-radius:999px;
      padding:2px 10px;font-size:11px;cursor:pointer}
.chip:hover{background:#3a2429}
#film{display:flex;gap:5px;overflow-x:auto;padding:6px 2px 10px}
.fr{position:relative;flex:0 0 auto;border:2px solid transparent;border-radius:5px;
    cursor:pointer;background:#0c0e12;line-height:0}
.fr img{display:block;width:var(--tw);height:auto;border-radius:3px}
.fr .n{position:absolute;left:2px;top:2px;background:#000a;color:#fff;font-size:9px;
       padding:0 3px;border-radius:3px;line-height:14px}
.fr .q{position:absolute;left:0;right:0;bottom:0;height:3px;background:var(--line)}
.fr.off img{opacity:.2;filter:grayscale(1)}
.fr.off{border-color:var(--bad)}
.fr.span{border-color:#7a3b3b}
.fr.cur{border-color:var(--acc)}
.fr.ref .n{background:var(--warn);color:#111}
#note{color:var(--dim);font-size:12px;margin-top:8px;min-height:18px}
#out{width:100%;height:96px;background:#12151a;color:var(--dim);border:1px solid var(--line);
     border-radius:6px;font:inherit;font-size:11px;padding:8px;margin-top:8px;resize:vertical}
</style></head><body>
<header>
  <h1>__HEADING__</h1>
  <span class="meta" id="hmeta"></span>
  <span class="sp"></span>
  <button id="btnAuto">auto-flag outliers</button>
  <button id="btnAll">reset all</button>
  <button class="primary" id="btnSave">download selection.json</button>
</header>

<main>
  <div id="stage">
    <div class="wrap"><img id="view" alt="frame"><canvas id="ov"></canvas></div>
    <div class="bar" style="margin-top:10px">
      <button id="btnPlay">▶ play</button>
      <input type="range" id="scrub" min="0" value="0" style="flex:1;min-width:180px">
      <span class="meta" id="curinfo"></span>
    </div>
  </div>
  <aside>
    <h2>region of interest</h2>
    <div class="row"><label>x</label><input type="number" id="rx"><label>y</label><input type="number" id="ry"></div>
    <div class="row"><label>w</label><input type="number" id="rw"><label>h</label><input type="number" id="rh"></div>
    <div class="kv"><span>drag on the image</span><b id="roisz">-</b></div>
    <h2>magnifier</h2>
    <canvas id="mag" width="268" height="148"></canvas>
    <h2>tally</h2>
    <div class="kv"><span>frames kept</span><b id="nkept">-</b></div>
    <div class="kv"><span>dropped</span><b id="ndrop">-</b></div>
    <div class="kv"><span>ignored spans</span><b id="nspan">0</b></div>
    <div class="kv"><span>reference frame</span><b id="nref">-</b></div>
    <h2>outlier threshold</h2>
    <input type="range" id="thr" min="5" max="60" value="25">
    <div class="kv"><span>flags frames unlike the rest</span><b id="thrv">2.5σ</b></div>
    <h2>next step</h2>
    <textarea id="out" readonly></textarea>
    <button id="btnCopy" style="margin-top:6px;width:100%">copy command</button>
  </aside>
</main>

<section>
  <div class="bar">
    <b>timeline</b>
    <span class="meta">drag to mark a span ·</span>
    <button id="mIgnore" class="on">drag = ignore</button>
    <button id="mKeep">drag = keep only this</button>
    <button id="mClear">clear spans</button>
    <span class="meta" id="tlinfo"></span>
  </div>
  <canvas id="tl" height="74"></canvas>
  <div id="spanlist"></div>
  <div id="note"></div>
</section>

<section>
  <div class="bar">
    <b>frames</b>
    <span class="meta">click = drop/keep one · shift+click = range · R = reference · X = toggle · ←/→ step · space play</span>
  </div>
  <div id="film"></div>
</section>

<script id="payload" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const F = D.frames, N = F.length, T = F.map(f => f.t);
const win = D.window;                       // [x,y,w,h] decoded window, source px
const T0 = T.length ? T[0] : 0, T1 = T.length ? T[N-1] : 1, TD = Math.max(1e-6, T1 - T0);
let roi = D.roi ? D.roi.slice() : null;     // source-pixel coords
let cur = D.ref_index|0, refIdx = D.ref_index|0, playing = false, lastPick = null;
let spans = (D.skip_spans || []).map(s => s.slice());   // ignored time spans
let mode = 'ignore';
const manual = new Array(N).fill(null);     // null = follow spans, else explicit
let scores = new Array(N).fill(0);
(D.excluded || []).forEach(i => { if (i >= 0 && i < N) manual[i] = false; });

const $ = id => document.getElementById(id);
const view = $('view'), ov = $('ov'), ctx = ov.getContext('2d');
const mag = $('mag'), mctx = mag.getContext('2d');
const tl = $('tl'), tctx = tl.getContext('2d');
const film = $('film'), scrub = $('scrub');

const hhmmss = t => {
  const h = Math.floor(t/3600), m = Math.floor(t%3600/60), s = t%60;
  return (h? h+':' : '') + String(m).padStart(2,'0') + ':' + s.toFixed(2).padStart(5,'0');
};
$('hmeta').textContent = `${D.video} · ${N} frames · ${hhmmss(T0)}–${hhmmss(T1)} · window ${win[2]}×${win[3]} @ ${win[0]},${win[1]}`;
scrub.max = N - 1; scrub.value = cur;
document.documentElement.style.setProperty('--tw', D.thumb_width + 'px');

const inSpans = t => spans.some(s => t >= s[0] && t <= s[1]);
const keepOf = i => manual[i] !== null ? manual[i] : !inSpans(T[i]);

/* ---------------- filmstrip ---------------- */
const cells = F.map((f, i) => {
  const d = document.createElement('div'); d.className = 'fr';
  d.innerHTML = `<img loading="lazy" src="${f.img}" alt=""><span class="n">${i}</span><span class="q"></span>`;
  d.title = `frame ${i} · t=${f.t.toFixed(3)}s`;
  d.addEventListener('click', ev => {
    if (ev.shiftKey && lastPick !== null) {
      const [a,b] = [Math.min(lastPick,i), Math.max(lastPick,i)], v = !keepOf(i);
      for (let k=a;k<=b;k++) manual[k] = v;
    } else { manual[i] = !keepOf(i); lastPick = i; }
    setCur(i); render();
  });
  film.appendChild(d); return d;
});

function setCur(i){
  cur = Math.max(0, Math.min(N-1, i));
  scrub.value = cur;
  view.src = (cur === refIdx && D.ref_hd) ? D.ref_hd : F[cur].img;
  $('curinfo').textContent = `frame ${cur} · t=${hhmmss(T[cur])}`;
  cells[cur].scrollIntoView({block:'nearest', inline:'nearest'});
  drawTimeline();
}

/* ---------------- ROI overlay ---------------- */
const sc = () => view.clientWidth / win[2];      // source px -> css px
function fitCanvas(){
  ov.width = view.clientWidth; ov.height = view.clientHeight;
  ov.style.width = view.clientWidth+'px'; ov.style.height = view.clientHeight+'px';
  drawOverlay();
}
function drawOverlay(){
  ctx.clearRect(0,0,ov.width,ov.height);
  if(!roi) return;
  const s = sc(), x = (roi[0]-win[0])*s, y = (roi[1]-win[1])*s, w = roi[2]*s, h = roi[3]*s;
  ctx.fillStyle = 'rgba(0,0,0,.45)';
  ctx.fillRect(0,0,ov.width,y); ctx.fillRect(0,y+h,ov.width,ov.height-y-h);
  ctx.fillRect(0,y,x,h); ctx.fillRect(x+w,y,ov.width-x-w,h);
  ctx.strokeStyle = '#4da3ff'; ctx.lineWidth = 1.5; ctx.strokeRect(x+.5,y+.5,w,h);
  ctx.strokeStyle = 'rgba(77,163,255,.3)'; ctx.lineWidth = 1;
  ctx.strokeRect(x-8.5,y-8.5,w+17,h+17);      // alignment context
}
let drag = null;
ov.addEventListener('pointerdown', e => {
  const r = ov.getBoundingClientRect(), s = sc();
  drag = {x:(e.clientX-r.left)/s+win[0], y:(e.clientY-r.top)/s+win[1]};
  ov.setPointerCapture(e.pointerId);
});
ov.addEventListener('pointermove', e => {
  const r = ov.getBoundingClientRect(), s = sc();
  const px = (e.clientX-r.left)/s+win[0], py = (e.clientY-r.top)/s+win[1];
  magnify(e.clientX-r.left, e.clientY-r.top);
  if(!drag) return;
  roi = [Math.round(Math.min(drag.x,px)), Math.round(Math.min(drag.y,py)),
         Math.max(2,Math.round(Math.abs(px-drag.x))), Math.max(2,Math.round(Math.abs(py-drag.y)))];
  syncFields(); drawOverlay();
});
ov.addEventListener('pointerup', () => {
  if(drag){ refIdx = cur; }   // the box is measured on the frame you drew it on
  drag = null; score(); render();
});
function magnify(cx, cy){
  const s = sc(), z = 6, sw = mag.width/z, sh = mag.height/z;
  const kx = view.naturalWidth/win[2], ky = view.naturalHeight/win[3];
  mctx.imageSmoothingEnabled = false;
  mctx.fillStyle = '#000'; mctx.fillRect(0,0,mag.width,mag.height);
  try{ mctx.drawImage(view, cx/s*kx - sw/2, cy/s*ky - sh/2, sw, sh, 0,0,mag.width,mag.height); }catch(e){}
  mctx.strokeStyle = 'rgba(255,255,255,.3)'; mctx.beginPath();
  mctx.moveTo(mag.width/2,0); mctx.lineTo(mag.width/2,mag.height);
  mctx.moveTo(0,mag.height/2); mctx.lineTo(mag.width,mag.height/2); mctx.stroke();
}
function syncFields(){
  if(!roi) return;
  rx.value=roi[0]; ry.value=roi[1]; rw.value=roi[2]; rh.value=roi[3];
  $('roisz').textContent = roi[2]+'×'+roi[3]+' px';
}
['rx','ry','rw','rh'].forEach(id => $(id).addEventListener('change', () => {
  roi = [+rx.value||0, +ry.value||0, Math.max(2,+rw.value||2), Math.max(2,+rh.value||2)];
  syncFields(); drawOverlay(); score(); render();
}));

/* ---------------- timeline: mark spans to ignore ---------------- */
let tdrag = null;
const xOfT = t => ((t - T0)/TD) * tl.width;
const tOfX = x => T0 + (x / tl.width) * TD;
function fitTimeline(){
  const r = tl.getBoundingClientRect();
  tl.width = Math.max(320, Math.round(r.width)); tl.height = 74;
  drawTimeline();
}
function drawTimeline(){
  const w = tl.width, h = tl.height;
  tctx.clearRect(0,0,w,h);
  tctx.fillStyle = '#0f1216'; tctx.fillRect(0,0,w,h);
  for(const s of spans){                                  // ignored regions
    const x0 = xOfT(s[0]), x1 = xOfT(s[1]);
    tctx.fillStyle = 'rgba(255,107,107,.20)'; tctx.fillRect(x0,0,Math.max(2,x1-x0),h);
    tctx.strokeStyle = 'rgba(255,107,107,.75)'; tctx.beginPath();
    tctx.moveTo(x0+.5,0); tctx.lineTo(x0+.5,h); tctx.moveTo(x1-.5,0); tctx.lineTo(x1-.5,h); tctx.stroke();
  }
  if(tdrag){
    const x0 = xOfT(Math.min(tdrag.a,tdrag.b)), x1 = xOfT(Math.max(tdrag.a,tdrag.b));
    tctx.fillStyle = mode==='ignore' ? 'rgba(255,107,107,.28)' : 'rgba(61,220,151,.22)';
    tctx.fillRect(x0,0,Math.max(2,x1-x0),h);
  }
  for(let i=0;i<N;i++){                                   // one tick per frame
    const x = xOfT(T[i]), on = keepOf(i);
    tctx.strokeStyle = on ? (i===cur?'#4da3ff':'rgba(61,220,151,.75)') : 'rgba(255,107,107,.85)';
    tctx.lineWidth = i===cur ? 2 : 1;
    tctx.beginPath(); tctx.moveTo(x+.5, h-8); tctx.lineTo(x+.5, h-8-(on?26:14)); tctx.stroke();
  }
  tctx.strokeStyle = '#2c323c'; tctx.beginPath();
  tctx.moveTo(0,h-8.5); tctx.lineTo(w,h-8.5); tctx.stroke();
  tctx.fillStyle = '#98a2b3'; tctx.font = '10px ui-monospace,monospace';
  const ticks = 6;
  for(let k=0;k<=ticks;k++){
    const t = T0 + TD*k/ticks, x = xOfT(t);
    tctx.fillText(hhmmss(t), Math.min(w-46, Math.max(2, x-20)), h-1);
  }
  const px = xOfT(T[cur]);                                 // playhead
  tctx.strokeStyle = '#4da3ff'; tctx.lineWidth = 1;
  tctx.beginPath(); tctx.moveTo(px+.5,0); tctx.lineTo(px+.5,h-8); tctx.stroke();
}
tl.addEventListener('pointerdown', e => {
  const r = tl.getBoundingClientRect();
  const t = tOfX((e.clientX-r.left) * tl.width / r.width);
  tdrag = {a:t, b:t, x0:e.clientX}; tl.setPointerCapture(e.pointerId);
});
tl.addEventListener('pointermove', e => {
  if(!tdrag) return;
  const r = tl.getBoundingClientRect();
  tdrag.b = tOfX((e.clientX-r.left) * tl.width / r.width);
  drawTimeline();
});
tl.addEventListener('pointerup', e => {
  if(!tdrag) return;
  const moved = Math.abs(e.clientX - tdrag.x0) > 3;
  const a = Math.min(tdrag.a,tdrag.b), b = Math.max(tdrag.a,tdrag.b);
  tdrag = null;
  if(!moved){
    const hit = spans.findIndex(s => a >= s[0] && a <= s[1]);
    if(hit >= 0) spans.splice(hit,1);                     // click a span to lift it
    else { let best=0; for(let i=0;i<N;i++) if(Math.abs(T[i]-a) < Math.abs(T[best]-a)) best=i; play(false); setCur(best); }
  } else if(mode === 'ignore'){
    spans.push([a,b]);
  } else {                                                 // keep only this span
    spans = [[T0 - 1, a], [b, T1 + 1]];
  }
  spans = mergeSpans(spans);
  for(let i=0;i<N;i++) if(inSpans(T[i])) manual[i] = null;  // spans win over old clicks
  render();
});
function mergeSpans(list){
  const s = list.map(x=>x.slice()).sort((p,q)=>p[0]-q[0]), out=[];
  for(const x of s){ if(out.length && x[0] <= out[out.length-1][1]) out[out.length-1][1] = Math.max(out[out.length-1][1], x[1]); else out.push(x); }
  return out;
}
$('mIgnore').addEventListener('click', ()=>{ mode='ignore'; $('mIgnore').classList.add('on'); $('mKeep').classList.remove('on'); });
$('mKeep').addEventListener('click', ()=>{ mode='keep'; $('mKeep').classList.add('on'); $('mIgnore').classList.remove('on'); });
$('mClear').addEventListener('click', ()=>{ spans=[]; render(); });

/* ---------------- per-frame outlier score inside the ROI ---------------- */
const SW=40, SH=24;
const sc1 = document.createElement('canvas'); sc1.width=SW; sc1.height=SH;
const sctx = sc1.getContext('2d', {willReadFrequently:true});
const vecs = new Array(N).fill(null);
const imgs = F.map(f => { const im = new Image(); im.src = f.img; return im; });
function vecOf(i){
  const im = imgs[i]; if(!im.complete || !im.naturalWidth || !roi) return null;
  const kx = im.naturalWidth/win[2], ky = im.naturalHeight/win[3];
  sctx.drawImage(im, (roi[0]-win[0])*kx, (roi[1]-win[1])*ky, roi[2]*kx, roi[3]*ky, 0,0,SW,SH);
  const d = sctx.getImageData(0,0,SW,SH).data, v = new Float32Array(SW*SH);
  for(let p=0;p<v.length;p++) v[p] = (d[p*4]*.114 + d[p*4+1]*.587 + d[p*4+2]*.299)/255;
  return v;
}
function score(){
  if(!roi) return;
  for(let i=0;i<N;i++) vecs[i] = vecOf(i);
  const ok = vecs.filter(Boolean); if(ok.length < 3) return;
  const med = new Float32Array(SW*SH), col = new Float64Array(ok.length);
  for(let p=0;p<med.length;p++){
    for(let k=0;k<ok.length;k++) col[k] = ok[k][p];
    const srt = Array.from(col).sort((a,b)=>a-b); med[p] = srt[srt.length>>1];
  }
  for(let i=0;i<N;i++){
    if(!vecs[i]){ scores[i]=0; continue; }
    let acc=0; for(let p=0;p<med.length;p++) acc += Math.abs(vecs[i][p]-med[p]);
    scores[i] = acc/med.length;
  }
}
function stats(){
  const v = scores.filter(x=>x>0); if(v.length<3) return null;
  const m = v.reduce((a,b)=>a+b,0)/v.length;
  return {m, sd: Math.sqrt(v.reduce((a,b)=>a+(b-m)*(b-m),0)/v.length) || 1e-6};
}
$('btnAuto').addEventListener('click', () => {
  if(!roi){ note('draw the ROI box first — outliers are judged inside it'); return; }
  score(); const st = stats(); if(!st){ note('not enough frames to score'); return; }
  const k = +$('thr').value/10; let n=0;
  for(let i=0;i<N;i++) if(scores[i] > st.m + k*st.sd){ manual[i] = false; n++; }
  note(`flagged ${n} frame(s) beyond ${k.toFixed(1)}σ of the ROI median — occlusions, glare, motion smear`);
  render();
});
$('thr').addEventListener('input', e => $('thrv').textContent = (e.target.value/10).toFixed(1)+'σ');
$('btnAll').addEventListener('click', ()=>{ manual.fill(null); spans=[]; render(); note(''); });
const note = m => $('note').textContent = m;

/* ---------------- playback ---------------- */
let timer=null;
function play(on){
  playing = on; $('btnPlay').textContent = on ? '❚❚ pause' : '▶ play';
  if(timer) clearInterval(timer);
  if(on) timer = setInterval(()=>setCur((cur+1)%N), 1000/Math.min(25, D.fps||12));
}
$('btnPlay').addEventListener('click', ()=>play(!playing));
scrub.addEventListener('input', e=>{ play(false); setCur(+e.target.value); });
addEventListener('keydown', e => {
  if(['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
  if(e.key===' '){ e.preventDefault(); play(!playing); }
  else if(e.key==='ArrowRight'){ play(false); setCur(cur+1); }
  else if(e.key==='ArrowLeft'){ play(false); setCur(cur-1); }
  else if(e.key==='x'||e.key==='X'){ manual[cur] = !keepOf(cur); render(); }
  else if(e.key==='r'||e.key==='R'){ refIdx = cur; render(); }
});

/* ---------------- export ---------------- */
function selection(){
  const keep = [], drop = [], dropT = [];
  for(let i=0;i<N;i++){ if(keepOf(i)) keep.push(i); else { drop.push(i); dropT.push(+T[i].toFixed(4)); } }
  return {version:D.version, video:D.video_abs, decode:D.decode, window:win, roi:roi,
          ref_index:refIdx, ref_time:+T[refIdx].toFixed(4), fps:D.fps,
          skip_spans:spans.map(s=>[+s[0].toFixed(4), +s[1].toFixed(4)]),
          include:keep, exclude:drop, exclude_times:dropT,
          times:T.map(t=>+t.toFixed(4)), created:new Date().toISOString()};
}
$('btnSave').addEventListener('click', () => {
  if(!roi){ note('draw the ROI box first'); return; }
  const b = new Blob([JSON.stringify(selection(),null,2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(b); a.download = 'selection.json'; a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href), 4000);
  note('saved selection.json → run the command from the panel (pass its path if it landed in ~/Downloads)');
});
$('btnCopy').addEventListener('click', async () => {
  const t = $('out'); t.select();
  try{ await navigator.clipboard.writeText(t.value); }catch(e){ document.execCommand('copy'); }
  $('btnCopy').textContent = 'copied'; setTimeout(()=>$('btnCopy').textContent='copy command', 1200);
});

/* ---------------- render ---------------- */
function render(){
  const st = stats();
  let nk = 0;
  for(let i=0;i<N;i++){
    const c = cells[i], on = keepOf(i);
    if(on) nk++;
    c.classList.toggle('off', !on);
    c.classList.toggle('span', !on && manual[i] === null);
    c.classList.toggle('cur', i===cur);
    c.classList.toggle('ref', i===refIdx);
    const q = c.querySelector('.q');
    if(st && scores[i] > 0){
      const z = (scores[i]-st.m)/st.sd;
      q.style.background = z>2.5 ? 'var(--bad)' : z>1.2 ? 'var(--warn)' : 'var(--ok)';
      q.style.width = Math.max(6, Math.min(100, 50+z*18))+'%';
    } else { q.style.width='100%'; q.style.background='var(--line)'; }
  }
  $('nkept').textContent = nk; $('ndrop').textContent = N-nk;
  $('nspan').textContent = spans.length; $('nref').textContent = refIdx;
  const sl = $('spanlist'); sl.innerHTML = '';
  spans.forEach((s,i) => {
    const c = document.createElement('span'); c.className = 'chip';
    c.textContent = `ignore ${hhmmss(Math.max(T0,s[0]))} – ${hhmmss(Math.min(T1,s[1]))}  ✕`;
    c.title = 'click to remove'; c.addEventListener('click', ()=>{ spans.splice(i,1); render(); });
    sl.appendChild(c);
  });
  $('tlinfo').textContent = spans.length ? `${spans.length} ignored span(s)` : 'no spans ignored';
  const skipFlags = spans.map(s=>`--skip ${Math.max(T0,s[0]).toFixed(2)}-${Math.min(T1,s[1]).toFixed(2)}`).join(' ');
  $('out').value = `${D.cmd_prefix} --select selection.json --out ${D.out_hint}` +
    (roi ? `\n\n# equivalent without the json:\n${D.cmd_prefix} --roi ${roi.join(',')}` +
           ` --use ${T0.toFixed(2)}-${T1.toFixed(2)} ${skipFlags}` +
           ` --ref-time ${T[refIdx].toFixed(3)} --out ${D.out_hint}` : '');
  drawTimeline(); drawOverlay();
}
view.addEventListener('load', fitCanvas);
addEventListener('resize', ()=>{ fitCanvas(); fitTimeline(); });
setCur(cur); syncFields(); fitTimeline();
setTimeout(()=>{ score(); render(); }, 350);
render();
</script></body></html>
"""


def write_picker(path: str, *, title: str, heading: str, video: str, decode_spec: DecodeSpec,
                 window: tuple[int, int, int, int], frames_uri: list[str], times: list[float],
                 fps: float, ref_index: int, ref_hd: Optional[str], roi=None,
                 excluded: Optional[Sequence[int]] = None, thumb_width: int = 120,
                 cmd_prefix: str = "", out_hint: str = "out",
                 skip_spans: Optional[Sequence[Sequence[float]]] = None) -> str:
    payload = {
        "version": __version__,
        "video": os.path.basename(video),
        "video_abs": os.path.abspath(video),
        "decode": decode_spec.to_json(),
        "window": list(window),
        "roi": list(roi) if roi else None,
        "ref_index": int(ref_index),
        "ref_hd": ref_hd,
        "fps": fps,
        "thumb_width": int(thumb_width),
        "excluded": list(excluded or []),
        "skip_spans": [[float(a), float(b)] for a, b in (skip_spans or [])],
        "cmd_prefix": cmd_prefix,
        "out_hint": out_hint,
        "range": f"{fmt_time(times[0])} - {fmt_time(times[-1])}" if times else "-",
        "frames": [{"t": float(t), "img": u} for t, u in zip(times, frames_uri)],
    }
    data = json.dumps(payload).replace("</", "<\\/")
    html = (PICKER_HTML.replace("__TITLE__", title)
                       .replace("__HEADING__", heading)
                       .replace("__DATA__", data))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


def fuse_stack(stack: np.ndarray, weights: np.ndarray, method: str, trim: float,
               band: int = 192) -> tuple[np.ndarray, np.ndarray]:
    """Row-banded wrapper around fuse() so big stacks stay inside memory."""
    h = stack.shape[1]
    if h <= band:
        return fuse(stack, weights, method, trim)
    out = np.empty(stack.shape[1:], np.float32)
    cov = np.empty(stack.shape[1:3], np.float32)
    for y in range(0, h, band):
        y1 = min(h, y + band)
        o, c = fuse(stack[:, y:y1], weights, method, trim)
        out[y:y1], cov[y:y1] = o, c
    return out, cov


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_info(a) -> int:
    info = probe(a.video)
    n = info.nb_frames or (int(info.duration * info.fps) if info.duration else 0)
    print(f"file        {info.path}")
    print(f"resolution  {info.width}x{info.height}   pix_fmt {info.pix_fmt}")
    print(f"codec       {info.codec}")
    print(f"fps         {info.fps:.4f}")
    print(f"duration    {fmt_time(info.duration)}  ({n} frames)")
    print(f"field order {info.field_order}" + ("   << interlaced: use --deint field" if info.interlaced else ""))
    print()
    print("next:  vidsr grid VIDEO --start 0 --end %s   # find the moment" % fmt_time(min(info.duration, 60)))
    return 0


def cmd_grid(a) -> int:
    info = probe(a.video)
    start = parse_time(a.start) or 0.0
    end = parse_time(a.end) if a.end else min(info.duration, start + 30.0)
    if end <= start:
        die("--end must be after --start")
    n = max(1, a.count)
    times = [start + (end - start) * i / max(1, n - 1) for i in range(n)]
    log(f"sampling {n} frames between {fmt_time(start)} and {fmt_time(end)}")
    tiles = []
    for t in times:
        f = grab_frame(info, t)
        tw = a.width
        th = max(1, int(round(f.shape[0] * tw / f.shape[1])))
        f = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
        tiles.append(label_strip(f, f"t={fmt_time(t)}  ({t:.3f}s)"))
    cols = a.cols
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    out = a.out or "contact_sheet.png"
    cv2.imwrite(out, sheet)
    print(f"wrote {out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print("next:  vidsr select VIDEO --start <t> --dur <seconds>")
    return 0


def resolve_spans(a, info: VideoInfo, default_len: float) -> list[tuple[float, float]]:
    """--use / --skip win; otherwise fall back to --start plus --end/--dur."""
    if getattr(a, "use", None):
        spans = parse_spans(a.use, info.duration)
    else:
        start = parse_time(a.start) or 0.0 if a.start is not None else 0.0
        if a.end:
            end = parse_time(a.end)
        elif a.dur:
            end = start + parse_time(a.dur)
        else:
            end = min(info.duration or (start + default_len), start + default_len)
        if end <= start:
            die("the time range is empty (check --start/--end/--dur)")
        spans = [(start, end)]
    if getattr(a, "skip", None):
        spans = subtract_spans(spans, parse_spans(a.skip, info.duration))
    if not spans:
        die("nothing left to process after --skip")
    return spans


def cmd_select(a) -> int:
    info = probe(a.video)
    spans = resolve_spans(a, info, 4.0)
    window = (0, 0, info.width, info.height)
    if a.window:
        window = even_rect(parse_rect(a.window), info.width, info.height)
    roi = clamp_rect(parse_rect(a.roi), info.width, info.height) if a.roi else None
    if roi and not a.window:
        # give the aligner room around the subject and the user room for context
        pad = a.pad if a.pad is not None else max(48, int(0.9 * max(roi[2], roi[3])))
        window = even_rect(expand_rect(roi, pad, pad, info.width, info.height),
                           info.width, info.height)

    spec = DecodeSpec(spans=spans, window=window, deint=a.deint,
                      step=a.step, max_frames=a.max_frames)
    stride = plan_stride(info, spec)
    log(f"decoding {fmt_spans(spans)}  ({spans_total(spans):.1f}s, every {stride} frame"
        f"{'s' if stride > 1 else ''})  window {window[2]}x{window[3]}"
        f"+{window[0]}+{window[1]} ...")
    frames, times = read_frames(info, spec)
    log(f"got {len(frames)} frames")
    if not frames:
        die("no frames decoded - check --start/--dur")

    sharps = [sharpness(to_gray32(f)) for f in frames]
    ref = int(np.argmax(sharps)) if a.ref is None else max(0, min(len(frames) - 1, a.ref))

    zoom = a.zoom
    uris = []
    for f in frames:
        img = f
        if roi and a.crop:
            x, y, w, h = roi
            img = f[y - window[1]:y - window[1] + h, x - window[0]:x - window[0] + w]
            img = cv2.resize(img, (img.shape[1] * zoom, img.shape[0] * zoom),
                             interpolation=cv2.INTER_NEAREST)
            uris.append(jpeg_data_uri(img, None, 88))
        else:
            uris.append(jpeg_data_uri(img, a.view_width, 82))
    ref_hd = jpeg_data_uri(frames[ref], None, 94) if not (roi and a.crop) else None

    os.makedirs(a.out, exist_ok=True)
    page = os.path.join(a.out, "select.html")
    write_picker(
        page, title="vidsr - select", heading="select frames + region",
        video=a.video, decode_spec=spec, window=window, frames_uri=uris, times=times,
        fps=info.fps * (2 if a.deint == "field" else 1), ref_index=ref, ref_hd=ref_hd,
        roi=roi, thumb_width=a.thumb_width,
        cmd_prefix=f"{PROG} sr {shlex_quote(os.path.abspath(a.video))}",
        out_hint=os.path.join(a.out, "result"),
    )
    size_mb = os.path.getsize(page) / 1e6
    print(f"wrote {page}  ({size_mb:.1f} MB, {len(frames)} frames)")
    print(f"sharpest frame: #{ref}   t={times[ref]:.3f}s")
    print()
    print("open it:  xdg-open " + page)
    print("  1. drag a box around the subject (plate, sign, badge, ...)")
    print("  2. scrub/play; click any frame to drop it (people walking through, blur, glare)")
    print("  3. 'auto-flag outliers' does a first pass for you")
    print("  4. download selection.json, then run the command shown in the panel")
    return 0


def shlex_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)


def _resolve_job(a):
    """Build (info, decode spec, window, roi, ref_index, excluded) from CLI or selection.json."""
    sel = None
    if a.select:
        if not os.path.exists(a.select):
            die(f"no such selection file: {a.select}")
        sel = json.load(open(a.select))
    video = a.video or (sel or {}).get("video")
    if not video:
        die("need a video (positional) or --select selection.json")
    if not os.path.exists(video) and sel and os.path.exists(sel.get("video", "")):
        video = sel["video"]
    info = probe(video)

    ref_time = None
    exclude_times: list[float] = []
    if sel:
        spec = DecodeSpec.from_json(sel.get("decode", {}))
        roi = tuple(sel["roi"]) if sel.get("roi") else None
        ref = int(sel.get("ref_index", -1))
        ref_time = sel.get("ref_time")
        excluded = set(int(i) for i in sel.get("exclude", []))
        # times survive a change of sampling stride; indices do not
        exclude_times = [float(t) for t in (sel.get("exclude_times") or [])]
        if sel.get("skip_spans"):
            spec.spans = subtract_spans(spec.spans,
                                        [tuple(x) for x in sel["skip_spans"]])
    else:
        spec = DecodeSpec(spans=resolve_spans(a, info, 3.0), deint=a.deint,
                          step=a.step or 1, max_frames=a.max_frames)
        roi = None
        ref = -1
        excluded = set()
    window = (0, 0, info.width, info.height)

    # CLI always wins over the file
    if a.roi:
        roi = parse_rect(a.roi)
    if not roi:
        die("no ROI: pass --roi x,y,w,h or a --select selection.json that has one")
    roi = clamp_rect(roi, info.width, info.height)
    if a.exclude:
        excluded |= parse_index_list(a.exclude)
    if a.ref is not None:
        ref, ref_time = a.ref, None
    if getattr(a, "ref_time", None) is not None:
        ref_time = a.ref_time
    if a.deint != "none":
        spec.deint = a.deint
    if a.max_frames is not None:
        spec.max_frames = a.max_frames
    if a.step and a.step > 1:
        spec.step = a.step
    # command-line spans override the selection; --skip always subtracts
    if a.use:
        spec.spans = parse_spans(a.use, info.duration)
    elif a.start is not None or a.end or a.dur:
        spec.spans = resolve_spans(a, info, 3.0)
    if a.skip:
        spec.spans = subtract_spans(spec.spans, parse_spans(a.skip, info.duration))
    if not spec.spans:
        die("nothing left to process after --skip")

    # cropping does not change frame indexing, so it is always safe to shrink the
    # decode window to the ROI plus its alignment context
    pad = a.pad if a.pad is not None else max(32, int(0.6 * max(roi[2], roi[3])))
    spec.window = even_rect(expand_rect(roi, pad, pad, info.width, info.height),
                            info.width, info.height)
    return info, spec, spec.window, roi, ref, excluded, ref_time, exclude_times


PRESETS = {
    # A still subject under a fixed camera has almost no sub-pixel diversity, so
    # the win comes from averaging noise/compression away, then deconvolving hard.
    "static": dict(motion="translation", fusion="trimmed", ibp=10, rl=14,
                   unsharp=0.5, denoise=3.0, clahe=1.5, trim=0.3),
    # A subject crossing the frame samples many sub-pixel phases: real resolution
    # gain, but fewer usable frames and more motion blur to reject.
    "moving": dict(motion="auto", fusion="trimmed", ibp=14, rl=10,
                   unsharp=0.4, denoise=0.0, clahe=1.5, sharp_keep=0.8),
}


def apply_preset(a, parser: argparse.ArgumentParser) -> None:
    if not getattr(a, "preset", None):
        return
    for k, v in PRESETS[a.preset].items():
        if getattr(a, k, None) == parser.get_default(k):   # user did not set it
            setattr(a, k, v)


def cmd_sr(a) -> int:
    t0 = _time.time()
    info, spec, window, roi, ref_idx, excluded, ref_time, exclude_times = _resolve_job(a)
    os.makedirs(a.out, exist_ok=True)
    wx, wy, ww, wh = window
    roi_in_win = (roi[0] - wx, roi[1] - wy, roi[2], roi[3])

    log(f"video    {info.path}  {info.width}x{info.height} @ {info.fps:.3f}fps"
        + ("  [interlaced]" if info.interlaced else ""))
    log(f"roi      {roi[2]}x{roi[3]} at {roi[0]},{roi[1]}   window {ww}x{wh} at {wx},{wy}")
    log(f"decode   {fmt_spans(spec.spans)}  ({spec.total:.1f}s, stride {plan_stride(info, spec)},"
        f" deint={spec.deint})")

    progress(0.02, "decoding frames")
    frames, times = read_frames(info, spec)
    n = len(frames)
    progress(0.18, f"{n} frames decoded")
    if n < 2:
        die("need at least 2 frames; widen --dur")
    # the decoded frames are the authority on the window's real size
    fh, fw = frames[0].shape[:2]
    if (fw, fh) != (ww, wh):
        window = (wx, wy, fw, fh)
        ww, wh = fw, fh
        spec.window = window
    roi_in_win = clamp_rect((roi[0] - wx, roi[1] - wy, roi[2], roi[3]), ww, wh)
    if roi_in_win[2] < roi[2] or roi_in_win[3] < roi[3]:
        log(f"note     ROI clipped to the decode window ({roi_in_win[2]}x{roi_in_win[3]})")
    log(f"frames   {n} decoded")

    # ---- reference -------------------------------------------------------
    if exclude_times:
        tol = 0.5 / max(1e-6, effective_fps(info, spec)) * max(1, plan_stride(info, spec))
        tarr = np.asarray(times)
        for et in exclude_times:
            j = int(np.argmin(np.abs(tarr - et)))
            if abs(tarr[j] - et) <= tol:
                excluded.add(j)
    if ref_time is not None and a.ref is None and times:
        ref_idx = int(np.argmin(np.abs(np.asarray(times) - float(ref_time))))

    grays = [to_gray32(f) for f in frames]
    x0, y0, rw, rh = roi_in_win
    sharps = [sharpness(g[y0:y0 + rh, x0:x0 + rw]) for g in grays]
    if ref_idx is None or ref_idx < 0 or ref_idx >= n:
        cand = [i for i in range(n) if i not in excluded] or list(range(n))
        ref_idx = max(cand, key=lambda i: sharps[i])
    log(f"ref      frame #{ref_idx} (t={times[ref_idx]:.3f}s, sharpness {sharps[ref_idx]:.5f})")

    mask = build_mask((wh, ww), roi_in_win, a.mask_grow)
    preps = [prep_for_align(g, a.hp_sigma) for g in grays]
    ref_prep, ref_gray = preps[ref_idx], grays[ref_idx]

    # ---- align -----------------------------------------------------------
    log(f"aligning {n} frames (motion={a.motion}) ...")
    results: list[AlignResult] = []
    init = None
    order = sorted(range(n), key=lambda i: abs(i - ref_idx))  # outward from the reference
    warps: dict[int, np.ndarray] = {}
    rhos: dict[int, float] = {}
    done = 0
    for i in order:
        if i == ref_idx:
            W = np.eye(3, dtype=np.float32)[:2] if a.motion != "homography" else np.eye(3, dtype=np.float32)
            warps[i], rhos[i] = W, 1.0
            continue
        prev = warps.get(i - 1 if i > ref_idx else i + 1)  # warm start from the neighbour
        W, rho = align_frame(ref_prep, preps[i], mask, a.motion, prev,
                             a.ecc_iters, a.ecc_eps, pyramid=not a.no_pyramid,
                             margin=a.model_margin)
        if W is None:
            W, rho = (np.eye(3, dtype=np.float32)[:2] if a.motion != "homography"
                      else np.eye(3, dtype=np.float32)), -1.0
        warps[i], rhos[i] = W, rho
        done += 1
        if done % 5 == 0 or done == n:
            progress(0.18 + 0.37 * done / max(1, n), f"aligning {done}/{n}")
        if a.verbose:
            log(f"    #{i:<4d} rho={rho:+.3f} dx={float(W[0, 2]):+.2f} dy={float(W[1, 2]):+.2f}")

    # ---- residual / occlusion score --------------------------------------
    # Encoders emit skip-blocks on static scenes: those frames are bit-identical
    # copies and contribute nothing, so find them before they skew the weights.
    dup: set[int] = set()
    prev_roi = None
    for i in range(n):
        cur_roi = grays[i][y0:y0 + rh, x0:x0 + rw]
        if prev_roi is not None and float(np.abs(cur_roi - prev_roi).mean()) < a.dup_eps:
            dup.add(i)
        prev_roi = cur_roi
    if dup:
        log(f"note     {len(dup)} frame(s) identical to their predecessor "
            f"(encoder skip) - they add no information")

    m = mask > 0
    for i in range(n):
        W = warps[i]
        aligned = warp_to_ref(grays[i], W, (ww, wh), cv2.INTER_LINEAR)
        cov = warp_to_ref(np.ones_like(grays[i]), W, (ww, wh), cv2.INTER_NEAREST) > 0.99
        sel = m & cov
        resid = float(np.mean(np.abs(aligned[sel] - ref_gray[sel]))) if sel.sum() > 32 else 1.0
        aligned_p = warp_to_ref(preps[i], W, (ww, wh), cv2.INTER_LINEAR)
        q = ncc(ref_prep, aligned_p, sel) if sel.sum() > 32 else 0.0
        dx, dy = float(warps[i][0, 2]), float(warps[i][1, 2])
        results.append(AlignResult(index=i, time=times[i], warp=warps[i], rho=rhos[i],
                                   sharp=sharps[i], residual=resid, shift=(dx, dy), ncc=q))

    res_vals = np.array([r.residual for r in results], np.float32)
    med = float(np.median(res_vals))
    mad = float(np.median(np.abs(res_vals - med))) * 1.4826 + 1e-6
    sharp_cut = 0.0
    if a.sharp_keep < 1.0:
        sharp_cut = float(np.quantile([r.sharp for r in results], 1.0 - a.sharp_keep))

    for r in results:
        if r.index in excluded:
            r.used, r.note = False, "manual"
        elif r.index in dup and r.index != ref_idx and not a.keep_duplicates:
            r.used, r.note = False, "duplicate"
        elif r.index != ref_idx and r.ncc < a.min_ncc:
            r.used, r.note = False, f"misaligned ncc={r.ncc:.2f}"
        elif r.index != ref_idx and r.residual > med + a.reject_k * mad:
            r.used, r.note = False, f"outlier {(r.residual - med) / mad:.1f}σ"
        elif r.index != ref_idx and r.sharp < sharp_cut:
            r.used, r.note = False, "soft"
    results[ref_idx].used = True
    used = [r for r in results if r.used]
    if not used:
        die("every frame was rejected; loosen --min-ecc / --reject-k")

    # sub-pixel diversity: without it, extra frames only reduce noise
    fr = np.array([[(r.shift[0] % 1.0), (r.shift[1] % 1.0)] for r in used])
    bins = min(int(a.scale), 4)
    hist = np.zeros((bins, bins))
    for fx, fy in fr:
        hist[min(bins - 1, int(fy * bins)), min(bins - 1, int(fx * bins))] += 1
    phase_cov = float((hist > 0).sum() / hist.size)
    log(f"kept     {len(used)}/{n} frames   median residual {med:.4f}   "
        f"sub-pixel phase coverage {phase_cov * 100:.0f}%")
    for r in results:
        if not r.used:
            log(f"  drop #{r.index:<4d} t={r.time:7.3f}  {r.note}")

    # ---- fuse ------------------------------------------------------------
    scale = float(a.scale)
    op = a.out_pad
    roi_out = expand_rect(roi_in_win, op, op, ww, wh)
    T = hr_transform(roi_out, scale)
    hr_w, hr_h = int(round(roi_out[2] * scale)), int(round(roi_out[3] * scale))
    hr_size = (hr_w, hr_h)
    need_gb = len(used) * hr_h * hr_w * 3 * 4 / 1e9
    log(f"fusing   {len(used)} frames -> {hr_w}x{hr_h} (x{scale:g}, {need_gb:.2f} GB stack)")
    if need_gb > a.mem_limit:
        die(f"stack would need {need_gb:.1f} GB (> --mem-limit {a.mem_limit}); "
            f"lower --scale, --max-frames or tighten the ROI")

    stack = np.empty((len(used), hr_h, hr_w, 3), np.float32)
    obs, wlist = [], []
    weights = np.ones(len(used), np.float32)
    # Weight by sharpness *relative to the median*, one-sided. On a static scene
    # every frame shows the same thing, so above-median Laplacian energy is noise,
    # not detail - rewarding it would preferentially stack the noisiest frames.
    smed = float(np.median([r.sharp for r in used])) or 1.0
    for k, r in enumerate(used):
        f32 = frames[r.index].astype(np.float32) / 255.0
        hr = warp_lr_to_hr(f32, r.warp, T, hr_size, cv2.INTER_CUBIC)
        cov = warp_lr_to_hr(np.ones(f32.shape[:2], np.float32), r.warp, T, hr_size,
                            cv2.INTER_NEAREST)
        hr[cov < 0.99] = np.nan
        stack[k] = hr
        obs.append(f32)
        wlist.append(r.warp)
        weights[k] = max(0.05, r.ncc) * min(1.0, r.sharp / smed) ** 0.5
    weights /= weights.mean()

    progress(0.58, f"fusing {len(used)} frames")
    fused, coverage = fuse_stack(stack, weights, a.fusion, a.trim)
    del stack
    thin = float((coverage < 1).mean())
    if thin > 0.02:
        log(f"note     {thin * 100:.1f}% of output pixels covered by no frame")

    # ---- back-projection + deconvolution ---------------------------------
    psf = a.psf if a.psf is not None else 0.35 * scale
    hr = fused
    if a.ibp > 0:
        log(f"refining  iterative back-projection x{a.ibp} (psf sigma {psf:.2f})")
        hr = back_project(fused, obs, wlist, T, weights, a.ibp, psf, a.lam)
    progress(0.86, "deconvolving")
    sharpened = richardson_lucy(hr, psf * a.rl_psf_scale, a.rl)
    final = unsharp(sharpened, max(0.6, psf * 0.7), a.unsharp)
    if a.denoise > 0:
        final = denoise(final, a.denoise)
    if a.clahe > 0:
        final = apply_clahe(final, a.clahe)
    if a.gamma != 1.0:
        final = np.clip(final, 0, 1) ** (1.0 / a.gamma)
    if a.gray:
        final = cv2.cvtColor(to_u8(final), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # ---- outputs ---------------------------------------------------------
    progress(0.92, "writing output")
    rx, ry, rw2, rh2 = roi_out
    ref_crop = frames[ref_idx][ry:ry + rh2, rx:rx + rw2].astype(np.float32) / 255.0
    bicubic = cv2.resize(ref_crop, hr_size, interpolation=cv2.INTER_CUBIC)
    nearest = cv2.resize(ref_crop, hr_size, interpolation=cv2.INTER_NEAREST)
    tag = f"x{scale:g}"
    paths = {
        "reference_crop": save_img(os.path.join(a.out, "01_reference_crop.png"), ref_crop),
        "baseline_bicubic": save_img(os.path.join(a.out, f"02_baseline_bicubic_{tag}.png"), bicubic),
        "fused": save_img(os.path.join(a.out, f"03_fused_{tag}.png"), fused),
        "backprojected": save_img(os.path.join(a.out, f"04_backprojected_{tag}.png"), hr),
        "result": save_img(os.path.join(a.out, f"05_result_{tag}.png"), final, a.bits),
    }
    cmp_img = side_by_side([
        (f"nearest {tag} (what you have)", nearest),
        (f"bicubic {tag} (naive upscale)", bicubic),
        (f"fused {len(used)} frames", fused),
        (f"+backproj +deconv (result)", final if final.ndim == 3 else cv2.cvtColor(to_u8(final), cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0),
    ])
    cv2.imwrite(os.path.join(a.out, "compare.png"), cmp_img)
    paths["compare"] = os.path.join(a.out, "compare.png")

    if not a.no_variants:
        tiles = []
        for rl in (0, max(4, a.rl // 2), a.rl, a.rl * 2):
            v = richardson_lucy(hr, psf * a.rl_psf_scale, rl)
            tiles.append((f"deconv {rl} it", v))
            tiles.append((f"deconv {rl} it + clahe", apply_clahe(v, 2.0)))
        rows = [side_by_side(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        wmax = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.zeros((r.shape[0], wmax - r.shape[1], 3), np.uint8)])
                if r.shape[1] < wmax else r for r in rows]
        cv2.imwrite(os.path.join(a.out, "variants.png"), np.vstack(rows))
        paths["variants"] = os.path.join(a.out, "variants.png")

    if a.corners:
        pts = [float(v) for v in re.findall(r"-?\d+\.?\d*", a.corners)]
        if len(pts) != 8:
            die("--corners needs 8 numbers: x1,y1,x2,y2,x3,y3,x4,y4 (TL,TR,BR,BL in source pixels)")
        src = np.array(pts, np.float64).reshape(4, 2)
        src[:, 0] -= wx
        src[:, 1] -= wy
        homo = np.hstack([src, np.ones((4, 1))])
        srch = (homo @ T.T)[:, :2].astype(np.float32)   # same LR->HR mapping as the fusion
        if a.rect_size:
            RW, RH = (int(v) for v in re.findall(r"\d+", a.rect_size)[:2])
        else:
            RW = int(round(max(np.linalg.norm(srch[1] - srch[0]), np.linalg.norm(srch[2] - srch[3]))))
            RH = int(round(max(np.linalg.norm(srch[3] - srch[0]), np.linalg.norm(srch[2] - srch[1]))))
        dst = np.array([[0, 0], [RW - 1, 0], [RW - 1, RH - 1], [0, RH - 1]], np.float32)
        M = cv2.getPerspectiveTransform(srch, dst)
        rect = cv2.warpPerspective(final, M, (max(2, RW), max(2, RH)), flags=cv2.INTER_CUBIC)
        paths["rectified"] = save_img(os.path.join(a.out, f"06_rectified_{tag}.png"), rect, a.bits)

    # ---- review page + report -------------------------------------------
    zoom = max(1, int(round(a.review_zoom)))
    uris = []
    for i in range(n):
        c = frames[i][roi_in_win[1]:roi_in_win[1] + roi_in_win[3],
                      roi_in_win[0]:roi_in_win[0] + roi_in_win[2]]
        c = cv2.resize(c, (c.shape[1] * zoom, c.shape[0] * zoom), interpolation=cv2.INTER_NEAREST)
        uris.append(jpeg_data_uri(c, None, 88))
    review = os.path.join(a.out, "review.html")
    write_picker(review, title="vidsr - review", heading="review frames used",
                 video=info.path, decode_spec=spec, window=roi, frames_uri=uris,
                 times=times, fps=info.fps * (2 if spec.deint == "field" else 1),
                 ref_index=ref_idx, ref_hd=None, roi=roi,
                 excluded=[r.index for r in results if not r.used],
                 thumb_width=max(90, roi[2] * zoom // 2),
                 cmd_prefix=f"{PROG} sr {shlex_quote(info.path)}",
                 out_hint=a.out)
    paths["review"] = review

    report = {
        "version": __version__,
        "video": os.path.abspath(info.path),
        "created": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        "roi": list(roi), "window": list(window), "roi_out": list(roi_out),
        "scale": scale, "decode": spec.to_json(),
        "reference_index": ref_idx, "reference_time": times[ref_idx],
        "frames_decoded": n, "frames_used": len(used),
        "phase_coverage": phase_cov,
        "median_residual": med,
        "settings": {k: getattr(a, k) for k in
                     ("motion", "fusion", "trim", "ibp", "lam", "rl", "unsharp",
                      "clahe", "denoise", "min_ncc", "reject_k", "mask_grow")},
        "psf_sigma": psf,
        "outputs": {k: os.path.abspath(v) for k, v in paths.items()},
        "frames": [{"i": r.index, "t": round(r.time, 4), "rho": round(r.rho, 4),
                    "sharp": round(r.sharp, 6), "residual": round(r.residual, 5),
                    "ncc": round(r.ncc, 4),
                    "dx": round(r.shift[0], 3), "dy": round(r.shift[1], 3),
                    "used": r.used, "note": r.note} for r in results],
    }
    with open(os.path.join(a.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    log("")
    print(f"result      {paths['result']}   ({hr_w}x{hr_h}, {len(used)} frames fused)")
    print(f"compare     {paths['compare']}")
    if "variants" in paths:
        print(f"variants    {paths['variants']}    <- pick the most readable, then tune --rl/--clahe")
    print(f"review      {paths['review']}    <- drop bad frames, export selection.json, re-run")
    print(f"report      {os.path.join(a.out, 'report.json')}")
    if phase_cov < 0.5:
        print(f"\nnote: sub-pixel phase coverage is only {phase_cov * 100:.0f}% - the object barely "
              f"moved between frames,\n      so extra frames mainly cut noise. A longer --dur (more "
              f"motion) usually helps more than a higher --scale.")
    progress(1.0, f"done in {_time.time() - t0:.1f}s")
    log(f"done in {_time.time() - t0:.1f}s")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

WORKFLOW = """\
typical session
  1  vidsr info  cam.mkv
  2  vidsr grid  cam.mkv --start 0 --end 2:00            # find the moment
  3  vidsr select cam.mkv --start 47 --dur 3 --out work  # opens a picker UI
       draw the box around the subject, drop the frames where someone walks
       through, download selection.json
  4  vidsr sr    cam.mkv --select work/selection.json --out out
  5  look at out/compare.png and out/variants.png; refine with
     out/review.html (re-export selection.json) or by tuning --rl / --clahe

no browser? skip step 3:
  vidsr sr cam.mkv --roi 812,430,96,34 --start 47 --dur 3 --out out
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=PROG,
        description="Multi-frame super-resolution for video: fuse many frames of one\n                 region (plate, sign, badge, face) into one sharper image.",
        epilog=WORKFLOW, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="stream properties (fps, codec, interlacing)")
    pi.add_argument("video")
    pi.set_defaults(func=cmd_info)

    pg = sub.add_parser("grid", help="contact sheet to locate the event")
    pg.add_argument("video")
    pg.add_argument("--start", default=0)
    pg.add_argument("--end")
    pg.add_argument("--count", type=int, default=12)
    pg.add_argument("--cols", type=int, default=4)
    pg.add_argument("--width", type=int, default=480, help="tile width in px")
    pg.add_argument("--out")
    pg.set_defaults(func=cmd_grid)

    ps = sub.add_parser("select", help="build the local HTML picker (ROI + frames)")
    ps.add_argument("video")
    ps.add_argument("--start", default=0)
    ps.add_argument("--end")
    ps.add_argument("--dur", help="seconds to take from --start (default 4)")
    ps.add_argument("--use", action="append", metavar="A-B",
                    help="only this span, repeatable: --use 10:00-15:00")
    ps.add_argument("--skip", action="append", metavar="A-B",
                    help="ignore this span, repeatable: --skip 12:30-14:30")
    ps.add_argument("--step", type=int, default=1, help="use every Nth frame")
    ps.add_argument("--max-frames", type=int, default=150)
    ps.add_argument("--deint", choices=list(DEINT_FILTERS), default="none",
                    help="'field' turns each interlaced field into its own sample")
    ps.add_argument("--roi", help="pre-set ROI x,y,w,h (second pass)")
    ps.add_argument("--window", help="restrict decoding to x,y,w,h")
    ps.add_argument("--pad", type=int, help="context around --roi for the window")
    ps.add_argument("--crop", action="store_true",
                    help="with --roi: embed native-resolution ROI crops instead of full frames")
    ps.add_argument("--zoom", type=int, default=4, help="magnification for --crop thumbs")
    ps.add_argument("--view-width", type=int, default=760)
    ps.add_argument("--thumb-width", type=int, default=120)
    ps.add_argument("--ref", type=int, help="reference frame index (default: sharpest)")
    ps.add_argument("--out", default="work")
    ps.set_defaults(func=cmd_select)

    pr = sub.add_parser("sr", help="align + fuse + deconvolve the selected frames",
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    pr.add_argument("video", nargs="?")
    pr.add_argument("--select", help="selection.json from the picker UI")
    pr.add_argument("--out", default="out")
    pr.add_argument("-v", "--verbose", action="store_true", help="per-frame alignment detail")
    pr.add_argument("--preset", choices=["static", "moving"],
                    help="static: subject and camera both still (noise+blur limited). "
                         "moving: subject crosses the frame (true sub-pixel SR). "
                         "Only fills in options you did not set yourself.")

    g = pr.add_argument_group("what to process")
    g.add_argument("--roi", help="x,y,w,h in source pixels (overrides the selection)")
    g.add_argument("--start")
    g.add_argument("--end")
    g.add_argument("--dur")
    g.add_argument("--use", action="append", metavar="A-B",
                   help="only these spans, repeatable: --use 10:00-15:00")
    g.add_argument("--skip", action="append", metavar="A-B",
                   help="ignore these spans, repeatable: --skip 12:30-14:30")
    g.add_argument("--step", type=int, default=1)
    g.add_argument("--max-frames", type=int, default=300,
                   help="frame budget, spread evenly over the selection")
    g.add_argument("--deint", choices=list(DEINT_FILTERS), default="none")
    g.add_argument("--ref", type=int, help="reference frame index (default: sharpest)")
    g.add_argument("--ref-time", type=float, help="pick the reference frame by timestamp")
    g.add_argument("--exclude", help="frame indices to drop, e.g. 12-19,44")
    g.add_argument("--pad", type=int, help="alignment context around the ROI (px)")
    g.add_argument("--out-pad", type=int, default=3, help="extra ROI margin in the output (px)")

    g = pr.add_argument_group("reconstruction")
    g.add_argument("--scale", type=float, default=4.0)
    g.add_argument("--motion", default="auto",
                   choices=["auto", "translation", "euclidean", "affine", "homography"],
                   help="auto = translation -> euclidean -> affine")
    g.add_argument("--fusion", default="trimmed", choices=["trimmed", "median", "mean"])
    g.add_argument("--trim", type=float, default=0.25, help="fraction trimmed per pixel")
    g.add_argument("--ibp", type=int, default=12, help="back-projection iterations (0 = off)")
    g.add_argument("--lam", type=float, default=0.7, help="back-projection step size")
    g.add_argument("--psf", type=float, help="PSF sigma on the HR grid (default 0.35*scale)")

    g = pr.add_argument_group("sharpening / look")
    g.add_argument("--rl", type=int, default=10, help="Richardson-Lucy iterations")
    g.add_argument("--rl-psf-scale", type=float, default=1.0)
    g.add_argument("--unsharp", type=float, default=0.4)
    g.add_argument("--clahe", type=float, default=1.5, help="local contrast clip (0 = off)")
    g.add_argument("--denoise", type=float, default=0.0)
    g.add_argument("--gamma", type=float, default=1.0)
    g.add_argument("--gray", action="store_true")
    g.add_argument("--bits", type=int, default=8, choices=[8, 16])

    g = pr.add_argument_group("frame rejection / alignment")
    g.add_argument("--min-ncc", type=float, default=0.5,
                   help="drop frames whose aligned ROI correlates below this with the reference")
    g.add_argument("--reject-k", type=float, default=3.0, help="outlier cut in MAD sigmas")
    g.add_argument("--sharp-keep", type=float, default=1.0, help="keep only the sharpest fraction")
    g.add_argument("--mask-grow", type=float, default=1.6, help="align on ROI x this factor")
    g.add_argument("--hp-sigma", type=float, default=3.0)
    g.add_argument("--ecc-iters", type=int, default=200)
    g.add_argument("--ecc-eps", type=float, default=1e-6)
    g.add_argument("--no-pyramid", action="store_true")
    g.add_argument("--model-margin", type=float, default=0.0,
                   help="min ECC gain required to accept a richer motion model")
    g.add_argument("--dup-eps", type=float, default=1e-5,
                   help="mean-abs-diff below which a frame counts as a duplicate")
    g.add_argument("--keep-duplicates", action="store_true")

    g = pr.add_argument_group("extras")
    g.add_argument("--corners", help="TL,TR,BR,BL of a flat subject in source px -> deskewed output")
    g.add_argument("--rect-size", help="WxH for --corners output")
    g.add_argument("--review-zoom", type=float, default=3)
    g.add_argument("--no-variants", action="store_true")
    g.add_argument("--mem-limit", type=float, default=6.0, help="max stack size in GB")
    pr.set_defaults(func=cmd_sr)
    p.set_defaults(_sr_parser=pr)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    a = parser.parse_args(argv)
    if a.cmd == "sr":
        apply_preset(a, a._sr_parser)
    try:
        return a.func(a)
    except KeyboardInterrupt:
        log("\ninterrupted")
        return 130
    except BrokenPipeError:
        return 141


if __name__ == "__main__":
    sys.exit(main())
