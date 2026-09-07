#!/usr/bin/env python3
"""Synthesise a security-cam clip with a known plate, to validate the pipeline.

The plate is rendered at 4x, blurred by a simulated lens, decimated to video
resolution, then noised and compressed - i.e. the same information loss a real
camera causes. A pedestrian occludes the plate for part of the clip so frame
rejection can be exercised too. Ground truth is printed on exit.
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

S = 4  # render supersampling


def plate_rect(W, H, t, mo, plate_px):
    """Where the plate is at time-index t, in render (S x video) pixels."""
    w, h = W * S, H * S
    zoom = 1.0 + mo["zoom"] * t
    pw = plate_px * S * zoom
    ph = pw * 64.0 / 272.0
    ch = pw * 150.0 / 272.0
    cx = w * 0.50 + mo["dx"] * S * t
    cy = h * 0.70 + mo["dy"] * S * t
    y0 = cy - ch / 2
    return cx - pw / 2, y0 + ch - ph - ch * 0.14, pw, ph


def plate_roi_video(W, H, t, mo, plate_px, margin=2):
    """Same rectangle in video pixels, padded - what you would draw in the UI."""
    px, py, pw, ph = plate_rect(W, H, t, mo, plate_px)
    return (int(px / S) - margin, int(py / S) - margin,
            int(round(pw / S)) + 2 * margin, int(round(ph / S)) + 2 * margin)


def plate_occlusion(W, H, t, mo, plate_px, occl):
    """Fraction of the plate's width the pedestrian actually covers at this
    frame. The walker starts and ends beside the plate, so only the middle of
    its crossing genuinely hides anything."""
    if occl is None:
        return 0.0
    px, py, pw, ph = plate_rect(W, H, t, mo, plate_px)
    ox = px + pw * 0.5 + (occl - 0.5) * pw * 2.4
    lo, hi = max(px, ox - 0.22 * pw), min(px + pw, ox + 0.22 * pw)
    return max(0.0, hi - lo) / pw


def render(W, H, t, plate_text, occl, mo, plate_px):
    """One ground-truth scene at S-times video resolution."""
    w, h = W * S, H * S
    img = np.full((h, w, 3), 60, np.uint8)
    cv2.rectangle(img, (0, int(h * 0.62)), (w, h), (72, 72, 76), -1)      # road
    for x in range(0, w, 420):                                            # lane marks
        cv2.rectangle(img, (x, int(h * 0.80)), (x + 120, int(h * 0.80) + 10), (150, 150, 150), -1)
    cv2.rectangle(img, (0, 0), (w, int(h * 0.62)), (48, 52, 58), -1)      # wall

    px, py, pw, ph = plate_rect(W, H, t, mo, plate_px)
    cw, ch = pw * 430.0 / 272.0, pw * 150.0 / 272.0
    cx = w * 0.50 + mo["dx"] * S * t
    cy = h * 0.70 + mo["dy"] * S * t
    x0, y0 = int(cx - cw / 2), int(cy - ch / 2)
    cv2.rectangle(img, (x0, y0), (int(x0 + cw), int(y0 + ch)), (36, 34, 40), -1)
    cv2.rectangle(img, (x0 + int(cw * .05), y0 + int(ch * .09)),
                  (int(x0 + cw * .95), y0 + int(ch * .36)), (60, 58, 66), -1)

    ipx, ipy, ipw, iph = int(px), int(py), int(pw), int(ph)
    cv2.rectangle(img, (ipx, ipy), (ipx + ipw, ipy + iph), (242, 242, 240), -1)
    cv2.rectangle(img, (ipx, ipy), (ipx + ipw, ipy + iph), (20, 20, 20), max(1, int(iph * .03)))
    cv2.rectangle(img, (ipx + 2, ipy + 2), (ipx + int(iph * .30), ipy + iph - 2), (150, 60, 20), -1)

    # shrink the glyphs until the whole registration fits inside the plate:
    # a real plate never has characters hanging off its edge
    th = max(1, int(round(iph * 0.052)))
    avail = ipw - int(iph * .34) - int(iph * .12)
    fs = 0.030 * iph
    (tw, tht), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_DUPLEX, fs, th)
    if tw > avail:
        fs *= avail / float(tw)
        (tw, tht), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_DUPLEX, fs, th)
    tx = ipx + int(iph * .34) + max(0, (ipw - int(iph * .34) - tw) // 2)
    cv2.putText(img, plate_text, (tx, ipy + (iph + tht) // 2),
                cv2.FONT_HERSHEY_DUPLEX, fs, (18, 18, 22), th, cv2.LINE_AA)

    if occl is not None:                            # pedestrian crossing the plate
        ox = int(px + pw * 0.5 + (occl - 0.5) * pw * 2.4)
        cv2.rectangle(img, (ox - int(pw * .22), y0 - int(ch * .8)),
                      (ox + int(pw * .22), int(y0 + ch * 1.3)), (28, 30, 44), -1)
        cv2.circle(img, (ox, y0 - int(ch * 1.0)), int(pw * .18), (40, 44, 60), -1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test_cam.mkv")
    ap.add_argument("--plate", default="MKX 8317")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--noise", type=float, default=4.0)
    ap.add_argument("--occlude", default="38-52", help="frame range hidden by a pedestrian")
    ap.add_argument("--codec", default="auto")
    ap.add_argument("--plate-px", type=float, default=68, help="plate width in video pixels")
    ap.add_argument("--dx", type=float, default=1.37, help="object drift px/frame (0 = parked)")
    ap.add_argument("--dy", type=float, default=0.42)
    ap.add_argument("--zoom-rate", type=float, default=0.00055)
    ap.add_argument("--jitter", type=float, default=0.25, help="camera shake px (0 = rigid mount)")
    ap.add_argument("--bitrate", default="1400k")
    a = ap.parse_args()

    oc0, oc1 = (int(v) for v in a.occlude.split("-")) if a.occlude else (-1, -1)
    rng = np.random.default_rng(7)

    # whichever H.264 encoder this ffmpeg has; mpeg4/ffv1 only as a last resort,
    # since their artefacts differ enough to move the quality thresholds
    codecs = ([a.codec] if a.codec != "auto"
              else ["libx264", "libopenh264", "mpeg4", "ffv1"])
    last = ""
    for codec in codecs:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{a.width}x{a.height}", "-r", str(a.fps), "-i", "-",
               "-c:v", codec]
        if codec in ("libx264", "libopenh264"):
            cmd += ["-b:v", a.bitrate]
        elif codec == "mpeg4":
            cmd += ["-q:v", "6"]
        cmd += ["-pix_fmt", "yuv420p", a.out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for i in range(a.frames):
                occ = None
                if oc0 <= i <= oc1:
                    occ = (i - oc0) / max(1, (oc1 - oc0))
                mo = {"dx": a.dx, "dy": a.dy, "zoom": a.zoom_rate}
                hi = render(a.width, a.height, i, a.plate, occ, mo, a.plate_px)
                hi = cv2.GaussianBlur(hi, (0, 0), 1.5 * S / 4)          # lens
                lo = cv2.resize(hi, (a.width, a.height), interpolation=cv2.INTER_AREA)
                jx, jy = rng.normal(0, a.jitter, 2) if a.jitter > 0 else (0.0, 0.0)
                M = np.float32([[1, 0, jx], [0, 1, jy]])
                lo = cv2.warpAffine(lo, M, (a.width, a.height), flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
                lo = lo.astype(np.float32) * (1.0 + rng.normal(0, 0.012))
                lo += rng.normal(0, a.noise, lo.shape)                   # sensor noise
                p.stdin.write(np.clip(lo, 0, 255).astype(np.uint8).tobytes())
            p.stdin.close()
            p.wait(timeout=120)
        except Exception as e:
            last = f"{codec}: {e}"
            continue
        if p.returncode == 0 and os.path.exists(a.out) and os.path.getsize(a.out) > 1000:
            err = p.stderr.read().decode(errors="replace")[:200]
            print(f"wrote {a.out}  codec={codec}  {a.frames} frames  "
                  f"{a.width}x{a.height}@{a.fps}  {os.path.getsize(a.out)/1e6:.2f} MB")
            if err.strip():
                print("ffmpeg:", err.strip()[:200], file=sys.stderr)
            print(f"ground truth plate text: {a.plate!r}")
            print(f"occluded frames: {a.occlude}")
            return 0
        last = f"{codec}: rc={p.returncode} {p.stderr.read().decode(errors='replace')[:200]}"
    sys.exit(f"encoding failed. {last}")


if __name__ == "__main__":
    sys.exit(main())
