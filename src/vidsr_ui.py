"""
vidsr_ui - a small desktop UI for picking what to reconstruct.

Everything here is a front end for the same CLI the headless path uses: the
window collects a selection, turns it into an argument list, and hands that to
vidsr's own parser. There is no second code path, so what you can do in the UI
you can always reproduce on a headless box - the window shows you the exact
command it is about to run.

Tkinter only, so it works on a stock Python without extra packages.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import sys
import threading
import traceback
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

import vidsr

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover - depends on the OS package
    tk = None


TK_HELP = """\
This UI needs Tk, which your Python was built without.

  Debian/Kali/Ubuntu   sudo apt install python3-tk
  Fedora               sudo dnf install python3-tkinter
  macOS (homebrew)     brew install python-tk

Everything the UI does is available headless, for example:
  vidsr select cam.mkv --use 10:00-15:00 --out work   # HTML picker
  vidsr sr cam.mkv --roi X,Y,W,H --use 10:00-15:00 --preset static --out out
"""


# ---------------------------------------------------------------------------
# selection -> command line (pure, so it can be tested without a display)
# ---------------------------------------------------------------------------


@dataclass
class Selection:
    video: str = ""
    roi: Optional[tuple] = None          # x, y, w, h in source pixels
    start: float = 0.0
    end: float = 0.0
    skips: list = field(default_factory=list)   # [(a, b), ...] seconds
    preset: str = "static"               # static | moving | auto
    scale: float = 4.0
    max_frames: int = 300
    out: str = "out"

    def spans(self) -> list:
        """What actually gets read, after the ignored spans are removed."""
        return vidsr.subtract_spans([(self.start, self.end)],
                                    [tuple(s) for s in self.skips])


def build_args(sel: Selection) -> list:
    """The exact `vidsr sr ...` argument list for this selection."""
    if not sel.video:
        raise ValueError("no video selected")
    if not sel.roi:
        raise ValueError("draw a box around the subject first")
    if sel.end <= sel.start:
        raise ValueError("the time range is empty")
    x, y, w, h = (int(v) for v in sel.roi)
    args = ["sr", sel.video,
            "--roi", f"{x},{y},{w},{h}",
            "--use", f"{sel.start:.3f}-{sel.end:.3f}"]
    for a, b in sel.skips:
        args += ["--skip", f"{float(a):.3f}-{float(b):.3f}"]
    if sel.preset in ("static", "moving"):
        args += ["--preset", sel.preset]
    args += ["--scale", f"{sel.scale:g}",
             "--max-frames", str(int(sel.max_frames)),
             "--out", sel.out]
    return args


def run_selection(sel: Selection, on_progress=None) -> dict:
    """Run the reconstruction in-process and return its report."""
    parser = vidsr.build_parser()
    a = parser.parse_args(build_args(sel))
    vidsr.apply_preset(a, a._sr_parser)
    if on_progress:
        vidsr.set_progress(on_progress)
    try:
        vidsr.cmd_sr(a)
    finally:
        vidsr.set_progress(None)
    with open(os.path.join(sel.out, "report.json")) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# geometry (pure, and therefore testable without a display)
# ---------------------------------------------------------------------------


def fit_view(src_w: int, src_h: int, view_w: int, view_h: int):
    """Letterbox a frame into the canvas: (scale, x offset, y offset)."""
    s = min(view_w / max(1, src_w), view_h / max(1, src_h))
    dw, dh = max(1, int(src_w * s)), max(1, int(src_h * s))
    return s, (view_w - dw) // 2, (view_h - dh) // 2


def view_to_source(cx: float, cy: float, disp):
    """Canvas point -> source pixel. Getting this wrong picks the wrong ROI."""
    s, ox, oy = disp
    return (cx - ox) / s, (cy - oy) / s


def source_to_view(x: float, y: float, disp):
    s, ox, oy = disp
    return x * s + ox, y * s + oy


def rect_from_drag(p0, p1):
    """Two source-space points -> an (x, y, w, h) rect, dragged any direction."""
    (x0, y0), (x1, y1) = p0, p1
    return (int(min(x0, x1)), int(min(y0, y1)),
            max(2, int(abs(x1 - x0))), max(2, int(abs(y1 - y0))))


def time_to_x(t: float, dur: float, width: int) -> float:
    return (t / max(1e-9, dur)) * max(1, width)


def x_to_time(x: float, dur: float, width: int) -> float:
    return max(0.0, min(dur, x / max(1, width) * dur))


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------


def _photo(bgr: np.ndarray):
    """A Tk image from a BGR array, without needing PIL."""
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("failed to encode frame")
    return tk.PhotoImage(data=base64.b64encode(buf.tobytes()))


class App:
    VIEW_W, VIEW_H = 780, 440
    TL_H = 58

    def __init__(self, root, video: Optional[str], out: str):
        self.root = root
        self.sel = Selection(out=out)
        self.info: Optional[vidsr.VideoInfo] = None
        self.frame_cache: dict = {}
        self.cur_t = 0.0
        self.disp = None                 # (scale, ox, oy) of the shown frame
        self.photo = None
        self.roi_photo = None
        self.res_photo = None
        self.drag = None
        self.tl_drag = None
        self.pending_skip = None
        self.q: queue.Queue = queue.Queue()
        self.running = False
        self.report = None

        root.title("vidsr")
        root.minsize(1100, 720)
        self._build()
        if video:
            self.load(video)
        self.root.after(80, self._poll)

    # -- layout ------------------------------------------------------------
    def _build(self):
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill="x")
        ttk.Button(top, text="Open video…", command=self.pick_file).pack(side="left")
        self.file_lbl = ttk.Label(top, text="no video loaded", foreground="#666")
        self.file_lbl.pack(side="left", padx=10)

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=10)
        self.tab_sel = ttk.Frame(self.nb)
        self.tab_res = ttk.Frame(self.nb)
        self.nb.add(self.tab_sel, text="  Select  ")
        self.nb.add(self.tab_res, text="  Result  ")
        self._build_select(self.tab_sel)
        self._build_result(self.tab_res)

        bar = ttk.Frame(self.root, padding=(10, 8))
        bar.pack(fill="x")
        self.run_btn = ttk.Button(bar, text="Run reconstruction", command=self.run)
        self.run_btn.pack(side="left")
        ttk.Button(bar, text="Copy CLI command", command=self.copy_cmd).pack(side="left", padx=6)
        self.pbar = ttk.Progressbar(bar, mode="determinate", maximum=1000, length=320)
        self.pbar.pack(side="left", padx=12)
        self.status = ttk.Label(bar, text="load a video to begin", foreground="#666")
        self.status.pack(side="left")

    def _build_select(self, parent):
        left = ttk.Frame(parent)
        left.pack(side="left", fill="both", expand=True, pady=8)
        self.canvas = tk.Canvas(left, width=self.VIEW_W, height=self.VIEW_H,
                                bg="#111", highlightthickness=1,
                                highlightbackground="#444", cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        nav = ttk.Frame(left)
        nav.pack(fill="x", pady=(8, 2))
        ttk.Button(nav, text="◀", width=3, command=lambda: self.step(-1)).pack(side="left")
        ttk.Button(nav, text="▶", width=3, command=lambda: self.step(1)).pack(side="left", padx=(2, 8))
        self.tvar = tk.DoubleVar(value=0.0)
        self.slider = ttk.Scale(nav, from_=0, to=1, variable=self.tvar,
                                command=lambda _v: self._slider_moved())
        self.slider.pack(side="left", fill="x", expand=True)
        self.time_lbl = ttk.Label(nav, text="0:00.000", width=12)
        self.time_lbl.pack(side="left", padx=6)

        self.timeline = tk.Canvas(left, height=self.TL_H, bg="#15181c",
                                  highlightthickness=1, highlightbackground="#444")
        self.timeline.pack(fill="x", pady=(6, 0))
        self.timeline.bind("<ButtonPress-1>", self.tl_press)
        self.timeline.bind("<B1-Motion>", self.tl_move)
        self.timeline.bind("<ButtonRelease-1>", self.tl_release)
        ttk.Label(left, foreground="#666",
                  text="drag on the timeline = set the range to use   ·   "
                       "shift+drag = mark a span to ignore   ·   click = seek").pack(anchor="w", pady=(3, 0))

        right = ttk.Frame(parent, padding=(12, 8))
        right.pack(side="left", fill="y")

        g = ttk.LabelFrame(right, text="Range to use", padding=8)
        g.pack(fill="x")
        self.start_var, self.end_var = tk.StringVar(value="0"), tk.StringVar(value="0")
        row = ttk.Frame(g); row.pack(fill="x")
        ttk.Label(row, text="from", width=5).pack(side="left")
        ttk.Entry(row, textvariable=self.start_var, width=11).pack(side="left")
        ttk.Button(row, text="⤓ here", width=7,
                   command=lambda: self.set_edge("start")).pack(side="left", padx=3)
        row = ttk.Frame(g); row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="to", width=5).pack(side="left")
        ttk.Entry(row, textvariable=self.end_var, width=11).pack(side="left")
        ttk.Button(row, text="⤓ here", width=7,
                   command=lambda: self.set_edge("end")).pack(side="left", padx=3)
        self.range_lbl = ttk.Label(g, text="", foreground="#666")
        self.range_lbl.pack(anchor="w", pady=(5, 0))
        for v in (self.start_var, self.end_var):
            v.trace_add("write", lambda *_: self.sync_range())

        g = ttk.LabelFrame(right, text="Spans to ignore", padding=8)
        g.pack(fill="x", pady=(10, 0))
        self.skip_list = tk.Listbox(g, height=4, activestyle="none")
        self.skip_list.pack(fill="x")
        row = ttk.Frame(g); row.pack(fill="x", pady=(5, 0))
        self.skip_btn = ttk.Button(row, text="Start ignoring here", command=self.mark_skip)
        self.skip_btn.pack(side="left")
        ttk.Button(row, text="Remove", command=self.remove_skip).pack(side="left", padx=4)

        g = ttk.LabelFrame(right, text="Subject", padding=8)
        g.pack(fill="x", pady=(10, 0))
        self.preset_var = tk.StringVar(value="static")
        for val, txt in (("static", "Still (parked car, fixed camera)"),
                         ("moving", "Moving through the frame"),
                         ("auto", "Neither / let me tune it")):
            ttk.Radiobutton(g, text=txt, value=val, variable=self.preset_var).pack(anchor="w")

        g = ttk.LabelFrame(right, text="Region", padding=8)
        g.pack(fill="x", pady=(10, 0))
        self.roi_lbl = ttk.Label(g, text="drag a box on the frame", foreground="#666")
        self.roi_lbl.pack(anchor="w")
        self.roi_canvas = tk.Canvas(g, width=250, height=90, bg="#111",
                                    highlightthickness=1, highlightbackground="#444")
        self.roi_canvas.pack(pady=(5, 0))
        ttk.Button(g, text="Clear", command=self.clear_roi).pack(anchor="w", pady=(5, 0))

        g = ttk.LabelFrame(right, text="Output", padding=8)
        g.pack(fill="x", pady=(10, 0))
        row = ttk.Frame(g); row.pack(fill="x")
        ttk.Label(row, text="zoom", width=6).pack(side="left")
        self.scale_var = tk.StringVar(value="4")
        ttk.Combobox(row, textvariable=self.scale_var, width=5, state="readonly",
                     values=["2", "3", "4", "6", "8"]).pack(side="left")
        row = ttk.Frame(g); row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="frames", width=6).pack(side="left")
        self.frames_var = tk.StringVar(value="300")
        ttk.Spinbox(row, textvariable=self.frames_var, from_=10, to=2000,
                    increment=10, width=7).pack(side="left")

    def _build_result(self, parent):
        bar = ttk.Frame(parent, padding=(8, 8))
        bar.pack(fill="x")
        ttk.Button(bar, text="Save image as…", command=self.save_result).pack(side="left")
        ttk.Button(bar, text="Open output folder", command=self.open_folder).pack(side="left", padx=6)
        self.view_var = tk.StringVar(value="result")
        for val, txt in (("result", "Result"), ("compare", "Side by side"),
                         ("variants", "Sharpening variants")):
            ttk.Radiobutton(bar, text=txt, value=val, variable=self.view_var,
                            command=self.show_result).pack(side="left", padx=(8, 0))
        self.res_lbl = ttk.Label(parent, text="no reconstruction yet", foreground="#666",
                                 padding=(8, 4))
        self.res_lbl.pack(anchor="w")
        self.res_canvas = tk.Canvas(parent, bg="#111", highlightthickness=0)
        self.res_canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # -- video -------------------------------------------------------------
    def pick_file(self):
        p = filedialog.askopenfilename(
            title="Open video",
            filetypes=[("Video", "*.mkv *.mp4 *.avi *.mov *.ts *.m2ts *.wmv *.flv"),
                       ("All files", "*")])
        if p:
            self.load(p)

    def load(self, path: str):
        try:
            self.info = vidsr.probe(path)
        except SystemExit as e:
            messagebox.showerror("vidsr", str(e))
            return
        self.sel.video = path
        self.frame_cache.clear()
        dur = self.info.duration or 1.0
        self.sel.start, self.sel.end = 0.0, dur
        self.sel.skips.clear()
        self.skip_list.delete(0, "end")
        self.start_var.set(f"{0:.3f}")
        self.end_var.set(f"{dur:.3f}")
        self.slider.configure(to=max(0.001, dur))
        self.file_lbl.configure(
            text=f"{os.path.basename(path)}   {self.info.width}×{self.info.height}   "
                 f"{self.info.fps:.2f} fps   {vidsr.fmt_time(dur)}"
                 + ("   [interlaced]" if self.info.interlaced else ""),
            foreground="#000")
        self.status.configure(text="draw a box around the subject")
        self.seek(0.0)
        self.draw_timeline()

    def seek(self, t: float):
        if not self.info:
            return
        t = max(0.0, min(t, max(0.0, (self.info.duration or 1.0) - 1e-3)))
        self.cur_t = t
        self.tvar.set(t)
        self.time_lbl.configure(text=vidsr.fmt_time(t))
        key = round(t, 2)
        if key not in self.frame_cache:
            try:
                self.frame_cache[key] = vidsr.grab_frame(self.info, t)
            except SystemExit:
                return
            if len(self.frame_cache) > 40:
                self.frame_cache.pop(next(iter(self.frame_cache)))
        self.show_frame(self.frame_cache[key])
        self.draw_timeline()

    def show_frame(self, bgr):
        h, w = bgr.shape[:2]
        s, ox, oy = fit_view(w, h, self.VIEW_W, self.VIEW_H)
        img = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
        self.disp = (s, ox, oy)
        self.photo = _photo(img)
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self.photo)
        self.draw_roi()

    def step(self, n: int):
        if self.info:
            self.seek(self.cur_t + n / max(1.0, self.info.fps))

    def _slider_moved(self):
        if not self.info:
            return
        t = float(self.tvar.get())
        self.time_lbl.configure(text=vidsr.fmt_time(t))
        self.cur_t = t
        self.root.after_cancel(getattr(self, "_seek_job", "")) if getattr(self, "_seek_job", None) else None
        self._seek_job = self.root.after(120, lambda: self.seek(float(self.tvar.get())))

    # -- ROI ---------------------------------------------------------------
    def to_src(self, cx, cy):
        return view_to_source(cx, cy, self.disp)

    def on_press(self, e):
        if self.disp:
            self.drag = self.to_src(e.x, e.y)

    def on_move(self, e):
        if not self.drag:
            return
        self.sel.roi = rect_from_drag(self.drag, self.to_src(e.x, e.y))
        self.draw_roi()

    def on_release(self, e):
        self.drag = None
        if self.sel.roi and self.info:
            self.sel.roi = vidsr.clamp_rect(self.sel.roi, self.info.width, self.info.height)
            self.draw_roi()
            self.update_status()

    def draw_roi(self):
        self.canvas.delete("roi")
        if not (self.sel.roi and self.disp):
            return
        x, y, w, h = self.sel.roi
        vx0, vy0 = source_to_view(x, y, self.disp)
        vx1, vy1 = source_to_view(x + w, y + h, self.disp)
        self.canvas.create_rectangle(vx0, vy0, vx1, vy1,
                                     outline="#4da3ff", width=2, tags="roi")
        self.roi_lbl.configure(text=f"{w} × {h} px at {x},{y}", foreground="#000")
        key = round(self.cur_t, 2)
        if key in self.frame_cache:
            crop = self.frame_cache[key][y:y + h, x:x + w]
            if crop.size:
                z = max(1, min(250 // max(1, w), 90 // max(1, h), 12))
                big = cv2.resize(crop, (w * z, h * z), interpolation=cv2.INTER_NEAREST)
                self.roi_photo = _photo(big)
                self.roi_canvas.delete("all")
                self.roi_canvas.create_image(125, 45, image=self.roi_photo)

    def clear_roi(self):
        self.sel.roi = None
        self.roi_canvas.delete("all")
        self.roi_lbl.configure(text="drag a box on the frame", foreground="#666")
        self.canvas.delete("roi")

    # -- range and skips ---------------------------------------------------
    def set_edge(self, which):
        (self.start_var if which == "start" else self.end_var).set(f"{self.cur_t:.3f}")

    def sync_range(self):
        try:
            self.sel.start = float(self.start_var.get() or 0)
            self.sel.end = float(self.end_var.get() or 0)
        except ValueError:
            return
        self.update_status()
        self.draw_timeline()

    def mark_skip(self):
        if self.pending_skip is None:
            self.pending_skip = self.cur_t
            self.skip_btn.configure(text="…end it here")
        else:
            a, b = sorted((self.pending_skip, self.cur_t))
            self.pending_skip = None
            self.skip_btn.configure(text="Start ignoring here")
            if b - a > 1e-3:
                self.add_skip(a, b)

    def add_skip(self, a, b):
        self.sel.skips = vidsr.merge_spans(self.sel.skips + [(a, b)])
        self.skip_list.delete(0, "end")
        for s0, s1 in self.sel.skips:
            self.skip_list.insert("end", f"{vidsr.fmt_time(s0)} – {vidsr.fmt_time(s1)}")
        self.update_status()
        self.draw_timeline()

    def remove_skip(self):
        for i in reversed(self.skip_list.curselection()):
            del self.sel.skips[i]
        self.skip_list.delete(0, "end")
        for s0, s1 in self.sel.skips:
            self.skip_list.insert("end", f"{vidsr.fmt_time(s0)} – {vidsr.fmt_time(s1)}")
        self.update_status()
        self.draw_timeline()

    # -- timeline ----------------------------------------------------------
    def tl_x(self, t):
        dur = (self.info.duration if self.info else 1.0) or 1.0
        return time_to_x(t, dur, self.timeline.winfo_width())

    def tl_t(self, x):
        dur = (self.info.duration if self.info else 1.0) or 1.0
        return x_to_time(x, dur, self.timeline.winfo_width())

    def draw_timeline(self):
        c = self.timeline
        c.delete("all")
        if not self.info:
            return
        w, h = max(1, c.winfo_width()), self.TL_H
        c.create_rectangle(0, 0, w, h, fill="#15181c", outline="")
        c.create_rectangle(self.tl_x(self.sel.start), 6, self.tl_x(self.sel.end), h - 16,
                           fill="#123a2a", outline="#3ddc97")
        for a, b in self.sel.skips:
            c.create_rectangle(self.tl_x(a), 6, self.tl_x(b), h - 16,
                               fill="#3a1b1e", outline="#ff6b6b")
        if self.tl_drag:
            a, b = sorted(self.tl_drag[:2])
            c.create_rectangle(self.tl_x(a), 6, self.tl_x(b), h - 16,
                               fill="#2a3550", outline="#4da3ff")
        if self.pending_skip is not None:
            c.create_line(self.tl_x(self.pending_skip), 0, self.tl_x(self.pending_skip), h,
                          fill="#ff6b6b", dash=(3, 2))
        x = self.tl_x(self.cur_t)
        c.create_line(x, 0, x, h - 12, fill="#4da3ff", width=2)
        dur = self.info.duration or 1.0
        for k in range(7):
            t = dur * k / 6
            c.create_text(min(w - 26, max(24, self.tl_x(t))), h - 6,
                          text=vidsr.fmt_time(t), fill="#98a2b3", font=("TkDefaultFont", 7))

    def tl_press(self, e):
        self.tl_drag = (self.tl_t(e.x), self.tl_t(e.x), bool(e.state & 0x1))

    def tl_move(self, e):
        if self.tl_drag:
            self.tl_drag = (self.tl_drag[0], self.tl_t(e.x), self.tl_drag[2])
            self.draw_timeline()

    def tl_release(self, e):
        if not self.tl_drag:
            return
        a, b, shift = self.tl_drag
        self.tl_drag = None
        a, b = sorted((a, b))
        if b - a < 0.05:                       # a click, not a drag
            self.seek(a)
            return
        if shift:
            self.add_skip(a, b)
        else:
            self.start_var.set(f"{a:.3f}")
            self.end_var.set(f"{b:.3f}")
        self.draw_timeline()

    # -- running -----------------------------------------------------------
    def collect(self) -> Selection:
        self.sel.preset = self.preset_var.get()
        self.sel.scale = float(self.scale_var.get())
        self.sel.max_frames = int(self.frames_var.get())
        return self.sel

    def update_status(self):
        if not self.info:
            return
        try:
            sel = self.collect()
            spans = sel.spans()
            secs = vidsr.spans_total(spans)
            n = int(secs * self.info.fps)
            self.range_lbl.configure(
                text=f"{secs:.1f}s of video, ~{n} frames"
                     + (f", using {min(n, sel.max_frames)}" if n > sel.max_frames else ""))
            if sel.roi:
                self.status.configure(text="ready", foreground="#666")
        except Exception:
            pass

    def copy_cmd(self):
        try:
            args = build_args(self.collect())
        except ValueError as e:
            messagebox.showwarning("vidsr", str(e))
            return
        import shlex
        line = "vidsr " + " ".join(shlex.quote(a) for a in args)
        self.root.clipboard_clear()
        self.root.clipboard_append(line)
        self.status.configure(text="command copied to the clipboard")

    def run(self):
        if self.running:
            return
        try:
            sel = self.collect()
            build_args(sel)
        except ValueError as e:
            messagebox.showwarning("vidsr", str(e))
            return
        self.running = True
        self.run_btn.configure(state="disabled")
        self.pbar.configure(value=0)
        self.status.configure(text="starting…")

        def work():
            try:
                rep = run_selection(sel, lambda f, m: self.q.put(("p", f, m)))
                self.q.put(("done", rep, ""))
            except SystemExit as e:
                self.q.put(("err", str(e), ""))
            except Exception as e:
                self.q.put(("err", f"{e}\n\n{traceback.format_exc()}", ""))

        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, a, b = self.q.get_nowait()
                if kind == "p":
                    self.pbar.configure(value=a * 1000)
                    self.status.configure(text=b)
                elif kind == "done":
                    self.running = False
                    self.run_btn.configure(state="normal")
                    self.pbar.configure(value=1000)
                    self.report = a
                    self.show_result()
                    self.nb.select(self.tab_res)
                elif kind == "err":
                    self.running = False
                    self.run_btn.configure(state="normal")
                    self.pbar.configure(value=0)
                    self.status.configure(text="failed")
                    messagebox.showerror("vidsr", a)
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    # -- result ------------------------------------------------------------
    def result_path(self) -> Optional[str]:
        if not self.report:
            return None
        outs = self.report.get("outputs", {})
        return outs.get({"result": "result", "compare": "compare",
                         "variants": "variants"}[self.view_var.get()]) or outs.get("result")

    def show_result(self):
        p = self.result_path()
        if not p or not os.path.exists(p):
            return
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            return
        cw = max(400, self.res_canvas.winfo_width())
        ch = max(300, self.res_canvas.winfo_height())
        s = min(cw / img.shape[1], ch / img.shape[0])
        if s > 1:                          # show small results big, but crisply
            img = cv2.resize(img, None, fx=int(s), fy=int(s),
                             interpolation=cv2.INTER_NEAREST)
        elif s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        self.res_photo = _photo(img)
        self.res_canvas.delete("all")
        self.res_canvas.create_image(cw // 2, ch // 2, image=self.res_photo)
        r = self.report
        cov = r.get("phase_coverage", 0) * 100
        hint = ("" if cov >= 50 else
                "  ·  low sub-pixel coverage: the subject barely moved, so this is "
                "mostly noise removal — a longer range helps more than more zoom")
        self.res_lbl.configure(
            text=f"{r['frames_used']}/{r['frames_decoded']} frames fused  ·  ×{r['scale']:g}  ·  "
                 f"sub-pixel coverage {cov:.0f}%{hint}", foreground="#000")

    def save_result(self):
        p = self.result_path()
        if not p:
            messagebox.showinfo("vidsr", "run a reconstruction first")
            return
        dest = filedialog.asksaveasfilename(defaultextension=".png",
                                            initialfile=os.path.basename(p),
                                            filetypes=[("PNG", "*.png")])
        if dest:
            import shutil
            shutil.copyfile(p, dest)
            self.status.configure(text=f"saved {os.path.basename(dest)}")

    def open_folder(self):
        if not self.report:
            return
        d = os.path.dirname(self.report["outputs"]["result"])
        import subprocess
        for cmd in (["xdg-open", d], ["open", d], ["explorer", d]):
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except FileNotFoundError:
                continue


def run_app(video: Optional[str] = None, out: str = "out") -> int:
    if tk is None:
        print(TK_HELP, file=sys.stderr)
        return 2
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    app = App(root, video, out)
    root.bind("<Configure>", lambda e: app.draw_timeline())
    root.mainloop()
    return 0
