from __future__ import annotations
import sys, json, math
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle, Polygon as MplPolygon

from preproc import PreprocSettings, load_grayscale_with_preproc

if not hasattr(tk, "Notebook") and hasattr(ttk, "Notebook"):
    tk.Notebook = ttk.Notebook


class AnalysisConfirmationDialog(tk.Toplevel):
    """A small, borderless dialog with accept/reject buttons."""

    def __init__(self, parent, accept_callback, reject_callback, x, y):
        super().__init__(parent)
        self.accept_callback = accept_callback
        self.reject_callback = reject_callback

        # Make window borderless and stay on top
        self.overrideredirect(True)
        self.wm_attributes("-topmost", True)

        # Positioning
        self.geometry(f"+{int(x)}+{int(y)}")

        # Frame for content
        frame = tk.Frame(self, background='white', highlightbackground="black", highlightthickness=1)
        frame.pack()

        try:
            # Attempt to load icons if available
            # Green checkmark icon
            check_icon_data = b"iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAADnSURBVDhPnY5BCsJgEETfZsE3iCDeQDxD8SbeoZ15gJcQvEknl3Zl7wYKSJb58nwzBpkfOElYmJqgmz8hJ04m6A6STpyUaA7STpySaA/STpyEaA/STpyBaA/STpyCaA/STpyBaA/STpyBaA/STpwoBeoGg2/iV0WwM9vxvUvAtb2P6b0LoTVQ5A0oVbQ6/hbNsgV/ycb/1k/gC6jX8f79wJ24+2WqlwCWv5u0FhROO2LgZ1xU8f07gW0e5Z0E0s+B3A/yQ+wlgQ/y1A/q5f/gB/829kFh2BfTQAAAABJRU5ErkJggg=="
            self.check_img = tk.PhotoImage(data=check_icon_data)
            btn_accept = tk.Button(frame, image=self.check_img, command=self.accept_callback, borderwidth=0,
                                   relief="flat", bg="white")

            # Red cross icon
            cross_icon_data = b"iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAACYSURBVDhPzY1BCsAwCEMv9xVeQPEMxTcUvEBv15kXkLwJ7+7sSxWE+C8L/5nF5H8gA8fS4qQfMBNf9CdoFvGkbyAaxJOnIFrEk6cgWsSTpyAaxJOvIFrEk68gWsSTr6BaxJOvIFrEk68gWsSTr6AaxJMnA7hX5oAZuDmz+W8Crf130/oWQLuAZ38DToPj/5v/6/4BN8V2yEaG41QAAAAASUVORK5CYII="
            self.cross_img = tk.PhotoImage(data=cross_icon_data)
            btn_reject = tk.Button(frame, image=self.cross_img, command=self.reject_callback, borderwidth=0,
                                   relief="flat", bg="white")

        except tk.TclError:  # Fallback to text if icons fail
            btn_accept = tk.Button(frame, text="✔", command=self.accept_callback, fg="green", relief="flat", bg="white")
            btn_reject = tk.Button(frame, text="✖", command=self.reject_callback, fg="red", relief="flat", bg="white")

        btn_accept.pack(side="left", padx=2, pady=2)
        btn_reject.pack(side="left", padx=2, pady=2)


class _Tooltip:
    # ... (code for Tooltip remains unchanged)
    def __init__(self, widget: tk.Widget, text: str, *, delay: int = 400):
        self.widget = widget
        self.text = text
        self.delay = max(0, int(delay))
        self._after_id: Optional[str] = None
        self._tip_window: Optional[tk.Toplevel] = None
        self._last_pointer: Optional[tuple[int, int]] = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<Motion>", self._on_motion, add="+")

    def _on_enter(self, event):
        self._last_pointer = (event.x_root, event.y_root)
        self._schedule()

    def _on_leave(self, _event):
        self._cancel()
        self._hide()

    def _on_motion(self, event):
        self._last_pointer = (event.x_root, event.y_root)
        self._position()

    def _schedule(self):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self._tip_window is not None or not self.text:
            return
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_attributes("-topmost", True)
        label = tk.Label(
            tip, text=self.text, justify="left", background="#ffffe0",
            relief="solid", borderwidth=1, wraplength=360,
        )
        label.pack(ipadx=8, ipady=4)
        self._tip_window = tip
        self._position()

    def _hide(self):
        if self._tip_window is not None:
            self._tip_window.destroy()
            self._tip_window = None

    def _position(self):
        if self._tip_window is None:
            return
        tip = self._tip_window
        tip.update_idletasks()
        width = tip.winfo_reqwidth()
        height = tip.winfo_reqheight()
        if self._last_pointer is not None:
            x, y = self._last_pointer
        else:
            x = self.widget.winfo_rootx() + self.widget.winfo_width()
            y = self.widget.winfo_rooty() + self.widget.winfo_height()
        x += 12
        y += 10
        root = self.widget.winfo_toplevel()
        root.update_idletasks()
        left = root.winfo_rootx()
        top = root.winfo_rooty()
        right = left + root.winfo_width()
        bottom = top + root.winfo_height()
        if x + width > right - 4:
            x = right - width - 4
        if y + height > bottom - 4:
            y = bottom - height - 4
        x = max(x, left + 4)
        y = max(y, top + 4)
        tip.wm_geometry(f"+{int(x)}+{int(y)}")


class HoverTooltip:
    # ... (code for HoverTooltip remains unchanged)
    def __init__(self, widget: tk.Widget, text: str, delay: int = 400):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after_id: Optional[str] = None
        self._window: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        self._after_id = None
        if self._window is not None:
            return
        x = self.widget.winfo_pointerx() + 16
        y = self.widget.winfo_pointery() + 12
        self._window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, background="#2f2f2f", foreground="white",
                         relief="solid", borderwidth=1, padx=6, pady=3, justify=tk.LEFT)
        label.pack()

    def _hide(self, _event=None):
        self._cancel()
        if self._window is not None:
            try:
                self._window.destroy()
            except Exception:
                pass
            self._window = None


def _parse_cli(argv=None):
    # ... (code remains unchanged)
    import argparse
    p = argparse.ArgumentParser(description="fibonachi_analysis — load input data")
    p.add_argument("--payload", type=str, default=None, help="Path to fibo_input.json")
    p.add_argument("--image", type=str, default=None, help="Path to the image (fallback)")
    p.add_argument("--points", type=str, default=None, help="Path to JSON with points (fallback)")
    return p.parse_args(argv)


def _candidate_dirs(extra_image: Optional[Path]) -> List[Path]:
    # ... (code remains unchanged)
    cands: List[Path] = []
    try:
        cands.append(Path.cwd())
    except Exception:
        pass
    if getattr(sys, "frozen", False):
        try:
            cands.append(Path(sys.executable).resolve().parent)
        except Exception:
            pass
        try:
            cands.append(Path(getattr(sys, "_MEIPASS")))
        except Exception:
            pass
    else:
        try:
            cands.append(Path(__file__).resolve().parent)
        except Exception:
            pass
    if extra_image:
        try:
            cands.append(extra_image.resolve().parent)
        except Exception:
            pass
    uniq, seen = [], set()
    for d in cands:
        rp = str(d.resolve())
        if rp not in seen:
            uniq.append(d);
            seen.add(rp)
    return uniq


def _autofind_json(extra_image: Optional[Path]) -> Optional[Path]:
    # ... (code remains unchanged)
    pats = ["fibo_input.json", "*fibo*input*.json", "*.fibo.json", "*.json"]
    for base in _candidate_dirs(extra_image):
        for pat in pats:
            try:
                for p in base.glob(pat):
                    name = p.name.lower()
                    if "fibo" in name and "input" in name:
                        return p.resolve()
                    if pat == "*.json":
                        try:
                            obj = json.loads(p.read_text(encoding="utf-8"))
                            if isinstance(obj, dict) and "image" in obj and "points" in obj:
                                return p.resolve()
                        except Exception:
                            pass
            except Exception:
                continue
    return None


def find_default_json(base_dir: Path) -> Optional[Path]:
    # ... (code remains unchanged)
    cand = base_dir / "fibo_input.json"
    if cand.exists():
        try:
            d = json.loads(cand.read_text(encoding="utf-8"))
            if "image" in d and "points" in d:
                return cand.resolve()
        except Exception:
            pass
    for p in base_dir.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if "image" in d and "points" in d:
                return p.resolve()
        except Exception:
            pass
    return _autofind_json(None)


def load_input(json_path: Path):
    # ... (code remains unchanged)
    d = json.loads(json_path.read_text(encoding='utf-8'))
    img = Path(d['image']) if d.get('image') else None
    if not img:
        raise RuntimeError("JSON does not contain the key 'image'.")
    pts = np.array([[float(p['y']), float(p['x'])] for p in d.get('points', [])], float)
    center = None;
    dead = 0.0;
    srch = 0.0
    if isinstance(d.get('centers'), dict):
        c = d['centers'].get('overlay') or d['centers'].get('geometric')
        if c and 'x' in c and 'y' in c:
            center = (float(c['y']), float(c['x']))
    if isinstance(d.get('radii'), dict):
        if d['radii'].get('dead') is not None:  dead = float(d['radii']['dead'])
        if d['radii'].get('search') is not None: srch = float(d['radii']['search'])
    fallback_mode = d.get('preproc_mode')
    if not isinstance(fallback_mode, str):
        fallback_mode = None
    preproc = PreprocSettings.from_json(d.get('preproc'), fallback_mode=fallback_mode)
    return img, pts, center, dead, srch, preproc


def cluster_lengths(lengths: np.ndarray):
    # ... (code remains unchanged)
    if lengths.size == 0:
        return np.array([], dtype=int), float("nan"), float("nan"), 0, 1
    c0, c1 = float(lengths.min()), float(lengths.max())
    if c0 == c1:
        lab = np.zeros(len(lengths), dtype=int)
        return lab, c0, float("nan"), 0, 1
    lab = np.zeros(len(lengths), dtype=int)
    for _ in range(60):
        d0 = np.abs(lengths - c0)
        d1 = np.abs(lengths - c1)
        lab = (d1 < d0).astype(int)
        nc0 = float(lengths[lab == 0].mean()) if np.any(lab == 0) else c0
        nc1 = float(lengths[lab == 1].mean()) if np.any(lab == 1) else c1
        if abs(nc0 - c0) < 1e-6 and abs(nc1 - c1) < 1e-6:
            c0, c1 = nc0, nc1;
            break
        c0, c1 = nc0, nc1
    m0 = float(lengths[lab == 0].mean()) if np.any(lab == 0) else float("nan")
    m1 = float(lengths[lab == 1].mean()) if np.any(lab == 1) else float("nan")
    if (not math.isnan(m0)) and (not math.isnan(m1)) and m0 > m1:
        lab = 1 - lab
        m0, m1 = m1, m0
    return lab, m0, m1, 0, 1


def fib_list_upto(n: int) -> List[int]:
    # ... (code remains unchanged)
    if n <= 0: return []
    seq = [1, 1]
    while seq[-1] < n:
        seq.append(seq[-1] + seq[-2])
    return [k for k in seq if k <= n]


def gen_fibonacci_words(max_len: int, start: str = "L") -> List[str]:
    # ... (code remains unchanged)
    if max_len <= 0: return []
    words = ["L" if start.upper() == "L" else "S"]
    while len(words[-1]) <= max_len:
        nxt = "".join(("LS" if ch == "L" else "L") for ch in words[-1])
        if len(nxt) > max_len: break
        words.append(nxt)
    return words


class FibonacciAnalysisFrame(tk.Frame):
    def __init__(self, master: tk.Misc, controller=None, auto_load: bool = True, license_manager=None):
        super().__init__(master)
        self.controller = controller
        self.license_manager = license_manager

        # Core data
        self.img_path: Optional[Path] = None
        self.points: Optional[np.ndarray] = None
        self.center: Optional[Tuple[float, float]] = None
        self.dead: float = 0.0
        self.srch: float = 0.0
        self.preproc: PreprocSettings = PreprocSettings(mode="raw")
        self.img_arr: Optional[np.ndarray] = None

        # View state
        self.zoom_val: float = 0.0
        self.view_cx: Optional[float] = None
        self.view_cy: Optional[float] = None
        self._img_shape: Optional[Tuple[int, int]] = None
        self._full_view_bounds: Optional[Tuple[float, float, float, float]] = None

        # UI and interaction state
        self.pick_tol = 10.0
        self.max_dist_line = 20.0
        self.mode_buttons: Dict[str, tk.Button] = {}
        self.analysis_mode: str = 'sl'
        self.anchor_idx: Optional[int] = None
        self.rubber_line = None

        # Polygon state
        self.polygon_current_idx: List[int] = []
        self.polygon_rubber_line = None

        # New analysis management state
        self.pending_analysis: Optional[Dict[str, Any]] = None
        self.permanent_analyses: List[Dict[str, Any]] = []
        self.active_analysis_idx: Optional[int] = None
        self.confirmation_dialog: Optional[AnalysisConfirmationDialog] = None
        self.list_index_map: Dict[int, Any] = {}

        # UI components
        self._right_scroll_canvas: Optional[tk.Canvas] = None
        self._right_scroll_window: Optional[int] = None

        # Build UI
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_ui()

        # Final setup
        if auto_load:
            self._initial_load()

    def _build_ui(self):
        container = tk.Frame(self)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=0, minsize=470)

        left = tk.Frame(container)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=12)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        # Matplotlib Canvas
        self.fig = plt.Figure(figsize=(9.6, 6.6))
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

        # Right Panel Setup
        right_host = tk.Frame(container)
        right_host.grid(row=0, column=1, sticky="ns", pady=12)
        right_host.grid_columnconfigure(0, weight=1)
        right_host.grid_rowconfigure(0, weight=1)

        scroll_host = ttk.Frame(right_host)
        scroll_host.grid(row=0, column=0, sticky="nsew")
        scroll_host.grid_columnconfigure(0, weight=1)
        scroll_host.grid_rowconfigure(0, weight=1)

        canvas = tk.Canvas(scroll_host, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=canvas.yview)
        right = tk.Frame(canvas)
        self._right_scroll_canvas = canvas
        self._right_scroll_window = canvas.create_window((0, 0), window=right, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        right.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(self._right_scroll_window, width=e.width))
        right.bind("<Enter>", self._activate_right_scroll)
        right.bind("<Leave>", self._deactivate_right_scroll)
        canvas.bind("<Enter>", self._activate_right_scroll)
        canvas.bind("<Leave>", self._deactivate_right_scroll)
        right.columnconfigure(0, weight=1)

        # Right Panel Widgets
        controls = tk.Frame(right)
        controls.grid(row=0, column=0, sticky="ew", padx=6, pady=(0, 10))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        tk.Button(controls, text='Open JSON…', command=self.open_json).grid(row=0, column=0, sticky='ew', padx=4,
                                                                            pady=2)
        tk.Button(controls, text='Save PNG', command=self.save_png).grid(row=0, column=1, sticky='ew', padx=4, pady=2)

        tk.Label(controls, text='Selection thickness (px):').grid(row=2, column=0, sticky='w', padx=4, pady=(8, 2))
        self.entBand = tk.Spinbox(controls, from_=2, to=100, width=6, command=self._onBand)
        self.entBand.delete(0, 'end');
        self.entBand.insert(0, str(int(self.max_dist_line)))
        self.entBand.grid(row=2, column=1, sticky='ew', padx=4, pady=(8, 2))

        mode_switch = tk.Frame(controls)
        mode_switch.grid(row=3, column=0, columnspan=2, sticky='ew', padx=4, pady=(8, 2))
        for col in range(3): mode_switch.columnconfigure(col, weight=1)

        self.mode_buttons['sl'] = tk.Button(mode_switch, text='🔗 Chain', command=lambda: self._set_analysis_mode('sl'))
        self.mode_buttons['sl'].grid(row=0, column=0, sticky='ew', padx=(0, 2))
        self.mode_buttons['ratio'] = tk.Button(mode_switch, text='📊 Ratio',
                                               command=lambda: self._set_analysis_mode('ratio'))
        self.mode_buttons['ratio'].grid(row=0, column=1, sticky='ew', padx=2)
        self.mode_buttons['polygon'] = tk.Button(mode_switch, text='🔺 Polygon',
                                                 command=lambda: self._set_analysis_mode('polygon'))
        self.mode_buttons['polygon'].grid(row=0, column=2, sticky='ew', padx=(2, 0))

        is_full_version = self.license_manager is None or self.license_manager.has_valid_license()
        HoverTooltip(self.mode_buttons['sl'], 'Chain analysis: use Left Click to select two endpoints.')
        if is_full_version:
            HoverTooltip(self.mode_buttons['ratio'], 'Neighbor ratios: use Left Click to select two endpoints.')
            HoverTooltip(self.mode_buttons['polygon'],
                         'Polygon analysis: add vertices with Left Click, close by clicking the first point.')
        else:
            for mode in ['ratio', 'polygon']:
                self.mode_buttons[mode].config(state=tk.DISABLED, relief='sunken')
                HoverTooltip(self.mode_buttons[mode], 'Available in the full version')

        zoom_group = ttk.LabelFrame(controls, text='Scale')
        zoom_group.grid(row=4, column=0, columnspan=2, sticky='ew', padx=4, pady=(8, 4))
        self.zoom_var = tk.DoubleVar(value=self.zoom_val)
        self.zoom_scale = ttk.Scale(zoom_group, from_=0, to=100, variable=self.zoom_var, command=self._on_zoom_change)
        self.zoom_scale.pack(fill=tk.X, padx=6, pady=(0, 4))
        self.zoom_hint = ttk.Label(zoom_group, anchor='w')
        self.zoom_hint.pack(fill=tk.X, padx=6, pady=(0, 2))
        self._update_zoom_hint()

        self.status = tk.Label(right, text='', anchor='w')
        self.status.grid(row=1, column=0, sticky='ew', padx=6, pady=(0, 10))
        self._set_analysis_mode('sl')

        self.lst_header = tk.Label(right, text='Select an analysis to view details')
        self.lst_header.grid(row=2, column=0, sticky='w', padx=6, pady=(0, 2))

        self.results_notebook = tk.Notebook(right)
        self.results_notebook.grid(row=3, column=0, sticky='nsew', padx=6, pady=(0, 8))
        right.grid_rowconfigure(3, weight=1)

        # Tabs for results
        tab_subsegments = ttk.Frame(self.results_notebook);
        tab_subsegments.grid_columnconfigure(0, weight=1);
        tab_subsegments.grid_rowconfigure(0, weight=1)
        tab_prefixes = ttk.Frame(self.results_notebook);
        tab_prefixes.grid_columnconfigure(0, weight=1);
        tab_prefixes.grid_rowconfigure(1, weight=1)
        self.results_notebook.add(tab_subsegments, text='Subsegments')
        self.results_notebook.add(tab_prefixes, text='Fib-Words')

        self.lst = tk.Listbox(tab_subsegments, width=66, height=15)
        self.lst.grid(row=0, column=0, sticky='nsew')
        self.lst.bind('<<ListboxSelect>>', self._on_list_select)

        info_frame = tk.Frame(tab_subsegments)
        info_frame.grid(row=1, column=0, sticky='ew', pady=(6, 0))
        self.lbl_ratio = tk.Label(info_frame, text='Average L/S along chain: —')
        self.lbl_ratio.pack(anchor='w')
        self.lbl_ratio_neigh = tk.Label(info_frame, text='Average neighboring segment ratio: —')
        self.lbl_ratio_neigh.pack(anchor='w')
        self.lbl_ratio_polygons = tk.Label(info_frame, text='Average neighboring polygon linear ratio: —')
        self.lbl_ratio_polygons.pack(anchor='w')

        tk.Label(tab_subsegments, text='S/L sequence (full):').grid(row=2, column=0, sticky='w', pady=(4, 2))
        self.txt_sl = tk.Text(tab_subsegments, height=4, wrap='word')
        self.txt_sl.grid(row=3, column=0, sticky='ew', pady=(0, 4))
        self.txt_sl.bind('<KeyPress>', self._on_sl_keypress)  # Note: editing is disabled, but hook is here

        prefixes_header = tk.Label(tab_prefixes, text='Prefixes of "fib-words" (L→LS, S→L)')
        prefixes_header.grid(row=0, column=0, sticky='w', pady=(0, 4))
        prefixes_frame = tk.Frame(tab_prefixes);
        prefixes_frame.grid(row=1, column=0, sticky='nsew')
        prefixes_frame.grid_columnconfigure(0, weight=1);
        prefixes_frame.grid_rowconfigure(0, weight=1)
        self.txt_words = tk.Text(prefixes_frame, height=10, state='disabled')
        self.txt_words.grid(row=0, column=0, sticky='nsew')
        scroll_words = tk.Scrollbar(prefixes_frame, orient='vertical', command=self.txt_words.yview)
        scroll_words.grid(row=0, column=1, sticky='ns')
        self.txt_words.configure(yscrollcommand=scroll_words.set)

        # Key bindings
        self.bind_all('<Return>', self._on_enter_key)
        self.bind_all('<Delete>', self._on_delete_key)
        self.bind_all('<Escape>', self._on_escape_key)

    def _initial_load(self):
        base = Path(getattr(sys, '_MEIPASS', Path(__file__).parent)) if getattr(sys, 'frozen', False) else Path(
            __file__).parent
        auto = find_default_json(base)
        if auto:
            try:
                self.load_json(auto)
                self.status.config(text=f'Loaded: {auto.name}')
            except Exception as e:
                messagebox.showerror('Load error', str(e))
        else:
            self.status.config(text='JSON not found. Select a file manually.')

    # --- NEW: Core Analysis Management ---

    def _prompt_for_confirmation(self, analysis_data: Dict[str, Any]):
        self._reject_pending_analysis(ask_user=False)  # Clear any previous pending analysis
        self.pending_analysis = analysis_data

        # Position the dialog
        fig_widget = self.canvas.get_tk_widget()
        pos_data = analysis_data.get('dialog_pos', (0, 0))

        # Transform data coordinates to screen coordinates
        x_display, y_display = self.ax.transData.transform(pos_data)
        x_screen = fig_widget.winfo_rootx() + x_display
        y_screen = fig_widget.winfo_rooty() + (fig_widget.winfo_height() - y_display)

        self.confirmation_dialog = AnalysisConfirmationDialog(
            self,
            accept_callback=self._accept_pending_analysis,
            reject_callback=lambda: self._reject_pending_analysis(ask_user=False),
            x=x_screen + 10,
            y=y_screen - 15,
        )
        self._redraw_canvas()

    def _accept_pending_analysis(self):
        if self.pending_analysis is None: return

        self.permanent_analyses.append(self.pending_analysis)
        self.active_analysis_idx = len(self.permanent_analyses) - 1

        if self.confirmation_dialog:
            self.confirmation_dialog.destroy()
            self.confirmation_dialog = None

        self.pending_analysis = None
        self._redraw_canvas()
        self._update_display_for_active_analysis()
        self.status.config(text=f"Analysis #{self.active_analysis_idx + 1} saved.")

    def _reject_pending_analysis(self, ask_user=True):
        if self.pending_analysis is None: return

        if self.confirmation_dialog:
            self.confirmation_dialog.destroy()
            self.confirmation_dialog = None

        self.pending_analysis = None
        self._redraw_canvas()
        self._update_display_for_active_analysis()
        self.status.config(text="Analysis discarded.")

    def _update_display_for_active_analysis(self):
        # Clear all info panels first
        self.lst.delete(0, tk.END)
        self.list_index_map.clear()
        self.lbl_ratio.config(text='Average L/S along chain: —')
        self.lbl_ratio_neigh.config(text='Average neighboring segment ratio: —')
        self.lbl_ratio_polygons.config(text='Average neighboring polygon linear ratio: —')
        self._set_sl_text("")
        self.txt_words.configure(state='normal');
        self.txt_words.delete('1.0', tk.END);
        self.txt_words.configure(state='disabled')
        self.lst_header.config(text='Select an analysis to view details')

        if self.active_analysis_idx is None or self.active_analysis_idx >= len(self.permanent_analyses):
            return

        analysis = self.permanent_analyses[self.active_analysis_idx]
        analysis_type = analysis['type']

        if analysis_type == 'sl':
            self._populate_sl_info(analysis)
        elif analysis_type == 'ratio':
            self._populate_ratio_info(analysis)
        elif analysis_type == 'polygon':
            # Polygon info is special, it aggregates *all* saved polygons
            self._populate_polygon_info()

    def _populate_sl_info(self, analysis_data):
        self.lst_header.config(text='Found words (Fibonacci subsegments)')
        self.list_index_map.clear()
        self.lst.delete(0, tk.END)

        ratio = analysis_data['data'].get('ratio', float('nan'))
        found = analysis_data['data'].get('found_words', [])
        SL = analysis_data['data'].get('sl_chain', [])

        groups: Dict[int, List] = {}
        for entry in found: groups.setdefault(entry[0], []).append(entry)

        row = 0
        if groups:
            for n in sorted(groups.keys()):
                self.lst.insert(tk.END, f'— n={n} —')
                self.list_index_map[row] = {'type': 'header'}
                row += 1
                for (n_, i0, word, Lc, Sc) in groups[n]:
                    self.lst.insert(tk.END, f'  i={i0}  word={word}  L={Lc} S={Sc}')
                    self.list_index_map[row] = {'analysis_idx': self.active_analysis_idx, 'type': 'sl', 'i0': i0,
                                                'n': n_}
                    row += 1
                self.lst.insert(tk.END, '')
                self.list_index_map[row] = {'type': 'spacer'}
                row += 1
        else:
            self.lst.insert(tk.END, 'No matches (n≥3)')

        if math.isfinite(ratio): self.lbl_ratio.config(text=f'Average L/S along chain: {ratio:.3f}')
        self._set_sl_text(''.join(SL))

        max_len_ref = max(groups.keys(), default=min(len(SL), 34))
        self.txt_words.configure(state='normal');
        self.txt_words.delete('1.0', tk.END)
        for w in gen_fibonacci_words(max_len=max_len_ref, start='L'):
            self.txt_words.insert(tk.END, f'len={len(w)} → {w}\n')
        self.txt_words.configure(state='disabled')

    def _populate_ratio_info(self, analysis_data):
        self.lst_header.config(text='Neighboring segment ratios (Ratio mode)')
        self.list_index_map.clear()
        self.lst.delete(0, tk.END)

        ratios = analysis_data['data'].get('ratios', [])
        mean_ratio = analysis_data['data'].get('mean_ratio', float('nan'))

        row = 0
        if not ratios:
            self.lst.insert(tk.END, 'Not enough segments for ratios.')
        else:
            for i, r in enumerate(ratios, start=2):
                entry_text = f'  ({i + 1}-{i}) / ({i}-{i - 1})  ≈  {r:.6g}'
                self.lst.insert(tk.END, entry_text)
                k = i - 1  # This 'k' is the index of the *first* segment in the ratio
                self.list_index_map[row] = {'analysis_idx': self.active_analysis_idx, 'type': 'ratio', 'k': k}
                row += 1

        if math.isfinite(mean_ratio): self.lbl_ratio_neigh.config(
            text=f'Average neighboring segment ratio: {mean_ratio:.6g}')

    def _populate_polygon_info(self):
        self.lst_header.config(text='Polygons (Polygon mode) — areas and ratios')
        self.list_index_map.clear()
        self.lst.delete(0, tk.END)

        # Find all *active* polygon analyses
        polygon_analyses: List[Tuple[int, Dict]] = []
        for i, p in enumerate(self.permanent_analyses):
            if p['type'] == 'polygon':
                polygon_analyses.append((i, p))

        if not polygon_analyses:
            self.lst.insert(tk.END, 'No saved polygons to analyze.')
            return

        areas = [p['data']['area'] for _, p in polygon_analyses]

        row = 0
        for i, (analysis_idx, p) in enumerate(polygon_analyses, start=1):
            area = p['data']['area']
            label = p['data']['label']
            self.lst.insert(tk.END, f'Polygon {label} (Area: {area:.6g})')
            self.list_index_map[row] = {'analysis_idx': analysis_idx, 'type': 'polygon'}
            row += 1

        if len(areas) >= 2:
            self.lst.insert(tk.END, '')
            self.list_index_map[row] = {'type': 'spacer'}
            row += 1

            polygon_linear_ratios = []
            for i in range(1, len(areas)):
                prev_area, curr_area = areas[i - 1], areas[i]
                prev_label = polygon_analyses[i - 1][1]['data']['label']
                curr_label = polygon_analyses[i][1]['data']['label']

                if prev_area > 0:
                    size_ratio = curr_area / prev_area
                    linear_ratio = math.sqrt(size_ratio) if size_ratio > 0 else 0
                    polygon_linear_ratios.append(linear_ratio)
                    self.lst.insert(tk.END, f'Lin. Ratio {curr_label}/{prev_label}: {linear_ratio:.6g}')
                else:
                    self.lst.insert(tk.END, f'Lin. Ratio {curr_label}/{prev_label}: undefined')

                self.list_index_map[row] = {'type': 'info'}
                row += 1

            finite_linear = [r for r in polygon_linear_ratios if math.isfinite(r) and r > 0]
            if finite_linear:
                mean_linear = float(np.mean(finite_linear))
                self.lbl_ratio_polygons.config(text=f'Average neighboring polygon linear ratio: {mean_linear:.6g}')

    # --- Event Handlers (Modified) ---

    def _on_click(self, event):
        if self.points is None or event.xdata is None or event.ydata is None: return

        # Any click rejects a pending analysis
        if self.pending_analysis:
            # Check if click was *on* the dialog
            if self.confirmation_dialog and self.confirmation_dialog.winfo_containing(event.x_root,
                                                                                      event.y_root) == self.confirmation_dialog:
                return  # Click was on the dialog, let its buttons handle it
            self._reject_pending_analysis(ask_user=False)
            return

        # Right click to select an active analysis
        if event.button == 3:
            self._handle_right_click(event)
            return

        if event.button != 1: return

        x, y = float(event.xdata), float(event.ydata)
        d2 = (self.points[:, 1] - x) ** 2 + (self.points[:, 0] - y) ** 2
        j = int(np.argmin(d2))
        if math.sqrt(d2[j]) > self.pick_tol: return

        self._focus_on(float(self.points[j, 1]), float(self.points[j, 0]))

        if self.analysis_mode in ('sl', 'ratio'):
            if self.anchor_idx is None:
                self.anchor_idx = j
                self._redraw_canvas()
            else:
                indices = self._collect_points_along_segment(self.anchor_idx, j, self.max_dist_line)
                if len(indices) < (3 if self.analysis_mode == 'ratio' else 2):
                    self.status.config(text=f"Not enough points found for {self.analysis_mode} analysis.")
                    self.anchor_idx = None
                    self._redraw_canvas()
                    return

                if self.analysis_mode == 'sl':
                    self.run_analysis(indices)
                else:
                    self.run_ratio_analysis(indices)

                self.anchor_idx = None

        elif self.analysis_mode == 'polygon':
            self._handle_polygon_click(j)

    def _handle_right_click(self, event):
        if not self.permanent_analyses or self.points is None: return
        x, y = float(event.xdata), float(event.ydata)

        min_dist_sq = float('inf')
        best_idx = None

        for i, analysis in enumerate(self.permanent_analyses):
            pts = self.points[analysis['indices']]
            if analysis['type'] == 'polygon':
                center = pts.mean(axis=0)  # y, x
                dist_sq = (center[0] - y) ** 2 + (center[1] - x) ** 2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_idx = i
            else:  # chain or ratio
                for k in range(len(pts) - 1):
                    p1_yx = pts[k]
                    p2_yx = pts[k + 1]
                    # Simple distance to segment midpoint for hit-testing
                    mid_yx = (p1_yx + p2_yx) / 2
                    dist_sq = (mid_yx[0] - y) ** 2 + (mid_yx[1] - x) ** 2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        best_idx = i

        # Use a larger tolerance for selection than for point picking
        if best_idx is not None and math.sqrt(min_dist_sq) < self.pick_tol * 3:
            if self.active_analysis_idx != best_idx:
                self.active_analysis_idx = best_idx
                self._redraw_canvas()
                self._update_display_for_active_analysis()
                self.status.config(text=f"Analysis #{best_idx + 1} is now active.")

    def _on_motion(self, event):
        if event.xdata is None or event.ydata is None or self.points is None: return

        if self.anchor_idx is not None:
            ax, ay = self.points[self.anchor_idx, [1, 0]]
            bx, by = float(event.xdata), float(event.ydata)
            if self.rubber_line is None:
                self.rubber_line, = self.ax.plot([ax, bx], [ay, by], color='yellow', lw=2.0, alpha=0.9, zorder=5)
            else:
                self.rubber_line.set_data([ax, bx], [ay, by])
            self.canvas.draw_idle()
        elif self.polygon_current_idx:
            ax, ay = self.points[self.polygon_current_idx[-1], [1, 0]]
            bx, by = float(event.xdata), float(event.ydata)
            if self.polygon_rubber_line is None:
                self.polygon_rubber_line, = self.ax.plot([ax, bx], [ay, by], color='orange', lw=2.2, alpha=0.8,
                                                         zorder=3.4)
            else:
                self.polygon_rubber_line.set_data([ax, bx], [ay, by])
            self.canvas.draw_idle()
        else:
            self._clear_rubber_lines()

    def _on_list_select(self, event):
        # A selection in the listbox was made, just trigger a redraw.
        # The redraw function will handle the highlighting.
        self._redraw_canvas()

    def _on_enter_key(self, event):
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)): return
        if self.pending_analysis:
            self._accept_pending_analysis()

    def _on_delete_key(self, event):
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)): return
        if self.pending_analysis:
            self._reject_pending_analysis(ask_user=False)

    def _on_escape_key(self, event):
        if self.pending_analysis:
            self._reject_pending_analysis(ask_user=False)
        elif self.anchor_idx is not None or self.polygon_current_idx:
            self.anchor_idx = None
            self.polygon_current_idx.clear()
            self._redraw_canvas()
            self.status.config(text="Selection cancelled.")

    # --- Drawing Logic (Modified) ---

    def _redraw_canvas(self):
        self.ax.clear()

        # 1. Base image and points
        img_shape = None
        if self.img_arr is not None:
            img_shape = self.img_arr.shape[:2]
            self.ax.imshow(self.img_arr, cmap='gray', interpolation='nearest')

        self._img_shape = img_shape
        if self.points is not None:
            self.ax.scatter(self.points[:, 1], self.points[:, 0], s=24, c='cyan', edgecolors='black', linewidths=0.4,
                            label='peaks')

        if self.center is not None:
            cy, cx = self.center
            self.ax.scatter([cx], [cy], s=40, c='red', marker='o')
            if self.dead > 0: self.ax.add_patch(Circle((cx, cy), self.dead, fill=False, ec='red', ls='--', lw=1.5))
            if self.srch > 0: self.ax.add_patch(Circle((cx, cy), self.srch, fill=False, ec='red', ls=':', lw=1.0))

        # 2. Permanent analyses
        for i, analysis in enumerate(self.permanent_analyses):
            is_active = (i == self.active_analysis_idx)
            self._draw_one_analysis(analysis, is_active=is_active)

        # 3. Pending analysis
        if self.pending_analysis:
            self._draw_one_analysis(self.pending_analysis, is_pending=True)

        # 4. Highlighted list selection (NEW)
        self._draw_list_selection_highlight()

        # 5. Interactive elements (anchors, rubber lines)
        if self.anchor_idx is not None:
            y, x = self.points[self.anchor_idx]
            self.ax.scatter([x], [y], s=52, c='yellow', edgecolors='k', linewidths=0.6, zorder=4)

        if self.polygon_current_idx:
            self._draw_polygon_construction()

        self._update_full_view_bounds()
        self.ax.axis('off')
        self._apply_zoom()
        self.canvas.draw_idle()

    def _draw_one_analysis(self, analysis_data: Dict[str, Any], is_active: bool = False, is_pending: bool = False):
        """Dispatches to the correct V1-style drawing function."""
        if self.points is None: return

        analysis_type = analysis_data['type']

        # Determine style based on state
        if is_pending:
            color, ls, lw, zorder = 'lime', '-', 2.2, 3.0
        elif is_active:
            color, ls, lw, zorder = 'magenta', '--', 2.0, 2.5
        else:
            color, ls, lw, zorder = 'deepskyblue', ':', 1.8, 2.0

        style = {'color': color, 'ls': ls, 'lw': lw, 'zorder': zorder, 'active': is_active or is_pending}

        if analysis_type in ('sl', 'ratio'):
            self._draw_one_analysis_chain(analysis_data, style)
        elif analysis_type == 'polygon':
            self._draw_one_analysis_polygon(analysis_data, style)

    def _draw_one_analysis_chain(self, analysis_data: Dict[str, Any], style: Dict):
        """Draws a chain (SL or Ratio) with V1 style (numbers, labels)."""
        indices = analysis_data['indices']
        pts = self.points[indices]
        color = style['color']

        self.ax.scatter(pts[:, 1], pts[:, 0], s=36, c=color, edgecolors='k', linewidths=0.6,
                        zorder=style['zorder'] + 0.1)

        for i in range(len(pts) - 1):
            y1, x1 = pts[i];
            y2, x2 = pts[i + 1]
            self.ax.plot([x1, x2], [y1, y2], color=style['color'], lw=style['lw'], ls=style['ls'],
                         zorder=style['zorder'])

        # Draw numbers and labels only for active/pending
        if style['active']:
            for i in range(len(pts)):
                yN, xN = pts[i]
                self.ax.text(xN, yN, str(i + 1), color=style['color'], fontsize=8, ha='right', va='bottom',
                             zorder=style['zorder'] + 0.2)

            if analysis_data['type'] == 'sl':
                sl_chain = analysis_data['data']['sl_chain']
                for k in range(len(sl_chain)):
                    y1, x1 = pts[k];
                    y2, x2 = pts[k + 1]
                    my, mx = (y1 + y2) / 2, (x1 + x2) / 2
                    self.ax.text(mx, my, sl_chain[k], color='red', fontsize=9, ha='center', va='center',
                                 zorder=style['zorder'] + 0.2)

    def _draw_one_analysis_polygon(self, analysis_data: Dict[str, Any], style: Dict):
        """Draws a polygon with V1 style (fill, label)."""
        indices = analysis_data['indices']
        pts = self.points[indices]

        poly = MplPolygon(pts[:, ::-1], closed=True,
                          facecolor=style['color'], alpha=0.25,
                          edgecolor=style['color'], linewidth=style['lw'],
                          ls=style['ls'], zorder=style['zorder'])
        self.ax.add_patch(poly)

        if style['active']:
            label = analysis_data['data']['label']
            cy, cx = pts.mean(axis=0)
            self.ax.text(cx, cy, label, color=style['color'], fontsize=9, ha='center', va='center',
                         zorder=style['zorder'] + 0.1)

    def _draw_list_selection_highlight(self):
        """Draws the bright green highlight for the selected listbox item."""
        sel = self.lst.curselection()
        if not sel: return

        row = sel[0]
        meta = self.list_index_map.get(row)
        if not meta or 'analysis_idx' not in meta: return

        analysis_idx = meta['analysis_idx']
        analysis_type = meta['type']

        if analysis_idx >= len(self.permanent_analyses): return

        analysis_data = self.permanent_analyses[analysis_idx]

        if analysis_type == 'sl':
            self._highlight_word_V1(analysis_data, meta['i0'], meta['n'])
        elif analysis_type == 'ratio':
            self._highlight_ratio_pair_V1(analysis_data, meta['k'] - 1, meta['k'])
        elif analysis_type == 'polygon':
            self._highlight_polygon_V1(analysis_data)

    def _highlight_word_V1(self, analysis_data: Dict, i0: int, n: int):
        """Draws lime green highlight for a 'word' (from V1)"""
        indices = analysis_data['indices']
        chain = self.points[indices]
        sl_chain = analysis_data['data']['sl_chain']

        for k in range(i0, i0 + n):
            y1, x1 = chain[k];
            y2, x2 = chain[k + 1]
            self.ax.plot([x1, x2], [y1, y2], color='lime', lw=3.2, zorder=10)
            my, mx = (y1 + y2) / 2, (x1 + x2) / 2
            self.ax.text(mx, my, sl_chain[k], color='red', fontsize=9, ha='center', va='center', zorder=10.1)

    def _highlight_ratio_pair_V1(self, analysis_data: Dict, seg_a: int, seg_b: int):
        """Draws lime green highlight for a ratio pair (from V1)"""
        indices = analysis_data['indices']
        chain = self.points[indices]
        M = len(chain) - 1
        if not (0 <= seg_a < M and 0 <= seg_b < M): return

        for k in (seg_a, seg_b):
            y1, x1 = chain[k];
            y2, x2 = chain[k + 1]
            self.ax.plot([x1, x2], [y1, y2], color='lime', lw=3.2, zorder=10)
            my, mx = (y1 + y2) / 2, (x1 + x2) / 2
            self.ax.text(mx, my, f'{k + 2}-{k + 1}', color='red', fontsize=9, ha='center', va='center', zorder=10.1)

    def _highlight_polygon_V1(self, analysis_data: Dict):
        """Draws lime green highlight for a polygon."""
        indices = analysis_data['indices']
        pts = self.points[indices]
        poly = MplPolygon(pts[:, ::-1], closed=True, fill=False, edgecolor='lime', linewidth=3.2, zorder=10)
        self.ax.add_patch(poly)

    def _draw_polygon_construction(self):
        if not self.polygon_current_idx or self.points is None: return
        pts_cur = self.points[self.polygon_current_idx]
        xs, ys = pts_cur[:, 1], pts_cur[:, 0]
        self.ax.plot(xs, ys, color='orange', lw=2.2, zorder=3.1)
        self.ax.scatter(xs, ys, s=46, c='orange', edgecolors='k', linewidths=0.6, zorder=3.2)
        first_x, first_y = xs[0], ys[0]
        self.ax.scatter([first_x], [first_y], s=70, facecolors='none', edgecolors='orange', linewidths=1.5, zorder=3.3)
        if len(xs) >= 2:
            self.ax.plot([xs[-1], first_x], [ys[-1], first_y], color='orange', lw=1.2, ls=':', zorder=3.0)

    # --- Analysis Logic (Modified) ---

    def run_analysis(self, indices: List[int]):
        chain = self.points[indices].copy()
        seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        labels, Slen, Llen, sidx, lidx = cluster_lengths(seg)
        sl_chain = ['S' if labels[i] == sidx else 'L' for i in range(len(seg))]
        ratio = (Llen / Slen) if (Slen and not math.isnan(Slen) and Slen > 0) else float('nan')

        # Find words
        found_words: List[Tuple[int, int, str, int, int]] = []
        fibNs = [n for n in fib_list_upto(len(sl_chain)) if n >= 3]
        for n in fibNs:
            fibs = fib_list_upto(n)
            k = len(fibs) - 1
            exp1 = (fibs[k - 1], fibs[k - 2]) if k >= 2 else (1, 0)
            for i in range(0, len(sl_chain) - n + 1):
                sub = sl_chain[i:i + n]
                Lc, Sc = sub.count('L'), sub.count('S')
                if (Lc, Sc) == exp1:
                    found_words.append((n, i, ''.join(sub), Lc, Sc))

        analysis_data = {
            'type': 'sl',
            'indices': indices,
            'dialog_pos': chain.mean(axis=0)[::-1],  # (x,y)
            'data': {
                'sl_chain': sl_chain,
                'ratio': ratio,
                'found_words': found_words,
            }
        }
        self._prompt_for_confirmation(analysis_data)
        self.status.config(text=f"Chain analysis complete. Please confirm or reject.")

    def run_ratio_analysis(self, indices: List[int]):
        chain = self.points[indices].copy()
        seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)

        ratios = [seg[i] / seg[i - 1] if seg[i - 1] > 0 else float('nan') for i in range(1, len(seg))]
        finite = [r for r in ratios if math.isfinite(r)]
        mean_ratio = float(np.mean(finite)) if finite else float('nan')

        analysis_data = {
            'type': 'ratio',
            'indices': indices,
            'dialog_pos': chain.mean(axis=0)[::-1],  # (x,y)
            'data': {
                'ratios': ratios,
                'mean_ratio': mean_ratio,
            }
        }
        self._prompt_for_confirmation(analysis_data)
        self.status.config(text=f"Ratio analysis complete. Please confirm or reject.")

    def _handle_polygon_click(self, point_idx: int):
        if point_idx in self.polygon_current_idx:
            # Closing the polygon
            if point_idx == self.polygon_current_idx[0] and len(self.polygon_current_idx) >= 3:
                indices = self.polygon_current_idx.copy()
                area = self._polygon_area(indices)

                poly_num = len([p for p in self.permanent_analyses if p['type'] == 'polygon']) + 1

                analysis_data = {
                    'type': 'polygon',
                    'indices': indices,
                    'dialog_pos': self.points[indices[-1]][::-1],  # (x,y)
                    'data': {
                        'area': area,
                        'label': f'P{poly_num}'
                    }
                }
                self.polygon_current_idx.clear()
                self._prompt_for_confirmation(analysis_data)
                self.status.config(text="Polygon closed. Please confirm or reject.")
            else:
                self.status.config(text="Vertex already added or polygon too small.")
            return

        # Adding a vertex
        self.polygon_current_idx.append(point_idx)
        self._redraw_canvas()
        self.status.config(text=f"Polygon vertices: {len(self.polygon_current_idx)}. Click first point to close.")

    # --- Utility and Helper functions ---

    def open_json(self):
        p = filedialog.askopenfilename(filetypes=[('JSON', '*.json'), ('All', '*.*')])
        if not p: return
        self.load_json(Path(p))

    def load_json(self, json_path: Path):
        self.img_path, self.points, self.center, self.dead, self.srch, self.preproc = load_input(json_path)
        self.img_arr = None
        if self.img_path:
            try:
                self.img_arr = load_grayscale_with_preproc(self.img_path, self.preproc)
            except Exception as exc:
                messagebox.showerror('Error', f'Failed to prepare the image:\n{exc}')

        # Reset everything
        self.permanent_analyses.clear()
        self.pending_analysis = None
        self.active_analysis_idx = None
        self.anchor_idx = None
        self.polygon_current_idx.clear()
        self._reject_pending_analysis(ask_user=False)
        self._reset_zoom_state()
        self._redraw_canvas()
        self._update_display_for_active_analysis()

        self.status.config(text=f'Loaded: {json_path.name}')
        self._flash_right_scroll()

    def _polygon_area(self, idxs: List[int]) -> float:
        if self.points is None or len(idxs) < 3: return 0.0
        pts = self.points[idxs]
        xs, ys = pts[:, 1], pts[:, 0]
        return 0.5 * abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))

    # ... (Most of the unchanged helper methods like _on_scroll, _set_analysis_mode, etc. are here)
    def _on_scroll(self, event):
        if event.xdata is None or event.ydata is None: return
        zoom_step = 5
        self.zoom_val += zoom_step if event.button == 'up' else -zoom_step
        self.zoom_val = max(0.0, min(100.0, self.zoom_val))
        self._on_zoom_change(self.zoom_val)

    def _on_zoom_change(self, value):
        self.zoom_val = max(0.0, min(100.0, float(value)))
        if hasattr(self, 'zoom_var'): self.zoom_var.set(self.zoom_val)
        self._apply_zoom()
        self._update_zoom_hint()
        if hasattr(self, 'canvas'): self.canvas.draw_idle()

    def _activate_right_scroll(self, _event):
        if self._right_scroll_canvas:
            self._right_scroll_canvas.bind_all("<MouseWheel>", self._on_right_scroll_mousewheel)
            self._right_scroll_canvas.bind_all("<Button-4>", self._on_right_scroll_mousewheel)
            self._right_scroll_canvas.bind_all("<Button-5>", self._on_right_scroll_mousewheel)

    def _deactivate_right_scroll(self, _event):
        if self._right_scroll_canvas:
            self._right_scroll_canvas.unbind_all("<MouseWheel>")
            self._right_scroll_canvas.unbind_all("<Button-4>")
            self._right_scroll_canvas.unbind_all("<Button-5>")

    def _on_right_scroll_mousewheel(self, event):
        if self._right_scroll_canvas is None: return
        if getattr(event, 'delta', 0):
            self._right_scroll_canvas.yview_scroll(int(-event.delta / 120), 'units')
        elif getattr(event, 'num', None) == 4:
            self._right_scroll_canvas.yview_scroll(-1, 'units')
        elif getattr(event, 'num', None) == 5:
            self._right_scroll_canvas.yview_scroll(1, 'units')

    def _onBand(self):
        try:
            self.max_dist_line = max(2.0, float(self.entBand.get()))
        except Exception:
            pass

    def _set_analysis_mode(self, mode: str):
        if self.license_manager and not self.license_manager.has_valid_license() and mode in ['ratio', 'polygon']:
            return

        # Reject pending analysis if switching modes
        if self.pending_analysis:
            self._reject_pending_analysis(ask_user=False)

        self.analysis_mode = mode
        hints = {
            'sl': 'Chain mode: Left Click to select two endpoints.',
            'ratio': 'Ratio mode: Left Click to select two endpoints.',
            'polygon': 'Polygon mode: Left Click to add vertices, close on the first point.',
        }
        if hasattr(self, 'status'): self.status.config(text=hints.get(mode))
        for key, btn in self.mode_buttons.items():
            if btn.cget('state') != tk.DISABLED: btn.config(relief='sunken' if key == mode else 'raised')

        # Cancel any ongoing selection when changing mode
        self.anchor_idx = None
        self.polygon_current_idx.clear()
        self._redraw_canvas()

    def _update_zoom_hint(self):
        if hasattr(self, 'zoom_hint'):
            value = int(round(self.zoom_val))
            self.zoom_hint.config(text=f'Current zoom: {value}% (0 = full view)')

    def _update_full_view_bounds(self):
        if self._img_shape is not None:
            h, w = self._img_shape
            self._full_view_bounds = (-0.5, w - 0.5, h - 0.5, -0.5)

    def _apply_zoom(self):
        if self._full_view_bounds is None: self._update_full_view_bounds()
        if self._full_view_bounds is None: return

        x0_full, x1_full, y0_full, y1_full = self._full_view_bounds
        if self.view_cx is None or self.view_cy is None:
            self.view_cx = (x0_full + x1_full) / 2.0
            self.view_cy = (y0_full + y1_full) / 2.0

        if self.zoom_val <= 0:
            self.ax.set_xlim(x0_full, x1_full)
            self.ax.set_ylim(y0_full, y1_full)
            return

        w, h = x1_full - x0_full, y0_full - y1_full  # y is inverted
        cx, cy = self.view_cx, self.view_cy

        min_dim = min(w, h)
        L = max(50.0, min_dim - (min_dim - 50.0) * (self.zoom_val / 100.0))
        half = L / 2.0

        x0, x1 = max(x0_full, cx - half), min(x1_full, cx + half)
        y0, y1 = min(y0_full, cy + half), max(y1_full, cy - half)  # inverted for imshow

        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)

    def _reset_zoom_state(self):
        self.zoom_val = 0.0
        self.view_cx, self.view_cy = None, None
        if self.points is not None and len(self.points) > 0:
            self.view_cy, self.view_cx = self.points.mean(axis=0)
        if hasattr(self, 'zoom_var'): self.zoom_var.set(0.0)
        self._update_zoom_hint()

    def _focus_on(self, x: float, y: float):
        self.view_cx, self.view_cy = float(x), float(y)
        self._apply_zoom()
        self.canvas.draw_idle()

    def _flash_right_scroll(self):
        if self._right_scroll_canvas:
            self._right_scroll_canvas.yview_moveto(1.0)
            self._right_scroll_canvas.after(120, lambda: self._right_scroll_canvas.yview_moveto(0.0))

    def save_png(self):
        p = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])
        if p:
            self.fig.savefig(p, dpi=150)
            self.status.config(text=f'Saved: {Path(p).name}')

    def _clear_rubber_lines(self):
        if self.rubber_line:
            try:
                self.rubber_line.remove()
            except Exception:
                pass
            self.rubber_line = None
        if self.polygon_rubber_line:
            try:
                self.polygon_rubber_line.remove()
            except Exception:
                pass
            self.polygon_rubber_line = None

    def _on_sl_keypress(self, event):
        # Manual editing of the SL chain is disabled in this workflow
        # to prevent data mismatch.
        messagebox.showinfo("Info",
                            "Manual editing of the S/L chain is disabled. Please re-run the chain analysis if needed.")
        return "break"

    def _set_sl_text(self, s: str):
        self.txt_sl.delete('1.0', tk.END)
        self.txt_sl.insert('1.0', s)

    def _collect_points_along_segment(self, i0: int, i1: int, max_dist: float) -> List[int]:
        if self.points is None: return []
        p0, p1 = self.points[i0, [1, 0]], self.points[i1, [1, 0]]  # x, y
        v = p1 - p0
        vv = float(np.dot(v, v))
        if vv == 0: return [i0]

        idx = []
        pts_xy = self.points[:, ::-1]  # y,x -> x,y

        for k, p_xy in enumerate(pts_xy):
            w = p_xy - p0
            t = float(np.dot(w, v) / vv)
            if 0.0 <= t <= 1.0:
                proj = p0 + t * v
                dist = float(np.hypot(p_xy[0] - proj[0], p_xy[1] - proj[1]))
                if dist <= max_dist:
                    idx.append((t, k))

        idx.sort(key=lambda z: z[0])

        # Ensure start and end points are included and unique
        result = []
        seen = set()
        if i0 not in seen:
            result.append(i0);
            seen.add(i0)

        for _, k in idx:
            if k not in seen:
                result.append(k);
                seen.add(k)

        if i1 not in seen:
            result.append(i1);
            seen.add(i1)

        # Re-sort based on original list to get the final order
        final_indices = [k for _, k in sorted([(idx[0], idx[1]) for idx in idx if idx[1] in result])]

        # Final check to ensure i0 and i1 are at the ends if they were in the original list
        if i0 in final_indices: final_indices.remove(i0)
        if i1 in final_indices: final_indices.remove(i1)

        # Determine correct order
        p0_yx = self.points[i0]
        p1_yx = self.points[i1]

        final_points = self.points[final_indices]
        if len(final_points) > 0:
            vec = p1_yx - p0_yx
            projs = np.dot(final_points - p0_yx, vec) / np.dot(vec, vec)
            sorted_k = np.array(final_indices)[np.argsort(projs)]
            final_indices = [i0] + list(sorted_k) + [i1]
        else:
            final_indices = [i0, i1]

        # Deduplicate one last time
        seen_final = set()
        deduped_final = []
        for k in final_indices:
            if k not in seen_final:
                deduped_final.append(k)
                seen_final.add(k)

        return deduped_final


class App(tk.Tk):
    """Standalone wrapper compatible with the previous CLI."""

    def __init__(self):
        super().__init__()
        self.title('fibonachi_analysis')
        self.geometry('1520x980')
        self.resizable(True, True)

        # Mock license manager for standalone run
        class MockLicense:
            def has_valid_license(self): return True

        frame = FibonacciAnalysisFrame(self, license_manager=MockLicense())
        frame.pack(fill=tk.BOTH, expand=True)
        self.frame = frame


if __name__ == '__main__':
    app = App()
    app.mainloop()