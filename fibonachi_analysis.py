from __future__ import annotations  # 1
# 1
import sys, json, math  # 1
from pathlib import Path  # 1
from typing import Optional, Tuple, List, Dict  # 1
# 1
import numpy as np  # 1
from PIL import Image  # 1
import tkinter as tk  # 1
from tkinter import filedialog, messagebox, ttk  # 1
# 1
import matplotlib  # 1
matplotlib.use('TkAgg')  # 1
import matplotlib.pyplot as plt  # 1
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # 1
from matplotlib.patches import Circle  # 1
# 1
from preproc import PreprocSettings, load_grayscale_with_preproc  # 1
if not hasattr(tk, "Notebook") and hasattr(ttk, "Notebook"):  # 1
    tk.Notebook = ttk.Notebook  # 1
# 1
class _Tooltip:  # 1
    def __init__(self, widget: tk.Widget, text: str, *, delay: int = 400):  # 1
        self.widget = widget  # 1
        self.text = text  # 1
        self.delay = max(0, int(delay))  # 1
        self._after_id: Optional[str] = None  # 1
        self._tip_window: Optional[tk.Toplevel] = None  # 1
        self._last_pointer: Optional[tuple[int, int]] = None  # 1
        widget.bind("<Enter>", self._on_enter, add="+")  # 1
        widget.bind("<Leave>", self._on_leave, add="+")  # 1
        widget.bind("<Motion>", self._on_motion, add="+")  # 1
# 1
    def _on_enter(self, event):  # 1
        self._last_pointer = (event.x_root, event.y_root)  # 1
        self._schedule()  # 1
# 1
    def _on_leave(self, _event):  # 1
        self._cancel()  # 1
        self._hide()  # 1
# 1
    def _on_motion(self, event):  # 1
        self._last_pointer = (event.x_root, event.y_root)  # 1
        self._position()  # 1
# 1
    def _schedule(self):  # 1
        self._cancel()  # 1
        self._after_id = self.widget.after(self.delay, self._show)  # 1
# 1
    def _cancel(self):  # 1
        if self._after_id is not None:  # 1
            self.widget.after_cancel(self._after_id)  # 1
            self._after_id = None  # 1
# 1
    def _show(self):  # 1
        if self._tip_window is not None or not self.text:  # 1
            return  # 1
        tip = tk.Toplevel(self.widget)  # 1
        tip.wm_overrideredirect(True)  # 1
        tip.wm_attributes("-topmost", True)  # 1
        label = tk.Label(  # 1
            tip,  # 1
            text=self.text,  # 1
            justify="left",  # 1
            background="#ffffe0",  # 1
            relief="solid",  # 1
            borderwidth=1,  # 1
            wraplength=360,  # 1
        )  # 1
        label.pack(ipadx=8, ipady=4)  # 1
        self._tip_window = tip  # 1
        self._position()  # 1
# 1
    def _hide(self):  # 1
        if self._tip_window is not None:  # 1
            self._tip_window.destroy()  # 1
            self._tip_window = None  # 1
# 1
    def _position(self):  # 1
        if self._tip_window is None:  # 1
            return  # 1
        tip = self._tip_window  # 1
        tip.update_idletasks()  # 1
        width = tip.winfo_reqwidth()  # 1
        height = tip.winfo_reqheight()  # 1
        if self._last_pointer is not None:  # 1
            x, y = self._last_pointer  # 1
        else:  # 1
            x = self.widget.winfo_rootx() + self.widget.winfo_width()  # 1
            y = self.widget.winfo_rooty() + self.widget.winfo_height()  # 1
        x += 12  # 1
        y += 10  # 1
        root = self.widget.winfo_toplevel()  # 1
        root.update_idletasks()  # 1
        left = root.winfo_rootx()  # 1
        top = root.winfo_rooty()  # 1
        right = left + root.winfo_width()  # 1
        bottom = top + root.winfo_height()  # 1
        if x + width > right - 4:  # 1
            x = right - width - 4  # 1
        if y + height > bottom - 4:  # 1
            y = bottom - height - 4  # 1
        x = max(x, left + 4)  # 1
        y = max(y, top + 4)  # 1
        tip.wm_geometry(f"+{int(x)}+{int(y)}")  # 1
# 1
# 1
# 1
class HoverTooltip:  # 1
    """Display a tooltip when the mouse hovers over a widget."""  # 1
# 1
    def __init__(self, widget: tk.Widget, text: str, delay: int = 400):  # 1
        self.widget = widget  # 1
        self.text = text  # 1
        self.delay = delay  # 1
        self._after_id: Optional[str] = None  # 1
        self._window: Optional[tk.Toplevel] = None  # 1
        widget.bind("<Enter>", self._schedule)  # 1
        widget.bind("<Leave>", self._hide)  # 1
        widget.bind("<ButtonPress>", self._hide)  # 1
# 1
    def _schedule(self, _event=None):  # 1
        self._cancel()  # 1
        self._after_id = self.widget.after(self.delay, self._show)  # 1
# 1
    def _cancel(self):  # 1
        if self._after_id is not None:  # 1
            try:  # 1
                self.widget.after_cancel(self._after_id)  # 1
            except Exception:  # 1
                pass  # 1
            self._after_id = None  # 1
# 1
    def _show(self):  # 1
        self._after_id = None  # 1
        if self._window is not None:  # 1
            return  # 1
        x = self.widget.winfo_pointerx() + 16  # 1
        y = self.widget.winfo_pointery() + 12  # 1
        self._window = tw = tk.Toplevel(self.widget)  # 1
        tw.wm_overrideredirect(True)  # 1
        tw.wm_geometry(f"+{x}+{y}")  # 1
        label = tk.Label(tw, text=self.text, background="#2f2f2f", foreground="white",  # 1
                         relief="solid", borderwidth=1, padx=6, pady=3, justify=tk.LEFT)  # 1
        label.pack()  # 1
# 1
    def _hide(self, _event=None):  # 1
        self._cancel()  # 1
        if self._window is not None:  # 1
            try:  # 1
                self._window.destroy()  # 1
            except Exception:  # 1
                pass  # 1
            self._window = None  # 1
# 1
# 1
# 1
# 1
def _parse_cli(argv=None):  # 1
    import argparse  # 1
    p = argparse.ArgumentParser(description="fibonachi_analysis — load input data")  # 1
    p.add_argument("--payload", type=str, default=None, help="Path to fibo_input.json")  # 1
    p.add_argument("--image", type=str, default=None, help="Path to the image (fallback)")  # 1
    p.add_argument("--points", type=str, default=None, help="Path to JSON with points (fallback)")  # 1
    return p.parse_args(argv)  # 1
# 1
# 1
def _candidate_dirs(extra_image: Optional[Path]) -> List[Path]:  # 1
    cands: List[Path] = []  # 1
    try:  # 1
        cands.append(Path.cwd())  # 1
    except Exception:  # 1
        pass  # 1
    if getattr(sys, "frozen", False):  # 1
        try:  # 1
            cands.append(Path(sys.executable).resolve().parent)  # 1
        except Exception:  # 1
            pass  # 1
        try:  # 1
            cands.append(Path(getattr(sys, "_MEIPASS")))  # 1
        except Exception:  # 1
            pass  # 1
    else:  # 1
        try:  # 1
            cands.append(Path(__file__).resolve().parent)  # 1
        except Exception:  # 1
            pass  # 1
    if extra_image:  # 1
        try:  # 1
            cands.append(extra_image.resolve().parent)  # 1
        except Exception:  # 1
            pass  # 1
    uniq, seen = [], set()  # 1
    for d in cands:  # 1
        rp = str(d.resolve())  # 1
        if rp not in seen:  # 1
            uniq.append(d); seen.add(rp)  # 1
    return uniq  # 1
# 1
# 1
def _autofind_json(extra_image: Optional[Path]) -> Optional[Path]:  # 1
    pats = ["fibo_input.json", "*fibo*input*.json", "*.fibo.json", "*.json"]  # 1
    for base in _candidate_dirs(extra_image):  # 1
        for pat in pats:  # 1
            try:  # 1
                for p in base.glob(pat):  # 1
                    name = p.name.lower()  # 1
                    if "fibo" in name and "input" in name:  # 1
                        return p.resolve()  # 1
                    if pat == "*.json":  # 1
                        try:  # 1
                            obj = json.loads(p.read_text(encoding="utf-8"))  # 1
                            if isinstance(obj, dict) and "image" in obj and "points" in obj:  # 1
                                return p.resolve()  # 1
                        except Exception:  # 1
                            pass  # 1
            except Exception:  # 1
                continue  # 1
    return None  # 1
# 1
# 1
def find_default_json(base_dir: Path) -> Optional[Path]:  # 1
    cand = base_dir / "fibo_input.json"  # 1
    if cand.exists():  # 1
        try:  # 1
            d = json.loads(cand.read_text(encoding="utf-8"))  # 1
            if "image" in d and "points" in d:  # 1
                return cand.resolve()  # 1
        except Exception:  # 1
            pass  # 1
    for p in base_dir.glob("*.json"):  # 1
        try:  # 1
            d = json.loads(p.read_text(encoding="utf-8"))  # 1
            if "image" in d and "points" in d:  # 1
                return p.resolve()  # 1
        except Exception:  # 1
            pass  # 1
    return _autofind_json(None)  # 1
# 1
# 1
def load_input(json_path: Path):  # 1
    d = json.loads(json_path.read_text(encoding='utf-8'))  # 1
    img = Path(d['image']) if d.get('image') else None  # 1
    if not img:  # 1
        raise RuntimeError("JSON does not contain the key 'image'.")  # 1
    pts = np.array([[float(p['y']), float(p['x'])] for p in d.get('points', [])], float)  # 1
    center = None; dead = 0.0; srch = 0.0  # 1
    if isinstance(d.get('centers'), dict):  # 1
        c = d['centers'].get('overlay') or d['centers'].get('geometric')  # 1
        if c and 'x' in c and 'y' in c:  # 1
            center = (float(c['y']), float(c['x']))  # 1
    if isinstance(d.get('radii'), dict):  # 1
        if d['radii'].get('dead') is not None:  dead = float(d['radii']['dead'])  # 1
        if d['radii'].get('search') is not None: srch = float(d['radii']['search'])  # 1
    fallback_mode = d.get('preproc_mode')  # 1
    if not isinstance(fallback_mode, str):  # 1
        fallback_mode = None  # 1
    preproc = PreprocSettings.from_json(d.get('preproc'), fallback_mode=fallback_mode)  # 1
    return img, pts, center, dead, srch, preproc  # 1
# 1
# 1
# 1
def cluster_lengths(lengths: np.ndarray):  # 1
    """k=2 clustering of lengths into S/L; returns (labels, Slen, Llen, sidx, lidx)."""  # 1
    if lengths.size == 0:  # 1
        return np.array([], dtype=int), float("nan"), float("nan"), 0, 1  # 1
    c0, c1 = float(lengths.min()), float(lengths.max())  # 1
    if c0 == c1:  # 1
        lab = np.zeros(len(lengths), dtype=int)  # 1
        return lab, c0, float("nan"), 0, 1  # 1
    lab = np.zeros(len(lengths), dtype=int)  # 1
    for _ in range(60):  # 1
        d0 = np.abs(lengths - c0)  # 1
        d1 = np.abs(lengths - c1)  # 1
        lab = (d1 < d0).astype(int)  # 1
        nc0 = float(lengths[lab == 0].mean()) if np.any(lab == 0) else c0  # 1
        nc1 = float(lengths[lab == 1].mean()) if np.any(lab == 1) else c1  # 1
        if abs(nc0 - c0) < 1e-6 and abs(nc1 - c1) < 1e-6:  # 1
            c0, c1 = nc0, nc1; break  # 1
        c0, c1 = nc0, nc1  # 1
    # 1
    m0 = float(lengths[lab == 0].mean()) if np.any(lab == 0) else float("nan")  # 1
    m1 = float(lengths[lab == 1].mean()) if np.any(lab == 1) else float("nan")  # 1
    if (not math.isnan(m0)) and (not math.isnan(m1)) and m0 > m1:  # 1
        lab = 1 - lab  # 1
        m0, m1 = m1, m0  # 1
    return lab, m0, m1, 0, 1  # 1
# 1
def fib_list_upto(n: int) -> List[int]:  # 1
    """Fibonacci numbers up to n (inclusive), starting with 1,1,2,3,..."""  # 1
    if n <= 0: return []  # 1
    seq = [1, 1]  # 1
    while seq[-1] < n:  # 1
        seq.append(seq[-1] + seq[-2])  # 1
    return [k for k in seq if k <= n]  # 1
# 1
def gen_fibonacci_words(max_len: int, start: str = "L") -> List[str]:  # 1
    """Generate prefixes of the "fibo-word": L->LS, S->L, up to max_len."""  # 1
    if max_len <= 0: return []  # 1
    words = ["L" if start.upper() == "L" else "S"]  # 1
    while len(words[-1]) <= max_len:  # 1
        nxt = "".join(("LS" if ch == "L" else "L") for ch in words[-1])  # 1
        if len(nxt) > max_len: break  # 1
        words.append(nxt)  # 1
    return words  # 1
# 1
# 1
# 1
class FibonacciAnalysisFrame(tk.Frame):  # 1
    def __init__(self, master: tk.Misc, controller=None, auto_load: bool = True):  # 1
        super().__init__(master)  # 1
        self.controller = controller  # 1
# 1
        # 1
        self.img_path: Optional[Path] = None  # 1
        self.points: Optional[np.ndarray] = None      # 1
        self.center: Optional[Tuple[float, float]] = None  # 1
        self.dead: float = 0.0  # 1
        self.srch: float = 0.0  # 1
        self.preproc: PreprocSettings = PreprocSettings(mode="raw")  # 1
        self.img_arr: Optional[np.ndarray] = None  # 1
        self.zoom_val: float = 0.0  # 1
        self.view_cx: Optional[float] = None  # 1
        self.view_cy: Optional[float] = None  # 1
        self._img_shape: Optional[Tuple[int, int]] = None  # 1
        self._full_view_bounds: Optional[Tuple[float, float, float, float]] = None  # 1
# 1
        # 1
        self.pick_tol = 10.0  # 1
        self.max_dist_line = 20.0  # 1
# 1
        self.mode_buttons: Dict[str, tk.Button] = {}  # 1
        self._mode_tooltips: List[HoverTooltip] = []  # 1
        self.analysis_mode: str = 'sl'  # 1
        self._last_analysis_mode: Optional[str] = None  # 1
# 1
        self.selected_idx: List[int] = []  # 1
        self.ratio_selected_idx: List[int] = []  # 1
        self.polygon_current_idx: List[int] = []  # 1
        self.polygons_idx: List[List[int]] = []  # 1
        self._polygon_history: List[Tuple[List[int], List[List[int]]]] = []  # 1
        self._polygon_redo: List[Tuple[List[int], List[List[int]]]] = []  # 1
        self._right_scroll_canvas: Optional[tk.Canvas] = None  # 1
        self._right_scroll_window: Optional[int] = None  # 1
# 1
        self.list_index_map: Dict[int, Tuple] = {}  # 1
        self.curr_chain = None  # 1
        self.curr_seg = None  # 1
        self.curr_labels = None  # 1
        self.curr_ratio = float('nan')  # 1
# 1
        self.anchor_idx: Optional[int] = None  # 1
        self.ratio_anchor_idx: Optional[int] = None  # 1
        self.rubber_line = None  # 1
        self.rubber_line_ratio = None  # 1
        self.polygon_rubber_line = None  # 1
# 1
        self.columnconfigure(0, weight=1)  # 1
        self.rowconfigure(0, weight=1)  # 1
# 1
        container = tk.Frame(self)  # 1
        container.grid(row=0, column=0, sticky="nsew")  # 1
        container.columnconfigure(0, weight=1)  # 1
        container.columnconfigure(1, weight=0)  # 1
# 1
        left = tk.Frame(container)  # 1
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=12)  # 1
        left.rowconfigure(0, weight=1)  # 1
        left.columnconfigure(0, weight=1)  # 1
# 1
        right_host = tk.Frame(container, width=470)  # 1
        right_host.grid(row=0, column=1, sticky="ns", pady=12)  # 1
        right_host.grid_columnconfigure(0, weight=1)  # 1
        right_host.grid_rowconfigure(0, weight=1)  # 1
# 1
        scroll_host = ttk.Frame(right_host)  # 1
        scroll_host.grid(row=0, column=0, sticky="nsew")  # 1
        scroll_host.grid_columnconfigure(0, weight=1)  # 1
        scroll_host.grid_rowconfigure(0, weight=1)  # 1
# 1
        canvas = tk.Canvas(scroll_host, borderwidth=0, highlightthickness=0)  # 1
        vscroll = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=canvas.yview)  # 1
        right = tk.Frame(canvas)  # 1
        self._right_scroll_canvas = canvas  # 1
        self._right_scroll_window = canvas.create_window((0, 0), window=right, anchor="nw")  # 1
        canvas.configure(yscrollcommand=vscroll.set)  # 1
# 1
        right.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))  # 1
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(self._right_scroll_window, width=e.width))  # 1
# 1
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # 1
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)  # 1
# 1
        right.bind("<Enter>", self._activate_right_scroll)  # 1
        right.bind("<Leave>", self._deactivate_right_scroll)  # 1
        canvas.bind("<Enter>", self._activate_right_scroll)  # 1
        canvas.bind("<Leave>", self._deactivate_right_scroll)  # 1
# 1
        right.columnconfigure(0, weight=1)  # 1
# 1
        controls = tk.Frame(right)  # 1
        controls.grid(row=0, column=0, sticky="ew", padx=6, pady=(0, 10))  # 1
        controls.columnconfigure(0, weight=1)  # 1
        controls.columnconfigure(1, weight=1)  # 1
# 1
        tk.Button(controls, text='Open JSON…', command=self.open_json).grid(row=0, column=0, sticky='ew', padx=4, pady=2)  # 1
        tk.Button(controls, text='Save PNG', command=self.save_png).grid(row=0, column=1, sticky='ew', padx=4, pady=2)  # 1
        tk.Button(controls, text='Clear selection', command=self.clear_selection).grid(row=1, column=0, columnspan=2, sticky='ew', padx=4, pady=2)  # 1
# 1
        tk.Label(controls, text='Selection thickness (px):').grid(row=2, column=0, sticky='w', padx=4, pady=(8, 2))  # 1
        self.entBand = tk.Spinbox(controls, from_=2, to=100, width=6, command=self._onBand)  # 1
        self.entBand.delete(0, 'end'); self.entBand.insert(0, str(int(self.max_dist_line)))  # 1
        self.entBand.grid(row=2, column=1, sticky='ew', padx=4, pady=(8, 2))  # 1
# 1
        mode_switch = tk.Frame(controls)  # 1
        mode_switch.grid(row=3, column=0, columnspan=2, sticky='ew', padx=4, pady=(8, 2))  # 1
        for col in range(3):  # 1
            mode_switch.columnconfigure(col, weight=1)  # 1
# 1
        self.mode_buttons['sl'] = tk.Button(mode_switch, text='🔗 Chain', command=lambda: self._set_analysis_mode('sl'))  # 1
        self.mode_buttons['sl'].grid(row=0, column=0, sticky='ew', padx=(0, 2))  # 1
        self._mode_tooltips.append(HoverTooltip(self.mode_buttons['sl'], 'Chain analysis: use Left Click while this mode is active.'))  # 1
# 1
        self.mode_buttons['ratio'] = tk.Button(mode_switch, text='📊 Ratio', command=lambda: self._set_analysis_mode('ratio'))  # 1
        self.mode_buttons['ratio'].grid(row=0, column=1, sticky='ew', padx=2)  # 1
        self._mode_tooltips.append(HoverTooltip(self.mode_buttons['ratio'], 'Neighbor ratios: use Left Click while this mode is active.'))  # 1
# 1
        self.mode_buttons['polygon'] = tk.Button(mode_switch, text='🔺 Polygon', command=lambda: self._set_analysis_mode('polygon'))  # 1
        self.mode_buttons['polygon'].grid(row=0, column=2, sticky='ew', padx=(2, 0))  # 1
        self._mode_tooltips.append(HoverTooltip(self.mode_buttons['polygon'], 'Polygon analysis: add vertices with Left Click, close by clicking the first point.'))  # 1
# 1
        zoom_group = ttk.LabelFrame(controls, text='Scale')  # 1
        zoom_group.grid(row=4, column=0, columnspan=2, sticky='ew', padx=4, pady=(8, 4))  # 1
        self.zoom_var = tk.DoubleVar(value=self.zoom_val)  # 1
        self.zoom_scale = ttk.Scale(zoom_group, from_=0, to=100, variable=self.zoom_var, command=self._on_zoom_change)  # 1
        self.zoom_scale.pack(fill=tk.X, padx=6, pady=(0, 4))  # 1
        self.zoom_hint = ttk.Label(zoom_group, anchor='w')  # 1
        self.zoom_hint.pack(fill=tk.X, padx=6, pady=(0, 2))  # 1
        self._update_zoom_hint()  # 1
# 1
        self.status = tk.Label(right, text='', anchor='w')  # 1
        self.status.grid(row=1, column=0, sticky='ew', padx=6, pady=(0, 10))  # 1
        self._set_analysis_mode('sl')  # 1
        self.lst_header = tk.Label(right, text='Found words (Fibonacci subsegments)')  # 1
        self.lst_header.grid(row=2, column=0, sticky='w', padx=6, pady=(0, 2))  # 1
# 1
        self.results_notebook = tk.Notebook(right)  # 1
        self.results_notebook.grid(row=3, column=0, sticky='nsew', padx=6, pady=(0, 8))  # 1
        right.grid_rowconfigure(3, weight=1)  # 1
# 1
        tab_subsegments = ttk.Frame(self.results_notebook)  # 1
        tab_subsegments.grid_columnconfigure(0, weight=1)  # 1
        tab_subsegments.grid_rowconfigure(0, weight=1)  # 1
# 1
        tab_prefixes = ttk.Frame(self.results_notebook)  # 1
        tab_prefixes.grid_columnconfigure(0, weight=1)  # 1
        tab_prefixes.grid_rowconfigure(1, weight=1)  # 1
# 1
        self.results_notebook.add(tab_subsegments, text='Subsegments')  # 1
        self.results_notebook.add(tab_prefixes, text='Fib prefixes')  # 1
# 1
        self.lst = tk.Listbox(tab_subsegments, width=66, height=22)  # 1
        self.lst.grid(row=0, column=0, sticky='nsew')  # 1
        self.lst.bind('<<ListboxSelect>>', self._on_list_select)  # 1
# 1
        self.lbl_ratio = tk.Label(tab_subsegments, text='Average L/S along chain: —')  # 1
        self.lbl_ratio.grid(row=1, column=0, sticky='w', pady=(6, 4))  # 1
        self.lbl_ratio_neigh = tk.Label(tab_subsegments, text='Average neighboring segment ratio: —')  # 1
        self.lbl_ratio_neigh.grid(row=2, column=0, sticky='w', pady=(2, 4))  # 1
        self.lbl_ratio_polygons = tk.Label(tab_subsegments, text='Average neighboring polygon linear ratio: —')  # 1
        self.lbl_ratio_polygons.grid(row=3, column=0, sticky='w', pady=(2, 8))  # 1
        # 1
        tk.Label(tab_subsegments, text='S/L sequence (full):').grid(row=4, column=0, sticky='w', pady=(4, 2))  # 1
        self.txt_sl = tk.Text(tab_subsegments, height=6, wrap='word')  # 1
        self.txt_sl.grid(row=5, column=0, sticky='ew', pady=(0, 4))  # 1
        self.txt_sl.bind('<KeyPress>', self._on_sl_keypress)  # 1
# 1
        prefixes_header = tk.Label(tab_prefixes, text='Prefixes of "fib-words" (L→LS, S→L)')  # 1
        prefixes_header.grid(row=0, column=0, sticky='w', pady=(0, 4))  # 1
# 1
        prefixes_frame = tk.Frame(tab_prefixes)  # 1
        prefixes_frame.grid(row=1, column=0, sticky='nsew')  # 1
        prefixes_frame.grid_columnconfigure(0, weight=1)  # 1
        prefixes_frame.grid_rowconfigure(0, weight=1)  # 1
# 1
        self.txt_words = tk.Text(prefixes_frame, height=10, state='disabled')  # 1
        self.txt_words.grid(row=0, column=0, sticky='nsew')  # 1
        scroll_words = tk.Scrollbar(prefixes_frame, orient='vertical', command=self.txt_words.yview)  # 1
        scroll_words.grid(row=0, column=1, sticky='ns')  # 1
        self.txt_words.configure(yscrollcommand=scroll_words.set)  # 1
# 1
        self.fig = plt.Figure(figsize=(9.6, 6.6)); self.ax = self.fig.add_subplot(111)  # 1
        self.ax.axis('off')  # 1
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)  # 1
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')  # 1
        # 1
        self.canvas.mpl_connect('button_press_event', self._on_click)  # 1
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)  # 1
        self.bind('<Escape>', lambda e: self.clear_selection())  # 1
        self.bind_all('<Return>', self._on_enter_key)  # 1
        self.bind_all('<z>', self._on_polygon_undo)  # 1
        self.bind_all('<z>', self._on_polygon_undo)  # 1
        self.bind_all('<y>', self._on_polygon_redo)  # 1
        self.bind_all('<y>', self._on_polygon_redo)  # 1
# 1
        self._reset_polygon_history()  # 1
# 1
        # 1
        if auto_load:  # 1
            base = Path(getattr(sys, '_MEIPASS', Path(__file__).parent)) if getattr(sys, 'frozen', False) else Path(  # 1
                __file__).parent  # 1
            auto = find_default_json(base)  # 1
            if auto:  # 1
                try:  # 1
                    self.load_json(auto)  # 1
                    self.status.config(text=f'Loaded: {auto.name}')  # 1
                except Exception as e:  # 1
                    messagebox.showerror('Load error', str(e))  # 1
            else:  # 1
                self.status.config(text='JSON not found. Select a file manually.')  # 1
        else:  # 1
            self.status.config(text='JSON not loaded. Use "Open JSON…".')  # 1
    # 1
# 1
    def _activate_right_scroll(self, _event):  # 1
        if self._right_scroll_canvas is None:  # 1
            return  # 1
        self._right_scroll_canvas.bind_all("<MouseWheel>", self._on_right_scroll_mousewheel)  # 1
        self._right_scroll_canvas.bind_all("<Button-4>", self._on_right_scroll_mousewheel)  # 1
        self._right_scroll_canvas.bind_all("<Button-5>", self._on_right_scroll_mousewheel)  # 1
# 1
    def _deactivate_right_scroll(self, _event):  # 1
        if self._right_scroll_canvas is None:  # 1
            return  # 1
        self._right_scroll_canvas.unbind_all("<MouseWheel>")  # 1
        self._right_scroll_canvas.unbind_all("<Button-4>")  # 1
        self._right_scroll_canvas.unbind_all("<Button-5>")  # 1
# 1
    def _on_right_scroll_mousewheel(self, event):  # 1
        if self._right_scroll_canvas is None:  # 1
            return  # 1
        if getattr(event, 'delta', 0):  # 1
            self._right_scroll_canvas.yview_scroll(int(-event.delta / 120), 'units')  # 1
        elif getattr(event, 'num', None) == 4:  # 1
            self._right_scroll_canvas.yview_scroll(-1, 'units')  # 1
        elif getattr(event, 'num', None) == 5:  # 1
            self._right_scroll_canvas.yview_scroll(1, 'units')  # 1
# 1
    def _onBand(self):  # 1
        try:  # 1
            self.max_dist_line = max(2.0, float(self.entBand.get()))  # 1
        except Exception:  # 1
            pass  # 1
# 1
    def _set_analysis_mode(self, mode: str):  # 1
        if mode not in ('sl', 'ratio', 'polygon'):  # 1
            return  # 1
        self.analysis_mode = mode  # 1
        hints = {  # 1
            'sl': 'Chain mode: Left Click to anchor two peaks and analyze the Fibonacci split.',  # 1
            'ratio': 'Ratio mode: Left Click to anchor two peaks and inspect neighboring segment ratios.',  # 1
            'polygon': 'Polygon mode: Left Click to add vertices, close on the first point, use Z/Y to undo/redo.',  # 1
        }  # 1
        hint = hints.get(mode)  # 1
        if hint and hasattr(self, 'status'):  # 1
            self.status.config(text=hint)  # 1
        for key, btn in self.mode_buttons.items():  # 1
            relief = 'sunken' if key == mode else 'raised'  # 1
            btn.config(relief=relief)  # 1
        if mode != 'sl':  # 1
            self.anchor_idx = None  # 1
            self._clear_rubber()  # 1
        if mode != 'ratio':  # 1
            self.ratio_anchor_idx = None  # 1
            self._clear_rubber_ratio()  # 1
        if mode != 'polygon':  # 1
            self._clear_polygon_rubber()  # 1
# 1
    def _update_zoom_hint(self):  # 1
        if hasattr(self, 'zoom_hint'):  # 1
            try:  # 1
                value = int(round(float(self.zoom_var.get()))) if hasattr(self, 'zoom_var') else int(round(self.zoom_val))  # 1
            except Exception:  # 1
                value = int(round(self.zoom_val))  # 1
            self.zoom_hint.config(text=f'Current zoom: {value}% (0 = full view)')  # 1
# 1
    def _ensure_view_center(self):  # 1
        if self.view_cx is not None and self.view_cy is not None:  # 1
            return  # 1
        if self.center is not None:  # 1
            cy, cx = self.center  # 1
            self.view_cx = float(cx)  # 1
            self.view_cy = float(cy)  # 1
            return  # 1
        if self.points is not None and len(self.points):  # 1
            ys = self.points[:, 0]  # 1
            xs = self.points[:, 1]  # 1
            self.view_cx = float(xs.mean())  # 1
            self.view_cy = float(ys.mean())  # 1
            return  # 1
        if self._full_view_bounds is not None:  # 1
            x0, x1, y0, y1 = self._full_view_bounds  # 1
            self.view_cx = (x0 + x1) / 2.0  # 1
            self.view_cy = (y0 + y1) / 2.0  # 1
# 1
    def _update_full_view_bounds(self):  # 1
        bounds = None  # 1
        if self._img_shape is not None:  # 1
            h, w = self._img_shape  # 1
            bounds = (-0.5, w - 0.5, -0.5, h - 0.5)  # 1
        elif self.points is not None and len(self.points):  # 1
            xs = self.points[:, 1]  # 1
            ys = self.points[:, 0]  # 1
            min_x = float(xs.min())  # 1
            max_x = float(xs.max())  # 1
            min_y = float(ys.min())  # 1
            max_y = float(ys.max())  # 1
            margin_x = max((max_x - min_x) * 0.05, 10.0)  # 1
            margin_y = max((max_y - min_y) * 0.05, 10.0)  # 1
            bounds = (min_x - margin_x, max_x + margin_x, min_y - margin_y, max_y + margin_y)  # 1
        self._full_view_bounds = bounds  # 1
# 1
    def _apply_zoom(self):  # 1
        if self._full_view_bounds is None:  # 1
            return  # 1
        x0_full, x1_full, y0_full, y1_full = self._full_view_bounds  # 1
        width = x1_full - x0_full  # 1
        height = y1_full - y0_full  # 1
        self._ensure_view_center()  # 1
        cx = float(self.view_cx if self.view_cx is not None else (x0_full + x1_full) / 2.0)  # 1
        cy = float(self.view_cy if self.view_cy is not None else (y0_full + y1_full) / 2.0)  # 1
        if self.zoom_val <= 0:  # 1
            self.ax.set_xlim(x0_full, x1_full)  # 1
            self.ax.set_ylim(y1_full, y0_full)  # 1
            return  # 1
        min_dim = float(min(width, height))  # 1
        L = float(round(min_dim - (min_dim - 50.0) * (self.zoom_val / 100.0)))  # 1
        L = max(50.0, min_dim if L < 50.0 else L)  # 1
        half = L / 2.0  # 1
        x0 = max(x0_full, cx - half)  # 1
        x1 = min(x1_full, cx + half)  # 1
        if (x1 - x0) < L:  # 1
            if x0 <= x0_full:  # 1
                x1 = min(x0 + L, x1_full)  # 1
            elif x1 >= x1_full:  # 1
                x0 = max(x1 - L, x0_full)  # 1
        y0 = max(y0_full, cy - half)  # 1
        y1 = min(y1_full, cy + half)  # 1
        if (y1 - y0) < L:  # 1
            if y0 <= y0_full:  # 1
                y1 = min(y0 + L, y1_full)  # 1
            elif y1 >= y1_full:  # 1
                y0 = max(y1 - L, y0_full)  # 1
        self.ax.set_xlim(x0, x1)  # 1
        self.ax.set_ylim(y1, y0)  # 1
# 1
    def _on_zoom_change(self, value):  # 1
        try:  # 1
            new_val = float(value)  # 1
        except Exception:  # 1
            new_val = 0.0  # 1
        new_val = max(0.0, min(100.0, new_val))  # 1
        self.zoom_val = new_val  # 1
        if hasattr(self, 'zoom_var'):  # 1
            current = float(self.zoom_var.get())  # 1
            if abs(current - new_val) > 1e-3:  # 1
                self.zoom_var.set(new_val)  # 1
        self._apply_zoom()  # 1
        self._update_zoom_hint()  # 1
        if hasattr(self, 'canvas'):  # 1
            self.canvas.draw_idle()  # 1
# 1
    def _reset_zoom_state(self):  # 1
        self.zoom_val = 0.0  # 1
        self.view_cx = None  # 1
        self.view_cy = None  # 1
        if hasattr(self, 'zoom_var'):  # 1
            try:  # 1
                self.zoom_var.set(0.0)  # 1
            except Exception:  # 1
                pass  # 1
        self._update_zoom_hint()  # 1
# 1
    def _focus_on(self, x: float, y: float):  # 1
        self.view_cx = float(x)  # 1
        self.view_cy = float(y)  # 1
        if self._full_view_bounds is not None and hasattr(self, 'canvas'):  # 1
            self._apply_zoom()  # 1
            self.canvas.draw_idle()  # 1
# 1
    def _flash_right_scroll(self):  # 1
        canvas = self._right_scroll_canvas  # 1
        if canvas is None or not canvas.winfo_exists():  # 1
            return  # 1

        def _scroll_bottom():  # 1
            if not canvas.winfo_exists():  # 1
                return  # 1
            canvas.yview_moveto(1.0)  # 1
            canvas.after(120, _scroll_top)  # 1

        def _scroll_top():  # 1
            if not canvas.winfo_exists():  # 1
                return  # 1
            canvas.yview_moveto(0.0)  # 1

        canvas.after_idle(_scroll_bottom)  # 1
# 1
    def _on_click(self, event):  # 1
        """Route left-clicks to the currently selected analysis mode."""  # 1
        if self.points is None or event.xdata is None or event.ydata is None:  # 1
            return  # 1
        if event.button != 1:  # 1
            return  # 1
        x, y = float(event.xdata), float(event.ydata)  # 1
# 1
        # 1
        d2 = (self.points[:,1] - x)**2 + (self.points[:,0] - y)**2  # 1
        j = int(np.argmin(d2))  # 1
        if math.sqrt(d2[j]) > self.pick_tol:  # 1
            return  # 1
        focus_x = float(self.points[j, 1])  # 1
        focus_y = float(self.points[j, 0])  # 1
        self._focus_on(focus_x, focus_y)  # 1
# 1
        if self.analysis_mode == 'sl':  # 1
            # 1
            if self.anchor_idx is None:  # 1
                self.anchor_idx = j  # 1
                self._clear_rubber()  # 1
                self.draw_base(); self._draw_anchor(self.anchor_idx)  # 1
            else:  # 1
                i0, i1 = self.anchor_idx, j  # 1
                self.selected_idx = self._collect_points_along_segment(i0, i1, self.max_dist_line)  # 1
                self.anchor_idx = None  # 1
                self._clear_rubber()  # 1
                self.draw_base(); self._draw_selection(self.selected_idx)  # 1
                self.run_analysis()  # 1
        elif self.analysis_mode == 'ratio':  # 1
            # 1
            if self.ratio_anchor_idx is None:  # 1
                self.ratio_anchor_idx = j  # 1
                self._clear_rubber_ratio()  # 1
                self.draw_base(); self._draw_anchor(self.ratio_anchor_idx)  # 1
            else:  # 1
                i0, i1 = self.ratio_anchor_idx, j  # 1
                self.ratio_selected_idx = self._collect_points_along_segment(i0, i1, self.max_dist_line)  # 1
                self.ratio_anchor_idx = None  # 1
                self._clear_rubber_ratio()  # 1
                self.draw_base();  # 1
                self._draw_selection(self.ratio_selected_idx)  # 1
                self.run_ratio_analysis()  # 1
        elif self.analysis_mode == 'polygon':  # 1
            # 1
            self._handle_polygon_click(j)  # 1
# 1
    def _on_motion(self, event):  # 1
        if event.xdata is None or event.ydata is None or self.points is None:  # 1
            return  # 1
            # 1
        if self.anchor_idx is not None:  # 1
            ax = self.points[self.anchor_idx, 1]  # 1
            ay = self.points[self.anchor_idx, 0]  # 1
            bx = float(event.xdata); by = float(event.ydata)  # 1
            if self.rubber_line is None:  # 1
                (self.rubber_line,) = self.ax.plot([ax, bx], [ay, by], color='yellow', lw=2.0, alpha=0.9)  # 1
            else:  # 1
                self.rubber_line.set_data([ax, bx], [ay, by])  # 1
            self.canvas.draw_idle()  # 1
        # 1
        if self.ratio_anchor_idx is not None:  # 1
            ax = self.points[self.ratio_anchor_idx, 1]  # 1
            ay = self.points[self.ratio_anchor_idx, 0]  # 1
            bx = float(event.xdata); by = float(event.ydata)  # 1
        if self.rubber_line_ratio is None:  # 1
            (self.rubber_line_ratio,) = self.ax.plot([ax, bx], [ay, by], color='yellow', lw=2.0, alpha=0.9)  # 1
        else:  # 1
            self.rubber_line_ratio.set_data([ax, bx], [ay, by])  # 1
        self.canvas.draw_idle()  # 1
        # 1
        if self.polygon_current_idx:  # 1
            last_idx = self.polygon_current_idx[-1]  # 1
            ax = self.points[last_idx, 1]  # 1
            ay = self.points[last_idx, 0]  # 1
            bx = float(event.xdata);  # 1
            by = float(event.ydata)  # 1
            if self.polygon_rubber_line is None:  # 1
                (self.polygon_rubber_line,) = self.ax.plot([ax, bx], [ay, by], color='orange', lw=2.0, alpha=0.8,  # 1
                                                           zorder=3.4)  # 1
            else:  # 1
                self.polygon_rubber_line.set_data([ax, bx], [ay, by])  # 1
            self.canvas.draw_idle()  # 1
# 1
    def _on_list_select(self, event):  # 1
        """Highlight on list row click (both modes)."""  # 1
        if not self.list_index_map:  # 1
            return  # 1
        sel = self.lst.curselection()  # 1
        if not sel:  # 1
            return  # 1
        row = sel[0]  # 1
        meta = self.list_index_map.get(row)  # 1
        if not meta:  # 1
            return  # 1
        kind = meta[0]  # 1
# 1
        # 1
        self.draw_base()  # 1
        if kind == 'sl':  # 1
            # 1
            i0, n = meta[1], meta[2]  # 1
            if self.curr_chain is None or self.curr_labels is None:  # 1
                return  # 1
            # 1
            self._draw_selection(self.selected_idx)  # 1
            # 1
            self._highlight_word(self.curr_chain, self.curr_labels, i0, n)  # 1
        elif kind == 'ratio':  # 1
            # 1
            k = meta[1]  # 1
            if len(self.ratio_selected_idx) < 3:  # 1
                return  # 1
            chain = self.points[self.ratio_selected_idx].copy()  # 1
            self._draw_selection(self.ratio_selected_idx)  # 1
            # 1
            self._highlight_ratio_pair(chain, k-1, k)  # 1
        self.canvas.draw_idle()  # 1
# 1
    # 1
    def open_json(self):  # 1
        p = filedialog.askopenfilename(filetypes=[('JSON', '*.json'), ('All', '*.*')])  # 1
        if not p: return  # 1
        self.load_json(Path(p))  # 1
# 1
    def load_json(self, json_path: Path):  # 1
        self.img_path, self.points, self.center, self.dead, self.srch, self.preproc = load_input(json_path)  # 1
        self.img_arr = None  # 1
        self._img_shape = None  # 1
        self._full_view_bounds = None  # 1
        if self.img_path is not None:  # 1
            try:  # 1
                self.img_arr = load_grayscale_with_preproc(self.img_path, self.preproc)  # 1
            except Exception as exc:  # 1
                messagebox.showerror('Error', f'Failed to prepare the image:\n{exc}')  # 1
                self.img_arr = None  # 1
        self.clear_selection(redraw=False)  # 1
        self._reset_zoom_state()  # 1
        self.draw_base()  # 1
        self.status.config(text=f'Loaded: {json_path.name}')  # 1
        self._flash_right_scroll()  # 1
# 1
    def save_png(self):  # 1
        if self.img_path is None:  # 1
            messagebox.showinfo('Save', 'No data loaded.'); return  # 1
        p = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])  # 1
        if not p: return  # 1
        self.fig.savefig(p, dpi=150)  # 1
        self.status.config(text=f'Saved: {Path(p).name}')  # 1
# 1
    def draw_base(self):  # 1
        self.ax.clear()  # 1
        img_shape = None  # 1
        if self.img_arr is not None:  # 1
            img_shape = self.img_arr.shape[:2]  # 1
            self.ax.imshow(self.img_arr, cmap='gray', interpolation='nearest')  # 1
        elif self.img_path:  # 1
            im = Image.open(self.img_path).convert('L')  # 1
            img_arr = np.array(im)  # 1
            img_shape = img_arr.shape[:2]  # 1
            self.ax.imshow(img_arr, cmap='gray', interpolation='nearest')  # 1
        self._img_shape = img_shape  # 1
        if self.points is not None and len(self.points):  # 1
            self.ax.scatter(self.points[:, 1], self.points[:, 0],  # 1
                            s=24, c='cyan', edgecolors='black', linewidths=0.4, label='peaks')  # 1
        if self.center is not None:  # 1
            cy, cx = self.center  # 1
            self.ax.scatter([cx], [cy], s=40, c='red', marker='o', label='center')  # 1
            if self.dead and self.dead > 0:  # 1
                self.ax.add_patch(Circle((cx, cy), self.dead, fill=False, ec='red', ls='--', lw=1.5))  # 1
            if self.srch and self.srch > 0:  # 1
                self.ax.add_patch(Circle((cx, cy), self.srch, fill=False, ec='red', ls=':', lw=1.0))  # 1
        self._update_full_view_bounds()  # 1
        self.ax.axis('off')  # 1
        self._draw_polygons()  # 1
        self._apply_zoom()  # 1
        self.canvas.draw_idle()  # 1
# 1
    def _draw_anchor(self, idx: int):  # 1
        y, x = self.points[idx]  # 1
        self.ax.scatter([x], [y], s=52, c='yellow', edgecolors='k', linewidths=0.6, zorder=4)  # 1
        self.canvas.draw_idle()  # 1
# 1
    def _draw_selection(self, idxs: List[int]):  # 1
        if not idxs: return  # 1
        sel = self.points[idxs]  # 1
        self.ax.scatter(sel[:,1], sel[:,0], s=36, c='magenta', edgecolors='k', linewidths=0.6, zorder=3)  # 1
        for i in range(len(sel)-1):  # 1
            y1,x1 = sel[i]; y2,x2 = sel[i+1]  # 1
            self.ax.plot([x1,x2],[y1,y2], color='yellow', lw=1.8, ls='--', zorder=2)  # 1
            self.ax.text(x1, y1, str(i+1), color='magenta', fontsize=8, ha='right', va='bottom')  # 1
        yN, xN = sel[-1]  # 1
        self.ax.text(xN, yN, str(len(sel)), color='magenta', fontsize=8, ha='right', va='bottom')  # 1
        self.canvas.draw_idle()  # 1
# 1
    def _clear_rubber(self):  # 1
        if self.rubber_line is not None:  # 1
            try: self.rubber_line.remove()  # 1
            except Exception: pass  # 1
            self.rubber_line = None  # 1
            self.canvas.draw_idle()  # 1
# 1
    def _clear_rubber_ratio(self):  # 1
        if self.rubber_line_ratio is not None:  # 1
            try: self.rubber_line_ratio.remove()  # 1
            except Exception: pass  # 1
            self.rubber_line_ratio = None  # 1
            self.canvas.draw_idle()  # 1
# 1
    def _clear_polygon_rubber(self):  # 1
        if self.polygon_rubber_line is not None:  # 1
            try: self.polygon_rubber_line.remove()  # 1
            except Exception: pass  # 1
            self.polygon_rubber_line = None  # 1
            self.canvas.draw_idle()  # 1
# 1
    def clear_selection(self, redraw: bool=True):  # 1
        # 1
        self.selected_idx.clear()  # 1
        self.anchor_idx = None  # 1
        self._clear_rubber()  # 1
        self.curr_chain = None  # 1
        self.curr_seg = None  # 1
        self.curr_labels = None  # 1
        self.curr_ratio = float('nan')  # 1
        # 1
        self.ratio_selected_idx.clear()  # 1
        self.ratio_anchor_idx = None  # 1
        self._clear_rubber_ratio()  # 1
        # 1
        self.polygon_current_idx.clear()  # 1
        self.polygons_idx.clear()  # 1
        self._clear_polygon_rubber()  # 1
        self._reset_polygon_history()  # 1
        self._last_analysis_mode = None  # 1
        # 1
        self.lst.delete(0, tk.END)  # 1
        self.list_index_map.clear()  # 1
        self.lst_header.config(text='Found words (Fibonacci subsegments)')  # 1
        self.lbl_ratio.config(text='Average L/S along chain: —')  # 1
        self.lbl_ratio_neigh.config(text='Average neighboring segment ratio: —')  # 1
        self.lbl_ratio_polygons.config(text='Average neighboring polygon linear ratio: —')  # 1
        self.txt_sl.delete('1.0', tk.END)  # 1
        self.txt_words.configure(state='normal');  # 1
        self.txt_words.delete('1.0', tk.END);  # 1
        self.txt_words.configure(state='disabled')  # 1
        if redraw: self.draw_base()  # 1
# 1
    def _capture_polygon_state(self) -> Tuple[List[int], List[List[int]]]:  # 1
        return (  # 1
            list(self.polygon_current_idx),  # 1
            [list(poly) for poly in self.polygons_idx],  # 1
        )  # 1
# 1
    def _reset_polygon_history(self) -> None:  # 1
        self._polygon_history = [self._capture_polygon_state()]  # 1
        self._polygon_redo.clear()  # 1
# 1
    def _record_polygon_state(self) -> None:  # 1
        state = self._capture_polygon_state()  # 1
        if self._polygon_history and state == self._polygon_history[-1]:  # 1
            return  # 1
        self._polygon_history.append(state)  # 1
        self._polygon_redo.clear()  # 1
# 1
    def _restore_polygon_state(self, state: Tuple[List[int], List[List[int]]]) -> None:  # 1
        current, finished = state  # 1
        self.polygon_current_idx = list(current)  # 1
        self.polygons_idx = [list(poly) for poly in finished]  # 1
        self._clear_polygon_rubber()  # 1
# 1
    def _on_polygon_undo(self, event):  # 1
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)):  # 1
            return  # 1
        if len(self._polygon_history) <= 1:  # 1
            return "break"  # 1
        state = self._polygon_history.pop()  # 1
        self._polygon_redo.append(state)  # 1
        prev_state = self._polygon_history[-1]  # 1
        self._restore_polygon_state(prev_state)  # 1
        self._refresh_canvas_after_polygon()  # 1
        self.status.config(text='Last polygon construction action undone.')  # 1
        return "break"  # 1
# 1
    def _on_polygon_redo(self, event):  # 1
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)):  # 1
            return  # 1
        if not self._polygon_redo:  # 1
            return "break"  # 1
        state = self._polygon_redo.pop()  # 1
        self._polygon_history.append(state)  # 1
        self._restore_polygon_state(state)  # 1
        self._refresh_canvas_after_polygon()  # 1
        self.status.config(text='Polygon construction action redone.')  # 1
        return "break"  # 1
# 1
    # 1
# 1
    def _collect_points_along_segment(self, i0: int, i1: int, max_dist: float) -> List[int]:  # 1
        """Indices of points within max_dist from segment p0->p1, ordered by projection."""  # 1
        p0 = self.points[i0][[1,0]]  # 1
        p1 = self.points[i1][[1,0]]  # 1
        v = p1 - p0  # 1
        vv = float(np.dot(v, v))  # 1
        if vv == 0: return [i0]  # 1
        idx = []  # 1
        for k, (y, x) in enumerate(self.points):  # 1
            w = np.array([x, y]) - p0  # 1
            t = float(np.dot(w, v) / vv)  # 1
            if 0.0 <= t <= 1.0:  # 1
                proj = p0 + t * v  # 1
                dist = float(np.hypot(x - proj[0], y - proj[1]))  # 1
                if dist <= max_dist:  # 1
                    idx.append((t, k))  # 1
        idx.sort(key=lambda z: z[0])  # 1
        chain = [k for t,k in idx]  # 1
        # 1
        if chain and chain[0] != i0:  # 1
            if i0 in chain: chain.remove(i0)  # 1
            chain.insert(0, i0)  # 1
        if chain and chain[-1] != i1:  # 1
            if i1 in chain: chain.remove(i1)  # 1
            chain.append(i1)  # 1
        # 1
        seen=set(); out=[]  # 1
        for k in chain:  # 1
            if k not in seen:  # 1
                out.append(k); seen.add(k)  # 1
        return out  # 1
# 1
    # 1
# 1
    def _on_sl_keypress(self, event):  # 1
        """Invert selected L/S with keys: 'i', 'sh', 'Sh'."""  # 1
        if event.char not in ('i', 'I', 'ш', 'Ш'):  # 1
            return  # 1
        try:  # 1
            start = self.txt_sl.index("sel.first")  # 1
            end   = self.txt_sl.index("sel.last")  # 1
        except tk.TclError:  # 1
            return "break"  # 1
        segment = self.txt_sl.get(start, end)  # 1
        flipped = ''.join('S' if ch == 'L' else ('L' if ch == 'S' else ch) for ch in segment)  # 1
        self.txt_sl.delete(start, end)  # 1
        self.txt_sl.insert(start, flipped)  # 1
        # 1
        self._recompute_words_from_manual_SL()  # 1
        return "break"  # 1
# 1
    def _set_sl_text(self, s: str):  # 1
        self.txt_sl.delete('1.0', tk.END)  # 1
        self.txt_sl.insert('1.0', s)  # 1
# 1
    def _get_sl_text_letters(self) -> List[str]:  # 1
        raw = self.txt_sl.get('1.0', tk.END)  # 1
        return [ch for ch in raw if ch in ('L','S')]  # 1
# 1
    # 1
# 1
    def run_analysis(self):  # 1
        if self.points is None or len(self.points) < 2:  # 1
            messagebox.showinfo('Analysis', 'Not enough points (need ≥ 2).'); return  # 1
        if len(self.selected_idx) < 2:  # 1
            messagebox.showinfo('Analysis', 'Select two points first in Chain mode.'); return  # 1
# 1
        chain = self.points[self.selected_idx].copy()  # 1
        seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)  # 1
# 1
        labels, Slen, Llen, sidx, lidx = cluster_lengths(seg)  # 1
        SL = ['S' if labels[i] == sidx else 'L' for i in range(len(seg))]  # 1
        ratio = (Llen / Slen) if (Slen and not math.isnan(Slen)) else float('nan')  # 1
# 1
        self.curr_chain = chain  # 1
        self.curr_seg = seg  # 1
        self.curr_labels = SL[:]  # 1
        self.curr_ratio = ratio  # 1
# 1
        self._set_sl_text(''.join(SL))  # 1
        self._recompute_words_and_redraw()  # 1
        self._last_analysis_mode = 'sl'  # 1
# 1
    def _recompute_words_from_manual_SL(self):  # 1
        if self.curr_chain is None or self.curr_seg is None:  # 1
            return  # 1
        SL = self._get_sl_text_letters()  # 1
        m = len(self.curr_seg)  # 1
        if len(SL) < m: SL = SL + ['S']*(m-len(SL))  # 1
        if len(SL) > m: SL = SL[:m]  # 1
        self.curr_labels = SL  # 1
        self._recompute_words_and_redraw()  # 1
# 1
    def _highlight_word(self, chain: np.ndarray, SL: List[str], i0: int, n: int):  # 1
        # 1
        # 1
        # 1
        for k in range(i0, i0 + n):  # 1
            y1, x1 = chain[k]; y2, x2 = chain[k + 1]  # 1
            self.ax.plot([x1, x2], [y1, y2], color='lime', lw=3.2)  # 1
            my, mx = (y1 + y2) / 2, (x1 + x2) / 2  # 1
            self.ax.text(mx, my, SL[k], color='red', fontsize=9, ha='center', va='center')  # 1
        yN, xN = chain[i0 + n]  # 1
        self.ax.text(xN, yN, str(n), color='white', fontsize=8, ha='right', va='bottom')  # 1
# 1
    def _recompute_words_and_redraw(self):  # 1
        chain = self.curr_chain  # 1
        SL = self.curr_labels  # 1
        ratio = self.curr_ratio  # 1
# 1
        # 1
        found: List[Tuple[int, int, str, int, int]] = []  # 1
        fibNs = [n for n in fib_list_upto(len(SL)) if n >= 3]  # 1
        for n in fibNs:  # 1
            fibs = fib_list_upto(n)  # 1
            k = len(fibs) - 1  # 1
            exp1 = (fibs[k - 1], fibs[k - 2]) if k >= 2 else (1, 0)  # 1
            for i in range(0, len(SL) - n + 1):  # 1
                sub = SL[i:i + n]  # 1
                Lc, Sc = sub.count('L'), sub.count('S')  # 1
                if (Lc, Sc) == exp1:  # 1
                    found.append((n, i, ''.join(sub), Lc, Sc))  # 1
# 1
        # 1
        self.draw_base()  # 1
        self._draw_selection(self.selected_idx)  # 1
        if found:  # 1
            found.sort(key=lambda z: (-z[0], z[1]))  # 1
            n, i0, word, Lc, Sc = found[0]  # 1
            self._highlight_word(chain, SL, i0, n)  # 1
            self.ax.text(0.01, 0.02,  # 1
                         f'Chain: L/S≈{ratio:.3f}  | Best word: n={n}, L={Lc}, S={Sc}',  # 1
                         transform=self.ax.transAxes, color='lime', fontsize=10,  # 1
                         ha='left', va='bottom')  # 1
        else:  # 1
            self.ax.text(0.01, 0.02, f'Chain: L/S≈{ratio:.3f}. No matches (n≥3) found.',  # 1
                         transform=self.ax.transAxes, color='orange', fontsize=10,  # 1
                         ha='left', va='bottom')  # 1
        self.canvas.draw_idle()  # 1
# 1
        # 1
        self.lst.delete(0, tk.END)  # 1
        self.list_index_map.clear()  # 1
        self.lst_header.config(text='Found words (Fibonacci subsegments)')  # 1
        groups: Dict[int, List[Tuple[int,int,str,int,int]]] = {}  # 1
        for entry in found:  # 1
            groups.setdefault(entry[0], []).append(entry)  # 1
        row = 0  # 1
        if groups:  # 1
            for n in sorted(groups.keys()):  # 1
                self.lst.insert(tk.END, f'— n={n} —'); row += 1  # 1
                for (n_, i0, word, Lc, Sc) in groups[n]:  # 1
                    self.lst.insert(tk.END, f'  i={i0}  word={word}  L={Lc} S={Sc}')  # 1
                    self.list_index_map[row] = ('sl', i0, n)  # 1
                    row += 1  # 1
                self.lst.insert(tk.END, ''); row += 1  # 1
        else:  # 1
            self.lst.insert(tk.END, 'No matches (n≥3)')  # 1
# 1
        # 1
        if math.isfinite(ratio): self.lbl_ratio.config(text=f'Average L/S along chain: {ratio:.3f}')  # 1
        else:                    self.lbl_ratio.config(text='Average L/S along chain: —')  # 1
        self.lbl_ratio_neigh.config(text='Average neighboring segment ratio: —')  # 1
# 1
        # 1
        self.txt_words.configure(state='normal')  # 1
        self.txt_words.delete('1.0', tk.END)  # 1
        max_len_ref = max(groups.keys(), default=min(len(SL), 34))  # 1
        for w in gen_fibonacci_words(max_len=max_len_ref, start='L'):  # 1
            self.txt_words.insert(tk.END, f'len={len(w)} → {w}\n')  # 1
        self.txt_words.configure(state='disabled')  # 1
# 1
        self.status.config(text=f'Selected points (Chain mode): {len(chain)}. Segments: {len(SL)}. '  # 1
                                f'Words (n≥3): {sum(len(v) for v in groups.values())}.')  # 1
# 1
    # 1
# 1
    def _highlight_ratio_pair(self, chain: np.ndarray, seg_a: int, seg_b: int):  # 1
        """Highlight two neighboring segments by their indices (0..M-1)."""  # 1
        M = len(chain) - 1  # 1
        if not (0 <= seg_a < M and 0 <= seg_b < M):  # 1
            return  # 1
        for k in (seg_a, seg_b):  # 1
            y1, x1 = chain[k]; y2, x2 = chain[k + 1]  # 1
            self.ax.plot([x1, x2], [y1, y2], color='lime', lw=3.2)  # 1
            my, mx = (y1 + y2) / 2, (x1 + x2) / 2  # 1
            self.ax.text(mx, my, f'{k+2}-{k+1}', color='red', fontsize=9, ha='center', va='center')  # 1
# 1
    def run_ratio_analysis(self):  # 1
        """Analysis for the chain selected in Ratio mode: segment names 'n-(n-1)', neighboring ratios, mean value."""  # 1
        if self.points is None or len(self.points) < 2:  # 1
            messagebox.showinfo('Analysis', 'Not enough points.'); return  # 1
        if len(self.ratio_selected_idx) < 3:  # 1
            messagebox.showinfo('Analysis', 'Need ≥ 3 points in Ratio mode.'); return  # 1
# 1
        chain = self.points[self.ratio_selected_idx].copy()  # 1
        seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)  # 1
# 1
        # 1
        self.draw_base()  # 1
        self._draw_selection(self.ratio_selected_idx)  # 1
        for i in range(len(chain)-1):  # 1
            y1,x1 = chain[i]; y2,x2 = chain[i+1]  # 1
            my, mx = (y1+y2)/2, (x1+x2)/2  # 1
            label = f"{i+2}-{i+1}"  # 1
            self.ax.text(mx, my, label, color='yellow', fontsize=9, ha='center', va='center')  # 1
        self.canvas.draw_idle()  # 1
# 1
        # 1
        ratios = []  # 1
        for i in range(1, len(seg)):  # 1
            if seg[i-1] > 0:  # 1
                ratios.append(seg[i]/seg[i-1])  # 1
            else:  # 1
                ratios.append(float('nan'))  # 1
# 1
        # 1
        self.lst.delete(0, tk.END)  # 1
        self.list_index_map.clear()  # 1
        self.lst_header.config(text='Neighboring segment ratios (Ratio mode)')  # 1
        row = 0  # 1
        if len(ratios) == 0:  # 1
            self.lst.insert(tk.END, 'Not enough segments for ratios.')  # 1
        else:  # 1
            for i, r in enumerate(ratios, start=2):  # 1
                self.lst.insert(tk.END, f'  ({i + 1}-{i}) / ({i}-{i - 1})  ≈  {r:.6g}')  # 1
                k = i - 1  # 1
                self.list_index_map[row] = ('ratio', k)  # 1
                row += 1  # 1
# 1
        # 1
        finite = [r for r in ratios if math.isfinite(r)]  # 1
        mean_ratio = float(np.mean(finite)) if finite else float('nan')  # 1
        if math.isfinite(mean_ratio):  # 1
            self.lbl_ratio_neigh.config(text=f'Average neighboring segment ratio: {mean_ratio:.6g}')  # 1
        else:  # 1
            self.lbl_ratio_neigh.config(text='Average neighboring segment ratio: —')  # 1
# 1
        # 1
        self.status.config(text=f'Selected points (Ratio mode): {len(chain)}. Segments: {len(seg)}. Ratios: {len(ratios)}.')  # 1
        self._last_analysis_mode = 'ratio'  # 1
# 1
    # 1
# 1
    def _draw_polygons(self):  # 1
        """Draw finished polygons and the current one."""  # 1
        if self.points is None or (not self.polygons_idx and not self.polygon_current_idx):  # 1
            return  # 1
# 1
        # 1
        for num, idxs in enumerate(self.polygons_idx, start=1):  # 1
            if len(idxs) < 3:  # 1
                continue  # 1
            pts = self.points[idxs]  # 1
            xs = pts[:, 1]  # 1
            ys = pts[:, 0]  # 1
            self.ax.fill(xs, ys, facecolor='deepskyblue', alpha=0.25,  # 1
                         edgecolor='blue', linewidth=1.4, zorder=1.5)  # 1
            cx = float(np.mean(xs))  # 1
            cy = float(np.mean(ys))  # 1
            self.ax.text(cx, cy, f'P{num}', color='navy', fontsize=9,  # 1
                         ha='center', va='center', zorder=1.6)  # 1
# 1
        # 1
        if self.polygon_current_idx:  # 1
            pts_cur = self.points[self.polygon_current_idx]  # 1
            xs = pts_cur[:, 1]  # 1
            ys = pts_cur[:, 0]  # 1
            self.ax.plot(xs, ys, color='orange', lw=2.2, zorder=3.1)  # 1
            self.ax.scatter(xs, ys, s=46, c='orange', edgecolors='k', linewidths=0.6, zorder=3.2)  # 1
            first_x, first_y = xs[0], ys[0]  # 1
            self.ax.scatter([first_x], [first_y], s=70, facecolors='none', edgecolors='orange',  # 1
                            linewidths=1.5, zorder=3.3)  # 1
            if len(xs) >= 2:  # 1
                self.ax.plot([xs[-1], first_x], [ys[-1], first_y], color='orange', lw=1.2, ls=':', zorder=3.0)  # 1
            for idx, (xv, yv) in enumerate(zip(xs, ys), start=1):  # 1
                self.ax.text(xv, yv, str(idx), color='orange', fontsize=8,  # 1
                             ha='right', va='bottom', zorder=3.4)  # 1
# 1
    def _handle_polygon_click(self, point_idx: int):  # 1
        """Handle a left-click to build a polygon while Polygon mode is active."""  # 1
        if point_idx < 0 or point_idx >= len(self.points):  # 1
            return  # 1
# 1
        if not self.polygon_current_idx:  # 1
            self.polygon_current_idx.append(point_idx)  # 1
            self._refresh_canvas_after_polygon()  # 1
            self.status.config(text=f'Polygon construction: first vertex selected (#{point_idx + 1}).')  # 1
            self._record_polygon_state()  # 1
            return  # 1
# 1
        first_idx = self.polygon_current_idx[0]  # 1
        if point_idx == first_idx:  # 1
            if len(self.polygon_current_idx) < 3:  # 1
                self.status.config(text='Polygon requires ≥ 3 unique points.')  # 1
                return  # 1
            # 1
            self.polygons_idx.append(self.polygon_current_idx.copy())  # 1
            poly_num = len(self.polygons_idx)  # 1
            vertex_count = len(self.polygon_current_idx)  # 1
            self.polygon_current_idx.clear()  # 1
            self._clear_polygon_rubber()  # 1
            self._refresh_canvas_after_polygon()  # 1
            self.status.config(text=f'Polygon #{poly_num} closed. Vertices: {vertex_count}.')  # 1
            self._record_polygon_state()  # 1
            return  # 1
# 1
        if point_idx in self.polygon_current_idx:  # 1
            self.status.config(text='Vertex already added. Choose another point or close the polygon.')  # 1
            return  # 1
# 1
        self.polygon_current_idx.append(point_idx)  # 1
        self._refresh_canvas_after_polygon()  # 1
        self.status.config(text=f'Polygon construction: total vertices {len(self.polygon_current_idx)}.')  # 1
        self._record_polygon_state()  # 1
# 1
    def _refresh_canvas_after_polygon(self):  # 1
        """Redraw the image, taking current modes and polygons into account."""  # 1
        if self._last_analysis_mode == 'sl' and self.curr_chain is not None:  # 1
            self._recompute_words_and_redraw()  # 1
        elif self._last_analysis_mode == 'ratio' and len(self.ratio_selected_idx) >= 3:  # 1
            self.run_ratio_analysis()  # 1
        else:  # 1
            self.draw_base()  # 1
        self.polygon_rubber_line = None  # 1
# 1
    def _polygon_area(self, idxs: List[int]) -> float:  # 1
        if self.points is None or len(idxs) < 3:  # 1
            return 0.0  # 1
        pts = self.points[idxs]  # 1
        xs = pts[:, 1]  # 1
        ys = pts[:, 0]  # 1
        shifted_x = np.roll(xs, -1)  # 1
        shifted_y = np.roll(ys, -1)  # 1
        area = 0.5 * abs(float(np.dot(xs, shifted_y) - np.dot(ys, shifted_x)))  # 1
        return area  # 1
# 1
    def _on_enter_key(self, event):  # 1
        """Compute the areas of the selected polygons and their ratios when Enter is pressed."""  # 1
        # 1
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)):  # 1
            return  # 1
# 1
        if self.polygon_current_idx:  # 1
            messagebox.showinfo('Polygons', 'Finish the current polygon (return to the first point).')  # 1
            return  # 1
        if len(self.polygons_idx) < 2:  # 1
            messagebox.showinfo('Polygons', 'Need at least two completed polygons.')  # 1
            return  # 1
# 1
        areas = [self._polygon_area(poly) for poly in self.polygons_idx]  # 1
        self.lst.delete(0, tk.END)  # 1
        self.list_index_map.clear()  # 1
        self.lst_header.config(text='Polygons (Polygon mode) — areas and ratios')  # 1
# 1
        for i, area in enumerate(areas, start=1):  # 1
            self.lst.insert(tk.END, f'Polygon area {i}: {area:.6g}')  # 1
# 1
        self.lst.insert(tk.END, '')  # 1
        lines_added = False  # 1
        polygon_linear_ratios = []  # 1
        for idx in range(len(areas), 1, -1):  # 1
            prev_area = areas[idx - 2]  # 1
            curr_area = areas[idx - 1]  # 1
            if prev_area == 0:  # 1
                linear_text = 'undefined (previous area = 0)'  # 1
            else:  # 1
                size_ratio = curr_area / prev_area  # 1
                ratio_text = f'{size_ratio:.6g}'  # 1
                if size_ratio > 0:  # 1
                    linear_ratio = math.sqrt(size_ratio)  # 1
                    polygon_linear_ratios.append(linear_ratio)  # 1
                    linear_text = f'{linear_ratio:.6g}'  # 1
                else:  # 1
                    linear_text = 'undefined (ratio ≤ 0)'  # 1
            self.lst.insert(tk.END, f'Linear size ratio {idx} and {idx - 1}: {linear_text}')  # 1
            lines_added = True  # 1
        # 1
        if not lines_added:  # 1
            self.lst.insert(tk.END, 'Not enough polygons for ratios.')  # 1
        # 1
        finite_linear = [r for r in polygon_linear_ratios if math.isfinite(r)]  # 1
        if finite_linear:  # 1
            mean_linear = float(np.mean(finite_linear))  # 1
            self.lbl_ratio_polygons.config(text=f'Average neighboring polygon linear ratio: {mean_linear:.6g}')  # 1
        else:  # 1
            self.lbl_ratio_polygons.config(text='Average neighboring polygon linear ratio: —')  # 1
        # 1
        self.status.config(text=f'Areas computed: {len(areas)}. See the list on the right.')  # 1
# 1
class App(tk.Tk):  # 1
    """Standalone wrapper compatible with the previous CLI."""  # 1
# 1
    def __init__(self):  # 1
        super().__init__()  # 1
        self.title('fibonachi_analysis')  # 1
        self.geometry('1520x980')  # 1
        self.resizable(True, True)  # 1
        frame = FibonacciAnalysisFrame(self)  # 1
        frame.pack(fill=tk.BOTH, expand=True)  # 1
        self.frame = frame  # 1
# 1
# 1
# 1
if __name__ == '__main__':  # 1
    app = App()  # 1
    app.mainloop()  # 1