# 4
# 2
# 3
# 5
# 5
# 3
# 4
# 5
# 4
# 3
# 5
# 4
# 2
# 1
# 2
# 5
# 1
# 2
from __future__ import annotations  # 2
# 5
import sys, json, math  # 3
from pathlib import Path  # 1
from typing import Optional, Tuple, List, Dict  # 5
# 3
import numpy as np  # 1
from PIL import Image  # 2
import tkinter as tk  # 2
from tkinter import filedialog, messagebox  # 2
# 1
import matplotlib  # 1
matplotlib.use('TkAgg')  # 5
import matplotlib.pyplot as plt  # 1
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # 2
from matplotlib.patches import Circle  # 4
# 2
from preproc import PreprocSettings, load_grayscale_with_preproc  # 36
if not hasattr(tk, "Notebook") and hasattr(tk, "Notebook"):  # 41
    tk.Notebook = ttk.Notebook  # type: ignore[attr-defined]

class _Tooltip:  # 0
    def __init__(self, widget: tk.Widget, text: str, *, delay: int = 400):  # 0
        self.widget = widget  # 0
        self.text = text  # 0
        self.delay = max(0, int(delay))  # 0
        self._after_id: Optional[str] = None  # 0
        self._tip_window: Optional[tk.Toplevel] = None  # 0
        self._last_pointer: Optional[tuple[int, int]] = None  # 0
        widget.bind("<Enter>", self._on_enter, add="+")  # 0
        widget.bind("<Leave>", self._on_leave, add="+")  # 0
        widget.bind("<Motion>", self._on_motion, add="+")  # 0

    def _on_enter(self, event):  # 0
        self._last_pointer = (event.x_root, event.y_root)  # 0
        self._schedule()  # 0

    def _on_leave(self, _event):  # 0
        self._cancel()  # 0
        self._hide()  # 0

    def _on_motion(self, event):  # 0
        self._last_pointer = (event.x_root, event.y_root)  # 0
        self._position()  # 0

    def _schedule(self):  # 0
        self._cancel()  # 0
        self._after_id = self.widget.after(self.delay, self._show)  # 0

    def _cancel(self):  # 0
        if self._after_id is not None:  # 0
            self.widget.after_cancel(self._after_id)  # 0
            self._after_id = None  # 0

    def _show(self):  # 0
        if self._tip_window is not None or not self.text:  # 0
            return  # 0
        tip = tk.Toplevel(self.widget)  # 0
        tip.wm_overrideredirect(True)  # 0
        tip.wm_attributes("-topmost", True)  # 0
        label = tk.Label(  # 0
            tip,  # 0
            text=self.text,  # 0
            justify="left",  # 0
            background="#ffffe0",  # 0
            relief="solid",  # 0
            borderwidth=1,  # 0
            wraplength=360,  # 0
        )  # 0
        label.pack(ipadx=8, ipady=4)  # 0
        self._tip_window = tip  # 0
        self._position()  # 0

    def _hide(self):  # 0
        if self._tip_window is not None:  # 0
            self._tip_window.destroy()  # 0
            self._tip_window = None  # 0

    def _position(self):  # 0
        if self._tip_window is None:  # 0
            return  # 0
        tip = self._tip_window  # 0
        tip.update_idletasks()  # 0
        width = tip.winfo_reqwidth()  # 0
        height = tip.winfo_reqheight()  # 0
        if self._last_pointer is not None:  # 0
            x, y = self._last_pointer  # 0
        else:  # 0
            x = self.widget.winfo_rootx() + self.widget.winfo_width()  # 0
            y = self.widget.winfo_rooty() + self.widget.winfo_height()  # 0
        x += 12  # 0
        y += 10  # 0
        root = self.widget.winfo_toplevel()  # 0
        root.update_idletasks()  # 0
        left = root.winfo_rootx()  # 0
        top = root.winfo_rooty()  # 0
        right = left + root.winfo_width()  # 0
        bottom = top + root.winfo_height()  # 0
        if x + width > right - 4:  # 0
            x = right - width - 4  # 0
        if y + height > bottom - 4:  # 0
            y = bottom - height - 4  # 0
        x = max(x, left + 4)  # 0
        y = max(y, top + 4)  # 0
        tip.wm_geometry(f"+{int(x)}+{int(y)}")  # 0
# 5


class HoverTooltip:  # 1
    """Display a tooltip when the mouse hovers over a widget."""  # 3

    def __init__(self, widget: tk.Widget, text: str, delay: int = 400):  # 2
        self.widget = widget  # 1
        self.text = text  # 1
        self.delay = delay  # 4
        self._after_id: Optional[str] = None  # 1
        self._window: Optional[tk.Toplevel] = None  # 4
        widget.bind("<Enter>", self._schedule)  # 4
        widget.bind("<Leave>", self._hide)  # 4
        widget.bind("<ButtonPress>", self._hide)  # 5

    def _schedule(self, _event=None):  # 2
        self._cancel()  # 1
        self._after_id = self.widget.after(self.delay, self._show)  # 1

    def _cancel(self):  # 5
        if self._after_id is not None:  # 1
            try:  # 5
                self.widget.after_cancel(self._after_id)  # 5
            except Exception:  # 2
                pass  # 4
            self._after_id = None  # 3

    def _show(self):  # 1
        self._after_id = None  # 1
        if self._window is not None:  # 4
            return  # 2
        x = self.widget.winfo_pointerx() + 16  # 4
        y = self.widget.winfo_pointery() + 12  # 1
        self._window = tw = tk.Toplevel(self.widget)  # 4
        tw.wm_overrideredirect(True)  # 3
        tw.wm_geometry(f"+{x}+{y}")  # 2
        label = tk.Label(tw, text=self.text, background="#2f2f2f", foreground="white",  # 4
                         relief="solid", borderwidth=1, padx=6, pady=3, justify=tk.LEFT)  # 4
        label.pack()  # 1

    def _hide(self, _event=None):  # 2
        self._cancel()  # 1
        if self._window is not None:  # 1
            try:  # 2
                self._window.destroy()  # 1
            except Exception:  # 4
                pass  # 3
            self._window = None  # 4


# 1
# 1
def _parse_cli(argv=None):  # 5
    import argparse  # 3
    p = argparse.ArgumentParser(description="fibonachi_analysis — load input data")  # 5
    p.add_argument("--payload", type=str, default=None, help="Path to fibo_input.json")  # 5
    p.add_argument("--image", type=str, default=None, help="Path to the image (fallback)")  # 5
    p.add_argument("--points", type=str, default=None, help="Path to JSON with points (fallback)")  # 2
    return p.parse_args(argv)  # 3
# 2
# 3
def _candidate_dirs(extra_image: Optional[Path]) -> List[Path]:  # 2
    cands: List[Path] = []  # 1
    try:  # 3
        cands.append(Path.cwd())  # 4
    except Exception:  # 2
        pass  # 1
    if getattr(sys, "frozen", False):  # 1
        try:  # 4
            cands.append(Path(sys.executable).resolve().parent)  # 1
        except Exception:  # 5
            pass  # 3
        try:  # 2
            cands.append(Path(getattr(sys, "_MEIPASS")))  # 1
        except Exception:  # 2
            pass  # 2
    else:  # 3
        try:  # 5
            cands.append(Path(__file__).resolve().parent)  # 2
        except Exception:  # 4
            pass  # 4
    if extra_image:  # 3
        try:  # 5
            cands.append(extra_image.resolve().parent)  # 5
        except Exception:  # 3
            pass  # 5
    uniq, seen = [], set()  # 2
    for d in cands:  # 4
        rp = str(d.resolve())  # 2
        if rp not in seen:  # 2
            uniq.append(d); seen.add(rp)  # 4
    return uniq  # 3
# 4
# 3
def _autofind_json(extra_image: Optional[Path]) -> Optional[Path]:  # 2
    pats = ["fibo_input.json", "*fibo*input*.json", "*.fibo.json", "*.json"]  # 4
    for base in _candidate_dirs(extra_image):  # 2
        for pat in pats:  # 1
            try:  # 1
                for p in base.glob(pat):  # 1
                    name = p.name.lower()  # 5
                    if "fibo" in name and "input" in name:  # 3
                        return p.resolve()  # 4
                    if pat == "*.json":  # 3
                        try:  # 2
                            obj = json.loads(p.read_text(encoding="utf-8"))  # 3
                            if isinstance(obj, dict) and "image" in obj and "points" in obj:  # 1
                                return p.resolve()  # 2
                        except Exception:  # 4
                            pass  # 3
            except Exception:  # 5
                continue  # 1
    return None  # 5
# 1
# 5
def find_default_json(base_dir: Path) -> Optional[Path]:  # 1
    cand = base_dir / "fibo_input.json"  # 3
    if cand.exists():  # 4
        try:  # 1
            d = json.loads(cand.read_text(encoding="utf-8"))  # 2
            if "image" in d and "points" in d:  # 4
                return cand.resolve()  # 4
        except Exception:  # 5
            pass  # 1
    for p in base_dir.glob("*.json"):  # 2
        try:  # 1
            d = json.loads(p.read_text(encoding="utf-8"))  # 3
            if "image" in d and "points" in d:  # 2
                return p.resolve()  # 3
        except Exception:  # 3
            pass  # 1
    return _autofind_json(None)  # 2
# 4
# 3
def load_input(json_path: Path):  # 4
    d = json.loads(json_path.read_text(encoding='utf-8'))  # 3
    img = Path(d['image']) if d.get('image') else None  # 4
    if not img:  # 1
        raise RuntimeError("JSON does not contain the key 'image'.")  # 3
    pts = np.array([[float(p['y']), float(p['x'])] for p in d.get('points', [])], float)  # 4
    center = None; dead = 0.0; srch = 0.0  # 5
    if isinstance(d.get('centers'), dict):  # 2
        c = d['centers'].get('overlay') or d['centers'].get('geometric')  # 3
        if c and 'x' in c and 'y' in c:  # 2
            center = (float(c['y']), float(c['x']))  # 1
    if isinstance(d.get('radii'), dict):  # 4
        if d['radii'].get('dead') is not None:  dead = float(d['radii']['dead'])  # 4
        if d['radii'].get('search') is not None: srch = float(d['radii']['search'])  # 4
    fallback_mode = d.get('preproc_mode')  # 1
    if not isinstance(fallback_mode, str):  # 3
        fallback_mode = None  # 3
    preproc = PreprocSettings.from_json(d.get('preproc'), fallback_mode=fallback_mode)  # 4
    return img, pts, center, dead, srch, preproc  # 3
# 2
# 2
# 2
def cluster_lengths(lengths: np.ndarray):  # 1
    """k=2 clustering of lengths into S/L; returns (labels, Slen, Llen, sidx, lidx)."""  # 1
    if lengths.size == 0:  # 2
        return np.array([], dtype=int), float("nan"), float("nan"), 0, 1  # 5
    c0, c1 = float(lengths.min()), float(lengths.max())  # 1
    if c0 == c1:  # 5
        lab = np.zeros(len(lengths), dtype=int)  # 2
        return lab, c0, float("nan"), 0, 1  # 2
    lab = np.zeros(len(lengths), dtype=int)  # 2
    for _ in range(60):  # 2
        d0 = np.abs(lengths - c0)  # 4
        d1 = np.abs(lengths - c1)  # 2
        lab = (d1 < d0).astype(int)  # 4
        nc0 = float(lengths[lab == 0].mean()) if np.any(lab == 0) else c0  # 3
        nc1 = float(lengths[lab == 1].mean()) if np.any(lab == 1) else c1  # 3
        if abs(nc0 - c0) < 1e-6 and abs(nc1 - c1) < 1e-6:  # 3
            c0, c1 = nc0, nc1; break  # 1
        c0, c1 = nc0, nc1  # 5
    # 4
    m0 = float(lengths[lab == 0].mean()) if np.any(lab == 0) else float("nan")  # 1
    m1 = float(lengths[lab == 1].mean()) if np.any(lab == 1) else float("nan")  # 5
    if (not math.isnan(m0)) and (not math.isnan(m1)) and m0 > m1:  # 4
        lab = 1 - lab  # 4
        m0, m1 = m1, m0  # 1
    return lab, m0, m1, 0, 1  # 1
# 3
def fib_list_upto(n: int) -> List[int]:  # 4
    """Fibonacci numbers up to n (inclusive), starting with 1,1,2,3,..."""  # 2
    if n <= 0: return []  # 1
    seq = [1, 1]  # 1
    while seq[-1] < n:  # 3
        seq.append(seq[-1] + seq[-2])  # 4
    return [k for k in seq if k <= n]  # 1
# 3
def gen_fibonacci_words(max_len: int, start: str = "L") -> List[str]:  # 4
    """Generate prefixes of the "fibo-word": L->LS, S->L, up to max_len."""  # 3
    if max_len <= 0: return []  # 2
    words = ["L" if start.upper() == "L" else "S"]  # 1
    while len(words[-1]) <= max_len:  # 1
        nxt = "".join(("LS" if ch == "L" else "L") for ch in words[-1])  # 5
        if len(nxt) > max_len: break  # 3
        words.append(nxt)  # 2
    return words  # 4
# 5
# 5
# 4
class FibonacciAnalysisFrame(tk.Frame):  # 1
    def __init__(self, master: tk.Misc, controller=None, auto_load: bool = True):  # 1
        super().__init__(master)  # 4
        self.controller = controller  # 1
# 1
        # 3
        self.img_path: Optional[Path] = None  # 2
        self.points: Optional[np.ndarray] = None      # 1
        self.center: Optional[Tuple[float, float]] = None  # 4
        self.dead: float = 0.0  # 1
        self.srch: float = 0.0  # 2
        self.preproc: PreprocSettings = PreprocSettings(mode="raw")  # 2
        self.img_arr: Optional[np.ndarray] = None  # 3
# 2
        # 4
        self.pick_tol = 10.0  # 1
        self.max_dist_line = 20.0  # default selection thickness in pixels  # 1

        self.mode_buttons: Dict[str, tk.Button] = {}  # 1
        self._mode_tooltips: List[HoverTooltip] = []  # 1
        self.analysis_mode: str = 'sl'  # 1
        self._last_analysis_mode: Optional[str] = None  # 1

        self.selected_idx: List[int] = []  # 1
        self.ratio_selected_idx: List[int] = []  # 1
        self.polygon_current_idx: List[int] = []  # 1
        self.polygons_idx: List[List[int]] = []  # 1
        self._polygon_history: List[Tuple[List[int], List[List[int]]]] = []  # 1
        self._polygon_redo: List[Tuple[List[int], List[List[int]]]] = []  # 1

        self.list_index_map: Dict[int, Tuple] = {}  # 1
        self.curr_chain = None  # 1
        self.curr_seg = None  # 1
        self.curr_labels = None  # 1
        self.curr_ratio = float('nan')  # 1

        self.anchor_idx: Optional[int] = None  # 1
        self.ratio_anchor_idx: Optional[int] = None  # 1
        self.rubber_line = None  # 1
        self.rubber_line_ratio = None  # 1
        self.polygon_rubber_line = None  # 1

        self.columnconfigure(0, weight=1)  # 1
        self.rowconfigure(0, weight=1)  # 1

        container = tk.Frame(self)  # 1
        container.grid(row=0, column=0, sticky="nsew")  # 1
        container.columnconfigure(0, weight=1)  # 1
        container.columnconfigure(1, weight=0)  # 1

        left = tk.Frame(container)  # 1
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=12)  # 1
        left.rowconfigure(0, weight=1)  # 1
        left.columnconfigure(0, weight=1)  # 1

        right = tk.Frame(container, width=470)  # 1
        right.grid(row=0, column=1, sticky="ns", pady=12)  # 1
        right.columnconfigure(0, weight=1)  # 1

        controls = tk.Frame(right)  # 1
        controls.grid(row=0, column=0, sticky="ew", padx=6, pady=(0, 10))  # 1
        controls.columnconfigure(0, weight=1)  # 1
        controls.columnconfigure(1, weight=1)  # 1

        tk.Button(controls, text='Open JSON…', command=self.open_json).grid(row=0, column=0, sticky='ew', padx=4, pady=2)  # 1
        tk.Button(controls, text='Save PNG', command=self.save_png).grid(row=0, column=1, sticky='ew', padx=4, pady=2)  # 1
        tk.Button(controls, text='Start analysis (LMB chain)', command=self.run_analysis).grid(row=1, column=0, sticky='ew', padx=4, pady=2)  # 1
        tk.Button(controls, text='Clear selection', command=self.clear_selection).grid(row=1, column=1, sticky='ew', padx=4, pady=2)  # 1

        tk.Label(controls, text='Selection thickness (px):').grid(row=2, column=0, sticky='w', padx=4, pady=(8, 2))  # 1
        self.entBand = tk.Spinbox(controls, from_=2, to=100, width=6, command=self._onBand)  # 1
        self.entBand.delete(0, 'end'); self.entBand.insert(0, str(int(self.max_dist_line)))  # 1
        self.entBand.grid(row=2, column=1, sticky='ew', padx=4, pady=(8, 2))  # 1

        mode_switch = tk.Frame(controls)  # 1
        mode_switch.grid(row=3, column=0, columnspan=2, sticky='ew', padx=4, pady=(8, 2))  # 1
        for col in range(3):  # 1
            mode_switch.columnconfigure(col, weight=1)  # 1

        self.mode_buttons['sl'] = tk.Button(mode_switch, text='🔗 Chain', command=lambda: self._set_analysis_mode('sl'))  # 1
        self.mode_buttons['sl'].grid(row=0, column=0, sticky='ew', padx=(0, 2))  # 1
        self._mode_tooltips.append(HoverTooltip(self.mode_buttons['sl'], 'Chain analysis: use Left Click while this mode is active.'))  # 1

        self.mode_buttons['ratio'] = tk.Button(mode_switch, text='📊 Ratio', command=lambda: self._set_analysis_mode('ratio'))  # 1
        self.mode_buttons['ratio'].grid(row=0, column=1, sticky='ew', padx=2)  # 1
        self._mode_tooltips.append(HoverTooltip(self.mode_buttons['ratio'], 'Neighbor ratios: use Left Click while this mode is active.'))  # 1

        self.mode_buttons['polygon'] = tk.Button(mode_switch, text='🔺 Polygon', command=lambda: self._set_analysis_mode('polygon'))  # 1
        self.mode_buttons['polygon'].grid(row=0, column=2, sticky='ew', padx=(2, 0))  # 1
        self._mode_tooltips.append(HoverTooltip(self.mode_buttons['polygon'], 'Polygon analysis: add vertices with Left Click, close by clicking the first point.'))  # 1

        self.status = tk.Label(right, text='', anchor='w')  # 1
        self.status.grid(row=1, column=0, sticky='ew', padx=6, pady=(0, 10))  # 1
        self._set_analysis_mode('sl')  # ensure default mode and status hint  # 1
        self.lst_header = tk.Label(right, text='Found words (Fibonacci subsegments)')  # 1
        self.lst_header.grid(row=2, column=0, sticky='w', padx=6, pady=(0, 2))  # 1

        self.results_notebook = tk.Notebook(right)  # 1
        self.results_notebook.grid(row=3, column=0, sticky='nsew', padx=6, pady=(0, 8))  # 1
        right.rowconfigure(3, weight=1)  # 1

        tab_subsegments = ttk.Frame(self.results_notebook)  # 1
        tab_subsegments.grid_columnconfigure(0, weight=1)  # 1
        tab_subsegments.grid_rowconfigure(0, weight=1)  # 1

        tab_prefixes = ttk.Frame(self.results_notebook)  # 1
        tab_prefixes.grid_columnconfigure(0, weight=1)  # 1
        tab_prefixes.grid_rowconfigure(1, weight=1)  # 1

        self.results_notebook.add(tab_subsegments, text='Subsegments')  # 1
        self.results_notebook.add(tab_prefixes, text='Fib prefixes')  # 1

        self.lst = tk.Listbox(tab_subsegments, width=66, height=22)  # 1
        self.lst.grid(row=0, column=0, sticky='nsew')  # 1
        self.lst.bind('<<ListboxSelect>>', self._on_list_select)  # 1

        self.lbl_ratio = tk.Label(tab_subsegments, text='Average L/S along chain: —')  # 1
        self.lbl_ratio.grid(row=1, column=0, sticky='w', pady=(6, 4))  # 1
        self.lbl_ratio_neigh = tk.Label(tab_subsegments, text='Average neighboring segment ratio: —')  # 1
        self.lbl_ratio_neigh.grid(row=2, column=0, sticky='w', pady=(2, 8))  # 1

        tk.Label(tab_subsegments, text='S/L sequence (full):').grid(row=3, column=0, sticky='w', pady=(4, 2))  # 1
        self.txt_sl = tk.Text(tab_subsegments, height=6, wrap='word')  # 1
        self.txt_sl.grid(row=4, column=0, sticky='ew', pady=(0, 4))  # 1
        self.txt_sl.bind('<KeyPress>', self._on_sl_keypress)  # 1

        prefixes_header = tk.Label(tab_prefixes, text='Prefixes of "fib-words" (L→LS, S→L)')  # 1
        prefixes_header.grid(row=0, column=0, sticky='w', pady=(0, 4))  # 1

        prefixes_frame = tk.Frame(tab_prefixes)  # 1
        prefixes_frame.grid(row=1, column=0, sticky='nsew')  # 1
        prefixes_frame.grid_columnconfigure(0, weight=1)  # 1
        prefixes_frame.grid_rowconfigure(0, weight=1)  # 1

        self.txt_words = tk.Text(prefixes_frame, height=10, state='disabled')  # 1
        self.txt_words.grid(row=0, column=0, sticky='nsew')  # 1
        scroll_words = tk.Scrollbar(prefixes_frame, orient='vertical', command=self.txt_words.yview)  # 1
        scroll_words.grid(row=0, column=1, sticky='ns')  # 1
        self.txt_words.configure(yscrollcommand=scroll_words.set)  # 1
# 5
        self.fig = plt.Figure(figsize=(9.6, 6.6)); self.ax = self.fig.add_subplot(111)  # 4
        self.ax.axis('off')  # 5
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)  # 3
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')  # 5
        # 4
        self.canvas.mpl_connect('button_press_event', self._on_click)  # 3
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)  # 2
        self.bind('<Escape>', lambda e: self.clear_selection())  # 2
        self.bind_all('<Return>', self._on_enter_key)  # 2
        self.bind_all('<z>', self._on_polygon_undo)  # 3
        self.bind_all('<z>', self._on_polygon_undo)  # 1
        self.bind_all('<y>', self._on_polygon_redo)  # 3
        self.bind_all('<y>', self._on_polygon_redo)  # 5
# 1
        self._reset_polygon_history()  # 5
# 3
        # 4
        if auto_load:  # 2
            base = Path(getattr(sys, '_MEIPASS', Path(__file__).parent)) if getattr(sys, 'frozen', False) else Path(  # 4
                __file__).parent  # 1
            auto = find_default_json(base)  # 4
            if auto:  # 4
                try:  # 4
                    self.load_json(auto)  # 2
                    self.status.config(text=f'Loaded: {auto.name}')  # 2
                except Exception as e:  # 2
                    messagebox.showerror('Load error', str(e))  # 5
            else:  # 3
                self.status.config(text='JSON not found. Select a file manually.')  # 3
        else:  # 2
            self.status.config(text='JSON not loaded. Use "Open JSON…".')  # 4
    # 4
# 4
    def _onBand(self):  # 4
        try:  # 1
            self.max_dist_line = max(2.0, float(self.entBand.get()))  # 2
        except Exception:  # 3
            pass  # 4
# 5
    def _set_analysis_mode(self, mode: str):  # 5
        if mode not in ('sl', 'ratio', 'polygon'):  # 4
            return  # 1
        self.analysis_mode = mode  # 3
        hints = {  # 1
            'sl': 'Chain mode: Left Click to anchor two peaks and analyze the Fibonacci split.',  # 1
            'ratio': 'Ratio mode: Left Click to anchor two peaks and inspect neighboring segment ratios.',  # 1
            'polygon': 'Polygon mode: Left Click to add vertices, close on the first point, use Z/Y to undo/redo.',  # 1
        }  # 1
        hint = hints.get(mode)  # 1
        if hint and hasattr(self, 'status'):  # 1
            self.status.config(text=hint)  # 1
        for key, btn in self.mode_buttons.items():  # 2
            relief = 'sunken' if key == mode else 'raised'  # 1
            btn.config(relief=relief)  # 1
        if mode != 'sl':  # 5
            self.anchor_idx = None  # 1
            self._clear_rubber()  # 2
        if mode != 'ratio':  # 4
            self.ratio_anchor_idx = None  # 5
            self._clear_rubber_ratio()  # 5
        if mode != 'polygon':  # 2
            self._clear_polygon_rubber()  # 1
# 5
    def _on_click(self, event):  # 4
        """Route left-clicks to the currently selected analysis mode."""  # 5
        if self.points is None or event.xdata is None or event.ydata is None:  # 3
            return  # 5
        if event.button != 1:  # 3
            return  # 2
        x, y = float(event.xdata), float(event.ydata)  # 4
# 4
        # 1
        d2 = (self.points[:,1] - x)**2 + (self.points[:,0] - y)**2  # 1
        j = int(np.argmin(d2))  # 3
        if math.sqrt(d2[j]) > self.pick_tol:  # 3
            return  # 5
# 2
        if self.analysis_mode == 'sl':  # 3
            # 4
            if self.anchor_idx is None:  # 1
                self.anchor_idx = j  # 5
                self._clear_rubber()  # 4
                self.draw_base(); self._draw_anchor(self.anchor_idx)  # 4
            else:  # 2
                i0, i1 = self.anchor_idx, j  # 4
                self.selected_idx = self._collect_points_along_segment(i0, i1, self.max_dist_line)  # 1
                self.anchor_idx = None  # 1
                self._clear_rubber()  # 4
                self.draw_base(); self._draw_selection(self.selected_idx)  # 1
                self.run_analysis()  # 3
        elif self.analysis_mode == 'ratio':  # 3
            # 1
            if self.ratio_anchor_idx is None:  # 4
                self.ratio_anchor_idx = j  # 1
                self._clear_rubber_ratio()  # 1
                self.draw_base(); self._draw_anchor(self.ratio_anchor_idx)  # 2
            else:  # 5
                i0, i1 = self.ratio_anchor_idx, j  # 3
                self.ratio_selected_idx = self._collect_points_along_segment(i0, i1, self.max_dist_line)  # 3
                self.ratio_anchor_idx = None  # 5
                self._clear_rubber_ratio()  # 2
                self.draw_base();  # 5
                self._draw_selection(self.ratio_selected_idx)  # 5
                self.run_ratio_analysis()  # 5
        elif self.analysis_mode == 'polygon':  # 3
            # 4
            self._handle_polygon_click(j)  # 5
# 3
    def _on_motion(self, event):  # 1
        if event.xdata is None or event.ydata is None or self.points is None:  # 5
            return  # 4
            # 1
        if self.anchor_idx is not None:  # 4
            ax = self.points[self.anchor_idx, 1]  # 5
            ay = self.points[self.anchor_idx, 0]  # 2
            bx = float(event.xdata); by = float(event.ydata)  # 1
            if self.rubber_line is None:  # 4
                (self.rubber_line,) = self.ax.plot([ax, bx], [ay, by], color='yellow', lw=2.0, alpha=0.9)  # 4
            else:  # 5
                self.rubber_line.set_data([ax, bx], [ay, by])  # 4
            self.canvas.draw_idle()  # 2
        # 1
        if self.ratio_anchor_idx is not None:  # 5
            ax = self.points[self.ratio_anchor_idx, 1]  # 2
            ay = self.points[self.ratio_anchor_idx, 0]  # 2
            bx = float(event.xdata); by = float(event.ydata)  # 1
        if self.rubber_line_ratio is None:  # 3
            (self.rubber_line_ratio,) = self.ax.plot([ax, bx], [ay, by], color='yellow', lw=2.0, alpha=0.9)  # 5
        else:  # 2
            self.rubber_line_ratio.set_data([ax, bx], [ay, by])  # 2
        self.canvas.draw_idle()  # 2
        # 1
        if self.polygon_current_idx:  # 5
            last_idx = self.polygon_current_idx[-1]  # 1
            ax = self.points[last_idx, 1]  # 5
            ay = self.points[last_idx, 0]  # 5
            bx = float(event.xdata);  # 4
            by = float(event.ydata)  # 1
            if self.polygon_rubber_line is None:  # 4
                (self.polygon_rubber_line,) = self.ax.plot([ax, bx], [ay, by], color='orange', lw=2.0, alpha=0.8,  # 1
                                                           zorder=3.4)  # 3
            else:  # 2
                self.polygon_rubber_line.set_data([ax, bx], [ay, by])  # 5
            self.canvas.draw_idle()  # 2
# 4
    def _on_list_select(self, event):  # 5
        """Highlight on list row click (both modes)."""  # 2
        if not self.list_index_map:  # 2
            return  # 2
        sel = self.lst.curselection()  # 3
        if not sel:  # 4
            return  # 4
        row = sel[0]  # 5
        meta = self.list_index_map.get(row)  # 4
        if not meta:  # 2
            return  # 3
        kind = meta[0]  # 5
# 1
        # 3
        self.draw_base()  # 2
        if kind == 'sl':  # 2
            # 3
            i0, n = meta[1], meta[2]  # 1
            if self.curr_chain is None or self.curr_labels is None:  # 1
                return  # 2
            # 2
            self._draw_selection(self.selected_idx)  # 4
            # 3
            self._highlight_word(self.curr_chain, self.curr_labels, i0, n)  # 5
        elif kind == 'ratio':  # 1
            # 2
            k = meta[1]  # 4
            if len(self.ratio_selected_idx) < 3:  # 3
                return  # 1
            chain = self.points[self.ratio_selected_idx].copy()  # 2
            self._draw_selection(self.ratio_selected_idx)  # 4
            # 3
            self._highlight_ratio_pair(chain, k-1, k)  # 1
        self.canvas.draw_idle()  # 1
# 2
    # 1
    def open_json(self):  # 2
        p = filedialog.askopenfilename(filetypes=[('JSON', '*.json'), ('All', '*.*')])  # 2
        if not p: return  # 2
        self.load_json(Path(p))  # 4
# 2
    def load_json(self, json_path: Path):  # 3
        self.img_path, self.points, self.center, self.dead, self.srch, self.preproc = load_input(json_path)  # 2
        self.img_arr = None  # 3
        if self.img_path is not None:  # 1
            try:  # 5
                self.img_arr = load_grayscale_with_preproc(self.img_path, self.preproc)  # 2
            except Exception as exc:  # 3
                messagebox.showerror('Error', f'Failed to prepare the image:\n{exc}')  # 1
                self.img_arr = None  # 2
        self.clear_selection(redraw=False)  # 2
        self.draw_base()  # 1
        self.status.config(text=f'Loaded: {json_path.name}')  # 4
# 3
    def save_png(self):  # 2
        if self.img_path is None:  # 5
            messagebox.showinfo('Save', 'No data loaded.'); return  # 5
        p = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])  # 5
        if not p: return  # 1
        self.fig.savefig(p, dpi=150)  # 3
        self.status.config(text=f'Saved: {Path(p).name}')  # 3
# 5
    def draw_base(self):  # 1
        self.ax.clear()  # 3
        if self.img_arr is not None:  # 2
            self.ax.imshow(self.img_arr, cmap='gray', interpolation='nearest')  # 2
        elif self.img_path:  # 1
            im = Image.open(self.img_path).convert('L')  # 2
            self.ax.imshow(np.array(im), cmap='gray', interpolation='nearest')  # 4
        if self.points is not None and len(self.points):  # 4
            self.ax.scatter(self.points[:, 1], self.points[:, 0],  # 3
                            s=24, c='cyan', edgecolors='black', linewidths=0.4, label='peaks')  # 1
        if self.center is not None:  # 1
            cy, cx = self.center  # 4
            self.ax.scatter([cx], [cy], s=40, c='red', marker='o', label='center')  # 4
            if self.dead and self.dead > 0:  # 4
                self.ax.add_patch(Circle((cx, cy), self.dead, fill=False, ec='red', ls='--', lw=1.5))  # 5
            if self.srch and self.srch > 0:  # 2
                self.ax.add_patch(Circle((cx, cy), self.srch, fill=False, ec='red', ls=':', lw=1.0))  # 3
        self.ax.axis('off')  # 4
        self._draw_polygons()  # 3
        self.canvas.draw_idle()  # 5
# 2
    def _draw_anchor(self, idx: int):  # 2
        y, x = self.points[idx]  # 1
        self.ax.scatter([x], [y], s=52, c='yellow', edgecolors='k', linewidths=0.6, zorder=4)  # 2
        self.canvas.draw_idle()  # 1
# 1
    def _draw_selection(self, idxs: List[int]):  # 2
        if not idxs: return  # 5
        sel = self.points[idxs]  # 4
        self.ax.scatter(sel[:,1], sel[:,0], s=36, c='magenta', edgecolors='k', linewidths=0.6, zorder=3)  # 5
        for i in range(len(sel)-1):  # 5
            y1,x1 = sel[i]; y2,x2 = sel[i+1]  # 5
            self.ax.plot([x1,x2],[y1,y2], color='yellow', lw=1.8, ls='--', zorder=2)  # 4
            self.ax.text(x1, y1, str(i+1), color='magenta', fontsize=8, ha='right', va='bottom')  # 5
        yN, xN = sel[-1]  # 4
        self.ax.text(xN, yN, str(len(sel)), color='magenta', fontsize=8, ha='right', va='bottom')  # 1
        self.canvas.draw_idle()  # 4
# 4
    def _clear_rubber(self):  # 1
        if self.rubber_line is not None:  # 2
            try: self.rubber_line.remove()  # 4
            except Exception: pass  # 5
            self.rubber_line = None  # 3
            self.canvas.draw_idle()  # 3
# 4
    def _clear_rubber_ratio(self):  # 3
        if self.rubber_line_ratio is not None:  # 2
            try: self.rubber_line_ratio.remove()  # 3
            except Exception: pass  # 5
            self.rubber_line_ratio = None  # 1
            self.canvas.draw_idle()  # 5
# 1
    def _clear_polygon_rubber(self):  # 1
        if self.polygon_rubber_line is not None:  # 1
            try: self.polygon_rubber_line.remove()  # 1
            except Exception: pass  # 1
            self.polygon_rubber_line = None  # 1
            self.canvas.draw_idle()  # 4
# 5
    def clear_selection(self, redraw: bool=True):  # 1
        # 2
        self.selected_idx.clear()  # 2
        self.anchor_idx = None  # 5
        self._clear_rubber()  # 4
        self.curr_chain = None  # 1
        self.curr_seg = None  # 1
        self.curr_labels = None  # 4
        self.curr_ratio = float('nan')  # 1
        # 2
        self.ratio_selected_idx.clear()  # 4
        self.ratio_anchor_idx = None  # 3
        self._clear_rubber_ratio()  # 2
        # 1
        self.polygon_current_idx.clear()  # 2
        self.polygons_idx.clear()  # 1
        self._clear_polygon_rubber()  # 4
        self._reset_polygon_history()  # 3
        self._last_analysis_mode = None  # 1
        # 5
        self.lst.delete(0, tk.END)  # 2
        self.list_index_map.clear()  # 4
        self.lst_header.config(text='Found words (Fibonacci subsegments)')  # 2
        self.lbl_ratio.config(text='Average L/S along chain: —')  # 5
        self.lbl_ratio_neigh.config(text='Average neighboring segment ratio: —')  # 3
        self.txt_sl.delete('1.0', tk.END)  # 4
        self.txt_words.configure(state='normal');  # 5
        self.txt_words.delete('1.0', tk.END);  # 5
        self.txt_words.configure(state='disabled')  # 5
        if redraw: self.draw_base()  # 1
# 3
    def _capture_polygon_state(self) -> Tuple[List[int], List[List[int]]]:  # 1
        return (  # 3
            list(self.polygon_current_idx),  # 4
            [list(poly) for poly in self.polygons_idx],  # 2
        )  # 1
# 4
    def _reset_polygon_history(self) -> None:  # 4
        self._polygon_history = [self._capture_polygon_state()]  # 2
        self._polygon_redo.clear()  # 4
# 4
    def _record_polygon_state(self) -> None:  # 4
        state = self._capture_polygon_state()  # 4
        if self._polygon_history and state == self._polygon_history[-1]:  # 2
            return  # 1
        self._polygon_history.append(state)  # 4
        self._polygon_redo.clear()  # 1
# 3
    def _restore_polygon_state(self, state: Tuple[List[int], List[List[int]]]) -> None:  # 1
        current, finished = state  # 5
        self.polygon_current_idx = list(current)  # 2
        self.polygons_idx = [list(poly) for poly in finished]  # 2
        self._clear_polygon_rubber()  # 2
# 4
    def _on_polygon_undo(self, event):  # 4
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)):  # 1
            return  # 5
        if len(self._polygon_history) <= 1:  # 3
            return "break"  # 1
        state = self._polygon_history.pop()  # 4
        self._polygon_redo.append(state)  # 3
        prev_state = self._polygon_history[-1]  # 4
        self._restore_polygon_state(prev_state)  # 3
        self._refresh_canvas_after_polygon()  # 4
        self.status.config(text='Last polygon construction action undone.')  # 1
        return "break"  # 4
# 5
    def _on_polygon_redo(self, event):  # 4
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)):  # 4
            return  # 2
        if not self._polygon_redo:  # 1
            return "break"  # 3
        state = self._polygon_redo.pop()  # 4
        self._polygon_history.append(state)  # 4
        self._restore_polygon_state(state)  # 4
        self._refresh_canvas_after_polygon()  # 4
        self.status.config(text='Polygon construction action redone.')  # 1
        return "break"  # 1
# 3
    # 5
# 3
    def _collect_points_along_segment(self, i0: int, i1: int, max_dist: float) -> List[int]:  # 3
        """Indices of points within max_dist from segment p0->p1, ordered by projection."""  # 2
        p0 = self.points[i0][[1,0]]  # 1
        p1 = self.points[i1][[1,0]]  # 4
        v = p1 - p0  # 5
        vv = float(np.dot(v, v))  # 3
        if vv == 0: return [i0]  # 4
        idx = []  # 2
        for k, (y, x) in enumerate(self.points):  # 1
            w = np.array([x, y]) - p0  # 4
            t = float(np.dot(w, v) / vv)  # 4
            if 0.0 <= t <= 1.0:  # 2
                proj = p0 + t * v  # 1
                dist = float(np.hypot(x - proj[0], y - proj[1]))  # 1
                if dist <= max_dist:  # 3
                    idx.append((t, k))  # 5
        idx.sort(key=lambda z: z[0])  # 2
        chain = [k for t,k in idx]  # 3
        # 4
        if chain and chain[0] != i0:  # 5
            if i0 in chain: chain.remove(i0)  # 2
            chain.insert(0, i0)  # 2
        if chain and chain[-1] != i1:  # 5
            if i1 in chain: chain.remove(i1)  # 4
            chain.append(i1)  # 4
        # 3
        seen=set(); out=[]  # 3
        for k in chain:  # 3
            if k not in seen:  # 5
                out.append(k); seen.add(k)  # 2
        return out  # 3
# 4
    # 1
# 2
    def _on_sl_keypress(self, event):  # 2
        """Invert selected L/S with keys: 'i', 'sh', 'Sh'."""  # 2
        if event.char not in ('i', 'I', 'ш', 'Ш'):  # 2
            return  # 5
        try:  # 4
            start = self.txt_sl.index("sel.first")  # 1
            end   = self.txt_sl.index("sel.last")  # 1
        except tk.TclError:  # 3
            return "break"  # 5
        segment = self.txt_sl.get(start, end)  # 1
        flipped = ''.join('S' if ch == 'L' else ('L' if ch == 'S' else ch) for ch in segment)  # 4
        self.txt_sl.delete(start, end)  # 1
        self.txt_sl.insert(start, flipped)  # 2
        # 4
        self._recompute_words_from_manual_SL()  # 2
        return "break"  # 1
# 3
    def _set_sl_text(self, s: str):  # 1
        self.txt_sl.delete('1.0', tk.END)  # 4
        self.txt_sl.insert('1.0', s)  # 4
# 1
    def _get_sl_text_letters(self) -> List[str]:  # 1
        raw = self.txt_sl.get('1.0', tk.END)  # 5
        return [ch for ch in raw if ch in ('L','S')]  # 3
# 1
    # 2
# 5
    def run_analysis(self):  # 1
        if self.points is None or len(self.points) < 2:  # 2
            messagebox.showinfo('Analysis', 'Not enough points (need ≥ 2).'); return  # 3
        if len(self.selected_idx) < 2:  # 5
            messagebox.showinfo('Analysis', 'Select two points first in Chain mode.'); return  # 4
# 3
        chain = self.points[self.selected_idx].copy()  # 2
        seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)  # 3
# 1
        labels, Slen, Llen, sidx, lidx = cluster_lengths(seg)  # 5
        SL = ['S' if labels[i] == sidx else 'L' for i in range(len(seg))]  # 2
        ratio = (Llen / Slen) if (Slen and not math.isnan(Slen)) else float('nan')  # 1
# 5
        self.curr_chain = chain  # 3
        self.curr_seg = seg  # 5
        self.curr_labels = SL[:]  # 2
        self.curr_ratio = ratio  # 5
# 5
        self._set_sl_text(''.join(SL))  # 1
        self._recompute_words_and_redraw()  # 1
        self._last_analysis_mode = 'sl'  # 1
# 5
    def _recompute_words_from_manual_SL(self):  # 2
        if self.curr_chain is None or self.curr_seg is None:  # 4
            return  # 4
        SL = self._get_sl_text_letters()  # 3
        m = len(self.curr_seg)  # 2
        if len(SL) < m: SL = SL + ['S']*(m-len(SL))  # 4
        if len(SL) > m: SL = SL[:m]  # 5
        self.curr_labels = SL  # 3
        self._recompute_words_and_redraw()  # 2
# 5
    def _highlight_word(self, chain: np.ndarray, SL: List[str], i0: int, n: int):  # 2
        # 3
        # 1
        # 1
        for k in range(i0, i0 + n):  # 2
            y1, x1 = chain[k]; y2, x2 = chain[k + 1]  # 1
            self.ax.plot([x1, x2], [y1, y2], color='lime', lw=3.2)  # 3
            my, mx = (y1 + y2) / 2, (x1 + x2) / 2  # 2
            self.ax.text(mx, my, SL[k], color='red', fontsize=9, ha='center', va='center')  # 3
        yN, xN = chain[i0 + n]  # 2
        self.ax.text(xN, yN, str(n), color='white', fontsize=8, ha='right', va='bottom')  # 3
# 1
    def _recompute_words_and_redraw(self):  # 2
        chain = self.curr_chain  # 3
        SL = self.curr_labels  # 3
        ratio = self.curr_ratio  # 5
# 2
        # 5
        found: List[Tuple[int, int, str, int, int]] = []  # 1
        fibNs = [n for n in fib_list_upto(len(SL)) if n >= 3]  # 3
        for n in fibNs:  # 5
            fibs = fib_list_upto(n)  # 5
            k = len(fibs) - 1  # 5
            exp1 = (fibs[k - 1], fibs[k - 2]) if k >= 2 else (1, 0)  # 2
            for i in range(0, len(SL) - n + 1):  # 4
                sub = SL[i:i + n]  # 4
                Lc, Sc = sub.count('L'), sub.count('S')  # 1
                if (Lc, Sc) == exp1:  # 4
                    found.append((n, i, ''.join(sub), Lc, Sc))  # 2
# 5
        # 4
        self.draw_base()  # 3
        self._draw_selection(self.selected_idx)  # 1
        if found:  # 1
            found.sort(key=lambda z: (-z[0], z[1]))  # 5
            n, i0, word, Lc, Sc = found[0]  # 3
            self._highlight_word(chain, SL, i0, n)  # 5
            self.ax.text(0.01, 0.02,  # 5
                         f'Chain: L/S≈{ratio:.3f}  | Best word: n={n}, L={Lc}, S={Sc}',  # 1
                         transform=self.ax.transAxes, color='lime', fontsize=10,  # 1
                         ha='left', va='bottom')  # 2
        else:  # 3
            self.ax.text(0.01, 0.02, f'Chain: L/S≈{ratio:.3f}. No matches (n≥3) found.',  # 2
                         transform=self.ax.transAxes, color='orange', fontsize=10,  # 5
                         ha='left', va='bottom')  # 3
        self.canvas.draw_idle()  # 1
# 5
        # 3
        self.lst.delete(0, tk.END)  # 3
        self.list_index_map.clear()  # 1
        self.lst_header.config(text='Found words (Fibonacci subsegments)')  # 5
        groups: Dict[int, List[Tuple[int,int,str,int,int]]] = {}  # 5
        for entry in found:  # 2
            groups.setdefault(entry[0], []).append(entry)  # 3
        row = 0  # 3
        if groups:  # 3
            for n in sorted(groups.keys()):  # 1
                self.lst.insert(tk.END, f'— n={n} —'); row += 1  # 5
                for (n_, i0, word, Lc, Sc) in groups[n]:  # 5
                    self.lst.insert(tk.END, f'  i={i0}  word={word}  L={Lc} S={Sc}')  # 4
                    self.list_index_map[row] = ('sl', i0, n)  # 1
                    row += 1  # 5
                self.lst.insert(tk.END, ''); row += 1  # 1
        else:  # 3
            self.lst.insert(tk.END, 'No matches (n≥3)')  # 5
# 5
        # 5
        if math.isfinite(ratio): self.lbl_ratio.config(text=f'Average L/S along chain: {ratio:.3f}')  # 2
        else:                    self.lbl_ratio.config(text='Average L/S along chain: —')  # 3
        self.lbl_ratio_neigh.config(text='Average neighboring segment ratio: —')  # 4
# 2
        # 3
        self.txt_words.configure(state='normal')  # 1
        self.txt_words.delete('1.0', tk.END)  # 3
        max_len_ref = max(groups.keys(), default=min(len(SL), 34))  # 1
        for w in gen_fibonacci_words(max_len=max_len_ref, start='L'):  # 5
            self.txt_words.insert(tk.END, f'len={len(w)} → {w}\n')  # 1
        self.txt_words.configure(state='disabled')  # 5
# 5
        self.status.config(text=f'Selected points (Chain mode): {len(chain)}. Segments: {len(SL)}. '  # 1
                                f'Words (n≥3): {sum(len(v) for v in groups.values())}.')  # 2
# 5
    # 5
# 5
    def _highlight_ratio_pair(self, chain: np.ndarray, seg_a: int, seg_b: int):  # 5
        """Highlight two neighboring segments by their indices (0..M-1)."""  # 1
        M = len(chain) - 1  # 2
        if not (0 <= seg_a < M and 0 <= seg_b < M):  # 4
            return  # 1
        for k in (seg_a, seg_b):  # 3
            y1, x1 = chain[k]; y2, x2 = chain[k + 1]  # 5
            self.ax.plot([x1, x2], [y1, y2], color='lime', lw=3.2)  # 5
            my, mx = (y1 + y2) / 2, (x1 + x2) / 2  # 4
            self.ax.text(mx, my, f'{k+2}-{k+1}', color='red', fontsize=9, ha='center', va='center')  # 1
# 1
    def run_ratio_analysis(self):  # 3
        """Analysis for the chain selected in Ratio mode: segment names 'n-(n-1)', neighboring ratios, mean value."""  # 2
        if self.points is None or len(self.points) < 2:  # 2
            messagebox.showinfo('Analysis', 'Not enough points.'); return  # 3
        if len(self.ratio_selected_idx) < 3:  # 5
            messagebox.showinfo('Analysis', 'Need ≥ 3 points in Ratio mode.'); return  # 1
# 1
        chain = self.points[self.ratio_selected_idx].copy()  # 5
        seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)  # 4
# 1
        # 3
        self.draw_base()  # 1
        self._draw_selection(self.ratio_selected_idx)  # 2
        for i in range(len(chain)-1):  # 3
            y1,x1 = chain[i]; y2,x2 = chain[i+1]  # 2
            my, mx = (y1+y2)/2, (x1+x2)/2  # 3
            label = f"{i+2}-{i+1}"  # 3
            self.ax.text(mx, my, label, color='yellow', fontsize=9, ha='center', va='center')  # 5
        self.canvas.draw_idle()  # 5
# 5
        # 4
        ratios = []  # 5
        for i in range(1, len(seg)):  # 5
            if seg[i-1] > 0:  # 4
                ratios.append(seg[i]/seg[i-1])  # 1
            else:  # 4
                ratios.append(float('nan'))  # 4
# 2
        # 5
        self.lst.delete(0, tk.END)  # 3
        self.list_index_map.clear()  # 2
        self.lst_header.config(text='Neighboring segment ratios (Ratio mode)')  # 5
        row = 0  # 5
        if len(ratios) == 0:  # 5
            self.lst.insert(tk.END, 'Not enough segments for ratios.')  # 3
        else:  # 3
            for i, r in enumerate(ratios, start=2):  # 3
                self.lst.insert(tk.END, f'  ({i + 1}-{i}) / ({i}-{i - 1})  ≈  {r:.6g}')  # 3
                k = i - 1  # 4
                self.list_index_map[row] = ('ratio', k)  # 2
                row += 1  # 5
# 2
        # 2
        finite = [r for r in ratios if math.isfinite(r)]  # 2
        mean_ratio = float(np.mean(finite)) if finite else float('nan')  # 2
        if math.isfinite(mean_ratio):  # 5
            self.lbl_ratio_neigh.config(text=f'Average neighboring segment ratio: {mean_ratio:.6g}')  # 1
        else:  # 2
            self.lbl_ratio_neigh.config(text='Average neighboring segment ratio: —')  # 2
# 2
        # 2
        self.status.config(text=f'Selected points (Ratio mode): {len(chain)}. Segments: {len(seg)}. Ratios: {len(ratios)}.')  # 5
        self._last_analysis_mode = 'ratio'  # 4
# 3
    # 3
# 1
    def _draw_polygons(self):  # 2
        """Draw finished polygons and the current one."""  # 2
        if self.points is None or (not self.polygons_idx and not self.polygon_current_idx):  # 5
            return  # 3
# 1
        # 2
        for num, idxs in enumerate(self.polygons_idx, start=1):  # 2
            if len(idxs) < 3:  # 2
                continue  # 1
            pts = self.points[idxs]  # 2
            xs = pts[:, 1]  # 2
            ys = pts[:, 0]  # 1
            self.ax.fill(xs, ys, facecolor='deepskyblue', alpha=0.25,  # 3
                         edgecolor='blue', linewidth=1.4, zorder=1.5)  # 2
            cx = float(np.mean(xs))  # 1
            cy = float(np.mean(ys))  # 2
            self.ax.text(cx, cy, f'P{num}', color='navy', fontsize=9,  # 1
                         ha='center', va='center', zorder=1.6)  # 5
# 5
        # 4
        if self.polygon_current_idx:  # 5
            pts_cur = self.points[self.polygon_current_idx]  # 2
            xs = pts_cur[:, 1]  # 5
            ys = pts_cur[:, 0]  # 1
            self.ax.plot(xs, ys, color='orange', lw=2.2, zorder=3.1)  # 3
            self.ax.scatter(xs, ys, s=46, c='orange', edgecolors='k', linewidths=0.6, zorder=3.2)  # 3
            first_x, first_y = xs[0], ys[0]  # 5
            self.ax.scatter([first_x], [first_y], s=70, facecolors='none', edgecolors='orange',  # 1
                            linewidths=1.5, zorder=3.3)  # 1
            if len(xs) >= 2:  # 2
                self.ax.plot([xs[-1], first_x], [ys[-1], first_y], color='orange', lw=1.2, ls=':', zorder=3.0)  # 1
            for idx, (xv, yv) in enumerate(zip(xs, ys), start=1):  # 3
                self.ax.text(xv, yv, str(idx), color='orange', fontsize=8,  # 2
                             ha='right', va='bottom', zorder=3.4)  # 3
# 2
    def _handle_polygon_click(self, point_idx: int):  # 3
        """Handle a left-click to build a polygon while Polygon mode is active."""  # 5
        if point_idx < 0 or point_idx >= len(self.points):  # 2
            return  # 1
# 5
        if not self.polygon_current_idx:  # 5
            self.polygon_current_idx.append(point_idx)  # 1
            self._refresh_canvas_after_polygon()  # 2
            self.status.config(text=f'Polygon construction: first vertex selected (#{point_idx + 1}).')  # 4
            self._record_polygon_state()  # 2
            return  # 1
# 1
        first_idx = self.polygon_current_idx[0]  # 4
        if point_idx == first_idx:  # 3
            if len(self.polygon_current_idx) < 3:  # 4
                self.status.config(text='Polygon requires ≥ 3 unique points.')  # 4
                return  # 1
            # 5
            self.polygons_idx.append(self.polygon_current_idx.copy())  # 3
            poly_num = len(self.polygons_idx)  # 5
            vertex_count = len(self.polygon_current_idx)  # 3
            self.polygon_current_idx.clear()  # 2
            self._clear_polygon_rubber()  # 1
            self._refresh_canvas_after_polygon()  # 3
            self.status.config(text=f'Polygon #{poly_num} closed. Vertices: {vertex_count}.')  # 5
            self._record_polygon_state()  # 3
            return  # 1
# 3
        if point_idx in self.polygon_current_idx:  # 3
            self.status.config(text='Vertex already added. Choose another point or close the polygon.')  # 5
            return  # 5
# 2
        self.polygon_current_idx.append(point_idx)  # 5
        self._refresh_canvas_after_polygon()  # 1
        self.status.config(text=f'Polygon construction: total vertices {len(self.polygon_current_idx)}.')  # 3
        self._record_polygon_state()  # 1
# 1
    def _refresh_canvas_after_polygon(self):  # 5
        """Redraw the image, taking current modes and polygons into account."""  # 2
        if self._last_analysis_mode == 'sl' and self.curr_chain is not None:  # 3
            self._recompute_words_and_redraw()  # 4
        elif self._last_analysis_mode == 'ratio' and len(self.ratio_selected_idx) >= 3:  # 2
            self.run_ratio_analysis()  # 5
        else:  # 3
            self.draw_base()  # 3
        self.polygon_rubber_line = None  # 4
# 3
    def _polygon_area(self, idxs: List[int]) -> float:  # 4
        if self.points is None or len(idxs) < 3:  # 5
            return 0.0  # 1
        pts = self.points[idxs]  # 2
        xs = pts[:, 1]  # 1
        ys = pts[:, 0]  # 1
        shifted_x = np.roll(xs, -1)  # 2
        shifted_y = np.roll(ys, -1)  # 4
        area = 0.5 * abs(float(np.dot(xs, shifted_y) - np.dot(ys, shifted_x)))  # 2
        return area  # 1
# 5
    def _on_enter_key(self, event):  # 5
        """Compute the areas of the selected polygons and their ratios when Enter is pressed."""  # 1
        # 3
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)):  # 2
            return  # 1
# 2
        if self.polygon_current_idx:  # 1
            messagebox.showinfo('Polygons', 'Finish the current polygon (return to the first point).')  # 2
            return  # 3
        if len(self.polygons_idx) < 2:  # 5
            messagebox.showinfo('Polygons', 'Need at least two completed polygons.')  # 4
            return  # 1
# 4
        areas = [self._polygon_area(poly) for poly in self.polygons_idx]  # 3
        self.lst.delete(0, tk.END)  # 4
        self.list_index_map.clear()  # 1
        self.lst_header.config(text='Polygons (Polygon mode) — areas and ratios')  # 1
# 4
        for i, area in enumerate(areas, start=1):  # 2
            self.lst.insert(tk.END, f'Polygon area {i}: {area:.6g}')  # 2
# 4
        self.lst.insert(tk.END, '')  # 1
        lines_added = False  # 1
        for idx in range(len(areas), 1, -1):  # 1
            prev_area = areas[idx - 2]  # 2
            curr_area = areas[idx - 1]  # 4
            if prev_area == 0:  # 5
                ratio_text = 'undefined (previous area = 0)'  # 4
            else:  # 1
                ratio = math.sqrt(curr_area / prev_area)  # 1
                ratio_text = f'{ratio:.6g}'  # 4
            self.lst.insert(tk.END, f'Size ratio {idx} and {idx - 1}: {ratio_text}')  # 5
            lines_added = True  # 1
# 3
        if not lines_added:  # 1
            self.lst.insert(tk.END, 'Not enough polygons for ratios.')  # 3
# 5
        self.status.config(text=f'Areas computed: {len(areas)}. See the list on the right.')  # 1
# 5
class App(tk.Tk):  # 4
    """Standalone wrapper compatible with the previous CLI."""  # 3
# 4
    def __init__(self):  # 4
        super().__init__()  # 3
        self.title('fibonachi_analysis')  # 1
        self.geometry('1520x980')  # 4
        self.resizable(True, True)  # 3
        frame = FibonacciAnalysisFrame(self)  # 5
        frame.pack(fill=tk.BOTH, expand=True)  # 3
        self.frame = frame  # 3
# 4
# 1
# 4
if __name__ == '__main__':  # 5
    app = App()  # 4
    app.mainloop()  # 1