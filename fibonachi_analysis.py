from __future__ import annotations
import sys, json, math
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
from preproc import PreprocSettings, load_grayscale_with_preproc

if not hasattr(tk, "Notebook") and hasattr(ttk, "Notebook"):
    tk.Notebook = ttk.Notebook


# ... (Классы _Tooltip и HoverTooltip, функции _parse_cli и остальные до FibonacciAnalysisFrame без изменений)
class _Tooltip:
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
            tip,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            wraplength=360,
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
    """Display a tooltip when the mouse hovers over a widget."""

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
    import argparse
    p = argparse.ArgumentParser(description="fibonachi_analysis — load input data")
    p.add_argument("--payload", type=str, default=None, help="Path to fibo_input.json")
    p.add_argument("--image", type=str, default=None, help="Path to the image (fallback)")
    p.add_argument("--points", type=str, default=None, help="Path to JSON with points (fallback)")
    return p.parse_args(argv)


def _candidate_dirs(extra_image: Optional[Path]) -> List[Path]:
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
    """k=2 clustering of lengths into S/L; returns (labels, Slen, Llen, sidx, lidx)."""
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
    #
    m0 = float(lengths[lab == 0].mean()) if np.any(lab == 0) else float("nan")
    m1 = float(lengths[lab == 1].mean()) if np.any(lab == 1) else float("nan")
    if (not math.isnan(m0)) and (not math.isnan(m1)) and m0 > m1:
        lab = 1 - lab
        m0, m1 = m1, m0
    return lab, m0, m1, 0, 1


def fib_list_upto(n: int) -> List[int]:
    """Fibonacci numbers up to n (inclusive), starting with 1,1,2,3,..."""
    if n <= 0: return []
    seq = [1, 1]
    while seq[-1] < n:
        seq.append(seq[-1] + seq[-2])
    return [k for k in seq if k <= n]


def gen_fibonacci_words(max_len: int, start: str = "L") -> List[str]:
    """Generate prefixes of the "fibo-word": L->LS, S->L, up to max_len."""
    if max_len <= 0: return []
    words = ["L" if start.upper() == "L" else "S"]
    while len(words[-1]) <= max_len:
        nxt = "".join(("LS" if ch == "L" else "L") for ch in words[-1])
        if len(nxt) > max_len: break
        words.append(nxt)
    return words


class FibonacciAnalysisFrame(tk.Frame):
    def __init__(self, master: tk.Misc, controller=None, auto_load: bool = True):
        super().__init__(master)
        self.controller = controller

        self.img_path: Optional[Path] = None
        self.points: Optional[np.ndarray] = None
        self.center: Optional[Tuple[float, float]] = None
        self.dead: float = 0.0
        self.srch: float = 0.0
        self.preproc: PreprocSettings = PreprocSettings(mode="raw")
        self.img_arr: Optional[np.ndarray] = None
        self.zoom_val: float = 0.0
        self.view_cx: Optional[float] = None
        self.view_cy: Optional[float] = None
        self._img_shape: Optional[Tuple[int, int]] = None
        self._full_view_bounds: Optional[Tuple[float, float, float, float]] = None

        self.pick_tol = 10.0
        self.max_dist_line = 20.0

        self.mode_buttons: Dict[str, tk.Button] = {}
        self._mode_tooltips: List[HoverTooltip] = []
        self.analysis_mode: str = 'sl'
        self._last_analysis_mode: Optional[str] = None

        self.selected_idx: List[int] = []
        self.ratio_selected_idx: List[int] = []
        self.polygon_current_idx: List[int] = []
        self.polygons_idx: List[List[int]] = []
        self._polygon_history: List[Tuple[List[int], List[List[int]]]] = []
        self._polygon_redo: List[Tuple[List[int], List[List[int]]]] = []
        self._right_scroll_canvas: Optional[tk.Canvas] = None
        self._right_scroll_window: Optional[int] = None

        self.list_index_map: Dict[int, Tuple] = {}
        self.curr_chain = None
        self.curr_seg = None
        self.curr_labels = None
        self.curr_ratio = float('nan')

        self.anchor_idx: Optional[int] = None
        self.ratio_anchor_idx: Optional[int] = None
        self.rubber_line = None
        self.rubber_line_ratio = None
        self.polygon_rubber_line = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        container = tk.Frame(self)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=0)

        left = tk.Frame(container)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=12)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        right_host = tk.Frame(container, width=470)
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

        right.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(self._right_scroll_window, width=e.width))

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        right.bind("<Enter>", self._activate_right_scroll)
        right.bind("<Leave>", self._deactivate_right_scroll)
        canvas.bind("<Enter>", self._activate_right_scroll)
        canvas.bind("<Leave>", self._deactivate_right_scroll)

        right.columnconfigure(0, weight=1)

        controls = tk.Frame(right)
        controls.grid(row=0, column=0, sticky="ew", padx=6, pady=(0, 10))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        tk.Button(controls, text='Open JSON…', command=self.open_json).grid(row=0, column=0, sticky='ew', padx=4,
                                                                            pady=2)
        tk.Button(controls, text='Save PNG', command=self.save_png).grid(row=0, column=1, sticky='ew', padx=4, pady=2)
        tk.Button(controls, text='Clear selection', command=self.clear_selection).grid(row=1, column=0, columnspan=2,
                                                                                       sticky='ew', padx=4, pady=2)

        tk.Label(controls, text='Selection thickness (px):').grid(row=2, column=0, sticky='w', padx=4, pady=(8, 2))
        self.entBand = tk.Spinbox(controls, from_=2, to=100, width=6, command=self._onBand)
        self.entBand.delete(0, 'end');
        self.entBand.insert(0, str(int(self.max_dist_line)))
        self.entBand.grid(row=2, column=1, sticky='ew', padx=4, pady=(8, 2))

        mode_switch = tk.Frame(controls)
        mode_switch.grid(row=3, column=0, columnspan=2, sticky='ew', padx=4, pady=(8, 2))
        for col in range(3):
            mode_switch.columnconfigure(col, weight=1)

        self.mode_buttons['sl'] = tk.Button(mode_switch, text='🔗 Chain', command=lambda: self._set_analysis_mode('sl'))
        self.mode_buttons['sl'].grid(row=0, column=0, sticky='ew', padx=(0, 2))
        self._mode_tooltips.append(
            HoverTooltip(self.mode_buttons['sl'], 'Chain analysis: use Left Click while this mode is active.'))

        self.mode_buttons['ratio'] = tk.Button(mode_switch, text='📊 Ratio',
                                               command=lambda: self._set_analysis_mode('ratio'))
        self.mode_buttons['ratio'].grid(row=0, column=1, sticky='ew', padx=2)
        self._mode_tooltips.append(
            HoverTooltip(self.mode_buttons['ratio'], 'Neighbor ratios: use Left Click while this mode is active.'))

        self.mode_buttons['polygon'] = tk.Button(mode_switch, text='🔺 Polygon',
                                                 command=lambda: self._set_analysis_mode('polygon'))
        self.mode_buttons['polygon'].grid(row=0, column=2, sticky='ew', padx=(2, 0))
        self._mode_tooltips.append(HoverTooltip(self.mode_buttons['polygon'],
                                                'Polygon analysis: add vertices with Left Click, close by clicking the first point.'))

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
        self.lst_header = tk.Label(right, text='Found words (Fibonacci subsegments)')
        self.lst_header.grid(row=2, column=0, sticky='w', padx=6, pady=(0, 2))

        self.results_notebook = tk.Notebook(right)
        self.results_notebook.grid(row=3, column=0, sticky='nsew', padx=6, pady=(0, 8))
        right.grid_rowconfigure(3, weight=1)

        tab_subsegments = ttk.Frame(self.results_notebook)
        tab_subsegments.grid_columnconfigure(0, weight=1)
        tab_subsegments.grid_rowconfigure(0, weight=1)

        tab_prefixes = ttk.Frame(self.results_notebook)
        tab_prefixes.grid_columnconfigure(0, weight=1)
        tab_prefixes.grid_rowconfigure(1, weight=1)

        self.results_notebook.add(tab_subsegments, text='Subsegments')
        self.results_notebook.add(tab_prefixes, text='Fib prefixes')

        self.lst = tk.Listbox(tab_subsegments, width=66, height=22)
        self.lst.grid(row=0, column=0, sticky='nsew')
        self.lst.bind('<<ListboxSelect>>', self._on_list_select)

        self.lbl_ratio = tk.Label(tab_subsegments, text='Average L/S along chain: —')
        self.lbl_ratio.grid(row=1, column=0, sticky='w', pady=(6, 4))
        self.lbl_ratio_neigh = tk.Label(tab_subsegments, text='Average neighboring segment ratio: —')
        self.lbl_ratio_neigh.grid(row=2, column=0, sticky='w', pady=(2, 4))
        self.lbl_ratio_polygons = tk.Label(tab_subsegments, text='Average neighboring polygon linear ratio: —')
        self.lbl_ratio_polygons.grid(row=3, column=0, sticky='w', pady=(2, 8))

        tk.Label(tab_subsegments, text='S/L sequence (full):').grid(row=4, column=0, sticky='w', pady=(4, 2))
        self.txt_sl = tk.Text(tab_subsegments, height=6, wrap='word')
        self.txt_sl.grid(row=5, column=0, sticky='ew', pady=(0, 4))
        self.txt_sl.bind('<KeyPress>', self._on_sl_keypress)

        prefixes_header = tk.Label(tab_prefixes, text='Prefixes of "fib-words" (L→LS, S→L)')
        prefixes_header.grid(row=0, column=0, sticky='w', pady=(0, 4))

        prefixes_frame = tk.Frame(tab_prefixes)
        prefixes_frame.grid(row=1, column=0, sticky='nsew')
        prefixes_frame.grid_columnconfigure(0, weight=1)
        prefixes_frame.grid_rowconfigure(0, weight=1)

        self.txt_words = tk.Text(prefixes_frame, height=10, state='disabled')
        self.txt_words.grid(row=0, column=0, sticky='nsew')
        scroll_words = tk.Scrollbar(prefixes_frame, orient='vertical', command=self.txt_words.yview)
        scroll_words.grid(row=0, column=1, sticky='ns')
        self.txt_words.configure(yscrollcommand=scroll_words.set)

        self.fig = plt.Figure(figsize=(9.6, 6.6));
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)

        # --- ИЗМЕНЕНИЕ: Добавлена привязка колесика мыши для зума ---
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        # --------------------------------------------------------

        self.bind('<Escape>', lambda e: self.clear_selection())
        self.bind_all('<Return>', self._on_enter_key)

        self.bind_all('<Control-z>', self._on_polygon_undo)
        self.bind_all('<Control-y>', self._on_polygon_redo)

        self._reset_polygon_history()

        if auto_load:
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
        else:
            self.status.config(text='JSON not loaded. Use "Open JSON…".')

    # ... (весь остальной код класса FibonacciAnalysisFrame до _on_zoom_change без изменений)

    # --- НОВЫЙ МЕТОД: Обработка скролла мыши для зума ---
    def _on_scroll(self, event):
        if event.xdata is None or event.ydata is None:
            return  # Не зумировать, если курсор за пределами изображения

        zoom_step = 5  # Шаг изменения зума
        if event.button == 'up':
            self.zoom_val += zoom_step
        elif event.button == 'down':
            self.zoom_val -= zoom_step

        # Ограничиваем значение в пределах 0-100
        self.zoom_val = max(0.0, min(100.0, self.zoom_val))

        # Обновляем виджеты и перерисовываем, вызывая _on_zoom_change
        self._on_zoom_change(self.zoom_val)

    # ----------------------------------------------------

    # --- ИЗМЕНЕНИЕ: Метод _on_zoom_change теперь может принимать значение напрямую ---
    def _on_zoom_change(self, value):
        try:
            new_val = float(value)
        except Exception:
            new_val = 0.0

        new_val = max(0.0, min(100.0, new_val))
        self.zoom_val = new_val

        # Синхронизируем значение с переменной слайдера
        if hasattr(self, 'zoom_var'):
            current_var_val = self.zoom_var.get()
            if abs(current_var_val - new_val) > 1e-3:
                self.zoom_var.set(new_val)

        self._apply_zoom()
        self._update_zoom_hint()
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()

    # --------------------------------------------------------------------------

    # ... (Остальная часть файла `fibonachi_analysis.py` без изменений)
    # ... (Класс App и блок if __name__ == "__main__":)
    def _activate_right_scroll(self, _event):
        if self._right_scroll_canvas is None:
            return
        self._right_scroll_canvas.bind_all("<MouseWheel>", self._on_right_scroll_mousewheel)
        self._right_scroll_canvas.bind_all("<Button-4>", self._on_right_scroll_mousewheel)
        self._right_scroll_canvas.bind_all("<Button-5>", self._on_right_scroll_mousewheel)

    def _deactivate_right_scroll(self, _event):
        if self._right_scroll_canvas is None:
            return
        self._right_scroll_canvas.unbind_all("<MouseWheel>")
        self._right_scroll_canvas.unbind_all("<Button-4>")
        self._right_scroll_canvas.unbind_all("<Button-5>")

    def _on_right_scroll_mousewheel(self, event):
        if self._right_scroll_canvas is None:
            return
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
        if mode not in ('sl', 'ratio', 'polygon'):
            return
        self.analysis_mode = mode
        hints = {
            'sl': 'Chain mode: Left Click to anchor two peaks and analyze the Fibonacci split.',
            'ratio': 'Ratio mode: Left Click to anchor two peaks and inspect neighboring segment ratios.',
            'polygon': 'Polygon mode: Left Click to add vertices, close on the first point, use Ctrl+Z/Ctrl+Y to undo/redo.',
        }
        hint = hints.get(mode)
        if hint and hasattr(self, 'status'):
            self.status.config(text=hint)
        for key, btn in self.mode_buttons.items():
            relief = 'sunken' if key == mode else 'raised'
            btn.config(relief=relief)
        if mode != 'sl':
            self.anchor_idx = None
            self._clear_rubber()
        if mode != 'ratio':
            self.ratio_anchor_idx = None
            self._clear_rubber_ratio()
        if mode != 'polygon':
            self._clear_polygon_rubber()

    def _update_zoom_hint(self):
        if hasattr(self, 'zoom_hint'):
            try:
                value = int(round(float(self.zoom_var.get()))) if hasattr(self, 'zoom_var') else int(
                    round(self.zoom_val))
            except Exception:
                value = int(round(self.zoom_val))
            self.zoom_hint.config(text=f'Current zoom: {value}% (0 = full view)')

    def _ensure_view_center(self):
        if self.view_cx is not None and self.view_cy is not None:
            return
        if self.center is not None:
            cy, cx = self.center
            self.view_cx = float(cx)
            self.view_cy = float(cy)
            return
        if self.points is not None and len(self.points):
            ys = self.points[:, 0]
            xs = self.points[:, 1]
            self.view_cx = float(xs.mean())
            self.view_cy = float(ys.mean())
            return
        if self._full_view_bounds is not None:
            x0, x1, y0, y1 = self._full_view_bounds
            self.view_cx = (x0 + x1) / 2.0
            self.view_cy = (y0 + y1) / 2.0

    def _update_full_view_bounds(self):
        bounds = None
        if self._img_shape is not None:
            h, w = self._img_shape
            bounds = (-0.5, w - 0.5, -0.5, h - 0.5)
        elif self.points is not None and len(self.points):
            xs = self.points[:, 1]
            ys = self.points[:, 0]
            min_x = float(xs.min())
            max_x = float(xs.max())
            min_y = float(ys.min())
            max_y = float(ys.max())
            margin_x = max((max_x - min_x) * 0.05, 10.0)
            margin_y = max((max_y - min_y) * 0.05, 10.0)
            bounds = (min_x - margin_x, max_x + margin_x, min_y - margin_y, max_y + margin_y)
        self._full_view_bounds = bounds

    def _apply_zoom(self):
        if self._full_view_bounds is None:
            return
        x0_full, x1_full, y0_full, y1_full = self._full_view_bounds
        width = x1_full - x0_full
        height = y1_full - y0_full
        self._ensure_view_center()
        cx = float(self.view_cx if self.view_cx is not None else (x0_full + x1_full) / 2.0)
        cy = float(self.view_cy if self.view_cy is not None else (y0_full + y1_full) / 2.0)
        if self.zoom_val <= 0:
            self.ax.set_xlim(x0_full, x1_full)
            self.ax.set_ylim(y1_full, y0_full)
            return
        min_dim = float(min(width, height))
        L = float(round(min_dim - (min_dim - 50.0) * (self.zoom_val / 100.0)))
        L = max(50.0, min_dim if L < 50.0 else L)
        half = L / 2.0
        x0 = max(x0_full, cx - half)
        x1 = min(x1_full, cx + half)
        if (x1 - x0) < L:
            if x0 <= x0_full:
                x1 = min(x0 + L, x1_full)
            elif x1 >= x1_full:
                x0 = max(x1 - L, x0_full)
        y0 = max(y0_full, cy - half)
        y1 = min(y1_full, cy + half)
        if (y1 - y0) < L:
            if y0 <= y0_full:
                y1 = min(y0 + L, y1_full)
            elif y1 >= y1_full:
                y0 = max(y1 - L, y0_full)
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y1, y0)

    def _reset_zoom_state(self):
        self.zoom_val = 0.0
        self.view_cx = None
        self.view_cy = None
        if hasattr(self, 'zoom_var'):
            try:
                self.zoom_var.set(0.0)
            except Exception:
                pass
        self._update_zoom_hint()

    def _focus_on(self, x: float, y: float):
        self.view_cx = float(x)
        self.view_cy = float(y)
        if self._full_view_bounds is not None and hasattr(self, 'canvas'):
            self._apply_zoom()
            self.canvas.draw_idle()

    def _flash_right_scroll(self):
        canvas = self._right_scroll_canvas
        if canvas is None or not canvas.winfo_exists():
            return

        def _scroll_bottom():
            if not canvas.winfo_exists():
                return
            canvas.yview_moveto(1.0)
            canvas.after(120, _scroll_top)

        def _scroll_top():
            if not canvas.winfo_exists():
                return
            canvas.yview_moveto(0.0)

        canvas.after_idle(_scroll_bottom)

    def _on_click(self, event):
        """Route left-clicks to the currently selected analysis mode."""
        if self.points is None or event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return
        x, y = float(event.xdata), float(event.ydata)

        #
        d2 = (self.points[:, 1] - x) ** 2 + (self.points[:, 0] - y) ** 2
        j = int(np.argmin(d2))
        if math.sqrt(d2[j]) > self.pick_tol:
            return
        focus_x = float(self.points[j, 1])
        focus_y = float(self.points[j, 0])
        self._focus_on(focus_x, focus_y)

        if self.analysis_mode == 'sl':
            #
            if self.anchor_idx is None:
                self.anchor_idx = j
                self._clear_rubber()
                self.draw_base();
                self._draw_anchor(self.anchor_idx)
            else:
                i0, i1 = self.anchor_idx, j
                self.selected_idx = self._collect_points_along_segment(i0, i1, self.max_dist_line)
                self.anchor_idx = None
                self._clear_rubber()
                self.draw_base();
                self._draw_selection(self.selected_idx)
                self.run_analysis()
        elif self.analysis_mode == 'ratio':
            #
            if self.ratio_anchor_idx is None:
                self.ratio_anchor_idx = j
                self._clear_rubber_ratio()
                self.draw_base();
                self._draw_anchor(self.ratio_anchor_idx)
            else:
                i0, i1 = self.ratio_anchor_idx, j
                self.ratio_selected_idx = self._collect_points_along_segment(i0, i1, self.max_dist_line)
                self.ratio_anchor_idx = None
                self._clear_rubber_ratio()
                self.draw_base();
                self._draw_selection(self.ratio_selected_idx)
                self.run_ratio_analysis()
        elif self.analysis_mode == 'polygon':
            #
            self._handle_polygon_click(j)

    def _on_motion(self, event):
        if event.xdata is None or event.ydata is None or self.points is None:
            return
            #
        if self.anchor_idx is not None:
            ax = self.points[self.anchor_idx, 1]
            ay = self.points[self.anchor_idx, 0]
            bx = float(event.xdata);
            by = float(event.ydata)
            if self.rubber_line is None:
                (self.rubber_line,) = self.ax.plot([ax, bx], [ay, by], color='yellow', lw=2.0, alpha=0.9)
            else:
                self.rubber_line.set_data([ax, bx], [ay, by])
            self.canvas.draw_idle()
        #
        if self.ratio_anchor_idx is not None:
            ax = self.points[self.ratio_anchor_idx, 1]
            ay = self.points[self.ratio_anchor_idx, 0]
            bx = float(event.xdata);
            by = float(event.ydata)
        if self.rubber_line_ratio is None:
            (self.rubber_line_ratio,) = self.ax.plot([ax, bx], [ay, by], color='yellow', lw=2.0, alpha=0.9)
        else:
            self.rubber_line_ratio.set_data([ax, bx], [ay, by])
        self.canvas.draw_idle()
        #
        if self.polygon_current_idx:
            last_idx = self.polygon_current_idx[-1]
            ax = self.points[last_idx, 1]
            ay = self.points[last_idx, 0]
            bx = float(event.xdata);
            by = float(event.ydata)
            if self.polygon_rubber_line is None:
                (self.polygon_rubber_line,) = self.ax.plot([ax, bx], [ay, by], color='orange', lw=2.2, alpha=0.8,
                                                           zorder=3.4)
            else:
                self.polygon_rubber_line.set_data([ax, bx], [ay, by])
            self.canvas.draw_idle()

    def _on_list_select(self, event):
        """Highlight on list row click (both modes)."""
        if not self.list_index_map:
            return
        sel = self.lst.curselection()
        if not sel:
            return
        row = sel[0]
        meta = self.list_index_map.get(row)
        if not meta:
            return
        kind = meta[0]

        #
        self.draw_base()
        if kind == 'sl':
            #
            i0, n = meta[1], meta[2]
            if self.curr_chain is None or self.curr_labels is None:
                return
            #
            self._draw_selection(self.selected_idx)
            #
            self._highlight_word(self.curr_chain, self.curr_labels, i0, n)
        elif kind == 'ratio':
            #
            k = meta[1]
            if len(self.ratio_selected_idx) < 3:
                return
            chain = self.points[self.ratio_selected_idx].copy()
            self._draw_selection(self.ratio_selected_idx)
            #
            self._highlight_ratio_pair(chain, k - 1, k)
        self.canvas.draw_idle()

    #
    def open_json(self):
        p = filedialog.askopenfilename(filetypes=[('JSON', '*.json'), ('All', '*.*')])
        if not p: return
        self.load_json(Path(p))

    def load_json(self, json_path: Path):
        self.img_path, self.points, self.center, self.dead, self.srch, self.preproc = load_input(json_path)
        self.img_arr = None
        self._img_shape = None
        self._full_view_bounds = None
        if self.img_path is not None:
            try:
                self.img_arr = load_grayscale_with_preproc(self.img_path, self.preproc)
            except Exception as exc:
                messagebox.showerror('Error', f'Failed to prepare the image:\n{exc}')
                self.img_arr = None
        self.clear_selection(redraw=False)
        self._reset_zoom_state()
        self.draw_base()
        self.status.config(text=f'Loaded: {json_path.name}')
        self._flash_right_scroll()

    def save_png(self):
        if self.img_path is None:
            messagebox.showinfo('Save', 'No data loaded.');
            return
        p = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])
        if not p: return
        self.fig.savefig(p, dpi=150)
        self.status.config(text=f'Saved: {Path(p).name}')

    def draw_base(self):
        self.ax.clear()
        img_shape = None
        if self.img_arr is not None:
            img_shape = self.img_arr.shape[:2]
            self.ax.imshow(self.img_arr, cmap='gray', interpolation='nearest')
        elif self.img_path:
            im = Image.open(self.img_path).convert('L')
            img_arr = np.array(im)
            img_shape = img_arr.shape[:2]
            self.ax.imshow(img_arr, cmap='gray', interpolation='nearest')
        self._img_shape = img_shape
        if self.points is not None and len(self.points):
            self.ax.scatter(self.points[:, 1], self.points[:, 0],
                            s=24, c='cyan', edgecolors='black', linewidths=0.4, label='peaks')
        if self.center is not None:
            cy, cx = self.center
            self.ax.scatter([cx], [cy], s=40, c='red', marker='o', label='center')
            if self.dead and self.dead > 0:
                self.ax.add_patch(Circle((cx, cy), self.dead, fill=False, ec='red', ls='--', lw=1.5))
            if self.srch and self.srch > 0:
                self.ax.add_patch(Circle((cx, cy), self.srch, fill=False, ec='red', ls=':', lw=1.0))
        self._update_full_view_bounds()
        self.ax.axis('off')
        self._draw_polygons()
        self._apply_zoom()
        self.canvas.draw_idle()

    def _draw_anchor(self, idx: int):
        y, x = self.points[idx]
        self.ax.scatter([x], [y], s=52, c='yellow', edgecolors='k', linewidths=0.6, zorder=4)
        self.canvas.draw_idle()

    def _draw_selection(self, idxs: List[int]):
        if not idxs: return
        sel = self.points[idxs]
        self.ax.scatter(sel[:, 1], sel[:, 0], s=36, c='magenta', edgecolors='k', linewidths=0.6, zorder=3)
        for i in range(len(sel) - 1):
            y1, x1 = sel[i];
            y2, x2 = sel[i + 1]
            self.ax.plot([x1, x2], [y1, y2], color='yellow', lw=1.8, ls='--', zorder=2)
            self.ax.text(x1, y1, str(i + 1), color='magenta', fontsize=8, ha='right', va='bottom')
        yN, xN = sel[-1]
        self.ax.text(xN, yN, str(len(sel)), color='magenta', fontsize=8, ha='right', va='bottom')
        self.canvas.draw_idle()

    def _clear_rubber(self):
        if self.rubber_line is not None:
            try:
                self.rubber_line.remove()
            except Exception:
                pass
            self.rubber_line = None
            self.canvas.draw_idle()

    def _clear_rubber_ratio(self):
        if self.rubber_line_ratio is not None:
            try:
                self.rubber_line_ratio.remove()
            except Exception:
                pass
            self.rubber_line_ratio = None
            self.canvas.draw_idle()

    def _clear_polygon_rubber(self):
        if self.polygon_rubber_line is not None:
            try:
                self.polygon_rubber_line.remove()
            except Exception:
                pass
            self.polygon_rubber_line = None
            self.canvas.draw_idle()

    def clear_selection(self, redraw: bool = True):
        #
        self.selected_idx.clear()
        self.anchor_idx = None
        self._clear_rubber()
        self.curr_chain = None
        self.curr_seg = None
        self.curr_labels = None
        self.curr_ratio = float('nan')
        #
        self.ratio_selected_idx.clear()
        self.ratio_anchor_idx = None
        self._clear_rubber_ratio()
        #
        self.polygon_current_idx.clear()
        self.polygons_idx.clear()
        self._clear_polygon_rubber()
        self._reset_polygon_history()
        self._last_analysis_mode = None
        #
        self.lst.delete(0, tk.END)
        self.list_index_map.clear()
        self.lst_header.config(text='Found words (Fibonacci subsegments)')
        self.lbl_ratio.config(text='Average L/S along chain: —')
        self.lbl_ratio_neigh.config(text='Average neighboring segment ratio: —')
        self.lbl_ratio_polygons.config(text='Average neighboring polygon linear ratio: —')
        self.txt_sl.delete('1.0', tk.END)
        self.txt_words.configure(state='normal');
        self.txt_words.delete('1.0', tk.END);
        self.txt_words.configure(state='disabled')
        if redraw: self.draw_base()

    def _capture_polygon_state(self) -> Tuple[List[int], List[List[int]]]:
        return (
            list(self.polygon_current_idx),
            [list(poly) for poly in self.polygons_idx],
        )

    def _reset_polygon_history(self) -> None:
        self._polygon_history = [self._capture_polygon_state()]
        self._polygon_redo.clear()

    def _record_polygon_state(self) -> None:
        state = self._capture_polygon_state()
        if self._polygon_history and state == self._polygon_history[-1]:
            return
        self._polygon_history.append(state)
        self._polygon_redo.clear()

    def _restore_polygon_state(self, state: Tuple[List[int], List[List[int]]]) -> None:
        current, finished = state
        self.polygon_current_idx = list(current)
        self.polygons_idx = [list(poly) for poly in finished]
        self._clear_polygon_rubber()

    def _on_polygon_undo(self, event):
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)):
            return
        if len(self._polygon_history) <= 1:
            return "break"
        state = self._polygon_history.pop()
        self._polygon_redo.append(state)
        prev_state = self._polygon_history[-1]
        self._restore_polygon_state(prev_state)
        self._refresh_canvas_after_polygon()
        self.status.config(text='Last polygon construction action undone.')
        return "break"

    def _on_polygon_redo(self, event):
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)):
            return
        if not self._polygon_redo:
            return "break"
        state = self._polygon_redo.pop()
        self._polygon_history.append(state)
        self._restore_polygon_state(state)
        self._refresh_canvas_after_polygon()
        self.status.config(text='Polygon construction action redone.')
        return "break"

    #

    def _collect_points_along_segment(self, i0: int, i1: int, max_dist: float) -> List[int]:
        """Indices of points within max_dist from segment p0->p1, ordered by projection."""
        p0 = self.points[i0][[1, 0]]
        p1 = self.points[i1][[1, 0]]
        v = p1 - p0
        vv = float(np.dot(v, v))
        if vv == 0: return [i0]
        idx = []
        for k, (y, x) in enumerate(self.points):
            w = np.array([x, y]) - p0
            t = float(np.dot(w, v) / vv)
            if 0.0 <= t <= 1.0:
                proj = p0 + t * v
                dist = float(np.hypot(x - proj[0], y - proj[1]))
                if dist <= max_dist:
                    idx.append((t, k))
        idx.sort(key=lambda z: z[0])
        chain = [k for t, k in idx]
        #
        if chain and chain[0] != i0:
            if i0 in chain: chain.remove(i0)
            chain.insert(0, i0)
        if chain and chain[-1] != i1:
            if i1 in chain: chain.remove(i1)
            chain.append(i1)
        #
        seen = set();
        out = []
        for k in chain:
            if k not in seen:
                out.append(k);
                seen.add(k)
        return out

    #

    def _on_sl_keypress(self, event):
        """Invert selected L/S with keys: 'i', 'sh', 'Sh'."""
        if event.char not in ('i', 'I', 'ш', 'Ш'):
            return
        try:
            start = self.txt_sl.index("sel.first")
            end = self.txt_sl.index("sel.last")
        except tk.TclError:
            return "break"
        segment = self.txt_sl.get(start, end)
        flipped = ''.join('S' if ch == 'L' else ('L' if ch == 'S' else ch) for ch in segment)
        self.txt_sl.delete(start, end)
        self.txt_sl.insert(start, flipped)
        #
        self._recompute_words_from_manual_SL()
        return "break"

    def _set_sl_text(self, s: str):
        self.txt_sl.delete('1.0', tk.END)
        self.txt_sl.insert('1.0', s)

    def _get_sl_text_letters(self) -> List[str]:
        raw = self.txt_sl.get('1.0', tk.END)
        return [ch for ch in raw if ch in ('L', 'S')]

    #

    def run_analysis(self):
        if self.points is None or len(self.points) < 2:
            messagebox.showinfo('Analysis', 'Not enough points (need ≥ 2).');
            return
        if len(self.selected_idx) < 2:
            messagebox.showinfo('Analysis', 'Select two points first in Chain mode.');
            return

        chain = self.points[self.selected_idx].copy()
        seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)

        labels, Slen, Llen, sidx, lidx = cluster_lengths(seg)
        SL = ['S' if labels[i] == sidx else 'L' for i in range(len(seg))]
        ratio = (Llen / Slen) if (Slen and not math.isnan(Slen)) else float('nan')

        self.curr_chain = chain
        self.curr_seg = seg
        self.curr_labels = SL[:]
        self.curr_ratio = ratio

        self._set_sl_text(''.join(SL))
        self._recompute_words_and_redraw()
        self._last_analysis_mode = 'sl'

    def _recompute_words_from_manual_SL(self):
        if self.curr_chain is None or self.curr_seg is None:
            return
        SL = self._get_sl_text_letters()
        m = len(self.curr_seg)
        if len(SL) < m: SL = SL + ['S'] * (m - len(SL))
        if len(SL) > m: SL = SL[:m]
        self.curr_labels = SL
        self._recompute_words_and_redraw()

    def _highlight_word(self, chain: np.ndarray, SL: List[str], i0: int, n: int):
        #
        #
        #
        for k in range(i0, i0 + n):
            y1, x1 = chain[k];
            y2, x2 = chain[k + 1]
            self.ax.plot([x1, x2], [y1, y2], color='lime', lw=3.2)
            my, mx = (y1 + y2) / 2, (x1 + x2) / 2
            self.ax.text(mx, my, SL[k], color='red', fontsize=9, ha='center', va='center')
        yN, xN = chain[i0 + n]
        self.ax.text(xN, yN, str(n), color='white', fontsize=8, ha='right', va='bottom')

    def _recompute_words_and_redraw(self):
        chain = self.curr_chain
        SL = self.curr_labels
        ratio = self.curr_ratio

        #
        found: List[Tuple[int, int, str, int, int]] = []
        fibNs = [n for n in fib_list_upto(len(SL)) if n >= 3]
        for n in fibNs:
            fibs = fib_list_upto(n)
            k = len(fibs) - 1
            exp1 = (fibs[k - 1], fibs[k - 2]) if k >= 2 else (1, 0)
            for i in range(0, len(SL) - n + 1):
                sub = SL[i:i + n]
                Lc, Sc = sub.count('L'), sub.count('S')
                if (Lc, Sc) == exp1:
                    found.append((n, i, ''.join(sub), Lc, Sc))

        #
        self.draw_base()
        self._draw_selection(self.selected_idx)
        if found:
            found.sort(key=lambda z: (-z[0], z[1]))
            n, i0, word, Lc, Sc = found[0]
            self._highlight_word(chain, SL, i0, n)
            self.ax.text(0.01, 0.02,
                         f'Chain: L/S≈{ratio:.3f}  | Best word: n={n}, L={Lc}, S={Sc}',
                         transform=self.ax.transAxes, color='lime', fontsize=10,
                         ha='left', va='bottom')
        else:
            self.ax.text(0.01, 0.02, f'Chain: L/S≈{ratio:.3f}. No matches (n≥3) found.',
                         transform=self.ax.transAxes, color='orange', fontsize=10,
                         ha='left', va='bottom')
        self.canvas.draw_idle()

        #
        self.lst.delete(0, tk.END)
        self.list_index_map.clear()
        self.lst_header.config(text='Found words (Fibonacci subsegments)')
        groups: Dict[int, List[Tuple[int, int, str, int, int]]] = {}
        for entry in found:
            groups.setdefault(entry[0], []).append(entry)
        row = 0
        if groups:
            for n in sorted(groups.keys()):
                self.lst.insert(tk.END, f'— n={n} —');
                row += 1
                for (n_, i0, word, Lc, Sc) in groups[n]:
                    self.lst.insert(tk.END, f'  i={i0}  word={word}  L={Lc} S={Sc}')
                    self.list_index_map[row] = ('sl', i0, n)
                    row += 1
                self.lst.insert(tk.END, '');
                row += 1
        else:
            self.lst.insert(tk.END, 'No matches (n≥3)')

        #
        if math.isfinite(ratio):
            self.lbl_ratio.config(text=f'Average L/S along chain: {ratio:.3f}')
        else:
            self.lbl_ratio.config(text='Average L/S along chain: —')
        self.lbl_ratio_neigh.config(text='Average neighboring segment ratio: —')

        #
        self.txt_words.configure(state='normal')
        self.txt_words.delete('1.0', tk.END)
        max_len_ref = max(groups.keys(), default=min(len(SL), 34))
        for w in gen_fibonacci_words(max_len=max_len_ref, start='L'):
            self.txt_words.insert(tk.END, f'len={len(w)} → {w}\n')
        self.txt_words.configure(state='disabled')

        self.status.config(text=f'Selected points (Chain mode): {len(chain)}. Segments: {len(SL)}. '
                                f'Words (n≥3): {sum(len(v) for v in groups.values())}.')

    #

    def _highlight_ratio_pair(self, chain: np.ndarray, seg_a: int, seg_b: int):
        """Highlight two neighboring segments by their indices (0..M-1)."""
        M = len(chain) - 1
        if not (0 <= seg_a < M and 0 <= seg_b < M):
            return
        for k in (seg_a, seg_b):
            y1, x1 = chain[k];
            y2, x2 = chain[k + 1]
            self.ax.plot([x1, x2], [y1, y2], color='lime', lw=3.2)
            my, mx = (y1 + y2) / 2, (x1 + x2) / 2
            self.ax.text(mx, my, f'{k + 2}-{k + 1}', color='red', fontsize=9, ha='center', va='center')

    def run_ratio_analysis(self):
        """Analysis for the chain selected in Ratio mode: segment names 'n-(n-1)', neighboring ratios, mean value."""
        if self.points is None or len(self.points) < 2:
            messagebox.showinfo('Analysis', 'Not enough points.');
            return
        if len(self.ratio_selected_idx) < 3:
            messagebox.showinfo('Analysis', 'Need ≥ 3 points in Ratio mode.');
            return

        chain = self.points[self.ratio_selected_idx].copy()
        seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)

        #
        self.draw_base()
        self._draw_selection(self.ratio_selected_idx)
        for i in range(len(chain) - 1):
            y1, x1 = chain[i];
            y2, x2 = chain[i + 1]
            my, mx = (y1 + y2) / 2, (x1 + x2) / 2
            label = f"{i + 2}-{i + 1}"
            self.ax.text(mx, my, label, color='yellow', fontsize=9, ha='center', va='center')
        self.canvas.draw_idle()

        #
        ratios = []
        for i in range(1, len(seg)):
            if seg[i - 1] > 0:
                ratios.append(seg[i] / seg[i - 1])
            else:
                ratios.append(float('nan'))

        #
        self.lst.delete(0, tk.END)
        self.list_index_map.clear()
        self.lst_header.config(text='Neighboring segment ratios (Ratio mode)')
        row = 0
        if len(ratios) == 0:
            self.lst.insert(tk.END, 'Not enough segments for ratios.')
        else:
            for i, r in enumerate(ratios, start=2):
                self.lst.insert(tk.END, f'  ({i + 1}-{i}) / ({i}-{i - 1})  ≈  {r:.6g}')
                k = i - 1
                self.list_index_map[row] = ('ratio', k)
                row += 1

        #
        finite = [r for r in ratios if math.isfinite(r)]
        mean_ratio = float(np.mean(finite)) if finite else float('nan')
        if math.isfinite(mean_ratio):
            self.lbl_ratio_neigh.config(text=f'Average neighboring segment ratio: {mean_ratio:.6g}')
        else:
            self.lbl_ratio_neigh.config(text='Average neighboring segment ratio: —')

        #
        self.status.config(
            text=f'Selected points (Ratio mode): {len(chain)}. Segments: {len(seg)}. Ratios: {len(ratios)}.')
        self._last_analysis_mode = 'ratio'

    #

    def _draw_polygons(self):
        """Draw finished polygons and the current one."""
        if self.points is None or (not self.polygons_idx and not self.polygon_current_idx):
            return

        #
        for num, idxs in enumerate(self.polygons_idx, start=1):
            if len(idxs) < 3:
                continue
            pts = self.points[idxs]
            xs = pts[:, 1]
            ys = pts[:, 0]
            self.ax.fill(xs, ys, facecolor='deepskyblue', alpha=0.25,
                         edgecolor='blue', linewidth=1.4, zorder=1.5)
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            self.ax.text(cx, cy, f'P{num}', color='navy', fontsize=9,
                         ha='center', va='center', zorder=1.6)

        #
        if self.polygon_current_idx:
            pts_cur = self.points[self.polygon_current_idx]
            xs = pts_cur[:, 1]
            ys = pts_cur[:, 0]
            self.ax.plot(xs, ys, color='orange', lw=2.2, zorder=3.1)
            self.ax.scatter(xs, ys, s=46, c='orange', edgecolors='k', linewidths=0.6, zorder=3.2)
            first_x, first_y = xs[0], ys[0]
            self.ax.scatter([first_x], [first_y], s=70, facecolors='none', edgecolors='orange',
                            linewidths=1.5, zorder=3.3)
            if len(xs) >= 2:
                self.ax.plot([xs[-1], first_x], [ys[-1], first_y], color='orange', lw=1.2, ls=':', zorder=3.0)
            for idx, (xv, yv) in enumerate(zip(xs, ys), start=1):
                self.ax.text(xv, yv, str(idx), color='orange', fontsize=8,
                             ha='right', va='bottom', zorder=3.4)

    def _handle_polygon_click(self, point_idx: int):
        """Handle a left-click to build a polygon while Polygon mode is active."""
        if point_idx < 0 or point_idx >= len(self.points):
            return

        if not self.polygon_current_idx:
            self.polygon_current_idx.append(point_idx)
            self._refresh_canvas_after_polygon()
            self.status.config(text=f'Polygon construction: first vertex selected (#{point_idx + 1}).')
            self._record_polygon_state()
            return

        first_idx = self.polygon_current_idx[0]
        if point_idx == first_idx:
            if len(self.polygon_current_idx) < 3:
                self.status.config(text='Polygon requires ≥ 3 unique points.')
                return
            #
            self.polygons_idx.append(self.polygon_current_idx.copy())
            poly_num = len(self.polygons_idx)
            vertex_count = len(self.polygon_current_idx)
            self.polygon_current_idx.clear()
            self._clear_polygon_rubber()
            self._refresh_canvas_after_polygon()
            self.status.config(text=f'Polygon #{poly_num} closed. Vertices: {vertex_count}.')
            self._record_polygon_state()
            return

        if point_idx in self.polygon_current_idx:
            self.status.config(text='Vertex already added. Choose another point or close the polygon.')
            return

        self.polygon_current_idx.append(point_idx)
        self._refresh_canvas_after_polygon()
        self.status.config(text=f'Polygon construction: total vertices {len(self.polygon_current_idx)}.')
        self._record_polygon_state()

    def _refresh_canvas_after_polygon(self):
        """Redraw the image, taking current modes and polygons into account."""
        if self._last_analysis_mode == 'sl' and self.curr_chain is not None:
            self._recompute_words_and_redraw()
        elif self._last_analysis_mode == 'ratio' and len(self.ratio_selected_idx) >= 3:
            self.run_ratio_analysis()
        else:
            self.draw_base()
        self.polygon_rubber_line = None

    def _polygon_area(self, idxs: List[int]) -> float:
        if self.points is None or len(idxs) < 3:
            return 0.0
        pts = self.points[idxs]
        xs = pts[:, 1]
        ys = pts[:, 0]
        shifted_x = np.roll(xs, -1)
        shifted_y = np.roll(ys, -1)
        area = 0.5 * abs(float(np.dot(xs, shifted_y) - np.dot(ys, shifted_x)))
        return area

    def _on_enter_key(self, event):
        """Compute the areas of the selected polygons and their ratios when Enter is pressed."""
        #
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)):
            return

        if self.polygon_current_idx:
            messagebox.showinfo('Polygons', 'Finish the current polygon (return to the first point).')
            return
        if len(self.polygons_idx) < 2:
            messagebox.showinfo('Polygons', 'Need at least two completed polygons.')
            return

        areas = [self._polygon_area(poly) for poly in self.polygons_idx]
        self.lst.delete(0, tk.END)
        self.list_index_map.clear()
        self.lst_header.config(text='Polygons (Polygon mode) — areas and ratios')

        for i, area in enumerate(areas, start=1):
            self.lst.insert(tk.END, f'Polygon area {i}: {area:.6g}')

        self.lst.insert(tk.END, '')
        lines_added = False
        polygon_linear_ratios = []
        for idx in range(len(areas), 1, -1):
            prev_area = areas[idx - 2]
            curr_area = areas[idx - 1]
            if prev_area == 0:
                linear_text = 'undefined (previous area = 0)'
            else:
                size_ratio = curr_area / prev_area
                ratio_text = f'{size_ratio:.6g}'
                if size_ratio > 0:
                    linear_ratio = math.sqrt(size_ratio)
                    polygon_linear_ratios.append(linear_ratio)
                    linear_text = f'{linear_ratio:.6g}'
                else:
                    linear_text = 'undefined (ratio ≤ 0)'
            self.lst.insert(tk.END, f'Linear size ratio {idx} and {idx - 1}: {linear_text}')
            lines_added = True
        #
        if not lines_added:
            self.lst.insert(tk.END, 'Not enough polygons for ratios.')
        #
        finite_linear = [r for r in polygon_linear_ratios if math.isfinite(r)]
        if finite_linear:
            mean_linear = float(np.mean(finite_linear))
            self.lbl_ratio_polygons.config(text=f'Average neighboring polygon linear ratio: {mean_linear:.6g}')
        else:
            self.lbl_ratio_polygons.config(text='Average neighboring polygon linear ratio: —')
        #
        self.status.config(text=f'Areas computed: {len(areas)}. See the list on the right.')


class App(tk.Tk):
    """Standalone wrapper compatible with the previous CLI."""

    def __init__(self):
        super().__init__()
        self.title('fibonachi_analysis')
        self.geometry('1520x980')
        self.resizable(True, True)
        frame = FibonacciAnalysisFrame(self)
        frame.pack(fill=tk.BOTH, expand=True)
        self.frame = frame


if __name__ == '__main__':
    app = App()
    app.mainloop()