from __future__ import annotations
import sys, json, math, argparse
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

# Импортируем всю логику из нового файла
from fibonachi_analysis_utils import *
# Импортируем GUI-утилиты
from analysis_gui_utils import AnalysisConfirmationDialog, _Tooltip, HoverTooltip


if not hasattr(tk, "Notebook") and hasattr(ttk, "Notebook"):
    tk.Notebook = ttk.Notebook


# Эта функция используется в __main__, поэтому остается здесь
def _parse_cli(argv=None):
    p = argparse.ArgumentParser(description="fibonachi_analysis — load input data")
    p.add_argument("--payload", type=str, default=None, help="Path to fibo_input.json")
    p.add_argument("--image", type=str, default=None, help="Path to the image (fallback)")
    p.add_argument("--points", type=str, default=None, help="Path to JSON with points (fallback)")
    return p.parse_args(argv)


class FibonacciAnalysisFrame(tk.Frame):
    def __init__(self, master: tk.Misc, controller=None, auto_load: bool = True, license_manager=None):
        super().__init__(master)
        self.controller = controller
        self.license_manager = license_manager # License manager is kept but not used to restrict modes

        # Core data
        self.img_path: Optional[Path] = None
        self.points: Optional[np.ndarray] = None
        self.center: Optional[Tuple[float, float]] = None
        self.dead: float = 0.0
        self.srch: float = 0.0
        self.preproc: PreprocSettings = PreprocSettings(mode="raw")
        self.img_arr: Optional[np.ndarray] = None
        self.fibo_input_path: Optional[Path] = None

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
        # --- ИЗМЕНЕНИЕ: Убран 'sl', 'ratio' теперь по умолчанию ---
        self.analysis_mode: str = 'ratio'
        # --- КОНЕЦ ---
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

    # --- NEW: Session Save/Load (Без изменений) ---
    def get_state(self) -> Dict[str, Any]:
        return {
            "fibo_input_path": str(self.fibo_input_path) if self.fibo_input_path else None,
            "permanent_analyses": self.permanent_analyses,
            "active_analysis_idx": self.active_analysis_idx,
            "zoom_val": self.zoom_val,
            "view_cx": self.view_cx,
            "view_cy": self.view_cy,
            "max_dist_line": self.max_dist_line,
        }

    def set_state(self, state: Dict[str, Any]):
        fibo_input_path = state.get("fibo_input_path")
        if not fibo_input_path:
            self.permanent_analyses = []
            self.active_analysis_idx = None
            self.img_arr = None; self.img_path = None; self.points = None; self.center = None
            self.dead = 0.0; self.srch = 0.0; self.fibo_input_path = None; self.pending_analysis = None
            self._reject_pending_analysis(ask_user=False)
            self._redraw_canvas(); self._update_display_for_active_analysis()
            if hasattr(self, 'entBand'): self.entBand.delete(0, 'end'); self.entBand.insert(0, str(int(20.0)))
            self._set_status("Analysis tab cleared.")
            return
        try:
            self.load_json(Path(fibo_input_path))
            self.permanent_analyses = state.get("permanent_analyses", [])
            saved_active_idx = state.get("active_analysis_idx")
            if saved_active_idx is not None and 0 <= saved_active_idx < len(self.permanent_analyses):
                self.active_analysis_idx = saved_active_idx
            else: self.active_analysis_idx = None if not self.permanent_analyses else len(self.permanent_analyses) - 1
            self.zoom_val = state.get("zoom_val", 0); self.view_cx = state.get("view_cx"); self.view_cy = state.get("view_cy")
            self.max_dist_line = state.get("max_dist_line", 20.0)
            if hasattr(self, 'zoom_var'): self.zoom_var.set(self.zoom_val)
            if hasattr(self, 'entBand'): self.entBand.delete(0, 'end'); self.entBand.insert(0, str(int(self.max_dist_line)))
            self._redraw_canvas(); self._update_display_for_active_analysis(); self._flash_right_scroll()
            if self.controller: self.controller.set_status(f"Restored analysis session from {Path(fibo_input_path).name}")
        except Exception as e:
            messagebox.showerror("Analysis Load Error", f"Failed to restore analysis state:\n{e}")
            self.set_state({})


    def _build_ui(self):
        container = tk.Frame(self); container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1); container.columnconfigure(1, weight=0, minsize=470)
        left = tk.Frame(container); left.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=12)
        left.rowconfigure(0, weight=1); left.columnconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(9.6, 6.6)); self.ax = self.fig.add_subplot(111); self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=left); self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

        right_host = tk.Frame(container); right_host.grid(row=0, column=1, sticky="ns", pady=12)
        right_host.grid_columnconfigure(0, weight=1); right_host.grid_rowconfigure(0, weight=1)
        scroll_host = ttk.Frame(right_host); scroll_host.grid(row=0, column=0, sticky="nsew")
        scroll_host.grid_columnconfigure(0, weight=1); scroll_host.grid_rowconfigure(0, weight=1)
        canvas = tk.Canvas(scroll_host, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=canvas.yview)
        right = tk.Frame(canvas); self._right_scroll_canvas = canvas
        self._right_scroll_window = canvas.create_window((0, 0), window=right, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y); right.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(self._right_scroll_window, width=e.width))
        right.bind("<Enter>", self._activate_right_scroll); right.bind("<Leave>", self._deactivate_right_scroll)
        canvas.bind("<Enter>", self._activate_right_scroll); canvas.bind("<Leave>", self._deactivate_right_scroll)
        right.columnconfigure(0, weight=1)

        controls = tk.Frame(right); controls.grid(row=0, column=0, sticky="ew", padx=6, pady=(0, 10))
        controls.columnconfigure(0, weight=1); controls.columnconfigure(1, weight=1)
        tk.Button(controls, text='Open JSON…', command=self.open_json).grid(row=0, column=0, sticky='ew', padx=4, pady=2)
        tk.Button(controls, text='Save PNG', command=self.save_png).grid(row=0, column=1, sticky='ew', padx=4, pady=2)
        tk.Label(controls, text='Selection thickness (px):').grid(row=2, column=0, sticky='w', padx=4, pady=(8, 2))
        self.entBand = tk.Spinbox(controls, from_=2, to=100, width=6, command=self._onBand)
        self.entBand.delete(0, 'end'); self.entBand.insert(0, str(int(self.max_dist_line)))
        self.entBand.grid(row=2, column=1, sticky='ew', padx=4, pady=(8, 2))

        mode_switch = tk.Frame(controls); mode_switch.grid(row=3, column=0, columnspan=2, sticky='ew', padx=4, pady=(8, 2))
        # --- ИЗМЕНЕНИЕ: Убрана кнопка 'sl', оставлены 2 колонки ---
        mode_switch.columnconfigure(0, weight=1) # Колонка для Ratio
        mode_switch.columnconfigure(1, weight=1) # Колонка для Polygon

        # УДАЛЕНО: self.mode_buttons['sl']

        self.mode_buttons['ratio'] = tk.Button(mode_switch, text='📊 Ratio', command=lambda: self._set_analysis_mode('ratio'))
        self.mode_buttons['ratio'].grid(row=0, column=0, sticky='ew', padx=(0, 2)) # В колонку 0
        self.mode_buttons['polygon'] = tk.Button(mode_switch, text='🔺 Polygon', command=lambda: self._set_analysis_mode('polygon'))
        self.mode_buttons['polygon'].grid(row=0, column=1, sticky='ew', padx=(2, 0)) # В колонку 1

        # --- ИЗМЕНЕНИЕ: Убран код отключения кнопок в триале ---
        # is_full_version = self.license_manager is None or self.license_manager.has_valid_license()
        # УДАЛЕНО: HoverTooltip(self.mode_buttons['sl'], ...)
        HoverTooltip(self.mode_buttons['ratio'], 'Neighbor ratios: use Left Click to select two endpoints.')
        HoverTooltip(self.mode_buttons['polygon'], 'Polygon analysis: add vertices with Left Click, close by clicking the first point.')
        # УДАЛЕНА проверка is_full_version и отключение кнопок
        # --- КОНЕЦ ---

        zoom_group = ttk.LabelFrame(controls, text='Scale'); zoom_group.grid(row=4, column=0, columnspan=2, sticky='ew', padx=4, pady=(8, 4))
        self.zoom_var = tk.DoubleVar(value=self.zoom_val)
        self.zoom_scale = ttk.Scale(zoom_group, from_=0, to=100, variable=self.zoom_var, command=self._on_zoom_change)
        self.zoom_scale.pack(fill=tk.X, padx=6, pady=(0, 4))
        self.zoom_hint = ttk.Label(zoom_group, anchor='w'); self.zoom_hint.pack(fill=tk.X, padx=6, pady=(0, 2))
        self._update_zoom_hint()

        self.status = tk.Label(right, text='', anchor='w'); self.status.grid(row=1, column=0, sticky='ew', padx=6, pady=(0, 10))

        self.lst_header = tk.Label(right, text='Select an analysis to view details'); self.lst_header.grid(row=2, column=0, sticky='w', padx=6, pady=(0, 2))
        self.results_notebook = tk.Notebook(right); self.results_notebook.grid(row=3, column=0, sticky='nsew', padx=6, pady=(0, 8))
        right.grid_rowconfigure(3, weight=1)

        tab_subsegments = ttk.Frame(self.results_notebook); tab_subsegments.grid_columnconfigure(0, weight=1); tab_subsegments.grid_rowconfigure(0, weight=1)
        # --- ИЗМЕНЕНИЕ: Убрана вкладка Fib-Words ---
        # tab_prefixes = ttk.Frame(self.results_notebook); tab_prefixes.grid_columnconfigure(0, weight=1); tab_prefixes.grid_rowconfigure(1, weight=1)
        self.results_notebook.add(tab_subsegments, text='Details')
        # УДАЛЕНО: self.results_notebook.add(tab_prefixes, ...)
        # --- КОНЕЦ ---

        self.lst = tk.Listbox(tab_subsegments, width=66, height=15); self.lst.grid(row=0, column=0, sticky='nsew')
        self.lst.bind('<<ListboxSelect>>', self._on_list_select)

        info_frame = tk.Frame(tab_subsegments); info_frame.grid(row=1, column=0, sticky='ew', pady=(6,0))
        # --- ИЗМЕНЕНИЕ: Убрана метка для L/S ratio ---
        # self.lbl_ratio = tk.Label(info_frame, text='Average L/S along chain: —')
        # self.lbl_ratio.pack(anchor='w')
        # --- КОНЕЦ ---
        self.lbl_ratio_neigh = tk.Label(info_frame, text='Average neighboring segment ratio: —'); self.lbl_ratio_neigh.pack(anchor='w')
        self.lbl_ratio_polygons = tk.Label(info_frame, text='Average neighboring polygon linear ratio: —'); self.lbl_ratio_polygons.pack(anchor='w')

        # --- ИЗМЕНЕНИЕ: Убрано текстовое поле для S/L sequence ---
        # tk.Label(tab_subsegments, text='S/L sequence (full):').grid(row=2, column=0, sticky='w', pady=(4, 2))
        # self.txt_sl = tk.Text(tab_subsegments, height=4, wrap='word')
        # self.txt_sl.grid(row=3, column=0, sticky='ew', pady=(0, 4))
        # self.txt_sl.bind('<KeyPress>', self._on_sl_keypress)
        # --- КОНЕЦ ---

        # --- ИЗМЕНЕНИЕ: Убрана вся вкладка Fib-Words ---
        # prefixes_header = tk.Label(tab_prefixes, ...)
        # prefixes_frame = tk.Frame(tab_prefixes); ...
        # self.txt_words = tk.Text(prefixes_frame, ...); ...
        # scroll_words = tk.Scrollbar(prefixes_frame, ...); ...
        # --- КОНЕЦ ---

        self.bind_all('<Return>', self._on_enter_key)
        self.bind_all('<Delete>', self._on_delete_key)
        self.bind_all('<Escape>', self._on_escape_key)

        # --- ИЗМЕНЕНИЕ: Устанавливаем 'ratio' как начальный режим ---
        self._set_analysis_mode('ratio')
        # --- КОНЕЦ ---

    def _set_status(self, text: str):
        if hasattr(self, "status") and self.status.winfo_exists(): self.status.configure(text=text)
        if self.controller is not None and hasattr(self.controller, "set_status"):
            try: self.controller.set_status(f"Analysis: {text}")
            except Exception: pass

    def _initial_load(self):
        base = Path(getattr(sys, '_MEIPASS', Path(__file__).parent)) if getattr(sys, 'frozen', False) else Path(__file__).parent
        auto = find_default_json(base)
        if auto:
            try: self.load_json(auto); self._set_status(f'Loaded: {auto.name}')
            except Exception as e: messagebox.showerror('Load error', str(e))
        else: self._set_status('JSON not found. Select a file manually.')

    # --- Core Analysis Management (Без изменений) ---
    def _prompt_for_confirmation(self, analysis_data: Dict[str, Any]):
        self._reject_pending_analysis(ask_user=False)
        self.pending_analysis = analysis_data
        fig_widget = self.canvas.get_tk_widget()
        pos_data = analysis_data.get('dialog_pos', (0,0))
        x_display, y_display = self.ax.transData.transform(pos_data)
        x_screen = fig_widget.winfo_rootx() + x_display
        y_screen = fig_widget.winfo_rooty() + (fig_widget.winfo_height() - y_display)
        self.confirmation_dialog = AnalysisConfirmationDialog(
            self, accept_callback=self._accept_pending_analysis,
            reject_callback=lambda: self._reject_pending_analysis(ask_user=False),
            x=x_screen + 10, y=y_screen - 15,
        )
        self._redraw_canvas()

    def _accept_pending_analysis(self):
        if self.pending_analysis is None: return
        self.permanent_analyses.append(self.pending_analysis)
        self.active_analysis_idx = len(self.permanent_analyses) - 1
        if self.confirmation_dialog: self.confirmation_dialog.destroy(); self.confirmation_dialog = None
        self.pending_analysis = None; self._redraw_canvas(); self._update_display_for_active_analysis()
        self._set_status(f"Analysis #{self.active_analysis_idx + 1} saved.")

    def _reject_pending_analysis(self, ask_user=True):
        if self.pending_analysis is None: return
        if self.confirmation_dialog: self.confirmation_dialog.destroy(); self.confirmation_dialog = None
        self.pending_analysis = None; self._redraw_canvas(); self._update_display_for_active_analysis()
        self._set_status("Analysis discarded.")

    def _update_display_for_active_analysis(self):
        self.lst.delete(0, tk.END); self.list_index_map.clear()
        # УДАЛЕНО: self.lbl_ratio.config(...)
        self.lbl_ratio_neigh.config(text='Average neighboring segment ratio: —')
        self.lbl_ratio_polygons.config(text='Average neighboring polygon linear ratio: —')
        # УДАЛЕНО: self._set_sl_text("")
        # УДАЛЕНО: self.txt_words.configure(...)
        self.lst_header.config(text='Select an analysis to view details')

        if self.active_analysis_idx is None or self.active_analysis_idx >= len(self.permanent_analyses): return
        analysis = self.permanent_analyses[self.active_analysis_idx]
        analysis_type = analysis['type']

        # --- ИЗМЕНЕНИЕ: Убран вызов _populate_sl_info ---
        # if analysis_type == 'sl': self._populate_sl_info(analysis)
        # --- КОНЕЦ ---
        if analysis_type == 'ratio': self._populate_ratio_info(analysis)
        elif analysis_type == 'polygon': self._populate_polygon_info()

    # УДАЛЕНО: _populate_sl_info

    def _populate_ratio_info(self, analysis_data):
        self.lst_header.config(text='Neighboring segment ratios (Ratio mode)')
        self.list_index_map.clear(); self.lst.delete(0, tk.END)
        ratios = analysis_data['data'].get('ratios', [])
        mean_ratio = analysis_data['data'].get('mean_ratio', float('nan'))
        row = 0
        if not ratios: self.lst.insert(tk.END, 'Not enough segments for ratios.')
        else:
            for i, r in enumerate(ratios, start=1): # Start from i=1 (second segment index)
                # Text uses original point indices + 1 for user display
                entry_text = f'  ({i + 1}-{i}) / ({i}-{i - 1})  ≈  {r:.6g}'
                self.lst.insert(tk.END, entry_text)
                k = i # k is the index of the *second* segment (segment i+1 - i)
                self.list_index_map[row] = {'analysis_idx': self.active_analysis_idx, 'type': 'ratio', 'k': k}
                row += 1
        if math.isfinite(mean_ratio): self.lbl_ratio_neigh.config(text=f'Average neighboring segment ratio: {mean_ratio:.6g}')

    def _populate_polygon_info(self): # Без изменений
        self.lst_header.config(text='Polygons (Polygon mode) — areas and ratios')
        self.list_index_map.clear(); self.lst.delete(0, tk.END)
        polygon_analyses: List[Tuple[int, Dict]] = []
        for i, p in enumerate(self.permanent_analyses):
            if p['type'] == 'polygon': polygon_analyses.append((i, p))
        if not polygon_analyses: self.lst.insert(tk.END, 'No saved polygons to analyze.'); return
        areas = [p['data']['area'] for _, p in polygon_analyses]; row = 0
        for i, (analysis_idx, p) in enumerate(polygon_analyses, start=1):
            area = p['data']['area']; label = p['data']['label']
            self.lst.insert(tk.END, f'Polygon {label} (Area: {area:.6g})')
            self.list_index_map[row] = {'analysis_idx': analysis_idx, 'type': 'polygon'}; row += 1
        if len(areas) >= 2:
            self.lst.insert(tk.END, ''); self.list_index_map[row] = {'type': 'spacer'}; row += 1
            polygon_linear_ratios = []
            for i in range(1, len(areas)):
                prev_area, curr_area = areas[i-1], areas[i]
                prev_label = polygon_analyses[i-1][1]['data']['label']; curr_label = polygon_analyses[i][1]['data']['label']
                if prev_area > 0:
                    size_ratio = curr_area / prev_area
                    linear_ratio = math.sqrt(size_ratio) if size_ratio > 0 else 0
                    polygon_linear_ratios.append(linear_ratio)
                    self.lst.insert(tk.END, f'Lin. Ratio {curr_label}/{prev_label}: {linear_ratio:.6g}')
                else: self.lst.insert(tk.END, f'Lin. Ratio {curr_label}/{prev_label}: undefined')
                self.list_index_map[row] = {'type': 'info'}; row += 1
            finite_linear = [r for r in polygon_linear_ratios if math.isfinite(r) and r > 0]
            if finite_linear:
                mean_linear = float(np.mean(finite_linear))
                self.lbl_ratio_polygons.config(text=f'Average neighboring polygon linear ratio: {mean_linear:.6g}')

    # --- Event Handlers (Упрощено) ---

    def _on_click(self, event):
        if self.points is None or event.xdata is None or event.ydata is None: return
        if self.pending_analysis:
            if self.confirmation_dialog and self.confirmation_dialog.winfo_containing(event.x_root, event.y_root) == self.confirmation_dialog: return
            self._reject_pending_analysis(ask_user=False); return
        if event.button == 3: self._handle_right_click(event); return
        if event.button != 1: return
        x, y = float(event.xdata), float(event.ydata)
        d2 = (self.points[:, 1] - x) ** 2 + (self.points[:, 0] - y) ** 2
        j = int(np.argmin(d2))
        if math.sqrt(d2[j]) > self.pick_tol: return
        self._focus_on(float(self.points[j, 1]), float(self.points[j, 0]))

        # --- ИЗМЕНЕНИЕ: Убрана ветка 'sl' ---
        if self.analysis_mode == 'ratio':
            if self.anchor_idx is None:
                self.anchor_idx = j; self._redraw_canvas()
            else:
                indices = self._collect_points_along_segment(self.anchor_idx, j, self.max_dist_line)
                if len(indices) < 3: # Ratio needs at least 3 points
                    self._set_status(f"Not enough points found for ratio analysis.")
                    self.anchor_idx = None; self._redraw_canvas(); return
                self.run_ratio_analysis(indices)
                self.anchor_idx = None
        # --- КОНЕЦ ---
        elif self.analysis_mode == 'polygon':
            self._handle_polygon_click(j)

    def _handle_right_click(self, event): # Без изменений
        if not self.permanent_analyses or self.points is None: return
        x, y = float(event.xdata), float(event.ydata); min_dist_sq = float('inf'); best_idx = None
        for i, analysis in enumerate(self.permanent_analyses):
            pts = self.points[analysis['indices']]
            if analysis['type'] == 'polygon': center = pts.mean(axis=0); dist_sq = (center[0] - y)**2 + (center[1] - x)**2
            else: # chain or ratio
                dist_sq = float('inf')
                for k in range(len(pts) - 1):
                    p1_yx, p2_yx = pts[k], pts[k+1]; mid_yx = (p1_yx + p2_yx) / 2
                    d_sq = (mid_yx[0] - y)**2 + (mid_yx[1] - x)**2; dist_sq = min(dist_sq, d_sq)
            if dist_sq < min_dist_sq: min_dist_sq = dist_sq; best_idx = i
        if best_idx is not None and math.sqrt(min_dist_sq) < self.pick_tol * 3:
            if self.active_analysis_idx != best_idx:
                self.active_analysis_idx = best_idx; self._redraw_canvas(); self._update_display_for_active_analysis()
                self._set_status(f"Analysis #{best_idx + 1} is now active.")

    def _on_motion(self, event): # Без изменений
        if event.xdata is None or event.ydata is None or self.points is None: return
        if self.anchor_idx is not None:
            ax, ay = self.points[self.anchor_idx, [1, 0]]; bx, by = float(event.xdata), float(event.ydata)
            if self.rubber_line is None: self.rubber_line, = self.ax.plot([ax, bx], [ay, by], color='yellow', lw=2.0, alpha=0.9, zorder=5)
            else: self.rubber_line.set_data([ax, bx], [ay, by])
            self.canvas.draw_idle()
        elif self.polygon_current_idx:
            ax, ay = self.points[self.polygon_current_idx[-1], [1, 0]]; bx, by = float(event.xdata), float(event.ydata)
            if self.polygon_rubber_line is None: self.polygon_rubber_line, = self.ax.plot([ax, bx], [ay, by], color='orange', lw=2.2, alpha=0.8, zorder=3.4)
            else: self.polygon_rubber_line.set_data([ax, bx], [ay, by])
            self.canvas.draw_idle()
        else: self._clear_rubber_lines()

    def _on_list_select(self, event): self._redraw_canvas()
    def _on_enter_key(self, event):
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)): return
        if self.pending_analysis: self._accept_pending_analysis()
    def _on_delete_key(self, event): # Без изменений
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)): return
        if self.pending_analysis: self._reject_pending_analysis(ask_user=False); return
        if self.active_analysis_idx is not None and 0 <= self.active_analysis_idx < len(self.permanent_analyses):
            idx_del = self.active_analysis_idx; label = f"#{idx_del + 1}"
            if self.permanent_analyses[idx_del]['type'] == 'polygon': label = self.permanent_analyses[idx_del]['data'].get('label', label)
            if messagebox.askyesno("Delete Analysis", f"Delete analysis '{label}'?", parent=self):
                try: del self.permanent_analyses[idx_del]; self.active_analysis_idx = None; self._redraw_canvas(); self._update_display_for_active_analysis(); self._set_status(f"Deleted analysis '{label}'.")
                except IndexError: self._set_status("Error deleting analysis.")
            else: self._set_status("Deletion cancelled.")
        else: self._set_status("No analysis selected to delete.")
    def _on_escape_key(self, event): # Без изменений
        if self.pending_analysis: self._reject_pending_analysis(ask_user=False)
        elif self.anchor_idx is not None or self.polygon_current_idx:
            self.anchor_idx = None; self.polygon_current_idx.clear(); self._redraw_canvas(); self._set_status("Selection cancelled.")

    # --- Drawing Logic (Упрощено) ---

    def _redraw_canvas(self):
        self.ax.clear(); img_shape = None
        if self.img_arr is not None: img_shape = self.img_arr.shape[:2]; self.ax.imshow(self.img_arr, cmap='gray', interpolation='nearest')
        self._img_shape = img_shape
        if self.points is not None: self.ax.scatter(self.points[:, 1], self.points[:, 0], s=24, c='cyan', edgecolors='black', linewidths=0.4, label='peaks')
        if self.center is not None:
            cy, cx = self.center; self.ax.scatter([cx], [cy], s=40, c='red', marker='o')
            if self.dead > 0: self.ax.add_patch(Circle((cx, cy), self.dead, fill=False, ec='red', ls='--', lw=1.5))
            if self.srch > 0: self.ax.add_patch(Circle((cx, cy), self.srch, fill=False, ec='red', ls=':', lw=1.0))
        for i, analysis in enumerate(self.permanent_analyses): self._draw_one_analysis(analysis, is_active=(i == self.active_analysis_idx))
        if self.pending_analysis: self._draw_one_analysis(self.pending_analysis, is_pending=True)
        self._draw_list_selection_highlight()
        if self.anchor_idx is not None: y, x = self.points[self.anchor_idx]; self.ax.scatter([x], [y], s=52, c='yellow', edgecolors='k', linewidths=0.6, zorder=4)
        if self.polygon_current_idx: self._draw_polygon_construction()
        self._update_full_view_bounds(); self.ax.axis('off'); self._apply_zoom(); self.canvas.draw_idle()

    def _draw_one_analysis(self, analysis_data: Dict[str, Any], is_active: bool = False, is_pending: bool = False):
        if self.points is None: return
        analysis_type = analysis_data['type']
        if is_pending: color, ls, lw, zorder = 'lime', '-', 2.2, 3.0
        elif is_active: color, ls, lw, zorder = 'magenta', '--', 2.0, 2.5
        else: color, ls, lw, zorder = 'deepskyblue', ':', 1.8, 2.0
        style = {'color': color, 'ls': ls, 'lw': lw, 'zorder': zorder, 'active': is_active or is_pending}

        # --- ИЗМЕНЕНИЕ: Убрана ветка 'sl' ---
        if analysis_type == 'ratio': self._draw_one_analysis_chain(analysis_data, style)
        elif analysis_type == 'polygon': self._draw_one_analysis_polygon(analysis_data, style)
        # --- КОНЕЦ ---

    def _draw_one_analysis_chain(self, analysis_data: Dict[str, Any], style: Dict): # Теперь только для ratio
        indices = analysis_data['indices']; pts = self.points[indices]; color = style['color']
        self.ax.scatter(pts[:, 1], pts[:, 0], s=36, c=color, edgecolors='k', linewidths=0.6, zorder=style['zorder'] + 0.1)
        for i in range(len(pts) - 1):
            y1, x1 = pts[i]; y2, x2 = pts[i + 1]
            self.ax.plot([x1, x2], [y1, y2], color=style['color'], lw=style['lw'], ls=style['ls'], zorder=style['zorder'])
        if style['active']: # Рисуем только номера
            for i in range(len(pts)):
                yN, xN = pts[i]
                self.ax.text(xN, yN, str(i + 1), color=style['color'], fontsize=8, ha='right', va='bottom', zorder=style['zorder']+0.2)
            # Убрана отрисовка S/L

    def _draw_one_analysis_polygon(self, analysis_data: Dict[str, Any], style: Dict): # Без изменений
        indices = analysis_data['indices']; pts = self.points[indices]
        poly = MplPolygon(pts[:, ::-1], closed=True, facecolor=style['color'], alpha=0.25, edgecolor=style['color'], linewidth=style['lw'], ls=style['ls'], zorder=style['zorder'])
        self.ax.add_patch(poly)
        if style['active']: label = analysis_data['data']['label']; cy, cx = pts.mean(axis=0); self.ax.text(cx, cy, label, color=style['color'], fontsize=9, ha='center', va='center', zorder=style['zorder']+0.1)

    def _draw_list_selection_highlight(self):
        if not hasattr(self, 'lst'): return
        sel = self.lst.curselection()
        if not sel: return
        row = sel[0]; meta = self.list_index_map.get(row)
        if not meta or 'analysis_idx' not in meta: return
        analysis_idx = meta['analysis_idx']
        if analysis_idx >= len(self.permanent_analyses): return
        analysis_type = meta['type']; analysis_data = self.permanent_analyses[analysis_idx]

        # --- ИЗМЕНЕНИЕ: Убрана ветка 'sl' ---
        # if analysis_type == 'sl': self._highlight_word_V1(analysis_data, meta['i0'], meta['n'])
        # --- КОНЕЦ ---
        if analysis_type == 'ratio': self._highlight_ratio_pair_V1(analysis_data, meta['k']-1, meta['k']) # Adjust indices for ratio highlight
        elif analysis_type == 'polygon': self._highlight_polygon_V1(analysis_data)

    # УДАЛЕНО: _highlight_word_V1

    def _highlight_ratio_pair_V1(self, analysis_data: Dict, seg_idx_1: int, seg_idx_2: int): # seg_idx_1 is 0-based index of first segment
        if self.points is None: return
        indices = analysis_data['indices']; chain = self.points[indices]; M = len(chain)
        # Highlight requires 3 points (indices k, k+1, k+2) to show 2 segments
        if not (0 <= seg_idx_1 < M - 1 and 0 <= seg_idx_2 < M - 1 and seg_idx_2 == seg_idx_1 + 1): return

        k = seg_idx_1 # Index of the start point of the first segment
        # Highlight first segment (k to k+1)
        y1, x1 = chain[k]; y2, x2 = chain[k + 1]
        self.ax.plot([x1, x2], [y1, y2], color='lime', lw=3.2, zorder=10)
        my, mx = (y1 + y2) / 2, (x1 + x2) / 2
        # Label uses original point indices + 1
        self.ax.text(mx, my, f'{k + 2}-{k + 1}', color='red', fontsize=9, ha='center', va='center', zorder=10.1)

        # Highlight second segment (k+1 to k+2)
        if k + 2 < M:
            y1, x1 = chain[k + 1]; y2, x2 = chain[k + 2]
            self.ax.plot([x1, x2], [y1, y2], color='lime', lw=3.2, zorder=10)
            my, mx = (y1 + y2) / 2, (x1 + x2) / 2
            # Label uses original point indices + 1
            self.ax.text(mx, my, f'{k + 3}-{k + 2}', color='red', fontsize=9, ha='center', va='center', zorder=10.1)


    def _highlight_polygon_V1(self, analysis_data: Dict): # Без изменений
        if self.points is None: return
        indices = analysis_data['indices']; pts = self.points[indices]
        poly = MplPolygon(pts[:, ::-1], closed=True, fill=False, edgecolor='lime', linewidth=3.2, zorder=10)
        self.ax.add_patch(poly)

    def _draw_polygon_construction(self): # Без изменений
        if not self.polygon_current_idx or self.points is None: return
        pts_cur = self.points[self.polygon_current_idx]; xs, ys = pts_cur[:, 1], pts_cur[:, 0]
        self.ax.plot(xs, ys, color='orange', lw=2.2, zorder=3.1)
        self.ax.scatter(xs, ys, s=46, c='orange', edgecolors='k', linewidths=0.6, zorder=3.2)
        first_x, first_y = xs[0], ys[0]
        self.ax.scatter([first_x], [first_y], s=70, facecolors='none', edgecolors='orange', linewidths=1.5, zorder=3.3)
        if len(xs) >= 2: self.ax.plot([xs[-1], first_x], [ys[-1], first_y], color='orange', lw=1.2, ls=':', zorder=3.0)

    # --- Analysis Logic (Упрощено) ---

    # УДАЛЕНО: run_analysis (для 'sl')

    def run_ratio_analysis(self, indices: List[int]): # Без изменений в логике, только вызов
        chain = self.points[indices].copy()
        seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        ratios_for_display = [seg[i] / seg[i-1] if i > 0 and seg[i-1] > 0 else float('nan') for i in range(1, len(seg))]
        finite = [r for r in ratios_for_display if math.isfinite(r)]
        mean_ratio = float(np.mean(finite)) if finite else float('nan')
        analysis_data = {
            'type': 'ratio', 'indices': indices, 'dialog_pos': chain.mean(axis=0)[::-1].tolist(),
            'data': { 'ratios': ratios_for_display, 'mean_ratio': mean_ratio }
        }
        self._prompt_for_confirmation(analysis_data)
        self._set_status(f"Ratio analysis complete. Please confirm or reject.")

    def _handle_polygon_click(self, point_idx: int): # Без изменений в логике, только вызов
        if point_idx in self.polygon_current_idx:
            if point_idx == self.polygon_current_idx[0] and len(self.polygon_current_idx) >= 3:
                indices = self.polygon_current_idx.copy(); area = self._polygon_area(indices)
                poly_num = len([p for p in self.permanent_analyses if p['type'] == 'polygon']) + 1
                analysis_data = {
                    'type': 'polygon', 'indices': indices, 'dialog_pos': self.points[indices[-1]][::-1].tolist(),
                    'data': { 'area': area, 'label': f'P{poly_num}' }
                }
                self.polygon_current_idx.clear(); self._prompt_for_confirmation(analysis_data); self._set_status("Polygon closed. Please confirm or reject.")
            else: self._set_status("Vertex already added or polygon too small.")
            return
        self.polygon_current_idx.append(point_idx); self._redraw_canvas(); self._set_status(f"Polygon vertices: {len(self.polygon_current_idx)}. Click first point to close.")

    # --- Utility and Helper functions (Упрощено) ---

    def open_json(self):
        p = filedialog.askopenfilename(filetypes=[('JSON', '*.json'), ('All', '*.*')])
        if not p: return
        self.load_json(Path(p))

    def load_json(self, json_path: Path): # Без изменений
        self.fibo_input_path = json_path
        try:
            self.img_path, self.points, self.center, self.dead, self.srch, self.preproc = load_input(json_path)
            self.img_arr = None
            if self.img_path:
                try: self.img_arr = load_grayscale_with_preproc(self.img_path, self.preproc)
                except Exception as exc: messagebox.showerror('Error', f'Failed to prepare image:\n{exc}')
            self.permanent_analyses.clear(); self.pending_analysis = None; self.active_analysis_idx = None
            self.anchor_idx = None; self.polygon_current_idx.clear(); self._reject_pending_analysis(ask_user=False)
            self._reset_zoom_state(); self._redraw_canvas(); self._update_display_for_active_analysis()
            self._set_status(f'Loaded: {json_path.name}'); self._flash_right_scroll()
        except Exception as e:
             messagebox.showerror('Load Error', f'Failed to load analysis input file "{json_path.name}":\n{e}')
             self._set_status(f'Error loading {json_path.name}')


    def _polygon_area(self, idxs: List[int]) -> float: # Без изменений
        if self.points is None or len(idxs) < 3: return 0.0
        pts = self.points[idxs]; xs, ys = pts[:, 1], pts[:, 0]
        return 0.5 * abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))

    def _on_scroll(self, event): # Без изменений
        if event.xdata is None or event.ydata is None: return
        zoom_step = 5; self.zoom_val += zoom_step if event.button == 'up' else -zoom_step
        self.zoom_val = max(0.0, min(100.0, self.zoom_val)); self._on_zoom_change(self.zoom_val)
    def _on_zoom_change(self, value): # Без изменений
        self.zoom_val = max(0.0, min(100.0, float(value)))
        if hasattr(self, 'zoom_var'): self.zoom_var.set(self.zoom_val)
        self._apply_zoom(); self._update_zoom_hint(); self.canvas.draw_idle()
    def _activate_right_scroll(self, _event): # Без изменений
        if self._right_scroll_canvas: self._right_scroll_canvas.bind_all("<MouseWheel>", self._on_right_scroll_mousewheel); self._right_scroll_canvas.bind_all("<Button-4>", self._on_right_scroll_mousewheel); self._right_scroll_canvas.bind_all("<Button-5>", self._on_right_scroll_mousewheel)
    def _deactivate_right_scroll(self, _event): # Без изменений
        if self._right_scroll_canvas: self._right_scroll_canvas.unbind_all("<MouseWheel>"); self._right_scroll_canvas.unbind_all("<Button-4>"); self._right_scroll_canvas.unbind_all("<Button-5>")
    def _on_right_scroll_mousewheel(self, event): # Без изменений
        if self._right_scroll_canvas is None: return
        if getattr(event, 'delta', 0): self._right_scroll_canvas.yview_scroll(int(-event.delta / 120), 'units')
        elif getattr(event, 'num', None) == 4: self._right_scroll_canvas.yview_scroll(-1, 'units')
        elif getattr(event, 'num', None) == 5: self._right_scroll_canvas.yview_scroll(1, 'units')
    def _onBand(self): # Без изменений
        try: self.max_dist_line = max(2.0, float(self.entBand.get()))
        except Exception: pass

    def _set_analysis_mode(self, mode: str):
        # --- ИЗМЕНЕНИЕ: Убран 'sl', проверка лицензии удалена ---
        if mode not in ['ratio', 'polygon']: return
        # --- КОНЕЦ ---
        if self.pending_analysis: self._reject_pending_analysis(ask_user=False)
        self.analysis_mode = mode
        hints = {
            # 'sl': 'Chain mode: Left Click to select two endpoints.', # Удалено
            'ratio': 'Ratio mode: Left Click to select two endpoints.',
            'polygon': 'Polygon mode: Left Click to add vertices, close on the first point.',
        }
        self._set_status(hints.get(mode,""))
        for key, btn in self.mode_buttons.items():
             # Проверяем, существует ли кнопка перед конфигурацией
             if key in self.mode_buttons:
                 if btn.cget('state') != tk.DISABLED: btn.config(relief='sunken' if key == mode else 'raised')
        self.anchor_idx = None; self.polygon_current_idx.clear(); self._redraw_canvas()

    def _update_zoom_hint(self): # Без изменений
        if hasattr(self, 'zoom_hint'): value = int(round(self.zoom_val)); self.zoom_hint.config(text=f'Current zoom: {value}% (0 = full view)')
    def _update_full_view_bounds(self): # Без изменений
        if self._img_shape is not None: h, w = self._img_shape; self._full_view_bounds = (-0.5, w - 0.5, h - 0.5, -0.5)
    def _apply_zoom(self): # Без изменений
        if self._full_view_bounds is None: self._update_full_view_bounds()
        if self._full_view_bounds is None: return
        x0f, x1f, y0f, y1f = self._full_view_bounds
        if self.view_cx is None or self.view_cy is None: self.view_cx = (x0f + x1f)/2.0; self.view_cy = (y0f + y1f)/2.0
        if self.zoom_val <= 0: self.ax.set_xlim(x0f, x1f); self.ax.set_ylim(y0f, y1f); return
        w, h = x1f - x0f, y0f - y1f; cx, cy = self.view_cx, self.view_cy
        min_dim = min(w, h); L = max(50.0, min_dim - (min_dim - 50.0) * (self.zoom_val / 100.0))
        hw, hh = L/2.0, L/2.0; x0 = max(x0f, cx - hw); x1 = min(x1f, cx + hw); y1 = max(y1f, cy - hh); y0 = min(y0f, cy + hh)
        cw = x1 - x0; ch = y0 - y1
        if cw < L - 1e-6:
             if x0 == x0f: x1 = min(x1f, x0 + L)
             elif x1 == x1f: x0 = max(x0f, x1 - L)
        if ch < L - 1e-6:
             if y1 == y1f: y0 = min(y0f, y1 + L)
             elif y0 == y0f: y1 = max(y1f, y0 - L)
        self.ax.set_xlim(x0, x1); self.ax.set_ylim(y0, y1)
    def _reset_zoom_state(self): # Без изменений
        self.zoom_val = 0.0; self.view_cx, self.view_cy = None, None
        if self.points is not None and len(self.points) > 0: self.view_cy, self.view_cx = self.points.mean(axis=0)
        if hasattr(self, 'zoom_var'): self.zoom_var.set(0.0); self._update_zoom_hint()
    def _focus_on(self, x: float, y: float): # Без изменений
        self.view_cx, self.view_cy = float(x), float(y); self._apply_zoom(); self.canvas.draw_idle()
    def _flash_right_scroll(self): # Без изменений
        if self._right_scroll_canvas: self._right_scroll_canvas.yview_moveto(1.0); self._right_scroll_canvas.after(120, lambda: self._right_scroll_canvas.yview_moveto(0.0))
    def save_png(self): # Без изменений
        p = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])
        if p: self.fig.savefig(p, dpi=150); self._set_status(f'Saved: {Path(p).name}')
    def _clear_rubber_lines(self): # Без изменений
        removed = False
        if self.rubber_line: try: self.rubber_line.remove(); except Exception: pass; self.rubber_line = None; removed = True
        if self.polygon_rubber_line: try: self.polygon_rubber_line.remove(); except Exception: pass; self.polygon_rubber_line = None; removed = True
        # if removed: self.canvas.draw_idle() # Необязательно перерисовывать, если ничего не изменилось

    # УДАЛЕНО: _on_sl_keypress
    # УДАЛЕНО: _set_sl_text

    def _collect_points_along_segment(self, i0: int, i1: int, max_dist: float) -> List[int]: # Без изменений
        if self.points is None: return []
        p0, p1 = self.points[i0, [1, 0]], self.points[i1, [1, 0]]; v = p1 - p0; vv = float(np.dot(v, v))
        if vv < 1e-9: return [i0] if i0==i1 else [i0, i1]
        idx_t = []; pts_xy = self.points[:, ::-1]
        for k, p_xy in enumerate(pts_xy):
            w = p_xy - p0; t = float(np.dot(w, v) / vv)
            if t < 0: proj = p0
            elif t > 1: proj = p1
            else: proj = p0 + t * v
            dist_sq = (p_xy[0] - proj[0])**2 + (p_xy[1] - proj[1])**2
            if (0.0 <= t <= 1.0 and dist_sq <= max_dist**2) or k == i0 or k == i1:
                 t_val = 0.0 if k == i0 else (1.0 if k == i1 else t); idx_t.append((t_val, k))
        idx_t.sort(key=lambda z: z[0]); final_indices = []; seen = set()
        for _, k in idx_t:
            if k not in seen: final_indices.append(k); seen.add(k)
        return final_indices


class App(tk.Tk): # Без изменений
    def __init__(self):
        super().__init__(); self.title('fibonachi_analysis'); self.geometry('1520x980'); self.resizable(True, True)
        class MockLicense: def has_valid_license(self): return True
        frame = FibonacciAnalysisFrame(self, license_manager=MockLicense()); frame.pack(fill=tk.BOTH, expand=True); self.frame = frame

if __name__ == '__main__':
    app = App(); app.mainloop()