#!/usr/bin/env python3
# -*- coding: utf-8 -*-
(
    "SAED Symmetry – Launcher\n"
    "========================\n"
    "Uses centroid detection with post-filtering by intensity percentile.\n"
    "Geometric filters applied before intensity filter.\n"
    "NLM preprocessing.\n"
)
from __future__ import annotations
import json, subprocess, sys, cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
from scipy import ndimage as ndi # Нужен для center_of_mass

from percentile_utils import compute_percentile_map
from preproc import PreprocSettings, load_grayscale_with_preproc

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# -------------------------- Algorithm --------------------------
@dataclass
class CenterResult:
    cy: float
    cx: float
    method: str

def detect_spots_by_centroid(
    arr: np.ndarray,
    min_area: int = 3,
    # perc убран отсюда
    # Используем очень низкий перцентиль для бинаризации, чтобы найти почти все
    low_threshold_perc: float = 5.0
) -> np.ndarray:
    """
    Detects potential spot locations using centroiding of connected components
    above a very low threshold. Filters only by minimum area.
    Returns ALL potential spots with their intensity from the original array.
    """
    H, W = arr.shape

    # 1. Используем очень низкий порог для бинаризации
    # Можно использовать percentile_map или напрямую arr
    percent_map, _, _ = compute_percentile_map(arr) # Используем для порога
    try:
        low_thresh_val = float(np.percentile(percent_map, low_threshold_perc))
    except (ValueError, IndexError):
        low_thresh_val = 5.0 # Fallback

    # 2. Создаем бинарную маску
    binary_mask = np.where(percent_map >= low_thresh_val, 255, 0).astype(np.uint8)

    # 3. Находим все компоненты
    num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    kept = []
    # 4. Итерация по всем найденным объектам (кроме фона 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # 5. Фильтр только по минимальной площади
        if area < min_area:
            continue

        cx, cy = centroids[i] # Получаем x, y
        yi, xi = int(round(cy)), int(round(cx))

        # 6. Получаем реальную интенсивность из arr
        if 0 <= yi < H and 0 <= xi < W:
            v = float(arr[yi, xi])
            kept.append((float(cy), float(cx), float(v))) # Store as (y, x, v)

    # НЕ СОРТИРУЕМ и НЕ ОГРАНИЧИВАЕМ здесь
    return np.array(kept, dtype=float) if kept else np.zeros((0, 3), dtype=float)


def geometric_midpoint(arr: np.ndarray) -> CenterResult:
    H, W = arr.shape
    return CenterResult(cy=(H - 1) / 2.0, cx=(W - 1) / 2.0, method="midpoint")

def refine_center_antipodal(center: Tuple[float,float], pts: np.ndarray, tol_ang_deg: float=8.0, tol_rel_r: float=0.06, iters: int=3) -> CenterResult:
    cy, cx = float(center[0]), float(center[1])
    if len(pts) < 4:
        return CenterResult(cy=cy, cx=cx, method="midpoint (fallback)")
    for _ in range(max(0, int(iters))):
        dy = pts[:,0]-cy; dx = pts[:,1]-cx
        r = np.hypot(dx, dy)
        r_safe = np.where(r > 1e-9, r, 1e-9)
        u = np.column_stack((dx, dy)) / r_safe[:,None]
        cos_thr = -np.cos(np.deg2rad(180.0 - float(tol_ang_deg)))
        mids = []
        for i in range(len(pts)):
            if r[i] < 1e-6: continue
            dots = (u @ u[i])
            max_r_pair = np.maximum(r, r[i])
            rel_diff = np.divide(np.abs(r - r[i]), max_r_pair, out=np.zeros_like(r), where=max_r_pair > 1e-9)
            rad_ok = (rel_diff < float(tol_rel_r)) & (max_r_pair > 1e-9)
            ang_ok = (dots < cos_thr)
            valid_match = rad_ok & ang_ok & (np.arange(len(pts)) != i) & (r > 1e-6)
            idx = np.where(valid_match)[0]
            if idx.size == 0: continue
            j = idx[np.argmin(np.abs(dots[idx] + 1.0))]
            yi, xi = pts[i,0], pts[i,1]
            yj, xj = pts[j,0], pts[j,1]
            mids.append(((yi+yj)/2.0, (xi+xj)/2.0))
        if len(mids) < 4: break
        mids = np.array(mids, dtype=float)
        cy = float(np.median(mids[:,0])); cx = float(np.median(mids[:,1]))
    return CenterResult(cy=cy, cx=cx, method="antipodal-refined")

# -------------------------- GUI --------------------------
class SAEDLauncherFrame(ttk.Frame):
    ( "Launcher tab" )

    def __init__(self, master: tk.Misc, controller=None):
        super().__init__(master)
        self.controller = controller
        self._scroll_canvas = None
        self._scroll_window_id = None
        self._build_ui()

    def _get_default_output_path(self) -> str:
        if getattr(sys, "frozen", False): base_dir = Path(sys.executable).parent
        else:
            try: base_dir = Path.cwd()
            except OSError: base_dir = Path(__file__).parent
        base_name = "saed_results"; output_path = base_dir / base_name
        if not output_path.exists(): return str(output_path)
        counter = 1
        while True:
            new_name = f"{base_name}_{counter}"; new_path = base_dir / new_name
            if not new_path.exists(): return str(new_path)
            counter += 1
            if counter > 999: return str(base_dir / f"{base_name}_temp_{np.random.randint(1000)}")

    def _build_ui(self):
        outer = ttk.Frame(self); outer.pack(fill=tk.BOTH, expand=True)
        fixed = ttk.Frame(outer, padding=(16, 16, 16, 0)); fixed.pack(side=tk.TOP, fill=tk.X); fixed.grid_columnconfigure(0, weight=1)

        data_box = ttk.LabelFrame(fixed, text="Input data", padding=(12, 10, 12, 12)); data_box.grid(row=0, column=0, sticky="nsew"); data_box.grid_columnconfigure(1, weight=1)
        ttk.Label(data_box, text="Image:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.ent_img = ttk.Entry(data_box); self.ent_img.grid(row=0, column=1, columnspan=2, sticky="we", padx=6, pady=4)
        ttk.Button(data_box, text="Browse…", command=self._browse_img).grid(row=0, column=3, sticky="ew", padx=6, pady=4)
        ttk.Label(data_box, text="Output folder:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.ent_out = ttk.Entry(data_box); self.ent_out.insert(0, self._get_default_output_path()); self.ent_out.grid(row=1, column=1, sticky="we", padx=6, pady=4)
        ttk.Button(data_box, text="Load Session…", command=self._load_session).grid(row=1, column=2, sticky="ew", padx=6, pady=4)
        ttk.Button(data_box, text="Choose…", command=self._browse_out).grid(row=1, column=3, sticky="ew", padx=6, pady=4)
        ttk.Label(data_box, text="Center X (optional):").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.ent_cx = ttk.Entry(data_box, width=12); self.ent_cx.grid(row=2, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(data_box, text="Center Y:").grid(row=2, column=2, sticky="w", padx=6, pady=4)
        self.ent_cy = ttk.Entry(data_box, width=12); self.ent_cy.grid(row=2, column=3, sticky="w", padx=6, pady=4)
        ttk.Label(data_box, text="Leave coordinates empty for auto-detection.", wraplength=520, foreground="#555555").grid(row=3, column=0, columnspan=4, sticky="we", padx=6, pady=(0, 4))

        pre_box = ttk.LabelFrame(fixed, text="Preprocessing", padding=(12, 10, 12, 12)); pre_box.grid(row=1, column=0, sticky="nsew", pady=(10, 0)); pre_box.grid_columnconfigure(1, weight=1)
        ttk.Label(pre_box, text="Mode:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.cmb_pre = ttk.Combobox(pre_box, values=["No smoothing", "NLM Denoising"], state="readonly"); self.cmb_pre.current(0); self.cmb_pre.grid(row=0, column=1, sticky="w", padx=6, pady=4)
        self.cmb_pre.bind("<<ComboboxSelected>>", self._on_preproc_change)
        ttk.Label(pre_box, text="NLM 'h' parameter:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.spn_h = ttk.Spinbox(pre_box, from_=0.01, to=5.0, increment=0.1, width=8, justify="right"); self._set_spinbox_value(self.spn_h, 1.0); self.spn_h.grid(row=1, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(pre_box, text="NLM reduces noise. 'h' controls strength.", wraplength=520, foreground="#555555").grid(row=2, column=0, columnspan=3, sticky="we", padx=6, pady=(2, 0))

        scroll_host = ttk.Frame(outer); scroll_host.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(scroll_host, borderwidth=0, highlightthickness=0); vscroll = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=canvas.yview)
        scrollable = ttk.Frame(canvas, padding=(16, 12, 16, 12)); scrollable.grid_columnconfigure(0, weight=1)
        self._scroll_canvas = canvas; self._scroll_window_id = canvas.create_window((0, 0), window=scrollable, anchor="nw"); canvas.configure(yscrollcommand=vscroll.set)
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))); canvas.bind("<Configure>", lambda e: canvas.itemconfigure(self._scroll_window_id, width=e.width))
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        scrollable.bind("<Enter>", self._activate_scroll); scrollable.bind("<Leave>", self._deactivate_scroll)
        canvas.bind("<Enter>", self._activate_scroll); canvas.bind("<Leave>", self._deactivate_scroll)

        detect_box = ttk.LabelFrame(scrollable, text="Detector and Refinement", padding=(12, 10, 12, 12)); detect_box.grid(row=0, column=0, sticky="nsew"); detect_box.grid_columnconfigure(1, weight=1)

        ttk.Label(detect_box, text="Detection Filters", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))
        self.spn_min_area = self._spin_param(detect_box, 1, "Min. peak area (px)", 3, from_=1, to=500, increment=1)
        self.spn_perc = self._spin_param(detect_box, 2, "Intensity percentile (%)", 99.0, from_=80.0, to=100.0, increment=0.1, format_str="%.1f")
        self.spn_maxpts = self._spin_param(detect_box, 3, "Maximum points (after filters)", 6000, from_=100, to=20000, increment=100)

        ttk.Separator(detect_box).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 8))
        ttk.Label(detect_box, text="Center Refinement", font=("TkDefaultFont", 10, "bold")).grid(row=5, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))
        self.spn_iters = self._spin_param(detect_box, 6, "Refinement iterations", 4, from_=0, to=10, increment=1)
        self.spn_tolang = self._spin_param(detect_box, 7, "Antipode tolerance (°)", 8.0, from_=1.0, to=30.0, increment=0.5, format_str="%.1f")
        self.spn_tolr = self._spin_param(detect_box, 8, "Radius tolerance (relative)", 0.06, from_=0.01, to=0.5, increment=0.01, format_str="%.2f")

        ttk.Separator(detect_box).grid(row=9, column=0, columnspan=2, sticky="ew", pady=(6, 8))
        ttk.Label(detect_box, text="Geometric Filters", font=("TkDefaultFont", 10, "bold")).grid(row=10, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))
        self.spn_dead = self._spin_param(detect_box, 11, "Dead zone (px)", 0, from_=0, to=500, increment=1)
        self.spn_search = self._spin_param(detect_box, 12, "Search radius (px, 0=unlimited)", 0, from_=0, to=10000, increment=25)

        ttk.Label(detect_box, text="Min Area filters noise. Percentile filters dim spots. Dead zone filters center.", wraplength=520, foreground="#555555").grid(row=13, column=0, columnspan=2, sticky="we", padx=6, pady=(2, 0))

        action_box = ttk.Frame(scrollable, padding=(0, 12, 0, 0)); action_box.grid(row=1, column=0, sticky="nsew"); action_box.grid_columnconfigure(0, weight=1)
        ttk.Label(action_box, text="Review parameters and press button below.", wraplength=540, justify="left").grid(row=0, column=0, sticky="we", padx=4, pady=(0, 8))
        ttk.Button(action_box, text="Open point editor", command=self._go_editor).grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 12))

        filler_bg = ttk.Style().lookup("TFrame", "background") or self.winfo_toplevel().cget("background"); bottom_filler = tk.Frame(scrollable, height=56, bg=filler_bg); bottom_filler.grid(row=2, column=0, sticky="ew"); bottom_filler.grid_propagate(False)
        self._on_preproc_change(None)

    def _activate_scroll(self, _event):
        if self._scroll_canvas: self._scroll_canvas.bind_all("<MouseWheel>", self._on_scroll_mousewheel); self._scroll_canvas.bind_all("<Button-4>", self._on_scroll_mousewheel); self._scroll_canvas.bind_all("<Button-5>", self._on_scroll_mousewheel)
    def _deactivate_scroll(self, _event):
        if self._scroll_canvas: self._scroll_canvas.unbind_all("<MouseWheel>"); self._scroll_canvas.unbind_all("<Button-4>"); self._scroll_canvas.unbind_all("<Button-5>")
    def _on_scroll_mousewheel(self, event):
        if not self._scroll_canvas: return
        delta = 0
        if sys.platform == "win32": delta = -int(event.delta / 120)
        elif sys.platform == "darwin": delta = event.delta
        elif event.num == 4: delta = -1
        elif event.num == 5: delta = 1
        if delta != 0: self._scroll_canvas.yview_scroll(delta, "units")
    def _on_preproc_change(self, _evt):
        mode = self.cmb_pre.get(); nlm_enabled = (mode == "NLM Denoising"); state = "normal" if nlm_enabled else "disabled"; self.spn_h.configure(state=state)
    def _spin_param(self, parent, row, label, default, *, from_, to, increment, format_str=None):
        ttk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        spin = ttk.Spinbox(parent, from_=from_, to=to, increment=increment, width=10, justify="right")
        if format_str: spin.configure(format=format_str)
        self._set_spinbox_value(spin, default); spin.grid(row=row, column=1, sticky="w", padx=6, pady=4); return spin
    def _set_spinbox_value(self, spinbox: ttk.Spinbox, value):
        try: spinbox.set(value)
        except tk.TclError:
            try:
                 current_value = spinbox.get()
                 if str(current_value) != str(value): spinbox.delete(0, tk.END); spinbox.insert(0, str(value))
            except (tk.TclError, ValueError): print(f"Warning: Could not set spinbox value to {value}")
    def _browse_img(self):
        p = filedialog.askopenfilename(title="Select image", filetypes=[("Images","*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),("All","*.*")])
        if p: self.ent_img.delete(0, tk.END); self.ent_img.insert(0, p)
    def _browse_out(self):
        p = filedialog.askdirectory(title="Select output folder", mustexist=False)
        if p: self.ent_out.delete(0, tk.END); self.ent_out.insert(0, p)
    def _load_session(self):
        filepath = filedialog.askopenfilename(title="Load SAED Session", filetypes=[("SAED Session", "saed_session.json"), ("All files", "*.*")])
        if not filepath or not self.controller: return
        try:
            self.controller.load_session_from_file(filepath)
            if self.controller: self.controller.set_status(f"Session loaded from {Path(filepath).name}")
        except FileNotFoundError: messagebox.showerror("Load Error", "Session file not found.")
        except Exception as e: messagebox.showerror("Load Error", f"Failed to load session:\n{e}")

    def get_state(self) -> Dict[str, Any]:
        return {
            "image_path": self.ent_img.get(), "output_folder": self.ent_out.get(),
            "center_x": self.ent_cx.get(), "center_y": self.ent_cy.get(),
            "preproc_mode": self.cmb_pre.get(), "h_param": self.spn_h.get(),
            "min_area": self.spn_min_area.get(), # <-- Фильтр детектора
            "detect_perc": self.spn_perc.get(),  # <-- Фильтр интенсивности
            "max_pts": self.spn_maxpts.get(),    # <-- Финальное ограничение
            "refine_iters": self.spn_iters.get(), "tol_angle": self.spn_tolang.get(), "tol_radius": self.spn_tolr.get(),
            "dead_zone": self.spn_dead.get(), "search_radius": self.spn_search.get(), # <-- Геометрические фильтры
        }

    def set_state(self, state: Dict[str, Any]):
        def _set_entry(widget, value):
            if value is not None and isinstance(widget, (ttk.Entry, tk.Entry)): widget.delete(0, tk.END); widget.insert(0, str(value))
        _set_entry(self.ent_img, state.get("image_path")); _set_entry(self.ent_out, state.get("output_folder"))
        _set_entry(self.ent_cx, state.get("center_x")); _set_entry(self.ent_cy, state.get("center_y"))
        preproc_mode = state.get("preproc_mode")
        if preproc_mode and isinstance(self.cmb_pre, ttk.Combobox):
             if preproc_mode in self.cmb_pre['values']: self.cmb_pre.set(preproc_mode)
             else: print(f"W: Saved preproc_mode '{preproc_mode}' not found."); self.cmb_pre.current(0)
        elif isinstance(self.cmb_pre, ttk.Combobox): self.cmb_pre.current(0)
        self._on_preproc_change(None)
        self._set_spinbox_value(self.spn_h, state.get("h_param", 1.0))
        self._set_spinbox_value(self.spn_min_area, state.get("min_area", 3))
        self._set_spinbox_value(self.spn_perc, state.get("detect_perc", 99.0))
        self._set_spinbox_value(self.spn_maxpts, state.get("max_pts", 6000))
        self._set_spinbox_value(self.spn_iters, state.get("refine_iters", 4))
        self._set_spinbox_value(self.spn_tolang, state.get("tol_angle", 8.0))
        self._set_spinbox_value(self.spn_tolr, state.get("tol_radius", 0.06))
        self._set_spinbox_value(self.spn_dead, state.get("dead_zone", 0))
        self._set_spinbox_value(self.spn_search, state.get("search_radius", 0))

    def _go_editor(self):
        try:
            image_path_str = self.ent_img.get(); output_dir_str = self.ent_out.get()
            if not image_path_str: messagebox.showerror("Error", "Please select image."); return
            if not output_dir_str: messagebox.showerror("Error", "Please specify output folder."); return
            image_path = Path(image_path_str).expanduser().resolve(); outdir = Path(output_dir_str).expanduser().resolve(); outdir.mkdir(parents=True, exist_ok=True)
            if not image_path.exists(): messagebox.showerror("Error", f"Image not found: {image_path}"); return

            # --- Получение параметров ---
            min_area = int(float(self.spn_min_area.get())) # Для детектора
            perc = float(self.spn_perc.get())             # Для фильтра по интенсивности
            max_pts = int(float(self.spn_maxpts.get()))   # Финальное ограничение
            iters = int(float(self.spn_iters.get())); tol_ang = float(self.spn_tolang.get()); tol_relr = float(self.spn_tolr.get()) # Для уточнения центра
            dead_r = float(self.spn_dead.get()); search_r = float(self.spn_search.get()) # Геометрические фильтры

            pre_mode = self.cmb_pre.get()
            if pre_mode == "No smoothing": settings = PreprocSettings(mode="raw")
            else: h = float(self.spn_h.get()); settings = PreprocSettings(mode="nlm", h_param=h)

            try: arr = load_grayscale_with_preproc(image_path, settings)
            except RuntimeError as e: messagebox.showerror("Dependency Error", str(e)); return
            except Exception as e: messagebox.showerror("Image Error", f"Failed load/process:\n{e}"); return
            mode = settings.mode; preproc_payload = settings.to_json()

            # --- Детекция ВСЕХ кандидатов ---
            # Использует низкий порог и min_area
            potential_pts = detect_spots_by_centroid(arr, min_area=min_area)
            if len(potential_pts) == 0:
                messagebox.showwarning("Detection Warning", "No potential spots found (check Min Area).")
                pts_final = potential_pts # Пустой массив
            else:
                # --- Уточнение центра (на всех кандидатах!) ---
                # Важно уточнить центр ДО геометрических фильтров
                cx_txt = self.ent_cx.get().strip(); cy_txt = self.ent_cy.get().strip()
                if cx_txt and cy_txt:
                    try: center0 = CenterResult(cy=float(cy_txt), cx=float(cx_txt), method="user")
                    except ValueError: messagebox.showwarning("Input Warning", "Invalid center coords. Using auto."); center0 = geometric_midpoint(arr)
                else: center0 = geometric_midpoint(arr)
                center = refine_center_antipodal((center0.cy, center0.cx), potential_pts, tol_ang_deg=tol_ang, tol_rel_r=tol_relr, iters=iters)

                # --- Геометрические фильтры ---
                pts_geom_filtered = potential_pts
                if (dead_r > 0 or search_r > 0):
                    dy = pts_geom_filtered[:, 0] - center.cy; dx = pts_geom_filtered[:, 1] - center.cx; r = np.hypot(dx, dy)
                    mask = np.ones(len(pts_geom_filtered), dtype=bool)
                    if dead_r > 0:   mask &= (r >= dead_r)
                    if search_r > 0: mask &= (r <= search_r)
                    pts_geom_filtered = pts_geom_filtered[mask]

                if len(pts_geom_filtered) == 0:
                     messagebox.showwarning("Detection Warning", "All spots removed by geometric filters.")
                     pts_final = pts_geom_filtered # Пустой массив
                else:
                    # --- Фильтр по интенсивности (перцентиль) ---
                    # Вычисляем порог по точкам, прошедшим геом. фильтры
                    intensities = pts_geom_filtered[:, 2]
                    try:
                        intensity_threshold = np.percentile(intensities, perc)
                    except IndexError: # Если массив пуст
                        intensity_threshold = 0

                    intensity_mask = intensities >= intensity_threshold
                    pts_intensity_filtered = pts_geom_filtered[intensity_mask]

                    if len(pts_intensity_filtered) == 0:
                        messagebox.showwarning("Detection Warning", "All remaining spots removed by intensity filter.")
                        pts_final = pts_intensity_filtered
                    else:
                         # --- Финальное ограничение по количеству ---
                         # Сортируем по яркости (уже есть в массиве)
                         pts_intensity_filtered = pts_intensity_filtered[np.argsort(-pts_intensity_filtered[:, 2])]
                         if len(pts_intensity_filtered) > max_pts:
                             pts_final = pts_intensity_filtered[:max_pts]
                             print(f"Info: Kept top {max_pts} brightest spots out of {len(pts_intensity_filtered)}.")
                         else:
                             pts_final = pts_intensity_filtered

            # --- Сохранение и запуск редактора ---
            points_list = [{"y": float(y), "x": float(x), "intensity": float(v)} for (y, x, v) in pts_final.tolist()]
            saed_input_data = { "image": str(image_path), "preproc_mode": mode, "preproc": preproc_payload,
                                "center": {"x": float(center.cx), "y": float(center.cy), "method": center.method},
                                "radii": {"dead": float(dead_r), "search": float(search_r)}, "points": points_list }
            saed_input_path = outdir / "saed_input.json"
            saed_input_path.write_text(json.dumps(saed_input_data, ensure_ascii=False, indent=2), encoding="utf-8")

            if self.controller:
                try: self.controller.open_editor(saed_input_path)
                except Exception as e: messagebox.showerror("Error", f"Failed open editor:\n{e}")
            else: messagebox.showwarning("Standalone Mode", "Running standalone.")

            try:
                (outdir/"center_init.json").write_text(json.dumps({
                    "initial": {"x": center0.cx, "y": center0.cy, "method": center0.method},
                    "refined": {"x": center.cx,  "y": center.cy,  "method": center.method},
                    "dead_zone_px": dead_r, "search_radius_px": search_r, "preproc_mode": mode, "preproc": preproc_payload,
                    "image_size": {"H": int(arr.shape[0]), "W": int(arr.shape[1])}
                }, indent=2), encoding="utf-8")
            except Exception as e: print(f"W: Could not save center_init.json - {e}")

        except Exception as e: messagebox.showerror("Processing Error", f"Unexpected error:\n{e}")

class SAEDApp(tk.Tk):
    ( "Standalone app wrapper" )
    def __init__(self):
        super().__init__()
        self.title("SAED Symmetry – Launcher")
        self.geometry("980x720")
        self.resizable(True, False)
        frame = SAEDLauncherFrame(self)
        frame.pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    SAEDApp().mainloop()