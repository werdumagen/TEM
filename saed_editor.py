#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAED Editor + Analysis
----------------------
• Подсказка по средней кнопке мыши (MMB) на точке: показывает x, y, intensity; закрывается по Esc и при любых действиях.

Удалено по ТЗ:
• Перемещение обычных точек.
• Любые упоминания/функции панорамирования или перемещения окна средней кнопкой мыши.

Остальной функционал редактора сохранён.
"""
import sys, json, subprocess
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from percentile_utils import compute_percentile_map, map_values_to_percent
from preproc import PreprocSettings, load_grayscale_with_preproc
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
from scipy.signal import find_peaks

# ------ симметрия (для отчёта) ------
def pol_from(center, pts):
    cy, cx = center
    dy, dx = pts[:, 0] - cy, pts[:, 1] - cx
    r = np.hypot(dx, dy)
    a = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    return r, a

def cluster_rings(radii):
    if len(radii) == 0:
        return np.array([]), np.array([]), ([], [])
    hist, edges = np.histogram(radii, bins=60)
    centers = (edges[:-1] + edges[1:]) / 2
    pk, _ = find_peaks(hist, prominence=3)
    ring_centers = centers[pk]
    if len(ring_centers) == 0:
        return np.array([]), np.zeros_like(radii, int), (hist, edges)
    labels = np.argmin(np.abs(radii[:, None] - ring_centers[None, :]), axis=1)
    return ring_centers, labels, (hist, edges)

def symmetry_scores(angles, radii, ring_means, top_rings=3):
    out = {}
    if not ring_means:
        return out
    idx = min(top_rings - 1, len(ring_means) - 1)
    maxR = ring_means[idx] * 1.15
    ang_sel = angles[radii <= maxR]
    for k in [4, 6, 8, 10, 12]:
        period = 360.0 / k
        phases = np.deg2rad((ang_sel % period) * k)
        C = np.cos(phases).mean(); S = np.sin(phases).mean()
        out[f"{k}-fold"] = float(np.hypot(C, S))
    return out

class PointEditor(tk.Frame):
    def __init__(self, master: tk.Misc, controller=None,
                 input_json: str | None = None, auto_load: bool = True):
        super().__init__(master)
        self.controller = controller
        # данные
        self.points = np.zeros((0, 2), float)   # [y, x]
        self.values = np.zeros((0,), float)     # интенсивности (параллельно points)
        self.rect_start = None
        self.rect_artist = None
        self.overlay = None  # {center:{x,y}, dead_radius, search_radius}
        self.image_path: Optional[Path] = None
        self.img_arr: Optional[np.ndarray] = None
        self._percent_map: Optional[np.ndarray] = None
        self._percent_lookup: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._preproc_settings: PreprocSettings = PreprocSettings(mode="raw")

        # Undo/Redo
        self._undo = []
        self._redo = []
        self._history_cap = 300

        # Перетаскивание центра
        self.center_dragging = False
        self._center_hit_radius = 10.0  # пикселей

        # Масштаб (ползунок)
        self.zoom_val = 0  # 0..100
        self.view_cx = None
        self.view_cy = None

        # Tooltip (подсказка на MMB по точке)
        self._tooltip = None          # matplotlib.text.Annotation
        self._tooltip_idx = None

        # Измерение расстояний между точками
        self._measure_active = False
        self._measure_start_idx: Optional[int] = None
        self._measure_start_point: Optional[tuple[float, float]] = None
        self._measure_preview_end: Optional[tuple[float, float]] = None
        self._measure_preview_artist = None
        self._measure_line_artist = None
        self._measure_annotation = None
        self._measurement: Optional[dict[str, object]] = None

        self._build_ui()

        # первичная загрузка
        if auto_load and input_json:
            self.load_input_json(Path(input_json), push_undo=False)
        else:
            self._ensure_view_center()
            self._redraw()

    # ---------- UI ----------
    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        side_panel = ttk.Frame(self, padding=(16, 16, 12, 16))
        side_panel.grid(row=0, column=0, sticky="ns")
        side_panel.columnconfigure(0, weight=1)

        controls = ttk.Frame(side_panel)
        controls.pack(side=tk.TOP, fill=tk.X)

        header_row = ttk.Frame(controls)
        header_row.pack(fill=tk.X)

        history_group = ttk.Frame(header_row)
        history_group.pack(side=tk.LEFT)
        ttk.Button(history_group, text="◀", width=3, command=self._undo_btn, style="Toolbutton").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(history_group, text="▶", width=3, command=self._redo_btn, style="Toolbutton").pack(
            side=tk.LEFT, padx=(0, 4)
        )

        help_button = ttk.Button(header_row, text="?", width=3, command=self._toggle_help, style="Toolbutton")
        help_button.pack(side=tk.RIGHT)

        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 10))

        zoom_group = ttk.LabelFrame(controls, text="Масштаб", padding=(12, 8, 12, 10))
        zoom_group.pack(fill=tk.X)
        self.zoom_var = tk.DoubleVar(value=self.zoom_val)
        self.zoom_scale = ttk.Scale(zoom_group, from_=0, to=100, variable=self.zoom_var, command=self._on_zoom_change)
        self.zoom_scale.pack(fill=tk.X, padx=4, pady=(0, 6))
        self.zoom_hint = ttk.Label(zoom_group, anchor="w")
        self.zoom_hint.pack(fill=tk.X, padx=4)
        self.zoom_scale.set(self.zoom_val)

        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 10))

        file_group = ttk.Frame(controls)
        file_group.pack(fill=tk.X)
        ttk.Button(file_group, text="Открыть JSON…", command=self._open_json).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(file_group, text="Сохранить", command=self._save_points).pack(side=tk.LEFT, padx=(0, 6))

        analysis_group = ttk.Frame(controls)
        analysis_group.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(analysis_group, text="Начать анализ", command=self._start_analysis).pack(side=tk.LEFT, padx=(0, 6))

        self.help_panel = ttk.LabelFrame(side_panel, text="Подсказки", padding=(16, 12, 16, 12))
        help_text = (
            "ЛКМ по пустому месту — добавить точку\n"
            "ЛКМ по центру — перетащить центр и пересчитать фильтры\n"
            "ПКМ по точке — удалить\n"
            "Средняя кнопка по точке — показать координаты и интенсивность\n"
            "Зажатая ЛКМ от точки до точки — измерить расстояние\n"
            "Shift + перетаскивание — прямоугольное удаление диапазона"
        )
        ttk.Label(self.help_panel, text=help_text, justify="left", wraplength=780).pack(fill=tk.X)
        self._help_visible = False

        self._side_spacer = ttk.Frame(side_panel)
        self._side_spacer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        status_frame = ttk.Frame(side_panel, padding=(0, 0, 0, 0))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))
        self.status_label = ttk.Label(status_frame, anchor="w")
        self.status_label.pack(fill=tk.X)

        canvas_frame = ttk.Frame(self, padding=(0, 16, 16, 16))
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(9.4, 6.4))
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("button_press_event", self._on_down)
        self.canvas.mpl_connect("button_release_event", self._on_up)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("key_press_event", self._on_key)

        self._default_status = "Режим: редактор точек"
        self._status_message = ""
        self._update_zoom_hint()
        self._set_status(self._default_status)

    def _toggle_help(self):
        self._help_visible = not self._help_visible
        if self._help_visible:
            self.help_panel.pack(side=tk.TOP, fill=tk.X, pady=(12, 8), before=self._side_spacer)
            self._set_status("Подробные подсказки раскрыты")
        else:
            self.help_panel.pack_forget()

    def _update_zoom_hint(self):
        if hasattr(self, "zoom_hint"):
            value = int(round(self.zoom_var.get())) if hasattr(self, "zoom_var") else self.zoom_val
            self.zoom_hint.configure(text=f"Текущее увеличение: {value}% (0 = весь кадр)")

    def _set_status(self, text: str):
        self._status_message = text
        if hasattr(self, "status_label"):
            self.status_label.configure(text=text)
        if self.controller is not None and hasattr(self.controller, "set_status"):
            try:
                self.controller.set_status(text)
            except Exception:
                pass

    # ---------- IO ----------
    def _open_json(self):
        p = filedialog.askopenfilename(filetypes=[("SAED Input JSON","*saed_input.json;*.json"), ("All","*.*")])
        if p:
            self.load_input_json(Path(p), push_undo=True)

    def load_input_json(self, path: Path, *, push_undo: bool = False, reset_view: bool = True):
        """Публичный метод загрузки JSON, используется и контроллером вкладок."""
        if push_undo:
            self._push_undo()
        self._load_input_json(path)
        self._clear_tooltip()
        if reset_view:
            self.view_cx = None
            self.view_cy = None
        self._ensure_view_center()
        self._redraw()
        self._update_zoom_hint()
        self._set_status(f"Загружено: {path.name}")

    def _load_input_json(self, path: Path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать JSON:\n{e}")
            return

        self._percent_map = None
        self._percent_lookup = None

        # загрузка изображения по пути из JSON
        img_path = data.get("image")
        if not img_path:
            messagebox.showerror("Ошибка", "В JSON отсутствует поле 'image'.")
            return
        self.image_path = Path(img_path)
        fallback_mode = data.get("preproc_mode")
        if not isinstance(fallback_mode, str):
            fallback_mode = None
        self._preproc_settings = PreprocSettings.from_json(
            data.get("preproc"), fallback_mode=fallback_mode
        )
        try:
            self.img_arr = load_grayscale_with_preproc(self.image_path, self._preproc_settings)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось подготовить изображение:\n{e}")
            return

        self._percent_map, uniq_vals, uniq_perc = compute_percentile_map(self.img_arr)
        self._percent_lookup = (uniq_vals, uniq_perc)

        # overlay: центр и радиусы
        c = data.get("center") or {}
        r = data.get("radii") or {}
        self.overlay = {
            "center": {"x": float(c.get("x", (self.img_arr.shape[1]-1)/2.0)),
                       "y": float(c.get("y", (self.img_arr.shape[0]-1)/2.0))},
            "dead_radius": float(r.get("dead") or 0.0),
            "search_radius": float(r.get("search") or 0.0),
        }

        # точки: y,x,intensity (если интенсивности нет, рассчитать)
        pts = data.get("points", [])
        if pts:
            yy = [float(p.get("y")) for p in pts]
            xx = [float(p.get("x")) for p in pts]
            self.points = np.column_stack([yy, xx]).astype(float)
            if self._percent_map is not None:
                self.values = self._sample_intensities(self.points)
            elif any("intensity" in p for p in pts):
                vv = np.array([float(p.get("intensity", 0.0)) for p in pts], dtype=float)
                if self._percent_lookup is not None:
                    self.values = map_values_to_percent(vv, *self._percent_lookup)
                else:
                    self.values = vv
            else:
                self.values = np.zeros((len(self.points),), float)
        else:
            self.points = np.zeros((0,2), float)
            self.values = np.zeros((0,), float)

    def _save_points(self) -> Path:
        """
        Сохраняет точки (включая интенсивности) рядом с исходным input.json:
        - spots.json  — список точек (y,x,intensity)
        - saed_input.edited.json — исходный входной JSON с обновлёнными точками
        """
        base = self.image_path.with_name("spots.json") if self.image_path else Path("spots.json")
        pts = []
        # пересэмплируем интенсивности в текущих координатах
        if self._percent_map is not None or self.img_arr is not None:
            vals = self._sample_intensities(self.points)
        else:
            vals = self.values
        for (y, x), v in zip(self.points, vals):
            pts.append({"y": float(y), "x": float(x), "intensity": float(v)})
        base.write_text(json.dumps({"points": pts}, indent=2), encoding="utf-8")

        # также пересохраним обновлённый saed_input
        si = {
            "image": str(self.image_path) if self.image_path else None,
            "preproc_mode": self._preproc_settings.mode,
            "preproc": self._preproc_settings.to_json(),
            "center": (self.overlay.get("center") if self.overlay else None),
            "radii": {
                "dead": float(self.overlay.get("dead_radius") or 0.0) if self.overlay else 0.0,
                "search": float(self.overlay.get("search_radius") or 0.0) if self.overlay else 0.0
            },
            "points": pts
        }
        edited = self.image_path.with_name("saed_input.edited.json") if self.image_path else Path(
            "saed_input.edited.json")
        edited.write_text(json.dumps(si, ensure_ascii=False, indent=2), encoding="utf-8")
        self._set_status(f"Точки сохранены: {base.name}")
        return base

    # ---------- Helpers ----------
    def _sample_intensities(self, pts_yx: np.ndarray) -> np.ndarray:
        if len(pts_yx) == 0:
            return np.zeros((0,), float)

        if self._percent_map is not None:
            src = self._percent_map
            H, W = src.shape[:2]
            out = []
            for y, x in pts_yx:
                yi = int(round(y)); xi = int(round(x))
                yi = max(0, min(H - 1, yi)); xi = max(0, min(W - 1, xi))
                out.append(float(src[yi, xi]))
            return np.array(out, float)

        if self.img_arr is None:
            return np.zeros((len(pts_yx),), float)

        H, W = self.img_arr.shape[:2]
        raw = []
        for y, x in pts_yx:
            yi = int(round(y)); xi = int(round(x))
            yi = max(0, min(H - 1, yi)); xi = max(0, min(W - 1, xi))
            raw.append(float(self.img_arr[yi, xi]))
        raw = np.array(raw, float)
        if self._percent_lookup is not None:
            return map_values_to_percent(raw, *self._percent_lookup)
        return raw

    def _img_xy(self, e):
        return None if (e.xdata is None or e.ydata is None) else (e.ydata, e.xdata)

    def _near_idx(self, y, x, pix_tol=8):
        if len(self.points) == 0: return None
        d2 = (self.points[:,0]-y)**2 + (self.points[:,1]-x)**2
        i = int(np.argmin(d2))
        return i if d2[i]**0.5 <= pix_tol else None

    def _center_hit(self, y, x):
        if not (self.overlay and self.overlay.get("center")): return False
        cy = float(self.overlay["center"].get("y", 0.0))
        cx = float(self.overlay["center"].get("x", 0.0))
        return ((y - cy)**2 + (x - cx)**2) ** 0.5 <= self._center_hit_radius

    def _apply_center_filters(self):
        """Удаляет точки, оказавшиеся в мёртвой зоне или вне радиуса поиска после перемещения центра."""
        if not (self.overlay and self.overlay.get("center")): return
        cy = float(self.overlay["center"].get("y", 0.0))
        cx = float(self.overlay["center"].get("x", 0.0))
        dead = float(self.overlay.get("dead_radius") or 0.0)
        sr   = float(self.overlay.get("search_radius") or 0.0)
        if len(self.points) == 0 or (dead <= 0 and sr <= 0): return
        r = np.hypot(self.points[:,1]-cx, self.points[:,0]-cy)
        mask = np.ones(len(self.points), dtype=bool)
        if dead > 0: mask &= (r >= dead)
        if sr   > 0: mask &= (r <= sr)
        self.points = self.points[mask]
        self.values = self.values[mask] if len(self.values)==len(mask) else self._sample_intensities(self.points)

    # ---------- Tooltip и измерения ----------
    def _remove_measure_preview_artist(self) -> bool:
        if self._measure_preview_artist is not None:
            try:
                self._measure_preview_artist.remove()
            except Exception:
                pass
            self._measure_preview_artist = None
            return True
        return False

    def _remove_measurement_artists(self) -> bool:
        removed = False
        if self._measure_line_artist is not None:
            try:
                self._measure_line_artist.remove()
            except Exception:
                pass
            self._measure_line_artist = None
            removed = True
        if self._measure_annotation is not None:
            try:
                self._measure_annotation.remove()
            except Exception:
                pass
            self._measure_annotation = None
            removed = True
        return removed

    def _cancel_measurement_preview(self) -> bool:
        removed = self._remove_measure_preview_artist()
        has_state = (
            self._measure_active
            or self._measure_start_point is not None
            or self._measure_preview_end is not None
        )
        removed = removed or has_state
        self._measure_active = False
        self._measure_start_idx = None
        self._measure_start_point = None
        self._measure_preview_end = None
        return removed

    def _clear_measurement_result(self) -> bool:
        removed = self._measurement is not None
        removed = self._remove_measurement_artists() or removed
        self._measurement = None
        return removed

    def _start_measurement(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.points):
            return
        self._measure_active = True
        self._measure_start_idx = idx
        y0, x0 = self.points[idx]
        self._measure_start_point = (float(y0), float(x0))
        self._measure_preview_end = None
        self._remove_measure_preview_artist()

    def _update_measurement_preview(self, pos: Optional[tuple[float, float]]) -> None:
        if not self._measure_active or self._measure_start_point is None:
            return
        if pos is None:
            self._measure_preview_end = None
            if self._remove_measure_preview_artist():
                if hasattr(self, "canvas"):
                    self.canvas.draw_idle()
            return
        y1, x1 = pos
        self._measure_preview_end = (float(y1), float(x1))
        y0, x0 = self._measure_start_point
        if self._measure_preview_artist is None:
            (line,) = self.ax.plot(
                [x0, x1], [y0, y1], color="#ffcc33", linewidth=1.6, linestyle="--", alpha=0.9
            )
            self._measure_preview_artist = line
        else:
            self._measure_preview_artist.set_data([x0, x1], [y0, y1])
        if hasattr(self, "canvas"):
            self.canvas.draw_idle()

    def _finalize_measurement(self, end_idx: Optional[int]) -> None:
        if not self._measure_active or self._measure_start_point is None:
            self._cancel_measurement_preview()
            return
        if end_idx is None or end_idx == self._measure_start_idx or end_idx < 0 or end_idx >= len(self.points):
            if self._cancel_measurement_preview() and hasattr(self, "canvas"):
                self.canvas.draw_idle()
            return
        start_y, start_x = self._measure_start_point
        end_y, end_x = map(float, self.points[end_idx])
        length = float(np.hypot(end_x - start_x, end_y - start_y))
        self._measurement = {
            "start": (start_y, start_x),
            "end": (end_y, end_x),
            "length": length,
        }
        self._cancel_measurement_preview()
        self._redraw()

    def _draw_measurement_overlays(self) -> None:
        self._measure_line_artist = None
        self._measure_annotation = None
        self._measure_preview_artist = None

        if self._measurement is not None:
            start_y, start_x = self._measurement["start"]
            end_y, end_x = self._measurement["end"]
            length = float(self._measurement.get("length", 0.0))
            (line,) = self.ax.plot(
                [start_x, end_x], [start_y, end_y], color="#ffcc33", linewidth=1.8, alpha=0.95
            )
            self._measure_line_artist = line
            mid_x = (start_x + end_x) / 2.0
            mid_y = (start_y + end_y) / 2.0
            txt = f"L = {length:.1f} px"
            self._measure_annotation = self.ax.annotate(
                txt,
                xy=(mid_x, mid_y),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9),
                fontsize=9,
            )

        if (
            self._measure_active
            and self._measure_start_point is not None
            and self._measure_preview_end is not None
        ):
            y0, x0 = self._measure_start_point
            y1, x1 = self._measure_preview_end
            (pline,) = self.ax.plot(
                [x0, x1], [y0, y1], color="#ffcc33", linewidth=1.6, linestyle="--", alpha=0.9
            )
            self._measure_preview_artist = pline

    def _clear_tooltip(self, *, keep_measure: bool = False, keep_preview: bool = False):
        removed = False
        if self._tooltip is not None:
            try:
                self._tooltip.remove()
            except Exception:
                pass
            self._tooltip = None
            self._tooltip_idx = None
            removed = True
        if not keep_preview:
            if self._cancel_measurement_preview():
                removed = True
        if not keep_measure:
            if self._clear_measurement_result():
                removed = True
        if removed and hasattr(self, "canvas"):
            self.canvas.draw_idle()

    def _show_tooltip_for_idx(self, idx):
        if idx is None or idx < 0 or idx >= len(self.points):
            return
        y, x = self.points[idx]
        # intensity по текущему изображению/values
        inten = float(self.values[idx]) if idx < len(self.values) else 0.0
        if self._percent_map is not None:
            H, W = self._percent_map.shape[:2]
            yi = max(0, min(H - 1, int(round(y))))
            xi = max(0, min(W - 1, int(round(x))))
            inten = float(self._percent_map[yi, xi])
        elif self.img_arr is not None:
            H, W = self.img_arr.shape[:2]
            yi = max(0, min(H - 1, int(round(y))))
            xi = max(0, min(W - 1, int(round(x))))
            raw_val = float(self.img_arr[yi, xi])
            if self._percent_lookup is not None:
                inten = float(map_values_to_percent(np.array([raw_val], dtype=float), *self._percent_lookup)[0])
            else:
                inten = raw_val

        # удалить предыдущую подсказку
        self._clear_tooltip()

        # создать аннотацию возле точки
        txt = f"x={x:.1f}, y={y:.1f}, I={inten:.1f}%"
        self._tooltip = self.ax.annotate(
            txt, xy=(x, y), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9),
            fontsize=9
        )
        self._tooltip_idx = idx
        self.canvas.draw_idle()

    # ---------- Undo/Redo ----------
    def _make_snapshot(self):
        center = None
        if self.overlay and isinstance(self.overlay, dict) and self.overlay.get("center"):
            c = self.overlay["center"]
            if c is not None and "x" in c and "y" in c:
                center = {"x": float(c["x"]), "y": float(c["y"])}
        return {
            "points": self.points.copy(),
            "values": self.values.copy(),
            "center": center,
            "view_cx": self.view_cx, "view_cy": self.view_cy
        }

    def _push_undo(self):
        self._undo.append(self._make_snapshot())
        if len(self._undo) > self._history_cap:
            self._undo.pop(0)
        self._redo.clear()

    def _apply_snapshot(self, snap):
        self.center_dragging = False
        if self.rect_artist is not None:
            try: self.rect_artist.remove()
            except Exception: pass
            self.rect_artist = None
        self.rect_start = None

        self.points = snap["points"].copy()
        self.values = snap["values"].copy()
        if snap["center"] is None:
            if self.overlay and "center" in self.overlay: self.overlay.pop("center")
        else:
            if self.overlay is None: self.overlay = {}
            self.overlay["center"] = {"x": float(snap["center"]["x"]), "y": float(snap["center"]["y"])}
        self.view_cx = snap.get("view_cx", self.view_cx)
        self.view_cy = snap.get("view_cy", self.view_cy)
        self._ensure_view_center()

    def _undo_btn(self):
        self._clear_tooltip()
        if not self._undo: return
        self._redo.append(self._make_snapshot())
        if len(self._redo) > self._history_cap: self._redo.pop(0)
        snap = self._undo.pop(-1)
        self._apply_snapshot(snap)
        self._redraw()

    def _redo_btn(self):
        self._clear_tooltip()
        if not self._redo: return
        self._undo.append(self._make_snapshot())
        if len(self._redo) > self._history_cap: self._undo.pop(0)
        snap = self._redo.pop(-1)
        self._apply_snapshot(snap)
        self._redraw()

    # ---------- View-center helpers ----------
    def _ensure_view_center(self):
        if self.img_arr is None:
            return
        H, W = self.img_arr.shape[:2]
        if self.overlay and self.overlay.get("center"):
            cx = float(self.overlay["center"].get("x", (W-1)/2.0))
            cy = float(self.overlay["center"].get("y", (H-1)/2.0))
        else:
            cx = (W-1)/2.0; cy = (H-1)/2.0
        if self.view_cx is None: self.view_cx = cx
        if self.view_cy is None: self.view_cy = cy

    # ---------- Zoom ----------
    def _apply_zoom(self):
        if self.img_arr is None:
            return
        H, W = self.img_arr.shape[:2]
        self._ensure_view_center()

        if self.zoom_val <= 0:
            self.ax.set_xlim(-0.5, W - 0.5)
            self.ax.set_ylim(H - 0.5, -0.5)
            return

        min_dim = min(H, W)
        L = int(round(min_dim - (min_dim - 50) * (self.zoom_val / 100.0)))
        L = max(50, min_dim if L < 50 else L)
        half = L / 2.0

        cx = float(self.view_cx); cy = float(self.view_cy)
        x0 = max(-0.5, cx - half); x1 = min(W - 0.5, cx + half)
        y0 = max(-0.5, cy - half); y1 = min(H - 0.5, cy + half)

        # если окно "урезано" границами — подправим
        if (x1 - x0) < L:
            if x0 <= -0.5: x1 = x0 + L
            elif x1 >= (W - 0.5): x0 = x1 - L
        if (y1 - y0) < L:
            if y0 <= -0.5: y1 = y0 + L
            elif y1 >= (H - 0.5): y0 = y1 - L

        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y1, y0)

    def _on_zoom_change(self, val):
        try:
            self.zoom_val = int(float(val))
        except Exception:
            self.zoom_val = 0
        if hasattr(self, "zoom_var"):
            current = int(round(self.zoom_var.get()))
            if current != self.zoom_val:
                self.zoom_var.set(self.zoom_val)
        self._update_zoom_hint()
        self._clear_tooltip()
        self._redraw()

    # ---------- Mouse / Keyboard events ----------
    def _on_key(self, e):
        # закрыть tooltip по Esc
        if e.key == 'escape':
            self._clear_tooltip()

    def _on_down(self, e):
        pos = self._img_xy(e)
        # любое действие закрывает tooltip
        self._clear_tooltip()

        # Средняя кнопка: только подсказка, если попали в точку
        if e.button == 2:
            if pos is None:
                return
            y, x = pos
            idx = self._near_idx(y, x, pix_tol=8)
            if idx is not None:
                self._show_tooltip_for_idx(idx)
            return  # не считаем это добавлением/редактированием

        if e.button == 1 and not (e.key and "shift" in e.key):
            if pos is None: return
            y, x = pos
            # Перетаскивание центра — оставить
            if self._center_hit(y, x):
                self._push_undo()
                self.center_dragging = True
                self._redo.clear()
                return

            # Добавление новой точки по клику в пустое место (без последующего перетаскивания!)
            i = self._near_idx(y, x)
            if i is None:
                self._push_undo()
                self.points = np.vstack([self.points, [y, x]])
                self.values = np.append(self.values, self._sample_intensities(np.array([[y, x]]))[0])
                self._redo.clear()
            else:
                self._start_measurement(i)
            # если кликнули по существующей точке — перемещение запрещено (используем для измерения)

        elif e.button == 3:
            if pos is None: return
            y, x = pos
            i = self._near_idx(y, x)
            if i is not None:
                self._push_undo()
                self.points = np.delete(self.points, i, axis=0)
                self.values = np.delete(self.values, i, axis=0)
                self._redo.clear()

        elif e.button == 1 and e.key and "shift" in e.key:
            self._push_undo()
            self.rect_start = pos
            self._redo.clear()

        self._redraw()

    def _on_move(self, e):
        pos = self._img_xy(e)
        # любое движение закрывает tooltip
        keep = self._measure_active
        self._clear_tooltip(keep_measure=keep, keep_preview=keep)

        # Перетаскивание центра — оставить
        if self.center_dragging and pos is not None:
            y, x = pos
            if self.overlay is None: self.overlay = {}
            self.overlay["center"] = {"x": float(x), "y": float(y)}
            self._redraw()
            return

        if self._measure_active:
            self._update_measurement_preview(pos)
            return

        # Прямоугольник выделения для удаления
        if self.rect_start and e.xdata and e.ydata:
            y0, x0 = self.rect_start; y1, x1 = e.ydata, e.xdata
            self._redraw()
            self.rect_artist = self.ax.add_patch(
                plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", ls="--", lw=1.5)
            )
            self.canvas.draw_idle()
            return

    def _on_up(self, e):
        # любое действие закрывает tooltip
        keep = self._measure_active
        self._clear_tooltip(keep_measure=keep, keep_preview=keep)

        # Завершение перетаскивания центра
        if self.center_dragging:
            self.center_dragging = False
            self._apply_center_filters()
            if self.overlay and self.overlay.get("center"):
                self.view_cx = float(self.overlay["center"]["x"])
                self.view_cy = float(self.overlay["center"]["y"])
            self._redraw()
            return

        if self._measure_active:
            pos = self._img_xy(e)
            end_idx = None
            if pos is not None:
                end_idx = self._near_idx(pos[0], pos[1])
            self._finalize_measurement(end_idx)
            return

        # Завершение прямоугольника удаления
        if self.rect_start:
            y0, x0 = self.rect_start; y1, x1 = e.ydata, e.xdata
            if y1 is not None and x1 is not None:
                ymin, ymax = sorted([y0, y1]); xmin, xmax = sorted([x0, x1])
                mask = ~((self.points[:,0] >= ymin) & (self.points[:,0] <= ymax) &
                         (self.points[:,1] >= xmin) & (self.points[:,1] <= xmax))
                self.points = self.points[mask]
                self.values = self.values[mask]
            self.rect_start = None
            if self.rect_artist is not None:
                self.rect_artist.remove(); self.rect_artist = None
            self._redraw()

    # ---------- Draw ----------
    def _redraw(self):
        self.ax.clear()
        if self.img_arr is not None:
            self.ax.imshow(self.img_arr, cmap="gray", interpolation="nearest")
        self.ax.axis("off")

        if self.overlay and self.overlay.get("center"):
            cy = float(self.overlay["center"].get("y", 0))
            cx = float(self.overlay["center"].get("x", 0))
            self.ax.scatter([cx], [cy], s=36, c="red", marker="o")
            dead = float(self.overlay.get("dead_radius") or 0)
            sr   = float(self.overlay.get("search_radius") or 0)
            for R in [dead, sr]:
                if R > 0:
                    self.ax.add_patch(Circle((cx, cy), R, fill=False, ls="--", lw=2.0, ec="red"))

        if len(self.points):
            self.ax.scatter(self.points[:,1], self.points[:,0],
                            s=22, alpha=0.9, marker="o", linewidths=0.5, edgecolors="black")

        self._draw_measurement_overlays()
        self._apply_zoom()
        self.canvas.draw_idle()

    # ---------- Анализ ----------
    def _start_analysis(self):
        saved = self._save_points()  # сохраняет текущие точки (и интенсивности) и возвращает путь к spots.json
        self._set_status("Подготовка данных для анализа…")

        # payload для fibonachi_analysis
        payload_path = None
        try:
            if self.img_arr is not None:
                geo_cy = (self.img_arr.shape[0] - 1) / 2.0
                geo_cx = (self.img_arr.shape[1] - 1) / 2.0
            else:
                geo_cy = geo_cx = None

            # актуальные точки (после правок) + интенсивности из текущего изображения
            points_list = []
            if len(self.points):
                if self._percent_map is not None:
                    H, W = self._percent_map.shape[:2]
                elif self.img_arr is not None:
                    H, W = self.img_arr.shape[:2]
                else:
                    H = W = None
                for idx, (y, x) in enumerate(self.points.tolist()):
                    inten = None
                    if H is not None:
                        yi = max(0, min(H - 1, int(round(y))))
                        xi = max(0, min(W - 1, int(round(x))))
                        if self._percent_map is not None:
                            inten = float(self._percent_map[yi, xi])
                        elif self.img_arr is not None:
                            raw_val = float(self.img_arr[yi, xi])
                            if self._percent_lookup is not None:
                                inten = float(
                                    map_values_to_percent(
                                        np.array([raw_val], dtype=float), *self._percent_lookup
                                    )[0]
                                )
                            else:
                                inten = raw_val
                    if inten is None and idx < len(self.values):
                        inten = float(self.values[idx])
                    points_list.append({"x": float(x), "y": float(y), "intensity": inten})

            # актуальный центр из overlay
            overlay_center = None
            if self.overlay and isinstance(self.overlay, dict) and self.overlay.get("center") is not None:
                c = self.overlay["center"]
                overlay_center = {"x": float(c.get("x")), "y": float(c.get("y"))}

            # радиусы из overlay
            dead_val = None
            search_val = None
            if self.overlay:
                if self.overlay.get("dead_radius") is not None:
                    dead_val = float(self.overlay.get("dead_radius"))
                if self.overlay.get("search_radius") is not None:
                    search_val = float(self.overlay.get("search_radius"))

            payload = {
                "image": str(self.image_path) if self.image_path else None,
                "preproc_mode": self._preproc_settings.mode,
                "preproc": self._preproc_settings.to_json(),
                "points": points_list,
                "centers": {
                    "geometric": {"x": float(geo_cx), "y": float(geo_cy)} if geo_cx is not None else None,
                    "overlay": overlay_center
                },
                "radii": {
                    "dead": dead_val,
                    "search": search_val
                },
                "spots_json": str(saved) if saved else None
            }

            payload_path = (self.image_path.with_name("fibo_input.json") if self.image_path
                            else Path("fibo_input.json"))
            payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


        except Exception as e:

            payload_path = None

            messagebox.showerror("Ошибка подготовки данных",

                                 f"Не удалось подготовить данные для fibonachi_analysis:\n{e}")

        used_controller = False

        if self.controller is not None and payload_path is not None:

            try:

                self.controller.open_analysis(payload_path, self.image_path, saved)
                self._set_status("Анализ открыт во вкладке")

                used_controller = True

            except Exception as e:

                messagebox.showerror("Ошибка запуска",

                                     f"Не удалось переключиться к анализатору:\n{e}")

        if not used_controller:

            # запуск fibonachi_analysis.exe (совместимость со старыми ключами сохранена)

            try:

                if getattr(sys, "frozen", False):

                    fibexe = Path(sys.executable).with_name("fibonachi_analysis.exe")

                else:

                    fibexe = Path(__file__).with_name("fibonachi_analysis.exe")

                cmd = [str(fibexe)]

                if payload_path is not None:
                    cmd += ["--payload", str(payload_path)]

                if self.image_path is not None:
                    cmd += ["--image", str(self.image_path)]

                if saved is not None:
                    cmd += ["--points", str(saved)]

                subprocess.Popen(cmd, shell=False)
                self._set_status("Запущен внешний анализ")

            except Exception as e:

                messagebox.showerror("Ошибка запуска",

                                     f"Не удалось запустить fibonachi_analysis.exe:\n{e}")

        # быстрый отчёт (как раньше)
        if self.img_arr is None or len(self.points) == 0:
            messagebox.showinfo("Анализ", "Нет изображения или точек для анализа.")
            return

        cy, cx = (self.img_arr.shape[0] - 1) / 2.0, (self.img_arr.shape[1] - 1) / 2.0
        radii, angles = pol_from((cy, cx), self.points)
        rc, labels, _ = cluster_rings(radii)
        ring_means = [np.mean(radii[labels == i]) for i in np.unique(labels)] if len(rc) else []
        sym = symmetry_scores(angles, radii, ring_means)

        lines = ["SAED Symmetry Analysis", "=======================", "",
                 f"Файл: {self.image_path}",
                 f"Сохранено: {saved.name if saved else '-'}",
                 f"Точек: {len(self.points)}", "", "Симметрии:"]
        for k, v in sorted(sym.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k:>7}: {v:.3f}")
        self._show_report("\n".join(lines))

    def _show_report(self, text: str):
        win = tk.Toplevel(self)
        win.title("SAED Report")
        txt = tk.Text(win, wrap="word")
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", text)
        txt.config(state=tk.DISABLED)
        self._set_status("Сформирован отчёт по симметрии")

class PointEditorApp(tk.Tk):
    """Standalone-обёртка, встраивающая редактор в корневое окно."""

    def __init__(self, input_json: str | None = None):
        super().__init__()
        self.title("SAED Editor + Analysis")
        self.geometry("1100x800")
        self.resizable(True, True)
        self.editor = PointEditor(self, input_json=input_json)
        self.editor.pack(fill=tk.BOTH, expand=True)


# -------- CLI ---------
def _parse_args(argv):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=False, help="Путь к saed_input.json")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    root = PointEditorApp(args.input)
    root.mainloop()