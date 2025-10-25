#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mix-in класс для PointEditor:
Обрабатывает состояние (Undo/Redo, Tooltips, Measurement, Ring Selection)
"""
import numpy as np
import math
import tkinter as tk
from typing import Optional, Dict, Any, List, Tuple, Union # Добавлено Union

CENTER_AS_POINT_IDX = -1

class EditorState:

    # --- Инициализация состояния ---
    def _initialize_state(self):
        # Tooltip
        self._tooltip = None
        self._tooltip_idx = None
        # Measurement
        self._measure_start_idx: Optional[int] = None
        self._measure_start_point: Optional[tuple[float, float]] = None
        self._measure_preview_end: Optional[tuple[float, float]] = None
        self._measure_preview_artist = None
        self._measure_line_artist = None
        self._measure_annotation = None
        self._measurement: Optional[dict[str, object]] = None
        # Ring Selection
        self._ring_select_active: bool = False
        self._ring_select_center_yx: Optional[tuple[float, float]] = None
        self._ring_select_radius: float = 10.0
        self._ring_select_thickness: float = 5.0
        self._ring_select_indices: set[int] = set()
        self._ring_select_artist: Optional[list] = None
        # --- ИЗМЕНЕНИЕ: Типы точек (int ID или str "unknown") и ПЛОЩАДИ ---
        self.point_types: list[Union[str, int]] = []
        self.areas: np.ndarray = np.zeros((0,), dtype=float) # Массив для площадей
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    # ... (Методы Measurement без изменений) ...
    def _remove_measure_preview_artist(self) -> bool:
        if hasattr(self, '_measure_preview_artist') and self._measure_preview_artist is not None:
            try: self._measure_preview_artist.remove(); self._measure_preview_artist = None; return True
            except Exception: pass; self._measure_preview_artist = None
        return False
    def _remove_measurement_artists(self) -> bool:
        removed = False
        if hasattr(self, '_measure_line_artist') and self._measure_line_artist is not None:
            try: self._measure_line_artist.remove(); removed = True; self._measure_line_artist = None
            except Exception: pass; self._measure_line_artist = None
        if hasattr(self, '_measure_annotation') and self._measure_annotation is not None:
            try: self._measure_annotation.remove(); removed = True; self._measure_annotation = None
            except Exception: pass; self._measure_annotation = None
        return removed
    def _cancel_measurement_preview(self) -> bool:
        removed_artist = self._remove_measure_preview_artist()
        has_preview_state = self._measure_preview_end is not None
        self._measure_preview_end = None
        return removed_artist or has_preview_state
    def _clear_measurement_result(self) -> bool:
        removed_artists = self._remove_measurement_artists()
        had_measurement = self._measurement is not None
        self._measurement = None
        return removed_artists or had_measurement
    def _start_measurement(self, idx: int) -> None:
        if self.points is None or idx < 0 or idx >= len(self.points): return
        self._clear_measurement_result()
        self._measure_start_idx = idx
        y0, x0 = self.points[idx]; self._measure_start_point = (float(y0), float(x0))
        self._measure_preview_end = None; self._remove_measure_preview_artist()
    def _start_measurement_from_center(self) -> bool:
        if not self.overlay or not self.overlay.get("center"): return False
        self._clear_measurement_result()
        center_data = self.overlay["center"]; cy, cx = float(center_data["y"]), float(center_data["x"])
        self._measure_start_idx = CENTER_AS_POINT_IDX; self._measure_start_point = (cy, cx)
        self._measure_preview_end = None; self._remove_measure_preview_artist()
        return True
    def _update_measurement_preview(self, pos: tuple[float, float] | None) -> None:
        if self._measure_start_idx is None or self._measure_start_point is None: return
        if pos is None:
            self._measure_preview_end = None
            if self._remove_measure_preview_artist():
                if hasattr(self, "canvas"): self.canvas.draw_idle()
            return
        y1, x1 = pos; self._measure_preview_end = (float(y1), float(x1)); y0, x0 = self._measure_start_point
        current_xlim = self.ax.get_xlim(); current_ylim = self.ax.get_ylim()
        if not hasattr(self, '_measure_preview_artist') or self._measure_preview_artist is None:
            (line,) = self.ax.plot([x0, x1], [y0, y1], color="#ffcc33", lw=1.6, ls="--", alpha=0.9, scalex=False, scaley=False, zorder=10)
            self._measure_preview_artist = line
        else: self._measure_preview_artist.set_data([x0, x1], [y0, y1])
        self.ax.set_xlim(current_xlim); self.ax.set_ylim(current_ylim)
        if hasattr(self, "canvas"): self.canvas.draw_idle()
    def _finalize_measurement(self, end_yx: tuple[float, float]) -> bool:
        if self._measure_start_point is None or self._measure_start_idx is None: return False
        start_y, start_x = self._measure_start_point; end_y, end_x = end_yx
        length = float(np.hypot(end_x - start_x, end_y - start_y))
        self._measurement = {"start_yx": (start_y, start_x), "end_yx": (end_y, end_x), "length": length}
        self._remove_measurement_artists(); self._cancel_measurement_preview()
        self._measure_start_idx = None; self._measure_start_point = None
        return True
    def _finalize_measurement_to_center(self) -> bool:
        if not self.overlay or not self.overlay.get("center"):
             self._cancel_measurement_preview(); self._measure_start_idx = None; self._measure_start_point = None
             return False
        center_data = self.overlay["center"]; center_y, center_x = float(center_data["y"]), float(center_data["x"])
        return self._finalize_measurement(end_yx=(center_y, center_x))

    # --- ИЗМЕНЕНИЕ: Tooltip показывает Radius, Intensity(%), Area, Group ID / Type ---
    def _clear_tooltip(self, *, keep_measure: bool = False, keep_preview: bool = False):
        removed_tooltip = False
        if hasattr(self, '_tooltip') and self._tooltip is not None:
            try: self._tooltip.remove(); self._tooltip = None; self._tooltip_idx = None; removed_tooltip = True
            except Exception: pass; self._tooltip = None; self._tooltip_idx = None
        removed_preview = False
        if not keep_preview: removed_preview = self._cancel_measurement_preview()
        removed_measure = False
        if not keep_measure: removed_measure = self._clear_measurement_result()
        if (removed_tooltip or removed_preview or removed_measure) and hasattr(self, "canvas"): self.canvas.draw_idle()

    def _show_tooltip_for_idx(self, idx):
        if self.points is None or idx is None or idx < 0 or idx >= len(self.points): return

        y, x = self.points[idx]
        intensity = None
        if hasattr(self, 'values') and self.values is not None and idx < len(self.values):
            intensity = float(self.values[idx])

        area = None
        if hasattr(self, 'areas') and self.areas is not None and idx < len(self.areas):
            area = float(self.areas[idx])

        point_type_or_id = "N/A"
        type_label = "Type" # Метка для тултипа
        if hasattr(self, 'point_types') and idx < len(self.point_types):
            type_val = self.point_types[idx]
            if isinstance(type_val, int):
                 point_type_or_id = str(type_val)
                 type_label = "Group ID" # Меняем метку, если это ID
            elif isinstance(type_val, str):
                 point_type_or_id = type_val
                 # type_label остается "Type"

        radius = None
        if self.overlay and self.overlay.get("center"):
            center_data = self.overlay["center"]
            if isinstance(center_data, dict) and "x" in center_data and "y" in center_data:
                cy, cx = float(center_data["y"]), float(center_data["x"])
                radius = float(math.hypot(x - cx, y - cy))

        txt_lines = []
        if radius is not None:
            txt_lines.append(f"Radius: {radius:.1f} px")
        else:
            txt_lines.append("Radius: N/A (no center)")
        if intensity is not None:
            txt_lines.append(f"Intensity: {intensity:.1f} %")
        else:
            txt_lines.append("Intensity: N/A")
        if area is not None:
             txt_lines.append(f"Area: {area:.1f} px²")
        else:
             txt_lines.append("Area: N/A")
        txt_lines.append(f"{type_label}: {point_type_or_id}") # Используем обновленную метку

        txt = "\n".join(txt_lines)

        self._clear_tooltip()
        self._tooltip = self.ax.annotate(
            txt, xy=(x, y), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9),
            fontsize=9, zorder=20
        )
        self._tooltip_idx = idx
        self.canvas.draw_idle()
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    # ---------- Undo/Redo ----------
    def _make_snapshot(self):
        # ... (код центра, точек, значений без изменений) ...
        center_data = None
        if self.overlay and isinstance(self.overlay.get("center"), dict): c = self.overlay["center"];
        if "x" in c and "y" in c: center_data = {"x": float(c["x"]), "y": float(c["y"])}
        points_copy = self.points.copy() if self.points is not None else np.zeros((0, 2))
        values_copy = self.values.copy() if self.values is not None else np.zeros((0,))
        # --- ИЗМЕНЕНО: Копируем типы (int/str) и ПЛОЩАДИ ---
        types_copy = list(self.point_types) if hasattr(self, 'point_types') else [] # Копируем как есть
        areas_copy = self.areas.copy() if hasattr(self, 'areas') and self.areas is not None else np.zeros((0,))
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        return {
            "points": points_copy, "values": values_copy, "center": center_data,
            "view_cx": self.view_cx, "view_cy": self.view_cy, "zoom_val": self.zoom_val,
            "measurement": self._measurement,
            "dead_radius": self.overlay.get("dead_radius", 0.0) if self.overlay else 0.0,
            "search_radius": self.overlay.get("search_radius", 0.0) if self.overlay else 0.0,
            "ring_select_indices": list(self._ring_select_indices),
            "point_types": types_copy, # Сохраняем типы/ID
            "areas": areas_copy,       # Сохраняем площади
        }

    def _push_undo(self):
        # ... (код без изменений) ...
        if self._history_cap > 0:
            self._undo.append(self._make_snapshot())
            if len(self._undo) > self._history_cap: self._undo.pop(0)
            self._redo.clear()

    def _apply_snapshot(self, snap):
        # ... (код сброса состояния без изменений) ...
        self.center_dragging = False
        if hasattr(self, 'rect_artist') and self.rect_artist is not None:
            try:
                self.rect_artist.remove()
            except Exception:
                pass
            self.rect_artist = None
        self.rect_start = None
        self._cancel_measurement_preview()
        self._cancel_ring_selection()

        # Восстанавливаем точки, значения, типы, ПЛОЩАДИ
        self.points = snap["points"].copy()
        self.values = snap["values"].copy()
        # --- ИЗМЕНЕНО: Восстанавливаем типы (int/str) и ПЛОЩАДИ ---
        self.point_types = list(snap.get("point_types", [])) # Восстанавливаем как есть
        self.areas = snap.get("areas", np.zeros(len(self.points))).copy() # Восстанавливаем площади

        # Проверяем консистентность
        if len(self.point_types) != len(self.points):
             print("Warning: Snapshot point/type mismatch. Resetting types.")
             self.point_types = ["unknown"] * len(self.points)
        if len(self.areas) != len(self.points):
             print("Warning: Snapshot point/area mismatch. Resetting areas.")
             self.areas = np.zeros(len(self.points), dtype=float) # Заполняем нулями
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        # ... (код восстановления overlay, view, measurement, ring_select_indices без изменений) ...
        if self.overlay is None: self.overlay = {}
        self.overlay["center"] = snap["center"]
        self.overlay["dead_radius"] = snap.get("dead_radius", 0.0)
        self.overlay["search_radius"] = snap.get("search_radius", 0.0)
        self.view_cx = snap.get("view_cx"); self.view_cy = snap.get("view_cy")
        self.zoom_val = snap.get("zoom_val", 0)
        if hasattr(self, 'zoom_var'): self.zoom_var.set(self.zoom_val)
        self._measurement = snap.get("measurement")
        self._ring_select_indices = set(snap.get("ring_select_indices", []))


        self._ensure_view_center()
        self._update_zoom_hint()

    def _undo_btn(self, event=None):
        # ... (код без изменений) ...
        if hasattr(event, 'widget') and isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)): return
        if not self._undo: self._set_status("Nothing to undo."); return
        self._clear_tooltip()
        current_state = self._make_snapshot(); self._redo.append(current_state)
        if len(self._redo) > self._history_cap: self._redo.pop(0)
        snap_to_restore = self._undo.pop(); self._apply_snapshot(snap_to_restore)
        self._redraw(); self._set_status("Undo successful.")

    def _redo_btn(self, event=None):
        # ... (код без изменений) ...
        if hasattr(event, 'widget') and isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)): return
        if not self._redo: self._set_status("Nothing to redo."); return
        self._clear_tooltip()
        current_state = self._make_snapshot(); self._undo.append(current_state)
        if len(self._undo) > self._history_cap: self._undo.pop(0)
        snap_to_restore = self._redo.pop(); self._apply_snapshot(snap_to_restore)
        self._redraw(); self._set_status("Redo successful.")

    def _cancel_ring_selection(self) -> bool:
        """Сбрасывает состояние выбора кольцом."""
        # ... (код без изменений) ...
        cleared_artist = False
        if hasattr(self, '_ring_select_artist') and self._ring_select_artist:
            try:
                if isinstance(self._ring_select_artist, list):
                    for patch in self._ring_select_artist: patch.remove()
                elif self._ring_select_artist is not None: self._ring_select_artist.remove()
                cleared_artist = True
            except Exception: pass
            self._ring_select_artist = None
        was_active = self._ring_select_active
        had_indices = bool(self._ring_select_indices)
        self._ring_select_active = False
        self._ring_select_center_yx = None
        self._ring_select_indices.clear()
        return cleared_artist or was_active or had_indices