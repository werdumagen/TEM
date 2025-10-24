#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mix-in класс для PointEditor:
Обрабатывает состояние (Undo/Redo, Tooltips, Measurement, Ring Selection)
"""
import numpy as np
import math # Добавлен импорт math
import tkinter as tk # Добавлен импорт tk

# --- НОВОЕ: Специальный индекс для центра ---
CENTER_AS_POINT_IDX = -1
# --- КОНЕЦ НОВОГО ---


class EditorState:

    # --- Инициализация состояния ---
    def _initialize_state(self):
        # Tooltip
        self._tooltip = None
        self._tooltip_idx = None
        # Measurement
        self._measure_start_idx: Optional[int] = None # Может быть CENTER_AS_POINT_IDX
        self._measure_start_point: Optional[tuple[float, float]] = None # Координаты (y, x)
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

    # ---------- Tooltip и измерения ----------
    def _remove_measure_preview_artist(self) -> bool:
        if hasattr(self, '_measure_preview_artist') and self._measure_preview_artist is not None:
            try:
                self._measure_preview_artist.remove()
            except Exception: pass
            self._measure_preview_artist = None
            return True
        return False

    def _remove_measurement_artists(self) -> bool:
        removed = False
        if hasattr(self, '_measure_line_artist') and self._measure_line_artist is not None:
            try: self._measure_line_artist.remove(); removed = True
            except Exception: pass
            self._measure_line_artist = None
        if hasattr(self, '_measure_annotation') and self._measure_annotation is not None:
            try: self._measure_annotation.remove(); removed = True
            except Exception: pass
            self._measure_annotation = None
        return removed

    def _cancel_measurement_preview(self) -> bool:
        """Cancels an ongoing measurement preview (dashed line) *without* resetting start state."""
        removed_artist = self._remove_measure_preview_artist()
        has_preview_state = self._measure_preview_end is not None
        self._measure_preview_end = None
        # НЕ сбрасываем _measure_start_idx / _measure_start_point здесь
        return removed_artist or has_preview_state

    def _clear_measurement_result(self) -> bool:
        removed_artists = self._remove_measurement_artists()
        had_measurement = self._measurement is not None
        self._measurement = None
        return removed_artists or had_measurement

    def _start_measurement(self, idx: int) -> None:
        """Начинает замер от обычной точки."""
        if self.points is None or idx < 0 or idx >= len(self.points): return
        self._clear_measurement_result()
        self._measure_start_idx = idx
        y0, x0 = self.points[idx]
        self._measure_start_point = (float(y0), float(x0))
        self._measure_preview_end = None
        self._remove_measure_preview_artist()

    # --- НОВЫЙ МЕТОД: Начало замера от центра ---
    def _start_measurement_from_center(self) -> bool:
        """Начинает замер от центра."""
        if not self.overlay or not self.overlay.get("center"):
            return False # Центр не определен

        self._clear_measurement_result()
        center_data = self.overlay["center"]
        cy, cx = float(center_data["y"]), float(center_data["x"])

        self._measure_start_idx = CENTER_AS_POINT_IDX # Используем специальный индекс
        self._measure_start_point = (cy, cx)
        self._measure_preview_end = None
        self._remove_measure_preview_artist()
        return True
    # --- КОНЕЦ НОВОГО МЕТОДА ---

    def _update_measurement_preview(self, pos: tuple[float, float] | None) -> None:
        if self._measure_start_idx is None or self._measure_start_point is None: return

        if pos is None:
            self._measure_preview_end = None
            if self._remove_measure_preview_artist():
                if hasattr(self, "canvas"): self.canvas.draw_idle()
            return

        y1, x1 = pos
        self._measure_preview_end = (float(y1), float(x1))
        y0, x0 = self._measure_start_point

        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()

        if not hasattr(self, '_measure_preview_artist') or self._measure_preview_artist is None:
            (line,) = self.ax.plot(
                [x0, x1], [y0, y1], color="#ffcc33", linewidth=1.6, linestyle="--",
                alpha=0.9, scalex=False, scaley=False, zorder=10
            )
            self._measure_preview_artist = line
        else:
            self._measure_preview_artist.set_data([x0, x1], [y0, y1])

        self.ax.set_xlim(current_xlim)
        self.ax.set_ylim(current_ylim)

        if hasattr(self, "canvas"): self.canvas.draw_idle()

    # --- ОБНОВЛЕНО: _finalize_measurement теперь принимает end_yx ---
    def _finalize_measurement(self, end_yx: tuple[float, float]) -> bool:
        """Stores the final measurement data using provided end coordinates."""
        if self._measure_start_point is None or self._measure_start_idx is None:
             return False # Замер не был начат

        start_y, start_x = self._measure_start_point
        end_y, end_x = end_yx # Используем переданные координаты
        length = float(np.hypot(end_x - start_x, end_y - start_y))

        self._measurement = {
            "start_yx": (start_y, start_x),
            "end_yx": (end_y, end_x), # Сохраняем конечные координаты
            "length": length,
        }

        # Очищаем и сбрасываем состояние
        self._remove_measurement_artists()
        self._cancel_measurement_preview()
        self._measure_start_idx = None
        self._measure_start_point = None

        return True
    # --- КОНЕЦ ОБНОВЛЕНИЯ ---

    # --- НОВЫЙ МЕТОД: Завершение замера до центра ---
    def _finalize_measurement_to_center(self) -> bool:
        """Завершает измерение до текущего положения центра."""
        if not self.overlay or not self.overlay.get("center"):
             # Центр не определен, отменяем
             self._cancel_measurement_preview()
             self._measure_start_idx = None
             self._measure_start_point = None
             return False

        center_data = self.overlay["center"]
        center_y, center_x = float(center_data["y"]), float(center_data["x"])

        # Вызываем основной метод финализации с координатами центра
        return self._finalize_measurement(end_yx=(center_y, center_x))
    # --- КОНЕЦ НОВОГО МЕТОДА ---

    def _clear_tooltip(self, *, keep_measure: bool = False, keep_preview: bool = False):
        removed_tooltip = False
        if hasattr(self, '_tooltip') and self._tooltip is not None:
            try: self._tooltip.remove()
            except Exception: pass
            self._tooltip = None
            self._tooltip_idx = None
            removed_tooltip = True

        removed_preview = False
        if not keep_preview:
            # Отменяем только превью, но не сбрасываем старт
            removed_preview = self._cancel_measurement_preview()

        removed_measure = False
        if not keep_measure:
            removed_measure = self._clear_measurement_result()

        if (removed_tooltip or removed_preview or removed_measure) and hasattr(self, "canvas"):
            self.canvas.draw_idle()

    def _show_tooltip_for_idx(self, idx):
        if self.points is None or idx is None or idx < 0 or idx >= len(self.points): return

        y, x = self.points[idx]
        inten = None
        is_percentile_like = False

        if hasattr(self, '_percent_map') and self._percent_map is not None:
            H, W = self._percent_map.shape[:2]
            yi = max(0, min(H - 1, int(round(y))))
            xi = max(0, min(W - 1, int(round(x))))
            inten = float(self._percent_map[yi, xi])
            is_percentile_like = True
        elif hasattr(self, 'values') and self.values is not None and idx < len(self.values):
            inten = float(self.values[idx])
            is_percentile_like = 0 <= inten <= 100

        txt = f"x={x:.1f}, y={y:.1f}\n"
        if inten is not None:
            unit = "%" if is_percentile_like else "raw"
            txt += f"I={inten:.1f} {unit}"
        else:
            txt += "Intensity: N/A"

        self._clear_tooltip()

        self._tooltip = self.ax.annotate(
            txt, xy=(x, y), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9),
            fontsize=9, zorder=20
        )
        self._tooltip_idx = idx
        self.canvas.draw_idle()

    # ---------- Undo/Redo ----------
    def _make_snapshot(self):
        center_data = None
        if self.overlay and isinstance(self.overlay.get("center"), dict):
            c = self.overlay["center"]
            if "x" in c and "y" in c:
                center_data = {"x": float(c["x"]), "y": float(c["y"])}

        points_copy = self.points.copy() if self.points is not None else np.zeros((0, 2))
        values_copy = self.values.copy() if self.values is not None else np.zeros((0,))

        return {
            "points": points_copy,
            "values": values_copy,
            "center": center_data,
            "view_cx": self.view_cx,
            "view_cy": self.view_cy,
            "zoom_val": self.zoom_val,
            "measurement": self._measurement,
            "dead_radius": self.overlay.get("dead_radius", 0.0) if self.overlay else 0.0,
            "search_radius": self.overlay.get("search_radius", 0.0) if self.overlay else 0.0,
            "ring_select_indices": list(self._ring_select_indices),
        }

    def _push_undo(self):
        if self._history_cap > 0:
            self._undo.append(self._make_snapshot())
            if len(self._undo) > self._history_cap:
                self._undo.pop(0)
            self._redo.clear()

    def _apply_snapshot(self, snap):
        self.center_dragging = False
        if hasattr(self, 'rect_artist') and self.rect_artist is not None:
            try: self.rect_artist.remove()
            except Exception: pass
            self.rect_artist = None
        self.rect_start = None
        # Отменяем превью, но не сбрасываем старт
        self._cancel_measurement_preview()
        self._cancel_ring_selection()

        self.points = snap["points"].copy()
        self.values = snap["values"].copy()

        if self.overlay is None: self.overlay = {}
        self.overlay["center"] = snap["center"]
        self.overlay["dead_radius"] = snap.get("dead_radius", 0.0)
        self.overlay["search_radius"] = snap.get("search_radius", 0.0)

        self.view_cx = snap.get("view_cx")
        self.view_cy = snap.get("view_cy")
        self.zoom_val = snap.get("zoom_val", 0)
        if hasattr(self, 'zoom_var'): self.zoom_var.set(self.zoom_val)

        self._measurement = snap.get("measurement")
        self._ring_select_indices = set(snap.get("ring_select_indices", []))

        # --- ИЗМЕНЕНИЕ: НЕ сбрасываем состояние измерения при откате ---
        # self._measure_start_idx = None # Убрано
        # self._measure_start_point = None # Убрано
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        self._ensure_view_center()
        self._update_zoom_hint()

    def _undo_btn(self, event=None):
        if hasattr(event, 'widget') and isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)): return
        if not self._undo:
            self._set_status("Nothing to undo.")
            return

        self._clear_tooltip()
        current_state = self._make_snapshot()
        self._redo.append(current_state)
        if len(self._redo) > self._history_cap: self._redo.pop(0)

        snap_to_restore = self._undo.pop()
        self._apply_snapshot(snap_to_restore)
        self._redraw()
        self._set_status("Undo successful.")

    def _redo_btn(self, event=None):
        if hasattr(event, 'widget') and isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)): return
        if not self._redo:
            self._set_status("Nothing to redo.")
            return

        self._clear_tooltip()
        current_state = self._make_snapshot()
        self._undo.append(current_state)
        if len(self._undo) > self._history_cap: self._undo.pop(0)

        snap_to_restore = self._redo.pop()
        self._apply_snapshot(snap_to_restore)
        self._redraw()
        self._set_status("Redo successful.")

    def _cancel_ring_selection(self) -> bool:
        """Сбрасывает состояние выбора кольцом."""
        cleared_artist = False
        if hasattr(self, '_ring_select_artist') and self._ring_select_artist:
            try:
                if isinstance(self._ring_select_artist, list):
                    for patch in self._ring_select_artist: patch.remove()
                elif self._ring_select_artist is not None:
                    self._ring_select_artist.remove()
                cleared_artist = True
            except Exception: pass
            self._ring_select_artist = None

        was_active = self._ring_select_active
        had_indices = bool(self._ring_select_indices)

        self._ring_select_active = False
        self._ring_select_center_yx = None
        self._ring_select_indices.clear()

        return cleared_artist or was_active or had_indices