#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Класс Состояния UI (Refactored)
-------------------------------
Хранит состояние, не связанное с данными точек:
- Текущий замер
- Выделение кольцом
- Состояние подсказки (Tooltip)
- Состояние перетаскивания (центр, прямоугольник)
- Хранит ссылки на matplotlib artists (линии, текст)
"""
import numpy as np
import math
import tkinter as tk
from typing import Optional, Dict, Any, List, Tuple, Set

# Специальный индекс для обозначения центра (остается здесь)
CENTER_AS_POINT_IDX = -1


class EditorUIState:

    def __init__(self):
        # --- Состояние UI ---
        self.rect_start: Optional[Tuple[float, float]] = None
        self.rect_artist = None
        self.center_dragging: bool = False

        # --- Tooltip ---
        self.tooltip = None
        self.tooltip_idx: Optional[int] = None

        # --- Measurement ---
        self.measure_start_idx: Optional[int] = None
        self.measure_start_point: Optional[Tuple[float, float]] = None
        self.measure_preview_end: Optional[Tuple[float, float]] = None
        self.measure_preview_artist = None
        self.measure_line_artist = None
        self.measure_annotation = None
        self.measurement: Optional[Dict[str, Any]] = None

        # --- Ring Selection ---
        self.ring_select_active: bool = False
        self.ring_select_center_yx: Optional[Tuple[float, float]] = None
        self.ring_select_radius: float = 10.0
        self.ring_select_thickness: float = 5.0
        self.ring_select_indices: Set[int] = set()
        self.ring_select_artist: Optional[list] = None  # Храним artists кольца

    # --- Методы управления состоянием ---

    def cancel_all_interactions(self) -> bool:
        """Сбрасывает все текущие взаимодействия (замер, выделение)."""
        cleared_measure = False
        if self.cancel_measurement_preview():  # Отменяет превью (пунктир)
            cleared_measure = True
        if self.measure_start_idx is not None:  # Сбрасывает начало замера
            self.measure_start_idx = None
            self.measure_start_point = None
            cleared_measure = True

        cleared_ring = self.cancel_ring_selection()
        return cleared_measure or cleared_ring

    # --- Measurement Artists & State ---

    def remove_measure_preview_artist(self) -> bool:
        if self.measure_preview_artist is not None:
            try:
                self.measure_preview_artist.remove(); self.measure_preview_artist = None; return True
            except Exception:
                pass; self.measure_preview_artist = None
        return False

    def remove_measurement_artists(self) -> bool:
        removed = False
        if self.measure_line_artist is not None:
            try:
                self.measure_line_artist.remove(); removed = True; self.measure_line_artist = None
            except Exception:
                pass; self.measure_line_artist = None
        if self.measure_annotation is not None:
            try:
                self.measure_annotation.remove(); removed = True; self.measure_annotation = None
            except Exception:
                pass; self.measure_annotation = None
        return removed

    def cancel_measurement_preview(self) -> bool:
        removed_artist = self.remove_measure_preview_artist()
        has_preview_state = self.measure_preview_end is not None
        self.measure_preview_end = None
        return removed_artist or has_preview_state

    def clear_measurement_result(self, view=None) -> bool:
        removed_artists = self.remove_measurement_artists()
        had_measurement = self.measurement is not None
        self.measurement = None
        if (removed_artists or had_measurement) and view:
            view.canvas.draw_idle()
        return removed_artists or had_measurement

    def start_measurement(self, idx: int, model):
        """Начинает замер от точки."""
        if model.is_empty() or idx < 0 or idx >= len(model.points): return
        y0, x0 = model.points[idx]
        self.measure_start_idx = idx
        self.measure_start_point = (float(y0), float(x0))
        self.measure_preview_end = None
        self.remove_measure_preview_artist()

    def start_measurement_from_center(self, center: Tuple[float, float]) -> bool:
        """Начинает замер от центра."""
        cy, cx = center
        self.measure_start_idx = CENTER_AS_POINT_IDX
        self.measure_start_point = (cy, cx)
        self.measure_preview_end = None
        self.remove_measure_preview_artist()
        return True

    def update_measurement_preview(self, pos: Optional[Tuple[float, float]], view):
        """Обновляет пунктирную линию превью замера."""
        if self.measure_start_idx is None or self.measure_start_point is None: return
        if pos is None:
            self.measure_preview_end = None
            if self.remove_measure_preview_artist():
                view.canvas.draw_idle()
            return

        y1, x1 = pos
        self.measure_preview_end = (float(y1), float(x1))
        # Отрисовка делегирована view.draw_measurement_overlays()
        view.ax.set_xlim(view.ax.get_xlim())  # Фиксируем zoom
        view.ax.set_ylim(view.ax.get_ylim())
        view.draw_measurement_overlays(self)  # Перерисовываем только оверлей
        view.canvas.draw_idle()

    def finalize_measurement(self, end_yx: Tuple[float, float]) -> bool:
        if self.measure_start_point is None or self.measure_start_idx is None: return False
        start_y, start_x = self.measure_start_point
        end_y, end_x = end_yx
        length = float(np.hypot(end_x - start_x, end_y - start_y))
        self.measurement = {"start_yx": (start_y, start_x), "end_yx": (end_y, end_x), "length": length}

        self.remove_measurement_artists()
        self.cancel_measurement_preview()
        self.measure_start_idx = None
        self.measure_start_point = None
        return True

    def finalize_measurement_to_center(self, center: Optional[Tuple[float, float]]) -> bool:
        if not center:
            self.cancel_measurement_preview()
            self.measure_start_idx = None
            self.measure_start_point = None
            return False
        center_y, center_x = center
        return self.finalize_measurement(end_yx=(center_y, center_x))

    # --- Tooltip ---

    def clear_tooltip(self, view=None):
        """Очищает подсказку и/или замеры."""
        removed_tooltip = False
        if self.tooltip is not None:
            try:
                self.tooltip.remove(); self.tooltip = None; self.tooltip_idx = None; removed_tooltip = True
            except Exception:
                pass; self.tooltip = None; self.tooltip_idx = None

        # При очистке подсказки мы НЕ сбрасываем замеры

        if removed_tooltip and view:
            view.canvas.draw_idle()

    def show_tooltip_for_idx(self, idx: int, model, center: Optional[Tuple[float, float]], view):
        """Показывает подсказку (читает данные из Модели)."""
        if model.is_empty() or idx < 0 or idx >= len(model.points): return

        y, x = model.points[idx]

        # --- Запрашиваем данные у Модели ---
        txt = model.get_point_data_for_tooltip(idx, center)
        # ---

        self.clear_tooltip(view)  # Очищаем старую
        self.tooltip = view.ax.annotate(
            txt, xy=(x, y), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9),
            fontsize=9, zorder=20
        )
        self.tooltip_idx = idx
        view.canvas.draw_idle()

    # --- Ring Selection ---

    def cancel_ring_selection(self) -> bool:
        """Сбрасывает состояние выбора кольцом (artists удаляются во View)."""
        was_active = self.ring_select_active
        had_indices = bool(self.ring_select_indices)

        self.ring_select_active = False
        self.ring_select_center_yx = None
        self.ring_select_indices.clear()

        # Artist удаляется в view.remove_ring_preview_artist()
        cleared_artist = self.ring_select_artist is not None
        if cleared_artist:
            self.ring_select_artist = None  # View должен удалить его

        return cleared_artist or was_active or had_indices

    def update_ring_indices_after_delete(self, indices_to_delete: List[int]):
        """Обновляет self.ring_select_indices после удаления точек."""
        if not self.ring_select_indices:
            return

        indices_to_delete_set = set(indices_to_delete)
        deleted_selected_indices = self.ring_select_indices.intersection(indices_to_delete_set)

        if not deleted_selected_indices and not any(i < max(self.ring_select_indices) for i in indices_to_delete):
            # Если ничего не удалено из выделенных и все удаленные
            # индексы "после" выделенных, ничего не делаем
            return

        self.ring_select_indices.difference_update(deleted_selected_indices)

        # Считаем смещение для каждого оставшегося индекса
        indices_to_delete_sorted = sorted(indices_to_delete)
        new_ring_indices = set()

        for old_idx in self.ring_select_indices:
            # Считаем, сколько точек было удалено "до" old_idx
            num_deleted_before = sum(1 for del_idx in indices_to_delete_sorted if del_idx < old_idx)
            new_ring_indices.add(old_idx - num_deleted_before)

        self.ring_select_indices = new_ring_indices