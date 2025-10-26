#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Класс-обработчик для PointEditor (Рефакторинг)
-----------------------------------------------
Содержит всю логику обработки событий мыши, клавиатуры и
бизнес-логику (например, авто-группировку).

Больше не mix-in. Получает 'controller' (PointEditor)
для доступа к модели (controller.model),
состоянию UI (controller.ui_state) и
методам (controller.redraw()).
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import messagebox, filedialog
from pathlib import Path
import json
from typing import Optional, Dict, Any, List, Tuple, Union
from collections import Counter

# Импорт специального индекса
from saed_editor_state_ui import CENTER_AS_POINT_IDX

# --- Импорт функций анализа (остаются здесь, т.к. это логика "обработчика") ---
from scipy.signal import find_peaks


def pol_from(center, pts):
    """Рассчитывает полярные координаты (r, a) для точек (pts) относительно центра."""
    cy, cx = center
    dy = pts[:, 0] - cy
    dx = pts[:, 1] - cx
    r = np.hypot(dx, dy)
    a = np.degrees(np.arctan2(dy, dx))
    a = (a + 360) % 360
    return r, a


def symmetry_scores(angles, radii, ring_means, top_rings=3):
    """Рассчитывает оценки симметрии."""
    out = {}
    if ring_means.size == 0: return out
    effective_top_rings = min(top_rings, len(ring_means))
    if effective_top_rings == 0: return out

    idx = effective_top_rings - 1
    maxR = ring_means[idx] * 1.15
    mask = radii <= maxR
    ang_sel = angles[mask]
    if len(ang_sel) == 0: return out

    for k in [4, 6, 8, 10, 12]:
        period = 360.0 / k
        phases_deg = (ang_sel % period) * k
        phases_rad = np.deg2rad(phases_deg)
        C = np.cos(phases_rad).mean()
        S = np.sin(phases_rad).mean()
        out[f"{k}-fold"] = float(np.hypot(C, S))
    return out


def cluster_rings(radii, bins=100, prominence_factor=0.03, min_prominence=2):
    """Кластеризует радиусы для поиска колец."""
    if len(radii) == 0:
        return np.array([]), np.zeros(0, dtype=int), ([], [])
    hist, edges = np.histogram(radii, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    if hist.max() > 0:
        prominence = max(min_prominence, hist.max() * prominence_factor)
    else:
        prominence = min_prominence

    pk, _ = find_peaks(hist, prominence=prominence)
    ring_centers = centers[pk]
    if len(ring_centers) == 0:
        return np.array([]), np.zeros_like(radii, dtype=int), (hist.tolist(), edges.tolist())

    labels = np.argmin(np.abs(radii[:, None] - ring_centers[None, :]), axis=1)
    return ring_centers, labels, (hist.tolist(), edges.tolist())


# --- КОНЕЦ ФУНКЦИЙ АНАЛИЗА ---


class EditorEventHandlers:

    def __init__(self, controller):
        """
        Инициализирует обработчики.
        :param controller: Экземпляр PointEditor, дающий доступ к
                           .model, .ui_state, .view, .redraw(), .set_status() и т.д.
        """
        self.controller = controller
        self._auto_grouping_active = False

    def connect_mpl_events(self, canvas):
        """Подключает обработчики к Matplotlib canvas."""
        canvas.mpl_connect("button_press_event", self._on_down)
        canvas.mpl_connect("button_release_event", self._on_up)
        canvas.mpl_connect("motion_notify_event", self._on_move)
        canvas.mpl_connect("key_press_event", self._on_key)
        canvas.mpl_connect('scroll_event', self._on_scroll)

    def on_zoom_change_tk(self, val=None):
        """Callback для Tkinter-слайдера зума."""
        try:
            new_val = float(val) if val is not None else self.controller.zoom_val
            new_zoom = max(0, min(100, int(round(new_val))))
        except (ValueError, TypeError):
            new_zoom = 0

        if new_zoom != self.controller.zoom_val:
            self.controller.zoom_val = new_zoom
            # Обновляем значение слайдера, если оно отличается
            if hasattr(self.controller, "zoom_var"):
                current_slider_val = int(round(self.controller.zoom_var.get()))
                if current_slider_val != self.controller.zoom_val:
                    self.controller.zoom_var.set(self.controller.zoom_val)

            self.controller.update_zoom_hint()
            self.controller.ui_state.clear_tooltip()
            self.controller.redraw()

    # ---------- ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ОБРАБОТЧИКОВ ---

    def _cancel_all_interactions(self, *, keep_status: bool = False) -> bool:
        """Отменяет текущие UI-взаимодействия (замер, выделение)."""
        # Логика сброса перенесена в ui_state
        cleared = self.controller.ui_state.cancel_all_interactions()

        cleared_drag = self.controller.ui_state.center_dragging
        self.controller.ui_state.center_dragging = False

        if not keep_status and (cleared or cleared_drag):
            self.controller.set_status(self.controller.default_status)
        return cleared or cleared_drag

    # --- Методы для выбора кольцом (Ring Selection) ---

    def _start_ring_selection(self, pos_yx: tuple[float, float]) -> None:
        """Начинает режим выбора кольцом."""
        center = self.controller.get_center()
        if not center:
            self.controller.set_status("Cannot start ring selection: Center is not defined.")
            return

        if self._cancel_all_interactions():
            self.controller.redraw()

        center_y, center_x = center
        cursor_y, cursor_x = pos_yx
        radius = math.hypot(cursor_x - center_x, cursor_y - center_y)

        # Обновляем ui_state
        self.controller.ui_state.ring_select_center_yx = (center_y, center_x)
        self.controller.ui_state.ring_select_radius = max(1.0, radius)
        self.controller.ui_state.ring_select_active = True
        self.controller.ui_state.ring_select_indices.clear()

        self.controller.set_status(
            "Ring selection active. Drag radius. Use +/- for thickness. LMB click to select points.")
        self.controller.redraw()

    def _update_ring_preview(self, pos_yx: tuple[float, float] | None) -> None:
        """Обновляет радиус кольца превью."""
        ui_state = self.controller.ui_state
        if not ui_state.ring_select_active or ui_state.ring_select_center_yx is None or pos_yx is None:
            return

        center_y, center_x = ui_state.ring_select_center_yx
        cursor_y, cursor_x = pos_yx
        radius = math.hypot(cursor_x - center_x, cursor_y - center_y)
        new_radius = max(1.0, radius)

        if abs(new_radius - ui_state.ring_select_radius) > 0.1:
            ui_state.ring_select_radius = new_radius
            self.controller.redraw()

    def _adjust_ring_thickness(self, delta: float) -> None:
        """Изменяет толщину кольца превью."""
        ui_state = self.controller.ui_state
        if not ui_state.ring_select_active: return

        new_thickness = ui_state.ring_select_thickness + delta
        ui_state.ring_select_thickness = max(1.0, new_thickness)
        self.controller.set_status(f"Ring thickness: {ui_state.ring_select_thickness:.1f} px. LMB click to select.")
        self.controller.redraw()

    def _select_points_in_ring(self) -> None:
        """Выделяет точки, попадающие в текущее кольцо превью."""
        ui_state = self.controller.ui_state
        model = self.controller.model

        if not ui_state.ring_select_active or ui_state.ring_select_center_yx is None or model.is_empty():
            return

        center_y, center_x = ui_state.ring_select_center_yx
        radius = ui_state.ring_select_radius
        thickness = ui_state.ring_select_thickness
        r_min = radius - thickness / 2.0
        r_max = radius + thickness / 2.0

        dy = model.points[:, 0] - center_y
        dx = model.points[:, 1] - center_x
        point_radii = np.hypot(dx, dy)
        indices_in_ring = np.where((point_radii >= r_min) & (point_radii <= r_max))[0]

        ui_state.ring_select_indices = set(indices_in_ring.tolist())
        ui_state.ring_select_active = False  # Выходим из режима рисования
        self.controller.view.remove_ring_preview_artist(ui_state)  # Передаем ui_state во view

        count = len(ui_state.ring_select_indices)
        if count > 0:
            self.controller.set_status(f"Selected {count} points. Shift+Click to add/remove. Enter to average.")
        else:
            self.controller.set_status("No points found in the ring. Selection cleared.")
        self.controller.redraw()

    def _toggle_point_in_ring_selection(self, index: int, add: bool) -> None:
        """Добавляет или удаляет точку из набора выделенных кольцом."""
        ui_state = self.controller.ui_state
        if add:
            ui_state.ring_select_indices.add(index)
            self.controller.set_status(
                f"Added point {index} ({len(ui_state.ring_select_indices)} total). Enter to average.")
        else:
            if index in ui_state.ring_select_indices:
                ui_state.ring_select_indices.remove(index)
                self.controller.set_status(
                    f"Removed point {index} ({len(ui_state.ring_select_indices)} total). Enter to average.")
            else:
                self.controller.set_status(f"Point {index} was not selected.")
        self.controller.redraw()

    def _average_selected_points_to_ring(self) -> None:
        """Усредняет радиусы выделенных точек и перемещает их."""
        ui_state = self.controller.ui_state
        model = self.controller.model

        if not ui_state.ring_select_indices or model.is_empty():
            self.controller.set_status("No points selected for averaging.")
            return

        center = self.controller.get_center()
        if not center:
            self.controller.set_status("Cannot average: Center is not defined.")
            return

        indices = list(ui_state.ring_select_indices)
        valid_indices = [i for i in indices if 0 <= i < len(model.points)]
        if len(valid_indices) < 2:
            self.controller.set_status(f"Need at least 2 valid points to average (found {len(valid_indices)}).")
            return

        self.controller.push_undo()

        cy, cx = center
        selected_points = model.points[valid_indices]
        dy = selected_points[:, 0] - cy
        dx = selected_points[:, 1] - cx
        radii = np.hypot(dx, dy)
        zero_radius_mask = radii < 1e-9
        non_zero_radii = radii[~zero_radius_mask]

        if len(non_zero_radii) == 0:
            self.controller.set_status("Cannot average: All selected points are at the center.")
            self.controller.pop_undo()  # Отменяем push_undo
            return

        average_radius = np.mean(non_zero_radii)
        unit_dx = np.divide(dx, radii, out=np.zeros_like(dx), where=radii > 1e-9)
        unit_dy = np.divide(dy, radii, out=np.zeros_like(dy), where=radii > 1e-9)
        new_dx = unit_dx * average_radius
        new_dy = unit_dy * average_radius

        # --- Обновляем точки в Модели ---
        new_points_yx = np.column_stack([cy + new_dy, cx + new_dx])
        new_values = self.controller.sample_intensities(new_points_yx)

        # Модель сама обновит все связанные данные (углы, etc.)
        model.update_points(valid_indices, new_points_yx, new_values, center)
        # --- Конец обновления ---

        count = len(valid_indices)
        self.controller.set_status(f"Averaged {count} points to radius {average_radius:.2f} px.")
        ui_state.ring_select_indices.clear()
        self.controller.redo.clear()
        self.controller.redraw()

    # --- КОНЕЦ МЕТОДОВ ДЛЯ КОЛЬЦА ---

    # --- Авто-группировка ---

    def auto_group_and_save_wrapper(self):
        """ Обертка для кнопки Auto-Group & Save Debug """
        if self._auto_grouping_active:
            messagebox.showwarning("Busy", "Auto-grouping is already running.")
            return
        try:
            self._auto_grouping_active = True
            self.controller.btn_auto_group.config(state=tk.DISABLED)
            self.controller.update_idletasks()
            self._auto_group_rings_and_save()
        except Exception as e:
            messagebox.showerror("Auto-Grouping Error", f"An error occurred:\n{e}")
            import traceback
            traceback.print_exc()
        finally:
            self._auto_grouping_active = False
            if hasattr(self.controller, 'btn_auto_group') and self.controller.btn_auto_group.winfo_exists():
                self.controller.btn_auto_group.config(state=tk.NORMAL)

    def _auto_group_rings_and_save(self):
        """Логика авто-группировки (только по радиусу + симметрия)."""
        self.controller.set_status("Starting auto-grouping...")
        model = self.controller.model

        if model.is_empty():
            self.controller.set_status("No points to group.")
            return

        center = self.controller.get_center()
        if not center:
            self.controller.set_status("Cannot group: Center is not defined.")
            return

        try:
            radius_tol_px = float(self.controller.spn_auto_radius_tol.get())
            # area_tol_perc УБРАН
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Input Error", f"Invalid tolerance value entered:\n{e}")
            self.controller.set_status("Auto-grouping cancelled due to invalid input.")
            return

        self.controller.push_undo()  # Сохраняем состояние модели перед началом

        cy, cx = center
        dead_radius = self.controller.get_dead_radius()

        # --- Рассчитываем радиусы и углы и сохраняем в модель ---
        all_radii = model.recalculate_angles(center)
        # --- Конец расчета ---

        sorted_indices = np.argsort(all_radii)

        # --- Площади больше не используются ---
        # all_areas = model.areas
        # ---

        # Типы: сначала все "unknown"
        new_point_types: list[Union[str, int]] = ["unknown"] * len(model.points)
        # Словарь для хранения ИСХОДНОГО ID группы для каждой точки
        new_initial_group_ids: Dict[int, Optional[int]] = {i: None for i in range(len(model.points))}

        assigned_indices = set()
        group_id_counter = 0
        groups_data = {}  # {initial_group_id: [indices]}

        # --- Этап 1: Группировка по Радиусу ---
        print("Starting Pass 1 (Radius grouping)...")
        for i in range(len(sorted_indices)):
            current_idx = sorted_indices[i]
            if current_idx in assigned_indices: continue
            current_radius = all_radii[current_idx]

            # 1. Находим точки в пределах допуска по радиусу
            potential_group_indices = []
            for j in range(i, len(sorted_indices)):
                check_idx = sorted_indices[j]
                if check_idx in assigned_indices: continue
                radius_diff = abs(all_radii[check_idx] - current_radius)
                if radius_diff <= radius_tol_px:
                    potential_group_indices.append(check_idx)
                else:
                    # Т.к. точки отсортированы по радиусу, дальше можно не смотреть
                    break

            # 2. Фильтр по площади УБРАН

            # 3. Назначаем ID
            if potential_group_indices:
                initial_group_id = group_id_counter # Используем счетчик как ID
                groups_data[initial_group_id] = potential_group_indices
                for idx in potential_group_indices:
                    new_initial_group_ids[idx] = initial_group_id # Сохраняем ID
                    assigned_indices.add(idx)
                group_id_counter += 1

        print(f"Pass 1 finished. Found {group_id_counter} initial groups.")

        # --- Этап 2: Определить доминирующую симметрию ---
        potential_ring_means, _, _ = cluster_rings(all_radii[all_radii > dead_radius])
        sym_scores = symmetry_scores(model.angles, all_radii, potential_ring_means)
        dominant_symmetry = 0
        if sym_scores:
            best_sym_str = max(sym_scores, key=sym_scores.get)
            dominant_symmetry = int(best_sym_str.split('-')[0])
            print(f"Dominant symmetry: {dominant_symmetry}-fold")
            scores_str = "\n".join(f"{key}: {value:.4f}" for key, value in sym_scores.items())
            ring_str = f"Ring means used for scores:\n{np.array2string(potential_ring_means, precision=2)}\n\n"
            messagebox.showinfo("Symmetry Scores", f"{ring_str}Scores:\n{scores_str}")
        else:
            print("Could not determine dominant symmetry.")
            messagebox.showwarning("Symmetry Warning",
                                   "Could not determine dominant symmetry. Classification by symmetry skipped.")

        # --- Этап 3: Классификация групп на основе симметрии ---
        print("Starting Pass 2 (Symmetry classification)...")
        numeric_type_id_counter = 0 # Отдельный счетчик для числового ТИПА
        num_classified_structural = 0
        num_classified_superstructural = 0

        # Проходим по группам, созданным на Этапе 1
        for initial_group_id, indices in groups_data.items():
            N = len(indices)
            final_type: Union[str, int] # Тип, который пойдет в model.point_types

            if dominant_symmetry > 0:
                if N == dominant_symmetry:
                    final_type = "structural"
                    num_classified_structural += 1
                elif N > dominant_symmetry and N % dominant_symmetry == 0:
                    final_type = "superstructural"
                    num_classified_superstructural += 1
                else:
                    # Группа не соответствует симметрии -> "другая"
                    final_type = numeric_type_id_counter
                    numeric_type_id_counter += 1
            else:
                # Симметрия не определена -> все "другие"
                final_type = numeric_type_id_counter
                numeric_type_id_counter += 1

            # Присваиваем финальный ТИП всем точкам этой ИСХОДНОЙ группы
            for idx in indices:
                new_point_types[idx] = final_type
                # new_initial_group_ids[idx] уже присвоен на Этапе 1

        num_numeric_types = numeric_type_id_counter # Сколько числовых ТИПОВ было присвоено
        print(
            f"Pass 2 finished. Classified: {num_classified_structural} structural, {num_classified_superstructural} superstructural. Assigned {num_numeric_types} numeric types."
        )

        # --- Шаг 4: Обновляем Модель, Сохранение и UI ---

        # *** Ключевое изменение: Обновляем модель ***
        model.point_types = new_point_types # Содержит "structural", "superstructural" или int (numeric type id)
        model.initial_group_ids = new_initial_group_ids # Содержит int (group id from pass 1)
        # *** Конец ***

        # --- Обновляем панель групп ---
        self.controller._update_group_panel()
        # ---

        self.controller.redo.clear()
        self.controller.redraw() # Перерисовка покажет новые цвета

        final_type_counts = Counter(model.point_types)
        num_unknown = final_type_counts.get("unknown", 0) # Точки, не попавшие ни в одну группу
        summary_msg = f"Grouping done. Structural: {num_classified_structural}. Superstr: {num_classified_superstructural}. Numeric types: {num_numeric_types}. Unknown: {num_unknown}."
        print(summary_msg)
        self.controller.set_status(summary_msg)

        # Сохраняем отладочный файл (через controller.io)
        self.controller.io.save_debug_data(filename="auto_grouped_points_debug.json")

    # ---------- Mouse / Keyboard events ----------

    def _on_key(self, e):
        ui_state = self.controller.ui_state

        if e.key == "escape":
            cleared_tooltip = ui_state.tooltip is not None
            if ui_state.tooltip:
                ui_state.clear_tooltip(self.controller.view)  # Передаем view для clear

            cleared_interactions = self._cancel_all_interactions()
            if cleared_interactions:
                self.controller.redraw()
            elif cleared_tooltip:
                pass
            else:
                self.controller.set_status("Escape pressed, no action taken.")
            return

        elif ui_state.ring_select_active:
            delta = 0.0
            if e.key in ('+', '=', 'KP_Add'):
                delta = 1.0
            elif e.key in ('-', 'KP_Subtract'):
                delta = -1.0
            if delta != 0.0:
                self._adjust_ring_thickness(delta)
                return

        elif e.key in {"enter", "return", "KP_Enter"}:
            if ui_state.ring_select_indices:
                self._average_selected_points_to_ring()
            else:
                self.controller.set_status("Enter pressed, no action selected.")
            return
        pass

    def _on_down(self, e):
        pos_yx = self.controller.view.img_xy(e)
        ui_state = self.controller.ui_state
        model = self.controller.model
        view = self.controller.view

        ui_state.clear_tooltip(view)

        # MMB - Показать инфо
        if e.button == 2:
            if pos_yx is None: return
            idx = model.find_nearest_point_idx(pos_yx[0], pos_yx[1], pix_tol=8)
            if idx is not None:
                center = self.controller.get_center()
                ui_state.show_tooltip_for_idx(idx, model, center, view)  # Передаем view
            return

        # LMB или RMB
        if e.button == 1 or e.button == 3:
            is_left_click = e.button == 1
            is_right_click = e.button == 3
            is_shift_pressed = hasattr(e, 'key') and e.key is not None and "shift" in e.key.lower()
            is_ctrl_pressed = hasattr(e, 'key') and e.key is not None and (
                        "control" in e.key.lower() or "ctrl" in e.key.lower())

            if ui_state.ring_select_active and is_left_click:
                self._select_points_in_ring()
                return

            if is_ctrl_pressed and is_left_click:
                if ui_state.measure_start_idx is None and not ui_state.center_dragging and not ui_state.rect_start:
                    if pos_yx is not None: self._start_ring_selection(pos_yx)
                return

            if is_shift_pressed and ui_state.ring_select_indices:
                if pos_yx is None: return
                idx = model.find_nearest_point_idx(pos_yx[0], pos_yx[1])
                if idx is not None:
                    if is_left_click:
                        self._toggle_point_in_ring_selection(idx, add=True)
                    elif is_right_click:
                        self._toggle_point_in_ring_selection(idx, add=False)
                else:
                    self.controller.set_status("Shift+Click only works on existing points.")
                return

            if is_shift_pressed and is_left_click and not ui_state.ring_select_indices:
                if pos_yx is not None:
                    if self._cancel_all_interactions(): self.controller.redraw()
                    # self.controller.push_undo() # Undo push теперь в _on_up
                    ui_state.rect_start = pos_yx
                    # self.controller.redo.clear()
                    self.controller.set_status("Drag to select points for deletion.")
                return

            if pos_yx is None: return
            y, x = pos_yx

            center = self.controller.get_center()
            hit_center = view.center_hit(y, x, center)
            hit_point_idx = model.find_nearest_point_idx(y, x)

            # --- Логика для ЛКМ ---
            if is_left_click:
                if hit_center:
                    if ui_state.measure_start_idx is not None:
                        if ui_state.finalize_measurement_to_center(center):
                            length = ui_state.measurement.get("length", 0.0)
                            self.controller.set_status(f"Measured {length:.1f} px to center.")
                        else:
                            self.controller.set_status("Measurement failed or center undefined.")
                        self.controller.redraw()
                    else:
                        if self._cancel_all_interactions(): self.controller.redraw()
                        self.controller.push_undo()
                        ui_state.center_dragging = True
                        self.controller.redo.clear()
                        self.controller.set_status("Dragging center overlay. Release to finish.")

                elif hit_point_idx is not None:
                    if ui_state.measure_start_idx is None:
                        if self._cancel_all_interactions(): self.controller.redraw()
                        ui_state.clear_measurement_result(view)
                        ui_state.start_measurement(hit_point_idx, model)
                        self.controller.set_status(
                            f"Measurement started from point {hit_point_idx}. Click second point or center.")
                        self.controller.redraw()
                    elif ui_state.measure_start_idx != hit_point_idx:
                        pt_y, pt_x = model.points[hit_point_idx]
                        if ui_state.finalize_measurement(end_yx=(float(pt_y), float(pt_x))):
                            length = ui_state.measurement.get("length", 0.0)
                            self.controller.set_status(f"Measured {length:.1f} px between points.")
                        else:
                            self.controller.set_status("Measurement failed.")
                        self.controller.redraw()
                    else:
                        self._cancel_all_interactions()
                        self.controller.set_status(self.controller.default_status)
                        self.controller.redraw()

                else:  # Клик на пустое место
                    redraw_needed = self._cancel_all_interactions()
                    redraw_needed |= ui_state.clear_measurement_result(view)
                    if redraw_needed: self.controller.redraw()

                    # --- Добавление точки (через Модель) ---
                    self.controller.push_undo()
                    sampled_value = self.controller.sample_intensities(np.array([[y, x]]))[0]

                    # Модель сама добавит "unknown", и None ID
                    model.add_point(y, x, sampled_value, center)

                    # --- Обновляем панель (добавление 1 точки неэффективно, но необходимо для sync) ---
                    self.controller._update_group_panel()
                    # ---

                    self.controller.redo.clear()
                    self.controller.set_status(f"Added point at ({x:.1f}, {y:.1f}).")
                    self.controller.redraw()

            # --- Логика для ПКМ ---
            elif is_right_click:
                if hit_point_idx is not None:  # Удаление точки
                    if self._cancel_all_interactions(): self.controller.redraw()
                    self.controller.push_undo()

                    was_selected = hit_point_idx in ui_state.ring_select_indices
                    if was_selected:
                        ui_state.ring_select_indices.remove(hit_point_idx)

                    if hit_point_idx >= len(model.points):
                        self.controller.set_status("Error: Point index out of bounds during deletion.")
                        self.controller.pop_undo()  # Отменяем push_undo
                        return

                    # --- Удаление точки (через Модель) ---
                    # Обновляем индексы ВЫДЕЛЕННЫХ точек
                    ui_state.update_ring_indices_after_delete([hit_point_idx])

                    # Модель сама удалит всё (point, value, area, type, angle, ...)
                    model.delete_points_by_indices([hit_point_idx])
                    # --- КОНЕЦ Удаления ---

                    # --- Обновляем панель (удаление 1 точки неэффективно, но необходимо для sync) ---
                    self.controller._update_group_panel()
                    # ---

                    self.controller.redo.clear()
                    self.controller.set_status("Deleted point.")
                    self.controller.redraw()
                else:  # Клик на пустое место
                    if self._cancel_all_interactions(): self.controller.redraw()
                    if ui_state.clear_measurement_result(view): self.controller.redraw()
            return

    def _on_move(self, e):
        pos_yx = self.controller.view.img_xy(e)
        ui_state = self.controller.ui_state
        view = self.controller.view

        # Перетаскивание центра
        if ui_state.center_dragging and pos_yx is not None and ui_state.measure_start_idx is None:
            y, x = pos_yx
            self.controller.set_center(x, y)  # Обновляем центр в контроллере

            # --- Пересчитываем углы в Модели ---
            self.controller.model.recalculate_angles((y, x))
            # --- КОНЕЦ ---

            self.controller.redraw()
            self.controller.set_status("Dragging center...")
            return

        # Обновление превью кольца
        if ui_state.ring_select_active:
            self._update_ring_preview(pos_yx)
            return

        # Обновление превью замера
        if ui_state.measure_start_idx is not None:
            ui_state.update_measurement_preview(pos_yx, view)  # Передаем view
            return

        # Обновление прямоугольного выделения
        if ui_state.rect_start and e.xdata is not None and e.ydata is not None:
            view.draw_rect_preview(ui_state, (e.ydata, e.xdata))
            return

    def _on_up(self, e):
        ui_state = self.controller.ui_state
        model = self.controller.model
        view = self.controller.view

        # Отпускание после перетаскивания центра
        if ui_state.center_dragging:
            ui_state.center_dragging = False
            # Пересчет углов уже произошел в _on_move
            if ui_state.measure_start_idx is None:
                self._apply_center_filters()  # Применяем dead/search radius

            center = self.controller.get_center()
            if center:
                self.controller.view_cx = center[1]
                self.controller.view_cy = center[0]

            self.controller.redraw()
            self.controller.set_status("Center position updated.")
            return

        # Отпускание после прямоугольного выделения
        if ui_state.rect_start and e.button == 1:  # Только для ЛКМ
            y0, x0 = ui_state.rect_start
            num_to_delete = 0

            if e.ydata is not None and e.xdata is not None:
                y1, x1 = e.ydata, e.xdata
                ymin, ymax = sorted([y0, y1])
                xmin, xmax = sorted([x0, x1])

                mask_in_rect = (
                        (model.points[:, 0] >= ymin) & (model.points[:, 0] <= ymax) &
                        (model.points[:, 1] >= xmin) & (model.points[:, 1] <= xmax)
                )
                num_to_delete = np.count_nonzero(mask_in_rect)

                if num_to_delete > 0:
                    self.controller.push_undo()  # Сохраняем состояние ДО удаления
                    indices_to_delete = np.where(mask_in_rect)[0]

                    # Обновляем selected ring indices ПЕРЕД удалением
                    ui_state.update_ring_indices_after_delete(indices_to_delete)

                    # --- Удаляем точки через Модель ---
                    model.delete_points_by_mask(~mask_in_rect) # ИСПОЛЬЗУЕМ ~mask_in_rect (True=сохранить)
                    # --- КОНЕЦ ---

                    # --- Обновляем панель групп ---
                    self.controller._update_group_panel()
                    # ---

                    self.controller.redo.clear()
                    self.controller.set_status(f"Deleted {num_to_delete} points in selection.")
                else:
                    self.controller.set_status("Rectangular selection finished, no points deleted.")

            ui_state.rect_start = None
            view.remove_rect_artist(ui_state)

            if num_to_delete > 0:
                self.controller.redraw()
            return

    def _on_scroll(self, event):
        """Обработка зума колесом мыши."""
        if event.xdata is None or event.ydata is None: return

        # Обновляем центр зума
        self.controller.view_cx = event.xdata
        self.controller.view_cy = event.ydata

        zoom_step = 5
        if event.button == 'up':
            new_zoom_val = self.controller.zoom_val + zoom_step
        elif event.button == 'down':
            new_zoom_val = self.controller.zoom_val - zoom_step
        else:
            return

        new_zoom_val = max(0, min(100, new_zoom_val))

        # Вызываем стандартизированный обработчик
        self.on_zoom_change_tk(new_zoom_val)

    def _apply_center_filters(self):
        """Применяет dead_radius и search_radius к точкам."""
        model = self.controller.model
        if model.is_empty(): return

        center = self.controller.get_center()
        if not center: return

        cy, cx = center
        dead = self.controller.get_dead_radius()
        sr = self.controller.get_search_radius()

        if dead <= 0 and sr <= 0: return

        # Рассчитываем радиусы (углы не нужны)
        r = np.hypot(model.points[:, 1] - cx, model.points[:, 0] - cy)

        mask_keep = np.ones(len(model.points), dtype=bool)
        if dead > 0: mask_keep &= (r >= dead)
        if sr > 0: mask_keep &= (r <= sr)

        num_deleted = np.count_nonzero(~mask_keep)

        if num_deleted > 0:
            self.controller.push_undo()  # Сохраняем до удаления
            indices_to_delete = np.where(~mask_keep)[0]

            # Обновляем ring_select_indices ПЕРЕД удалением
            self.controller.ui_state.update_ring_indices_after_delete(indices_to_delete)

            # --- Удаляем точки через Модель ---
            model.delete_points_by_mask(mask_keep) # Передаем mask_keep (True=сохранить)
            # --- КОНЕЦ ---

            # --- Обновляем панель групп ---
            self.controller._update_group_panel()
            # ---

            self.controller.set_status(f"Applied center filters, removed {num_deleted} points.")
            self.controller.redo.clear()
            # self.controller.redraw() # Redraw будет вызван в _on_up