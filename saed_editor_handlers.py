#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mix-in класс для PointEditor:
Обрабатывает события мыши (mouse) и клавиатуры (key)
и новую функцию Auto-Group
"""
import numpy as np
import matplotlib.pyplot as plt
import math # Добавлен импорт
import tkinter as tk # Добавлен импорт tk
from tkinter import messagebox # Для показа ошибок
# Импорт специального индекса
from saed_editor_state import CENTER_AS_POINT_IDX
# --- ИЗМЕНЕНИЕ: Импорт функций анализа ---
from scipy.signal import find_peaks
# --- КОНЕЦ ИЗМЕНЕНИЯ ---


# --- Функции анализа (остаются здесь, но больше не используются для авто-классификации) ---
def pol_from(center, pts):
    cy, cx = center
    dy, dx = pts[:, 0] - cy, pts[:, 1] - cx
    r = np.hypot(dx, dy)
    a = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    return r, a

def symmetry_scores(angles, radii, ring_means, top_rings=3):
    out = {}
    if ring_means.size == 0:
        return out
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
        C = np.cos(phases_rad).mean();
        S = np.sin(phases_rad).mean()
        out[f"{k}-fold"] = float(np.hypot(C, S))
    return out

def cluster_rings(radii, bins=100, prominence_factor=0.03, min_prominence=2):
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
# --- КОНЕЦ ПЕРЕНЕСЕННЫХ ФУНКЦИЙ ---


class EditorEventHandlers:

    # ---------- ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ОБРАБОТЧИКОВ ---
    def _cancel_all_interactions(self, *, keep_status: bool = False) -> bool:
        """Cancels any ongoing drag, selection, or preview."""
        cleared_measure = False
        if self._cancel_measurement_preview(): # Отменяет превью (пунктир)
            cleared_measure = True
        if self._measure_start_idx is not None: # Сбрасывает начало замера
             self._measure_start_idx = None
             self._measure_start_point = None
             cleared_measure = True

        cleared_ring = self._cancel_ring_selection()
        cleared_drag = self.center_dragging
        self.center_dragging = False
        if not keep_status and (cleared_measure or cleared_ring or cleared_drag):
             self._set_status(self._default_status)

        return (cleared_measure or cleared_ring or cleared_drag)

    # --- Методы для выбора кольцом (без изменений) ---
    def _start_ring_selection(self, pos_yx: tuple[float, float]) -> None:
        """Начинает режим выбора кольцом."""
        if not self.overlay or not self.overlay.get("center"):
            self._set_status("Cannot start ring selection: Center is not defined.")
            return
        if self._cancel_all_interactions(): self._redraw()
        center_data = self.overlay["center"]
        center_y, center_x = float(center_data["y"]), float(center_data["x"])
        self._ring_select_center_yx = (center_y, center_x)
        cursor_y, cursor_x = pos_yx
        radius = math.hypot(cursor_x - center_x, cursor_y - center_y)
        self._ring_select_radius = max(1.0, radius)
        self._ring_select_active = True
        self._ring_select_indices.clear()
        self._set_status("Ring selection active. Drag radius. Use +/- for thickness. LMB click to select points.")
        self._redraw()

    def _update_ring_preview(self, pos_yx: tuple[float, float] | None) -> None:
        """Обновляет радиус кольца превью."""
        if not self._ring_select_active or self._ring_select_center_yx is None or pos_yx is None: return
        center_y, center_x = self._ring_select_center_yx
        cursor_y, cursor_x = pos_yx
        radius = math.hypot(cursor_x - center_x, cursor_y - center_y)
        new_radius = max(1.0, radius)
        if abs(new_radius - self._ring_select_radius) > 0.1:
             self._ring_select_radius = new_radius
             self._redraw()

    def _adjust_ring_thickness(self, delta: float) -> None:
        """Изменяет толщину кольца превью."""
        if not self._ring_select_active: return
        new_thickness = self._ring_select_thickness + delta
        self._ring_select_thickness = max(1.0, new_thickness)
        self._set_status(f"Ring thickness: {self._ring_select_thickness:.1f} px. LMB click to select.")
        self._redraw()

    def _select_points_in_ring(self) -> None:
        """Выделяет точки, попадающие в текущее кольцо превью."""
        if not self._ring_select_active or self._ring_select_center_yx is None or self.points is None:
            return

        center_y, center_x = self._ring_select_center_yx
        radius = self._ring_select_radius
        thickness = self._ring_select_thickness
        r_min = radius - thickness / 2.0
        r_max = radius + thickness / 2.0

        dy = self.points[:, 0] - center_y
        dx = self.points[:, 1] - center_x
        point_radii = np.hypot(dx, dy)
        indices_in_ring = np.where((point_radii >= r_min) & (point_radii <= r_max))[0]

        self._ring_select_indices = set(indices_in_ring.tolist())
        self._ring_select_active = False # Выходим из режима рисования
        self._remove_ring_preview_artist()

        count = len(self._ring_select_indices)
        if count > 0:
            self._set_status(f"Selected {count} points. Shift+Click to add/remove. Enter to average.")
        else:
            self._set_status("No points found in the ring. Selection cleared.")
        self._redraw()

    def _toggle_point_in_ring_selection(self, index: int, add: bool) -> None:
        """Добавляет или удаляет точку из набора выделенных кольцом."""
        if add:
            self._ring_select_indices.add(index)
            self._set_status(f"Added point {index} ({len(self._ring_select_indices)} total). Enter to average.")
        else:
            if index in self._ring_select_indices:
                self._ring_select_indices.remove(index)
                self._set_status(f"Removed point {index} ({len(self._ring_select_indices)} total). Enter to average.")
            else:
                 self._set_status(f"Point {index} was not selected.")
        self._redraw()

    def _average_selected_points_to_ring(self) -> None:
        """Усредняет радиусы выделенных точек и перемещает их."""
        if not self._ring_select_indices or self.points is None:
            self._set_status("No points selected for averaging.")
            return
        # ... (остальная логика усреднения без изменений) ...
        if not self.overlay or not self.overlay.get("center"):
            self._set_status("Cannot average: Center is not defined.")
            return

        indices = list(self._ring_select_indices)
        valid_indices = [i for i in indices if 0 <= i < len(self.points)]
        if len(valid_indices) < 2:
            self._set_status(f"Need at least 2 valid points to average (found {len(valid_indices)}).")
            return

        self._push_undo()

        center_data = self.overlay["center"]
        cy, cx = float(center_data["y"]), float(center_data["x"])

        selected_points = self.points[valid_indices]
        dy = selected_points[:, 0] - cy
        dx = selected_points[:, 1] - cx
        radii = np.hypot(dx, dy)
        zero_radius_mask = radii < 1e-9
        non_zero_radii = radii[~zero_radius_mask]

        if len(non_zero_radii) == 0:
             self._set_status("Cannot average: All selected points are at the center.")
             return

        average_radius = np.mean(non_zero_radii)
        unit_dx = np.divide(dx, radii, out=np.zeros_like(dx), where=radii > 1e-9)
        unit_dy = np.divide(dy, radii, out=np.zeros_like(dy), where=radii > 1e-9)
        new_dx = unit_dx * average_radius
        new_dy = unit_dy * average_radius

        self.points[valid_indices, 0] = cy + new_dy
        self.points[valid_indices, 1] = cx + new_dx

        # --- ИЗМЕНЕНИЕ: Обновляем intensity И area после усреднения ---
        # Пересчитываем intensity (% map)
        if self.values is not None and len(self.values) == len(self.points):
             self.values[valid_indices] = self._sample_intensities(self.points[valid_indices])
        # Для area оставляем старые значения (усреднять их не имеет смысла)
        if hasattr(self, 'areas') and self.areas is not None and len(self.areas) == len(self.points):
            pass
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        count = len(valid_indices)
        self._set_status(f"Averaged {count} points to radius {average_radius:.2f} px.")
        self._ring_select_indices.clear()
        self._redo.clear()
        self._redraw()

    # --- КОНЕЦ МЕТОДОВ ДЛЯ КОЛЬЦА ---

    # --- НОВЫЙ МЕТОД: Автоматическая группировка колец (v2) ---
    def _auto_group_rings(self):
        self._set_status("Starting auto-grouping...")
        if self.points is None or len(self.points) == 0:
            self._set_status("No points to group.")
            return
        if not self.overlay or not self.overlay.get("center"):
            self._set_status("Cannot group: Center is not defined.")
            return

        try:
            # Получаем допуски из UI
            radius_tol_px = float(self.spn_auto_radius_tol.get())
            area_tol_perc = float(self.spn_auto_area_tol.get()) / 100.0 # Преобразуем в долю 0..1
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Input Error", f"Invalid tolerance value entered:\n{e}")
            self._set_status("Auto-grouping cancelled due to invalid input.")
            return

        self._push_undo() # Сохраняем состояние перед началом

        center_data = self.overlay["center"]
        cy, cx = float(center_data["y"]), float(center_data["x"])

        # Рассчитываем радиусы для всех точек
        dy = self.points[:, 0] - cy
        dx = self.points[:, 1] - cx
        all_radii = np.hypot(dx, dy)

        # Сортируем точки по радиусу (получаем отсортированные ИНДЕКСЫ)
        sorted_indices = np.argsort(all_radii)

        # Инициализируем типы как "unknown" для всех
        self.point_types = ["unknown"] * len(self.points)
        # Набор индексов точек, которые уже включены в какую-то группу
        assigned_indices = set()
        group_id_counter = 0

        # Итерируем по отсортированным индексам
        for i in range(len(sorted_indices)):
            current_idx = sorted_indices[i]

            # Пропускаем, если точка уже в группе
            if current_idx in assigned_indices:
                continue

            current_radius = all_radii[current_idx]

            # 1. Находим кандидатов по радиусу (среди еще не назначенных)
            potential_group_indices = []
            for j in range(i, len(sorted_indices)): # Начинаем с текущей точки
                check_idx = sorted_indices[j]
                if check_idx in assigned_indices:
                    continue

                radius_diff = abs(all_radii[check_idx] - current_radius)
                if radius_diff <= radius_tol_px:
                    potential_group_indices.append(check_idx)
                else:
                    # Так как точки отсортированы, дальше радиусы будут только больше
                    break # Выходим из внутреннего цикла

            if not potential_group_indices: # Не должно случиться, т.к. сама точка включается
                 continue

            # 2. Фильтруем по площади
            candidate_areas = self.areas[potential_group_indices]
            final_group_indices = []

            if len(candidate_areas) > 0 and np.any(candidate_areas > 0): # Проверяем, есть ли ненулевые площади
                max_area = np.max(candidate_areas)
                min_allowed_area = max_area * (1.0 - area_tol_perc)
                # Max allowed area не нужен по описанию, только нижний порог

                area_mask = (candidate_areas >= min_allowed_area)
                final_group_indices = np.array(potential_group_indices)[area_mask].tolist()
            else: # Если площадей нет или все нулевые, берем всех кандидатов по радиусу
                final_group_indices = potential_group_indices

            # 3. Если группа не пуста, назначаем ID и помечаем как использованные
            if final_group_indices:
                print(f"Found group {group_id_counter} at r~{current_radius:.1f}px: {len(final_group_indices)} points")
                for idx in final_group_indices:
                    self.point_types[idx] = group_id_counter
                    assigned_indices.add(idx)
                group_id_counter += 1

        # Точки, оставшиеся в assigned_indices=False, уже имеют тип "unknown"

        self._redo.clear() # Очищаем redo после автоматического изменения
        self._redraw()
        summary_msg = f"Auto-grouping finished. Found {group_id_counter} groups."
        print(summary_msg)
        self._set_status(summary_msg)
    # --- КОНЕЦ НОВОГО МЕТОДА ---


    # ---------- Mouse / Keyboard events ----------
    def _on_key(self, e):
        # --- Обработка Escape ---
        if e.key == "escape":
            cleared_tooltip = self._tooltip is not None
            if self._tooltip: self._clear_tooltip()
            cleared_interactions = self._cancel_all_interactions()
            if cleared_interactions:
                # Статус уже сброшен в _cancel_all_interactions
                self._redraw()
            elif cleared_tooltip: pass
            else: self._set_status("Escape pressed, no action taken.")
            return

        # --- Обработка +/- ---
        elif self._ring_select_active:
            delta = 0.0
            if e.key in ('+', '=', 'KP_Add'): delta = 1.0
            elif e.key in ('-', 'KP_Subtract'): delta = -1.0
            if delta != 0.0:
                self._adjust_ring_thickness(delta)
                return

        # --- Обработка Enter (Только для усреднения) ---
        elif e.key in {"enter", "return", "KP_Enter"}:
             if self._ring_select_indices: # Если точки ВЫБРАНЫ вручную
                  self._average_selected_points_to_ring() # Enter усредняет
             else:
                  self._set_status("Enter pressed, no action selected.")
             return

        pass # Другие клавиши игнорируем

    def _on_down(self, e):
        pos_yx = self._img_xy(e)
        self._clear_tooltip()

        # MMB - Показать инфо
        if e.button == 2:
            if pos_yx is None: return
            idx = self._near_idx(pos_yx[0], pos_yx[1], pix_tol=8)
            if idx is not None: self._show_tooltip_for_idx(idx)
            return

        # LMB или RMB
        if e.button == 1 or e.button == 3:
            is_left_click = e.button == 1
            is_right_click = e.button == 3
            is_shift_pressed = hasattr(e, 'key') and e.key is not None and "shift" in e.key.lower()
            is_ctrl_pressed = hasattr(e, 'key') and e.key is not None and ("control" in e.key.lower() or "ctrl" in e.key.lower())

            # --- Приоритет 1: Завершение рисования кольца (ЛКМ) ---
            if self._ring_select_active and is_left_click:
                 self._select_points_in_ring()
                 return # Важно! Не проваливаемся в другую логику ЛКМ

            # --- Приоритет 2: Режим Выбора кольцом (Ctrl+Drag) ---
            if is_ctrl_pressed and is_left_click:
                 # Проверяем, что не активен другой режим (на всякий случай)
                 if self._measure_start_idx is None and not self.center_dragging and not self.rect_start:
                      if pos_yx is not None: self._start_ring_selection(pos_yx)
                 return

            # --- Приоритет 3: До/После-выделение (Shift+Click при активном _ring_select_indices) ---
            if is_shift_pressed and self._ring_select_indices:
                 if pos_yx is None: return
                 idx = self._near_idx(pos_yx[0], pos_yx[1])
                 if idx is not None:
                      if is_left_click: self._toggle_point_in_ring_selection(idx, add=True)
                      elif is_right_click: self._toggle_point_in_ring_selection(idx, add=False)
                 else: self._set_status("Shift+Click only works on existing points.")
                 return

            # --- Приоритет 4: Прямоугольное удаление (Shift+Drag, если НЕТ выбора кольцом) ---
            if is_shift_pressed and is_left_click and not self._ring_select_indices:
                if pos_yx is not None:
                    if self._cancel_all_interactions(): self._redraw() # Отменяем всё перед началом выделения
                    self._push_undo()
                    self.rect_start = pos_yx
                    self._redo.clear()
                    self._set_status("Drag to select points for deletion.")
                return

            # --- Обычные клики (без Shift, Ctrl и НЕ в режиме рисования кольца) ---
            if pos_yx is None: return # Клик вне холста

            y, x = pos_yx
            hit_center = self._center_hit(y, x)
            hit_point_idx = self._near_idx(y, x)

            # --- Логика для ЛКМ ---
            if is_left_click:
                 if hit_center:
                      if self._measure_start_idx is not None:
                           # Завершаем замер до центра
                           if self._finalize_measurement_to_center():
                                length = self._measurement.get("length", 0.0)
                                self._set_status(f"Measured {length:.1f} px to center.")
                           else:
                                self._set_status("Measurement failed or center undefined.")
                           self._redraw()
                      else:
                           # Начинаем ПЕРЕТАСКИВАНИЕ центра (если замер не активен)
                           if self._cancel_all_interactions(): self._redraw()
                           self._push_undo()
                           self.center_dragging = True
                           self._redo.clear()
                           self._set_status("Dragging center overlay. Release to finish.")
                 elif hit_point_idx is not None:
                      if self._measure_start_idx is None:
                           # Начинаем замер от точки
                           if self._cancel_all_interactions(): self._redraw()
                           self._clear_measurement_result()
                           self._start_measurement(hit_point_idx)
                           self._set_status(f"Measurement started from point {hit_point_idx}. Click second point or center.")
                           self._redraw()
                      elif self._measure_start_idx != hit_point_idx:
                           # Завершаем замер до точки
                           pt_y, pt_x = self.points[hit_point_idx]
                           if self._finalize_measurement(end_yx=(float(pt_y), float(pt_x))):
                                length = self._measurement.get("length", 0.0)
                                self._set_status(f"Measured {length:.1f} px between points.")
                           else:
                                self._set_status("Measurement failed.")
                           self._redraw()
                      else: # Клик на ту же точку
                           # Отменяем замер
                           self._cancel_all_interactions() # Отменит и превью, и старт
                           self._set_status(self._default_status)
                           self._redraw()
                 else: # Клик на пустое место
                      redraw_needed = self._cancel_all_interactions()
                      redraw_needed |= self._clear_measurement_result()
                      if redraw_needed: self._redraw()

                      # --- ИСПРАВЛЕНИЕ: Добавляем тип и ПЛОЩАДЬ при добавлении точки ---
                      self._push_undo()
                      self.points = np.vstack([self.points, [y, x]])
                      sampled_value = self._sample_intensities(np.array([[y, x]]))[0]
                      self.values = np.append(self.values, sampled_value)
                      # Добавляем тип "unknown"
                      if hasattr(self, 'point_types'):
                          self.point_types.append("unknown")
                      else:
                          self.point_types = ["unknown"] * len(self.points)
                      # Добавляем площадь 0.0
                      if hasattr(self, 'areas') and self.areas is not None:
                          self.areas = np.append(self.areas, 0.0)
                      else:
                          self.areas = np.zeros(len(self.points), dtype=float)
                      # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
                      self._redo.clear()
                      self._set_status(f"Added point at ({x:.1f}, {y:.1f}).")
                      self._redraw()

            # --- Логика для ПКМ ---
            elif is_right_click:
                 if hit_point_idx is not None: # Удаление точки
                      if self._cancel_all_interactions(): self._redraw()
                      self._push_undo()
                      was_selected = hit_point_idx in self._ring_select_indices
                      if was_selected: self._ring_select_indices.remove(hit_point_idx)
                      if hit_point_idx >= len(self.points):
                           self._set_status("Error: Point index out of bounds during deletion.")
                           return

                      # --- ИСПРАВЛЕНИЕ: Удаляем тип точки и ПЛОЩАДЬ ---
                      # Удаляем точку, значение, ТИП и ПЛОЩАДЬ
                      self.points = np.delete(self.points, hit_point_idx, axis=0)
                      if self.values is not None and len(self.values) > hit_point_idx:
                           self.values = np.delete(self.values, hit_point_idx, axis=0)
                      else:
                           self.values = self._sample_intensities(self.points)
                      # Удаляем тип
                      if hasattr(self, 'point_types') and len(self.point_types) > hit_point_idx:
                          del self.point_types[hit_point_idx]
                      # Удаляем площадь
                      if hasattr(self, 'areas') and self.areas is not None and len(self.areas) > hit_point_idx:
                          self.areas = np.delete(self.areas, hit_point_idx, axis=0)
                      # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

                      # Обновляем индексы ВЫДЕЛЕННЫХ точек
                      if self._ring_select_indices:
                           new_indices = set()
                           for idx in self._ring_select_indices:
                                if idx > hit_point_idx:
                                    new_indices.add(idx - 1)
                                elif idx < hit_point_idx:
                                    new_indices.add(idx)
                           self._ring_select_indices = new_indices

                      self._redo.clear()
                      self._set_status("Deleted point.")
                      self._redraw()
                 else: # Клик на пустое место
                      if self._cancel_all_interactions(): self._redraw()
                      if self._clear_measurement_result(): self._redraw()
            return


    def _on_move(self, e):
        # ... (остальной код _on_move без изменений) ...
        pos_yx = self._img_xy(e)

        # Перетаскивание центра (только если не замеряем)
        if self.center_dragging and pos_yx is not None and self._measure_start_idx is None:
            y, x = pos_yx
            if self.overlay is None: self.overlay = {}
            self.overlay["center"] = {"x": float(x), "y": float(y)}
            self._redraw()
            self._set_status("Dragging center...")
            return

        # Обновление превью кольца
        if self._ring_select_active:
             self._update_ring_preview(pos_yx)
             return

        # Обновление превью замера
        if self._measure_start_idx is not None:
            self._update_measurement_preview(pos_yx)
            return

        # Обновление прямоугольного выделения
        if self.rect_start and e.xdata is not None and e.ydata is not None:
            y0, x0 = self.rect_start
            y1, x1 = e.ydata, e.xdata

            if hasattr(self, 'rect_artist') and self.rect_artist is not None:
                try: self.rect_artist.remove()
                except Exception: pass
                self.rect_artist = None

            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()

            rect_x = min(x0, x1); rect_y = min(y0, y1)
            rect_w = abs(x1 - x0); rect_h = abs(y1 - y0)
            self.rect_artist = self.ax.add_patch(
                plt.Rectangle((rect_x, rect_y), rect_w, rect_h,
                              fill=False, ec="red", ls="--", lw=1.5, zorder=15)
            )

            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)
            self.canvas.draw_idle()
            return


    def _on_up(self, e):
        # Отпускание после перетаскивания центра
        if self.center_dragging:
            self.center_dragging = False
            if self._measure_start_idx is None: self._apply_center_filters()
            if self.overlay and isinstance(self.overlay.get("center"), dict):
                self.view_cx = float(self.overlay["center"]["x"])
                self.view_cy = float(self.overlay["center"]["y"])
            self._redraw()
            self._set_status("Center position updated.")
            return

        # --- ИЗМЕНЕНИЕ: Отпускание в режиме рисования кольца ---
        # Мы больше НЕ завершаем выбор по отпусканию ЛКМ
        # if self._ring_select_active and e.button == 1:
        #      # Ничего не делаем здесь, ждем ЛКМ клика
        #      return
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        # Отпускание после прямоугольного выделения
        if self.rect_start and e.button == 1: # Только для ЛКМ
            # ... (код удаления точек) ...
            y0, x0 = self.rect_start;
            if e.ydata is not None and e.xdata is not None:
                y1, x1 = e.ydata, e.xdata
                ymin, ymax = sorted([y0, y1]);
                xmin, xmax = sorted([x0, x1])

                mask_in_rect = (
                        (self.points[:, 0] >= ymin) & (self.points[:, 0] <= ymax) &
                        (self.points[:, 1] >= xmin) & (self.points[:, 1] <= xmax)
                )
                num_to_delete = np.count_nonzero(mask_in_rect)

                if num_to_delete > 0:
                    self._push_undo() # Сохраняем состояние ДО удаления

                    indices_to_delete = np.where(mask_in_rect)[0]

                    # Обновляем selected ring indices ПЕРЕД удалением
                    if self._ring_select_indices:
                        deleted_selected_indices = self._ring_select_indices.intersection(indices_to_delete)
                        if deleted_selected_indices:
                             self._ring_select_indices.difference_update(deleted_selected_indices)
                             new_ring_indices = set()
                             num_deleted_before = {i: np.count_nonzero(indices_to_delete < i) for i in self._ring_select_indices}
                             for old_idx in self._ring_select_indices:
                                  new_ring_indices.add(old_idx - num_deleted_before[old_idx])
                             self._ring_select_indices = new_ring_indices

                    # Теперь удаляем точки, значения, типы и ПЛОЩАДИ
                    mask_to_keep = ~mask_in_rect
                    self.points = self.points[mask_to_keep]
                    if self.values is not None and len(self.values) == len(mask_to_keep) + num_to_delete:
                        self.values = self.values[mask_to_keep]
                    else:
                        self.values = self._sample_intensities(self.points)

                    # --- ИСПРАВЛЕНИЕ: Удаляем типы точек и ПЛОЩАДИ ---
                    if hasattr(self, 'point_types') and len(self.point_types) == len(mask_to_keep) + num_to_delete:
                        types_array = np.array(self.point_types)
                        self.point_types = types_array[mask_to_keep].tolist()
                    else:
                        print("Warning: point_types length mismatch during rect delete. Resetting types.")
                        self.point_types = ["unknown"] * len(self.points)

                    if hasattr(self, 'areas') and self.areas is not None and len(self.areas) == len(mask_to_keep) + num_to_delete:
                        self.areas = self.areas[mask_to_keep]
                    else:
                        print("Warning: areas length mismatch during rect delete. Resetting areas.")
                        self.areas = np.zeros(len(self.points), dtype=float)
                    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

                    self._redo.clear()
                    self._set_status(f"Deleted {num_to_delete} points in selection.")
                else:
                    self._set_status("Rectangular selection finished, no points deleted.")

            self.rect_start = None
            if hasattr(self, 'rect_artist') and self.rect_artist is not None:
                try: self.rect_artist.remove()
                except Exception: pass
                self.rect_artist = None
            self._redraw()
            return


    # --- Остальные методы (_img_xy, _near_idx, _center_hit, _apply_center_filters) ---
    def _img_xy(self, e):
        return None if (e.xdata is None or e.ydata is None) else (e.ydata, e.xdata)

    def _near_idx(self, y, x, pix_tol=8):
        if self.points is None or len(self.points) == 0: return None
        dist_sq = (self.points[:, 0] - y) ** 2 + (self.points[:, 1] - x) ** 2
        i = int(np.argmin(dist_sq))
        return i if dist_sq[i] <= pix_tol ** 2 else None

    def _center_hit(self, y, x):
        if not (self.overlay and self.overlay.get("center")): return False
        center_data = self.overlay["center"]
        if not isinstance(center_data, dict): return False
        cy = float(center_data.get("y", 0.0))
        cx = float(center_data.get("x", 0.0))
        dist_sq = (y - cy) ** 2 + (x - cx) ** 2
        return dist_sq <= self._center_hit_radius ** 2

    def _apply_center_filters(self):
        # --- ИСПРАВЛЕНИЕ: Удаляем/обновляем типы и ПЛОЩАДИ и здесь ---
        if self.points is None or len(self.points) == 0: return
        if not (self.overlay and self.overlay.get("center")): return

        center_data = self.overlay["center"]
        if not isinstance(center_data, dict): return
        cy = float(center_data.get("y", 0.0)); cx = float(center_data.get("x", 0.0))
        dead = float(self.overlay.get("dead_radius", 0.0)); sr = float(self.overlay.get("search_radius", 0.0))
        if dead <= 0 and sr <= 0: return

        r = np.hypot(self.points[:, 1] - cx, self.points[:, 0] - cy)
        mask_keep = np.ones(len(self.points), dtype=bool)
        if dead > 0: mask_keep &= (r >= dead)
        if sr > 0: mask_keep &= (r <= sr)

        num_deleted = np.count_nonzero(~mask_keep)
        if num_deleted > 0:
            self._push_undo() # Сохраняем до удаления

            indices_to_delete = np.where(~mask_keep)[0]

            # Обновляем ring_select_indices ПЕРЕД удалением
            if self._ring_select_indices:
                 deleted_selected_indices = self._ring_select_indices.intersection(indices_to_delete)
                 if deleted_selected_indices:
                      self._ring_select_indices.difference_update(deleted_selected_indices)
                      new_ring_indices = set()
                      num_deleted_before = {i: np.count_nonzero(indices_to_delete < i) for i in self._ring_select_indices}
                      for old_idx in self._ring_select_indices:
                           new_ring_indices.add(old_idx - num_deleted_before[old_idx])
                      self._ring_select_indices = new_ring_indices

            # Удаляем точки, значения, типы и ПЛОЩАДИ
            self.points = self.points[mask_keep]
            if self.values is not None and len(self.values) == len(mask_keep) + num_deleted:
                self.values = self.values[mask_keep]
            else:
                self.values = self._sample_intensities(self.points) # Пересчитываем

            # Удаляем типы
            if hasattr(self, 'point_types') and len(self.point_types) == len(mask_keep) + num_deleted:
                types_array = np.array(self.point_types)
                self.point_types = types_array[mask_keep].tolist()
            else:
                print("Warning: point_types length mismatch during center filter. Resetting types.")
                self.point_types = ["unknown"] * len(self.points)

            # Удаляем площади
            if hasattr(self, 'areas') and self.areas is not None and len(self.areas) == len(mask_keep) + num_deleted:
                self.areas = self.areas[mask_keep]
            else:
                print("Warning: areas length mismatch during center filter. Resetting areas.")
                self.areas = np.zeros(len(self.points), dtype=float)


            self._set_status(f"Applied center filters, removed {num_deleted} points.")
            self._redo.clear()
            # self._redraw() # Не нужно здесь, т.к. redraw будет вызван после _on_up -> center_dragging=False
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---