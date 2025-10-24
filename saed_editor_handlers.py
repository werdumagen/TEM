#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mix-in класс для PointEditor:
Обрабатывает события мыши (mouse) и клавиатуры (key)
"""
import numpy as np
import matplotlib.pyplot as plt
import math  # Добавлен импорт

# --- ДОБАВЛЕН импорт tk для проверки типа виджета ---
import tkinter as tk


# --- КОНЕЦ ДОБАВЛЕНИЯ ---


class EditorEventHandlers:

    # ---------- ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ОБРАБОТЧИКОВ ---
    def _cancel_all_interactions(self, *, keep_status: bool = False) -> bool:
        """Cancels any ongoing drag, selection, or preview."""
        cleared_preview = self._cancel_measurement_preview()
        cleared_ring = self._cancel_ring_selection()
        cleared_drag = self.center_dragging
        self.center_dragging = False
        if not keep_status and (cleared_preview or cleared_ring or cleared_drag):
            self._set_status(self._default_status)  # Сброс статуса, если что-то отменили

        return (cleared_preview or cleared_ring or cleared_drag)

    # --- НОВЫЕ МЕТОДЫ ДЛЯ ВЫБОРА КОЛЬЦОМ ---
    def _start_ring_selection(self, pos_yx: tuple[float, float]) -> None:
        """Начинает режим выбора кольцом."""
        if not self.overlay or not self.overlay.get("center"):
            self._set_status("Cannot start ring selection: Center is not defined.")
            return

        # Отменяем другие активные режимы
        self._cancel_measurement_preview()
        self._clear_measurement_result()

        center_data = self.overlay["center"]
        center_y, center_x = float(center_data["y"]), float(center_data["x"])
        self._ring_select_center_yx = (center_y, center_x)

        cursor_y, cursor_x = pos_yx
        radius = math.hypot(cursor_x - center_x, cursor_y - center_y)

        self._ring_select_radius = max(1.0, radius)  # Минимальный радиус 1 пиксель
        # self._ring_select_thickness = 5.0 # Используем текущее значение или дефолтное
        self._ring_select_active = True
        self._ring_select_indices.clear()  # Очищаем предыдущее выделение

        self._set_status(
            "Ring selection active. Drag to set radius. Use +/- to change thickness. Enter to select points.")
        self._redraw()

    def _update_ring_preview(self, pos_yx: tuple[float, float] | None) -> None:
        """Обновляет радиус кольца превью."""
        if not self._ring_select_active or self._ring_select_center_yx is None or pos_yx is None:
            return

        center_y, center_x = self._ring_select_center_yx
        cursor_y, cursor_x = pos_yx
        radius = math.hypot(cursor_x - center_x, cursor_y - center_y)

        new_radius = max(1.0, radius)
        if abs(new_radius - self._ring_select_radius) > 0.1:  # Обновляем только при заметном изменении
            self._ring_select_radius = new_radius
            self._redraw()  # Перерисовываем кольцо

    def _adjust_ring_thickness(self, delta: float) -> None:
        """Изменяет толщину кольца превью."""
        if not self._ring_select_active: return

        new_thickness = self._ring_select_thickness + delta
        self._ring_select_thickness = max(1.0, new_thickness)  # Минимальная толщина 1 пиксель
        self._set_status(f"Ring thickness: {self._ring_select_thickness:.1f} px. Enter to select.")
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
        self._ring_select_active = False  # Выходим из режима рисования
        self._remove_ring_preview_artist()  # Удаляем превью

        count = len(self._ring_select_indices)
        if count > 0:
            self._set_status(f"Selected {count} points. Shift+Click to add/remove. Enter to average.")
        else:
            self._set_status("No points found in the ring. Selection cleared.")
        self._redraw()  # Показать выделенные точки

    def _toggle_point_in_ring_selection(self, index: int, add: bool) -> None:
        """Добавляет или удаляет точку из набора выделенных кольцом."""
        if add:
            self._ring_select_indices.add(index)
            self._set_status(
                f"Added point {index} to selection ({len(self._ring_select_indices)} total). Enter to average.")
        else:
            if index in self._ring_select_indices:
                self._ring_select_indices.remove(index)
                self._set_status(
                    f"Removed point {index} from selection ({len(self._ring_select_indices)} total). Enter to average.")
            else:
                self._set_status(f"Point {index} was not selected.")  # Информируем, если пытались удалить невыделенную
        self._redraw()

    def _average_selected_points_to_ring(self) -> None:
        """Усредняет радиусы выделенных точек и перемещает их."""
        if not self._ring_select_indices or self.points is None:
            self._set_status("No points selected for averaging.")
            return

        if not self.overlay or not self.overlay.get("center"):
            self._set_status("Cannot average: Center is not defined.")
            return

        indices = list(self._ring_select_indices)
        valid_indices = [i for i in indices if 0 <= i < len(self.points)]
        if len(valid_indices) < 2:  # Нужно хотя бы 2 точки для усреднения
            self._set_status(f"Need at least 2 valid points to average (found {len(valid_indices)}).")
            # --- ИЗМЕНЕНИЕ: Не очищаем выделение при ошибке ---
            # self._ring_select_indices.clear()
            # self._redraw()
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---
            return

        self._push_undo()  # Сохраняем состояние до усреднения

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

        # --- ИЗМЕНЕНИЕ: Пересчет интенсивностей после перемещения ---
        if self.values is not None and len(self.values) == len(self.points):
            self.values[valid_indices] = self._sample_intensities(self.points[valid_indices])
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        count = len(valid_indices)
        self._set_status(f"Averaged {count} points to radius {average_radius:.2f} px.")
        self._ring_select_indices.clear()  # Снимаем выделение после усреднения
        self._redo.clear()  # Очищаем redo после необратимого действия
        self._redraw()

    # --- КОНЕЦ НОВЫХ МЕТОДОВ ---

    # ---------- Mouse / Keyboard events ----------
    def _on_key(self, e):
        # Handles key presses on the canvas
        if e.key == "escape":
            cleared_tooltip = self._tooltip is not None
            if self._tooltip: self._clear_tooltip()

            cleared_ring = self._cancel_ring_selection()
            cleared_measure = self._cancel_measurement_preview()

            if cleared_ring or cleared_measure:
                self._set_status("Action cancelled.")
                self._redraw()
            elif cleared_tooltip:
                pass
            else:
                self._set_status("Escape pressed, no action taken.")
            return  # Обработали escape

        # --- ОБНОВЛЕНА ЛОГИКА +/- ---
        elif self._ring_select_active:  # Только если рисуем кольцо
            delta = 0.0
            if e.key == '+' or e.key == 'KP_Add' or e.key == '=':
                delta = 1.0
            elif e.key == '-' or e.key == 'KP_Subtract':
                delta = -1.0

            if delta != 0.0:
                self._adjust_ring_thickness(delta)
                return  # Обработали клавишу
        # --- КОНЕЦ ОБНОВЛЕНИЯ ---

        elif e.key in {"enter", "return"}:
            if self._ring_select_active:
                self._select_points_in_ring()
            elif self._ring_select_indices:
                self._average_selected_points_to_ring()
            else:
                self._set_status("Enter pressed, no action selected.")
            return  # Обработали Enter

        pass

    def _on_down(self, e):
        pos_yx = self._img_xy(e)
        self._clear_tooltip()

        if e.button == 2:  # MMB
            if pos_yx is None: return
            idx = self._near_idx(pos_yx[0], pos_yx[1], pix_tol=8)
            if idx is not None: self._show_tooltip_for_idx(idx)
            return

        if e.button == 1 or e.button == 3:  # LMB or RMB
            is_left_click = e.button == 1
            is_right_click = e.button == 3
            is_shift_pressed = e.key and "shift" in e.key.lower()
            is_ctrl_pressed = e.key and ("control" in e.key.lower() or "ctrl" in e.key.lower())

            # --- Режим Выбора кольцом (Ctrl+Drag) ---
            if is_ctrl_pressed and is_left_click:
                if pos_yx is not None:
                    self._start_ring_selection(pos_yx)
                return

            # --- Режим До/После-выделения (Shift+Click при активном _ring_select_indices) ---
            if is_shift_pressed and self._ring_select_indices:
                if pos_yx is None: return
                idx = self._near_idx(pos_yx[0], pos_yx[1])
                if idx is not None:
                    if is_left_click:  # Shift + ЛКМ -> Добавить
                        self._toggle_point_in_ring_selection(idx, add=True)
                    elif is_right_click:  # Shift + ПКМ -> Убрать
                        self._toggle_point_in_ring_selection(idx, add=False)
                else:
                    self._set_status("Shift+Click only works on existing points when points are selected.")
                return

            # --- Режим Прямоугольного удаления (Shift+Drag, *только* если НЕТ активного выбора кольцом) ---
            if is_shift_pressed and is_left_click and not self._ring_select_indices:
                if pos_yx is not None:
                    if self._cancel_all_interactions(): self._redraw()  # Отменяем всё перед началом выделения
                    self._push_undo()
                    self.rect_start = pos_yx
                    self._redo.clear()
                    self._set_status("Drag to select points for deletion.")
                return

            # --- Обычные клики (без Shift, Ctrl) ---
            if pos_yx is None: return

            y, x = pos_yx

            # --- Клик на центр (ЛКМ) ---
            if self._center_hit(y, x) and is_left_click:
                if self._cancel_all_interactions(): self._redraw()
                self._push_undo()
                self.center_dragging = True
                self._redo.clear()
                self._set_status("Dragging center overlay. Release to finish.")
                # self._redraw() # Перерисовка в _on_move
                return

            # --- Клик рядом с точкой ---
            i = self._near_idx(y, x)
            if i is not None:
                if is_left_click:
                    # --- ИСПРАВЛЕНА ЛОГИКА ЗАМЕРА (Click-Click) ---
                    if self._measure_start_idx is None:
                        # 1. Первый клик: Начинаем замер
                        if self._cancel_all_interactions(): self._redraw()  # Отменяем другие режимы
                        self._clear_measurement_result()  # Очищаем старый результат
                        self._start_measurement(i)
                        self._set_status(f"Measurement started from point {i}. Click second point.")
                        self._redraw()  # Показать подсветку начальной точки

                    # --- ИЗМЕНЕНИЕ: Второй клик ВСЕГДА завершает замер ---
                    elif self._measure_start_idx != i:
                        # 2. Клик на ДРУГУЮ точку: Завершаем замер
                        if self._finalize_measurement(i):
                            length = self._measurement.get("length", 0.0)
                            self._set_status(f"Measured {length:.1f} px between points.")
                        else:
                            self._set_status("Measurement failed.")  # (на всякий случай)
                        self._redraw()  # Показать результат (и снять подсветку)

                    else:  # Клик на ту же самую точку
                        # 3. Клик на ту же точку: Отмена
                        self._cancel_measurement_preview()
                        self._set_status(self._default_status)
                        self._redraw()  # Снять подсветку
                    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

                elif is_right_click:  # ПКМ на точке
                    # --- Удаление точки ---
                    if self._cancel_all_interactions(): self._redraw()
                    self._push_undo()

                    was_selected = i in self._ring_select_indices
                    if was_selected: self._ring_select_indices.remove(i)

                    # --- ИСПРАВЛЕНИЕ: Используем `len(self.points)` для проверки индекса ---
                    if i >= len(self.points):  # Защита от гонки состояний
                        self._set_status("Error: Point index out of bounds during deletion.")
                        return

                    self.points = np.delete(self.points, i, axis=0)
                    if self.values is not None and len(self.values) > i:
                        self.values = np.delete(self.values, i, axis=0)
                    else:
                        self.values = self._sample_intensities(self.points)

                    if self._ring_select_indices:
                        new_indices = set()
                        for idx in self._ring_select_indices:
                            if idx > i:
                                new_indices.add(idx - 1)
                            elif idx < i:
                                new_indices.add(idx)
                        self._ring_select_indices = new_indices

                    self._redo.clear()
                    self._set_status("Deleted point.")
                    self._redraw()
                    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
                return

            # --- Клик на пустом месте (ЛКМ) ---
            elif is_left_click:
                redraw_needed = self._cancel_all_interactions()
                redraw_needed |= self._clear_measurement_result()
                if redraw_needed: self._redraw()

                self._push_undo()
                self.points = np.vstack([self.points, [y, x]])
                sampled_value = self._sample_intensities(np.array([[y, x]]))[0]
                self.values = np.append(self.values, sampled_value)
                self._redo.clear()
                self._set_status(f"Added point at ({x:.1f}, {y:.1f}).")
                self._redraw()
                return

            # --- Клик на пустом месте (ПКМ) ---
            elif is_right_click:
                # Отменяем режимы, но ничего не делаем
                if self._cancel_all_interactions(): self._redraw()
                if self._clear_measurement_result(): self._redraw()
                return

    def _on_move(self, e):
        pos_yx = self._img_xy(e)

        if self.center_dragging and pos_yx is not None:
            y, x = pos_yx
            if self.overlay is None: self.overlay = {}
            self.overlay["center"] = {"x": float(x), "y": float(y)}
            self._redraw()
            self._set_status("Dragging center...")
            return

        if self._ring_select_active:
            self._update_ring_preview(pos_yx)
            return

        if self._measure_start_idx is not None:
            self._update_measurement_preview(pos_yx)
            return

        if self.rect_start and e.xdata is not None and e.ydata is not None:
            y0, x0 = self.rect_start
            y1, x1 = e.ydata, e.xdata

            if hasattr(self, 'rect_artist') and self.rect_artist is not None:
                try:
                    self.rect_artist.remove()
                except Exception:
                    pass
                self.rect_artist = None

            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()

            rect_x = min(x0, x1);
            rect_y = min(y0, y1)
            rect_w = abs(x1 - x0);
            rect_h = abs(y1 - y0)
            self.rect_artist = self.ax.add_patch(
                plt.Rectangle((rect_x, rect_y), rect_w, rect_h,
                              fill=False, ec="red", ls="--", lw=1.5, zorder=15)
            )

            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)
            self.canvas.draw_idle()
            return

    def _on_up(self, e):
        # --- Releasing after Dragging Center ---
        if self.center_dragging:
            self.center_dragging = False
            self._apply_center_filters()
            if self.overlay and isinstance(self.overlay.get("center"), dict):
                self.view_cx = float(self.overlay["center"]["x"])
                self.view_cy = float(self.overlay["center"]["y"])
            self._redraw()
            self._set_status("Center position updated.")
            return

        # --- Releasing in Ring Selection Mode ---
        if self._ring_select_active:
            # Ничего не делаем при отпускании, ждем Enter/Escape
            return

        # --- Releasing after Rectangular Selection ---
        if self.rect_start:
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
                    self._push_undo()  # Сохраняем перед удалением

                    indices_to_delete = np.where(mask_in_rect)[0]

                    # Обновляем выделение кольцом, если точки удаляются
                    deleted_selected_indices = self._ring_select_indices.intersection(indices_to_delete)
                    if deleted_selected_indices:
                        self._ring_select_indices.difference_update(deleted_selected_indices)
                        new_ring_indices = set()
                        num_deleted_before = {i: np.count_nonzero(indices_to_delete < i) for i in
                                              self._ring_select_indices}
                        for old_idx in self._ring_select_indices:
                            new_ring_indices.add(old_idx - num_deleted_before[old_idx])
                        self._ring_select_indices = new_ring_indices

                    mask_to_keep = ~mask_in_rect
                    self.points = self.points[mask_to_keep]
                    if self.values is not None and len(self.values) == len(mask_to_keep) + num_to_delete:
                        self.values = self.values[mask_to_keep]
                    else:
                        self.values = self._sample_intensities(self.points)

                    self._redo.clear()
                    self._set_status(f"Deleted {num_to_delete} points in selection.")
                else:
                    self._set_status("Rectangular selection finished, no points deleted.")

            self.rect_start = None
            if hasattr(self, 'rect_artist') and self.rect_artist is not None:
                try:
                    self.rect_artist.remove()
                except Exception:
                    pass
                self.rect_artist = None
            self._redraw()
            return

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
        if self.points is None or len(self.points) == 0: return
        if not (self.overlay and self.overlay.get("center")): return

        center_data = self.overlay["center"]
        if not isinstance(center_data, dict): return
        cy = float(center_data.get("y", 0.0));
        cx = float(center_data.get("x", 0.0))
        dead = float(self.overlay.get("dead_radius", 0.0));
        sr = float(self.overlay.get("search_radius", 0.0))
        if dead <= 0 and sr <= 0: return

        r = np.hypot(self.points[:, 1] - cx, self.points[:, 0] - cy)
        mask_keep = np.ones(len(self.points), dtype=bool)
        if dead > 0: mask_keep &= (r >= dead)
        if sr > 0: mask_keep &= (r <= sr)

        if self._ring_select_indices:
            indices_to_delete = np.where(~mask_keep)[0]
            deleted_selected_indices = self._ring_select_indices.intersection(indices_to_delete)
            if deleted_selected_indices:
                self._ring_select_indices.difference_update(deleted_selected_indices)
                new_ring_indices = set()
                num_deleted_before = {i: np.count_nonzero(indices_to_delete < i) for i in self._ring_select_indices}
                for old_idx in self._ring_select_indices:
                    new_ring_indices.add(old_idx - num_deleted_before[old_idx])
                self._ring_select_indices = new_ring_indices

        num_deleted = np.count_nonzero(~mask_keep)
        if num_deleted > 0:
            self.points = self.points[mask_keep]
            if self.values is not None and len(self.values) == len(mask_keep) + num_deleted:
                self.values = self.values[mask_keep]
            else:
                self.values = self._sample_intensities(self.points)
            self._set_status(f"Applied center filters, removed {num_deleted} points.")
            # Перерисовка будет вызвана в _on_up после center_dragging