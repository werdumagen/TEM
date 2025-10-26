#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Класс Отрисовки (View) (Refactored)
------------------------------------
Отвечает ИСКЛЮЧИТЕЛЬНО за отрисовку данных из
Модели (model) и Состояния UI (ui_state) на
холсте (ax).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from typing import Optional, Tuple # <--- ИСПРАВЛЕНИЕ: Добавлен этот импорт

# Импорт UI State для type hint
try:
    from saed_editor_state_ui import EditorUIState, CENTER_AS_POINT_IDX
except ImportError:
    # Фоллбэк, если файл еще не создан или есть ошибка импорта
    print("Warning: Could not import EditorUIState. Using dummy class.")
    class EditorUIState: pass
    CENTER_AS_POINT_IDX = -1


class EditorDrawingView:

    def __init__(self, ax, canvas):
        self.ax = ax
        self.canvas = canvas
        self._center_hit_radius = 10.0 # Допуск нажатия на центр

    # --- Методы View/Zoom ---

    def _ensure_view_center(self, controller):
        """Гарантирует, что у controller есть view_cx, view_cy."""
        if controller.view_cx is not None and controller.view_cy is not None:
            return

        cx_def, cy_def = 0.0, 0.0
        center = controller.get_center()
        if center:
            cy_def, cx_def = center
        else:
            img_to_use = controller.get_image_to_display()
            if img_to_use is not None:
                H, W = img_to_use.shape[:2]
                cx_def = (W - 1) / 2.0
                cy_def = (H - 1) / 2.0

        if controller.view_cx is None: controller.view_cx = cx_def
        if controller.view_cy is None: controller.view_cy = cy_def

    def _apply_zoom(self, controller):
        """Применяет текущий зум и панорамирование к self.ax."""
        img_to_use = controller.get_image_to_display()
        if img_to_use is None:
             self.ax.set_xlim(0, 100); self.ax.set_ylim(100, 0); return

        H, W = img_to_use.shape[:2]
        self._ensure_view_center(controller)

        x0_full, x1_full = -0.5, W - 0.5
        y0_full, y1_full = H - 0.5, -0.5

        if controller.zoom_val <= 0:
            self.ax.set_xlim(x0_full, x1_full)
            self.ax.set_ylim(y0_full, y1_full) # ymax, ymin for imshow
            return

        min_dim = min(H, W)
        L = max(50.0, min_dim - (min_dim - 50.0) * (controller.zoom_val / 100.0))
        half_w, half_h = L/2.0, L/2.0 # Keep aspect ratio square for zoom window
        cx = float(controller.view_cx)
        cy = float(controller.view_cy)

        x0 = max(x0_full, cx - half_w)
        x1 = min(x1_full, cx + half_w)
        y1 = max(y1_full, cy - half_h) # ymin is max() because axis inverted
        y0 = min(y0_full, cy + half_h) # ymax is min()

        # Adjust if zoom window is smaller than L due to hitting image boundaries
        current_w = x1 - x0
        current_h = y0 - y1 # y0 > y1
        if current_w < L - 1e-6:
             if x0 == x0_full: x1 = min(x1_full, x0 + L)
             elif x1 == x1_full: x0 = max(x0_full, x1 - L)
        if current_h < L - 1e-6:
             if y1 == y1_full: y0 = min(y0_full, y1 + L)
             elif y0 == y0_full: y1 = max(y1_full, y0 - L)

        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1) # ymax, ymin


    # ---------- Методы Отрисовки Оверлеев ----------

    def draw_measurement_overlays(self, ui_state: EditorUIState) -> None:
        """Отрисовывает завершенный замер (линия + текст) или превью (пунктир)"""

        # 1. Очищаем старые artists
        ui_state.remove_measurement_artists()
        ui_state.remove_measure_preview_artist()

        # 2. Рисуем завершенный замер
        if ui_state.measurement is not None:
            start_y, start_x = ui_state.measurement.get("start_yx", (0, 0))
            end_y, end_x = ui_state.measurement.get("end_yx", (0, 0))
            length = float(ui_state.measurement.get("length", 0.0))

            (line,) = self.ax.plot([start_x, end_x], [start_y, end_y], color="#ffcc33", lw=1.8, alpha=0.95, scalex=False, scaley=False, zorder=5)
            ui_state.measure_line_artist = line # Сохраняем artist в UI State

            mid_x = (start_x + end_x) / 2.0; mid_y = (start_y + end_y) / 2.0
            txt = f"L = {length:.1f} px"
            annot = self.ax.annotate(txt, xy=(mid_x, mid_y), xytext=(0, -14), textcoords="offset points", ha="center", va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9), fontsize=9, zorder=6)
            ui_state.measure_annotation = annot # Сохраняем artist в UI State

        # 3. Рисуем превью (пунктир)
        if (ui_state.measure_start_point is not None and ui_state.measure_preview_end is not None):
            y0, x0 = ui_state.measure_start_point
            y1, x1 = ui_state.measure_preview_end

            (pline,) = self.ax.plot([x0, x1], [y0, y1], color="#ffcc33", lw=1.6, ls="--", alpha=0.9, scalex=False, scaley=False, zorder=10)
            ui_state.measure_preview_artist = pline # Сохраняем artist в UI State

    def remove_ring_preview_artist(self, ui_state: EditorUIState) -> bool:
        """Удаляет artists кольца из ui_state."""
        if ui_state.ring_select_artist:
            try:
                if isinstance(ui_state.ring_select_artist, list):
                    for patch in ui_state.ring_select_artist: patch.remove()
                else:
                    ui_state.ring_select_artist.remove()
                ui_state.ring_select_artist = None
                return True
            except Exception: pass
            ui_state.ring_select_artist = None
        return False

    def draw_ring_preview(self, ui_state: EditorUIState) -> None:
        """Рисует превью выбора кольцом."""
        self.remove_ring_preview_artist(ui_state)
        if not ui_state.ring_select_active or ui_state.ring_select_center_yx is None:
            return

        cy, cx = ui_state.ring_select_center_yx
        radius = ui_state.ring_select_radius
        thickness = ui_state.ring_select_thickness
        r_inner = max(0, radius - thickness / 2.0)
        r_outer = radius + thickness / 2.0

        circle_outer = Circle((cx, cy), r_outer, fill=False, ec="orange", ls="-", lw=1.5, alpha=0.8, zorder=12)
        circle_inner = Circle((cx, cy), r_inner, fill=False, ec="orange", ls=":", lw=1.0, alpha=0.8, zorder=12)

        self.ax.add_patch(circle_outer)
        self.ax.add_patch(circle_inner)
        ui_state.ring_select_artist = [circle_outer, circle_inner] # Сохраняем в UI State

    def highlight_selected_ring_points(self, model, ui_state: EditorUIState) -> None:
        """Подсвечивает точки, выбранные кольцом."""
        if not ui_state.ring_select_indices or (model is None or model.is_empty()): return # Добавлена проверка model

        indices = list(ui_state.ring_select_indices)
        valid_indices = [i for i in indices if 0 <= i < len(model.points)]
        if valid_indices:
            pts = model.points[valid_indices]
            self.ax.scatter(pts[:, 1], pts[:, 0], s=42, c="#ffd34d", alpha=0.95,
                            marker="o", lw=0.8, edgecolors="black", zorder=4)

    def remove_rect_artist(self, ui_state: EditorUIState):
        """Удаляет artist прямоугольного выделения."""
        if ui_state.rect_artist is not None:
            try: ui_state.rect_artist.remove()
            except Exception: pass
            ui_state.rect_artist = None

    def draw_rect_preview(self, ui_state: EditorUIState, pos_yx: Tuple[float, float]):
        """Рисует прямоугольник выделения."""
        self.remove_rect_artist(ui_state) # Удаляем старый

        if ui_state.rect_start is None: # Добавлена проверка
             return

        y0, x0 = ui_state.rect_start
        y1, x1 = pos_yx

        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()

        rect_x = min(x0, x1); rect_y = min(y0, y1)
        rect_w = abs(x1 - x0); rect_h = abs(y1 - y0)

        rect_artist = plt.Rectangle((rect_x, rect_y), rect_w, rect_h,
                                    fill=False, ec="red", ls="--", lw=1.5, zorder=15)
        self.ax.add_patch(rect_artist)
        ui_state.rect_artist = rect_artist # Сохраняем в UI State

        self.ax.set_xlim(current_xlim)
        self.ax.set_ylim(current_ylim)
        self.canvas.draw_idle()

    # ---------- Главный метод Redraw ----------

    def redraw(self, controller):
        """Полностью перерисовывает холст, читая данные из controller."""

        self.ax.clear()

        # Получаем компоненты из контроллера
        # Добавляем проверки на случай, если они еще не инициализированы
        model = getattr(controller, 'model', None)
        ui_state = getattr(controller, 'ui_state', None)

        if model is None or ui_state is None:
            print("Warning: Redraw called before model/ui_state are initialized.")
            self.ax.axis("off")
            self._apply_zoom(controller) # Применяем зум, чтобы показать пустую область
            self.canvas.draw_idle()
            return

        # --- 1. Фон ---
        img_to_display = controller.get_image_to_display()
        if img_to_display is not None:
            self.ax.imshow(img_to_display, cmap="gray", interpolation="nearest")
        self.ax.axis("off")

        # --- 2. Центр и Радиальные Оверлеи ---
        center = controller.get_center()
        if center:
            cy, cx = center
            center_color = "red"; center_zorder = 5; center_size = 40
            if ui_state.measure_start_idx == CENTER_AS_POINT_IDX:
                 center_color = "#FFA500"; center_zorder = 4; center_size = 50
            self.ax.scatter([cx], [cy], s=center_size, c=center_color, marker="o", zorder=center_zorder)

            dead = controller.get_dead_radius()
            sr = controller.get_search_radius()
            if dead > 0: self.ax.add_patch(Circle((cx, cy), dead, fill=False, ls="--", lw=1.5, ec="red", zorder=4))
            if sr > 0: self.ax.add_patch(Circle((cx, cy), sr, fill=False, ls=":", lw=1.0, ec="red", zorder=4))

        # --- 3. Точки ---
        if not model.is_empty():
            points_to_draw = model.points
            # *** КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Модель сама решает, какие цвета ***
            colors = model.get_colors_for_drawing()
            # ***

            if colors: # Убедимся, что список цветов не пуст
                self.ax.scatter(points_to_draw[:, 1], points_to_draw[:, 0],
                                s=22, c=colors, alpha=0.9, marker="o",
                                linewidths=0.5, edgecolors="black", zorder=3)

        # --- 4. Подсветка выделенных точек ---
        self.highlight_selected_ring_points(model, ui_state)

        # --- 5. Подсветка начальной точки замера ---
        if ui_state.measure_start_idx is not None and \
           ui_state.measure_start_idx != CENTER_AS_POINT_IDX:

            idx = ui_state.measure_start_idx
            # Добавлена проверка model
            if model is not None and 0 <= idx < len(model.points) and idx not in ui_state.ring_select_indices:
                y, x = model.points[idx]
                self.ax.scatter([x], [y], s=42, c="#FFA500",
                                alpha=0.95, marker="o", linewidths=0.8,
                                edgecolors="black", zorder=3.1)

        # --- 6. Оверлеи Замера (линии/текст) ---
        self.draw_measurement_overlays(ui_state)

        # --- 7. Превью Кольца ---
        self.draw_ring_preview(ui_state)

        # --- 8. Превью Прямоугольника (оно рисуется в _on_move, здесь не нужно) ---
        if ui_state.rect_start is None:
            self.remove_rect_artist(ui_state)

        # --- 9. Зум ---
        self._apply_zoom(controller)

        # --- 10. Обновление ---
        self.canvas.draw_idle()

    # --- Утилиты ---
    def img_xy(self, e):
        """Конвертирует событие matplotlib в (y, x) координаты изображения."""
        return None if (e.xdata is None or e.ydata is None) else (e.ydata, e.xdata)

    def center_hit(self, y: float, x: float, center: Optional[Tuple[float, float]]) -> bool:
        """Проверяет, было ли нажатие на центр."""
        if not center: return False
        cy, cx = center
        dist_sq = (y - cy) ** 2 + (x - cx) ** 2
        return dist_sq <= self._center_hit_radius ** 2