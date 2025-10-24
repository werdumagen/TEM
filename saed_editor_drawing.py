#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mix-in класс для PointEditor:
Обрабатывает отрисовку (matplotlib), зум и панорамирование
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

class EditorDrawingView:

    # ---------- View-center helpers ----------
    def _ensure_view_center(self):
        if self.view_cx is not None and self.view_cy is not None:
            return

        cx_def, cy_def = 0.0, 0.0
        if self.overlay and isinstance(self.overlay.get("center"), dict):
            c = self.overlay["center"]
            cx_def = float(c.get("x", 0.0))
            cy_def = float(c.get("y", 0.0))
        elif self.img_arr is not None:
            H, W = self.img_arr.shape[:2]
            cx_def = (W - 1) / 2.0
            cy_def = (H - 1) / 2.0
        if self.view_cx is None: self.view_cx = cx_def
        if self.view_cy is None: self.view_cy = cy_def

    def _update_zoom_hint(self):
        if hasattr(self, "zoom_hint"):
            value = int(round(self.zoom_var.get())) if hasattr(self, "zoom_var") else int(round(self.zoom_val))
            self.zoom_hint.configure(text=f"Current zoom: {value}% (0 = full frame)")

    # ---------- Zoom ----------
    def _apply_zoom(self):
        # ... (код без изменений) ...
        if self.img_arr is None:
            self.ax.set_xlim(0, 100)
            self.ax.set_ylim(100, 0)
            return

        H, W = self.img_arr.shape[:2]
        self._ensure_view_center()

        x0_full, x1_full = -0.5, W - 0.5
        y0_full, y1_full = H - 0.5, -0.5

        if self.zoom_val <= 0:
            self.ax.set_xlim(x0_full, x1_full)
            self.ax.set_ylim(y0_full, y1_full)
            return

        min_dim = min(H, W)
        L = max(50.0, min_dim - (min_dim - 50.0) * (self.zoom_val / 100.0))
        half_w, half_h = L / 2.0, L / 2.0
        cx = float(self.view_cx);
        cy = float(self.view_cy)

        x0 = cx - half_w;
        x1 = cx + half_w
        y1 = cy - half_h;
        y0 = cy + half_h

        x0 = max(x0_full, x0);
        x1 = min(x1_full, x1)
        y1 = max(y1_full, y1);
        y0 = min(y0_full, y0)

        current_w = x1 - x0
        current_h = y0 - y1
        if current_w < L - 1e-6:
            if x0 == x0_full:
                x1 = min(x1_full, x0 + L)
            elif x1 == x1_full:
                x0 = max(x0_full, x1 - L)

        if current_h < L - 1e-6:
            if y1 == y1_full:
                y0 = min(y0_full, y1 + L)
            elif y0 == y0_full:
                y1 = max(y1_full, y0 - L)

        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)


    def _on_zoom_change(self, val=None):
        # ... (код без изменений) ...
        try:
            new_val = float(val) if val is not None else self.zoom_val
            new_zoom = max(0, min(100, int(round(new_val))))
        except (ValueError, TypeError):
            new_zoom = 0

        if new_zoom != self.zoom_val:
            self.zoom_val = new_zoom
            if hasattr(self, "zoom_var"):
                current_slider_val = int(round(self.zoom_var.get()))
                if current_slider_val != self.zoom_val:
                    self.zoom_var.set(self.zoom_val)

            self._update_zoom_hint()
            self._clear_tooltip()
            self._redraw()


    def _on_scroll(self, event):
        # ... (код без изменений) ...
        if event.xdata is None or event.ydata is None:
            return

        self.view_cx = event.xdata
        self.view_cy = event.ydata

        zoom_step = 5
        if event.button == 'up':
            new_zoom_val = self.zoom_val + zoom_step
        elif event.button == 'down':
            new_zoom_val = self.zoom_val - zoom_step
        else:
            return

        new_zoom_val = max(0, min(100, new_zoom_val))
        self._on_zoom_change(new_zoom_val)


    # ---------- Draw ----------
    def _draw_measurement_overlays(self) -> None:
        """Отрисовывает завершенный замер (линия + текст) или превью (пунктир)"""
        # ... (код без изменений) ...
        self._remove_measurement_artists()

        if self._measurement is not None and isinstance(self._measurement, dict):
            start_y, start_x = self._measurement.get("start_yx", (0, 0))
            end_y, end_x = self._measurement.get("end_yx", (0, 0))
            length = float(self._measurement.get("length", 0.0))

            (line,) = self.ax.plot(
                [start_x, end_x], [start_y, end_y],
                color="#ffcc33", linewidth=1.8, alpha=0.95,
                scalex=False, scaley=False, zorder=5
            )
            self._measure_line_artist = line

            mid_x = (start_x + end_x) / 2.0
            mid_y = (start_y + end_y) / 2.0
            txt = f"L = {length:.1f} px"
            self._measure_annotation = self.ax.annotate(
                txt, xy=(mid_x, mid_y), xytext=(0, -14),
                textcoords="offset points", ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9),
                fontsize=9, zorder=6
            )

        if (self._measure_start_point is not None and self._measure_preview_end is not None):
            y0, x0 = self._measure_start_point
            y1, x1 = self._measure_preview_end
            self._remove_measure_preview_artist()
            (pline,) = self.ax.plot(
                [x0, x1], [y0, y1], color="#ffcc33", linewidth=1.6,
                linestyle="--", alpha=0.9, scalex=False, scaley=False, zorder=10
            )
            self._measure_preview_artist = pline


    # --- НОВЫЕ МЕТОДЫ ОТРИСОВКИ КОЛЬЦА И ПОДСВЕТКИ ---
    def _remove_ring_preview_artist(self) -> bool:
        """Удаляет артисты кольца превью."""
        if hasattr(self, '_ring_select_artist') and self._ring_select_artist:
            try:
                for patch in self._ring_select_artist: patch.remove()
                self._ring_select_artist = None
                return True
            except Exception: pass
        return False

    def _draw_ring_preview(self) -> None:
        """Отрисовывает кольцо превью."""
        self._remove_ring_preview_artist() # Удаляем старое
        if not self._ring_select_active or self._ring_select_center_yx is None:
            return

        cy, cx = self._ring_select_center_yx
        radius = self._ring_select_radius
        thickness = self._ring_select_thickness
        r_inner = max(0, radius - thickness / 2.0)
        r_outer = radius + thickness / 2.0

        # Рисуем две окружности
        circle_outer = Circle((cx, cy), r_outer, fill=False, ec="orange", ls="-", lw=1.5, alpha=0.8, zorder=12)
        circle_inner = Circle((cx, cy), r_inner, fill=False, ec="orange", ls=":", lw=1.0, alpha=0.8, zorder=12)

        self.ax.add_patch(circle_outer)
        self.ax.add_patch(circle_inner)
        self._ring_select_artist = [circle_outer, circle_inner] # Сохраняем оба

    def _highlight_selected_ring_points(self) -> None:
        """Подсвечивает точки, выбранные кольцом."""
        if not self._ring_select_indices or self.points is None:
            return

        indices_to_highlight = list(self._ring_select_indices)
        valid_indices = [i for i in indices_to_highlight if 0 <= i < len(self.points)]

        if valid_indices:
            pts = self.points[valid_indices]
            self.ax.scatter(pts[:, 1], pts[:, 0], s=42, c="#ffd34d", # Yellowish
                            alpha=0.95, marker="o", linewidths=0.8,
                            edgecolors="black", zorder=4) # zorder=4, выше обычных точек

    # --- КОНЕЦ НОВЫХ МЕТОДОВ ---

    def _redraw(self):
        self.ax.clear()
        if self.img_arr is not None:
            self.ax.imshow(self.img_arr, cmap="gray", interpolation="nearest")
        self.ax.axis("off")

        # Draw Center and Radii Overlay
        if self.overlay and isinstance(self.overlay.get("center"), dict):
            center_data = self.overlay["center"]
            cy = float(center_data.get("y", 0))
            cx = float(center_data.get("x", 0))
            self.ax.scatter([cx], [cy], s=40, c="red", marker="o", zorder=5)
            dead = float(self.overlay.get("dead_radius", 0))
            sr = float(self.overlay.get("search_radius", 0))
            if dead > 0:
                self.ax.add_patch(Circle((cx, cy), dead, fill=False, ls="--", lw=1.5, ec="red", zorder=4))
            if sr > 0:
                self.ax.add_patch(Circle((cx, cy), sr, fill=False, ls=":", lw=1.0, ec="red", zorder=4))

        # Draw Points
        if self.points is not None and len(self.points) > 0:
            points_to_draw = self.points
            colors = 'cyan'
            sizes = 22
            zorder = 3

            # --- ИЗМЕНЕНО: Рисуем все точки без специальной подсветки для merge ---
            self.ax.scatter(points_to_draw[:, 1], points_to_draw[:, 0],
                            s=sizes, c=colors, alpha=0.9, marker="o",
                            linewidths=0.5, edgecolors="black", zorder=zorder)
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        # --- НОВОЕ: Подсветка выделенных кольцом точек ---
        self._highlight_selected_ring_points()

        # --- ИЗМЕНЕНИЕ: Подсветка точки начала замера ---
        if self._measure_start_idx is not None and 0 <= self._measure_start_idx < len(self.points):
            # Проверяем, не выделена ли уже эта точка кольцом
            if self._measure_start_idx not in self._ring_select_indices:
                y, x = self.points[self._measure_start_idx]
                self.ax.scatter([x], [y], s=42, c="#FFA500", # Orange for measurement start
                                alpha=0.95, marker="o", linewidths=0.8,
                                edgecolors="black", zorder=zorder + 1)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        # Draw Measurement Overlays
        self._draw_measurement_overlays()

        # --- НОВОЕ: Отрисовка превью кольца ---
        self._draw_ring_preview()

        # Apply Zoom
        self._apply_zoom()

        # Update Canvas
        self.canvas.draw_idle()