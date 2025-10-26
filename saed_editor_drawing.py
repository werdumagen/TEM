#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mix-in класс для PointEditor:
Обрабатывает отрисовку (matplotlib), зум и панорамирование
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from saed_editor_state import CENTER_AS_POINT_IDX
# --- Убран импорт colormaps ---


class EditorDrawingView:

    # --- Методы View/Zoom (_ensure_view_center, ..., _on_scroll) без изменений ---
    def _ensure_view_center(self): # ...
        if self.view_cx is not None and self.view_cy is not None: return
        cx_def, cy_def = 0.0, 0.0
        if self.overlay and isinstance(self.overlay.get("center"), dict): c = self.overlay["center"]; cx_def = float(c.get("x", 0.0)); cy_def = float(c.get("y", 0.0))
        # --- ИЗМЕНЕНИЕ: Используем размер обработанного изображения ---
        elif self.img_arr_processed is not None: H, W = self.img_arr_processed.shape[:2]; cx_def = (W - 1) / 2.0; cy_def = (H - 1) / 2.0
        elif self.img_arr_raw is not None: H, W = self.img_arr_raw.shape[:2]; cx_def = (W - 1) / 2.0; cy_def = (H - 1) / 2.0
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
        if self.view_cx is None: self.view_cx = cx_def
        if self.view_cy is None: self.view_cy = cy_def

    def _update_zoom_hint(self): # ...
        if hasattr(self, "zoom_hint"): value = int(round(self.zoom_var.get())) if hasattr(self, "zoom_var") else int(round(self.zoom_val)); self.zoom_hint.configure(text=f"Current zoom: {value}% (0 = full frame)")

    def _apply_zoom(self): # ...
        # --- ИЗМЕНЕНИЕ: Используем размер изображения в зависимости от режима ---
        img_to_use = self.img_arr_raw if self.show_raw_background.get() else self.img_arr_processed
        if img_to_use is None: # Фоллбэк, если нужное изображение не загружено
             img_to_use = self.img_arr_processed if self.img_arr_processed is not None else self.img_arr_raw

        if img_to_use is None:
             self.ax.set_xlim(0, 100); self.ax.set_ylim(100, 0); return
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        H, W = img_to_use.shape[:2]; self._ensure_view_center()
        x0_full, x1_full = -0.5, W - 0.5; y0_full, y1_full = H - 0.5, -0.5
        if self.zoom_val <= 0: self.ax.set_xlim(x0_full, x1_full); self.ax.set_ylim(y0_full, y1_full); return
        min_dim = min(H, W); L = max(50.0, min_dim - (min_dim - 50.0) * (self.zoom_val / 100.0))
        half_w, half_h = L / 2.0, L / 2.0; cx = float(self.view_cx); cy = float(self.view_cy)
        x0 = cx - half_w; x1 = cx + half_w; y1 = cy - half_h; y0 = cy + half_h
        x0 = max(x0_full, x0); x1 = min(x1_full, x1); y1 = max(y1_full, y1); y0 = min(y0_full, y0)
        current_w = x1 - x0; current_h = y0 - y1
        if current_w < L - 1e-6:
            if x0 == x0_full: x1 = min(x1_full, x0 + L)
            elif x1 == x1_full: x0 = max(x0_full, x1 - L)
        if current_h < L - 1e-6:
            if y1 == y1_full: y0 = min(y0_full, y1 + L)
            elif y0 == y0_full: y1 = max(y1_full, y0 - L)
        self.ax.set_xlim(x0, x1); self.ax.set_ylim(y0, y1)

    def _on_zoom_change(self, val=None): # ...
        try: new_val = float(val) if val is not None else self.zoom_val; new_zoom = max(0, min(100, int(round(new_val))))
        except (ValueError, TypeError): new_zoom = 0
        if new_zoom != self.zoom_val:
            self.zoom_val = new_zoom
            if hasattr(self, "zoom_var"): current_slider_val = int(round(self.zoom_var.get()));
            if current_slider_val != self.zoom_val: self.zoom_var.set(self.zoom_val)
            self._update_zoom_hint(); self._clear_tooltip(); self._redraw()

    def _on_scroll(self, event): # ...
        if event.xdata is None or event.ydata is None: return
        self.view_cx = event.xdata; self.view_cy = event.ydata
        zoom_step = 5;
        if event.button == 'up': new_zoom_val = self.zoom_val + zoom_step
        elif event.button == 'down': new_zoom_val = self.zoom_val - zoom_step
        else: return
        new_zoom_val = max(0, min(100, new_zoom_val)); self._on_zoom_change(new_zoom_val)

    # ---------- Draw ----------
    def _draw_measurement_overlays(self) -> None:
        """Отрисовывает завершенный замер (линия + текст) или превью (пунктир)"""
        # ... (код без изменений) ...
        self._remove_measurement_artists()
        if self._measurement is not None and isinstance(self._measurement, dict):
            start_y, start_x = self._measurement.get("start_yx", (0, 0)); end_y, end_x = self._measurement.get("end_yx", (0, 0))
            length = float(self._measurement.get("length", 0.0))
            (line,) = self.ax.plot([start_x, end_x], [start_y, end_y], color="#ffcc33", lw=1.8, alpha=0.95, scalex=False, scaley=False, zorder=5)
            self._measure_line_artist = line
            mid_x = (start_x + end_x) / 2.0; mid_y = (start_y + end_y) / 2.0
            txt = f"L = {length:.1f} px"
            self._measure_annotation = self.ax.annotate(txt, xy=(mid_x, mid_y), xytext=(0, -14), textcoords="offset points", ha="center", va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9), fontsize=9, zorder=6)
        if (self._measure_start_point is not None and self._measure_preview_end is not None):
            y0, x0 = self._measure_start_point; y1, x1 = self._measure_preview_end
            self._remove_measure_preview_artist()
            (pline,) = self.ax.plot([x0, x1], [y0, y1], color="#ffcc33", lw=1.6, ls="--", alpha=0.9, scalex=False, scaley=False, zorder=10)
            self._measure_preview_artist = pline

    # --- Методы отрисовки кольца и подсветки (_remove_ring_preview_artist, _draw_ring_preview, _highlight_selected_ring_points) без изменений ---
    def _remove_ring_preview_artist(self) -> bool: # ...
        if hasattr(self, '_ring_select_artist') and self._ring_select_artist:
            try:
                if isinstance(self._ring_select_artist, list):
                    for patch in self._ring_select_artist: patch.remove()
                elif self._ring_select_artist is not None: self._ring_select_artist.remove()
                self._ring_select_artist = None; return True
            except Exception: pass; self._ring_select_artist = None
        return False
    def _draw_ring_preview(self) -> None: # ...
        self._remove_ring_preview_artist()
        if not self._ring_select_active or self._ring_select_center_yx is None: return
        cy, cx = self._ring_select_center_yx; radius = self._ring_select_radius; thickness = self._ring_select_thickness
        r_inner = max(0, radius - thickness / 2.0); r_outer = radius + thickness / 2.0
        circle_outer = Circle((cx, cy), r_outer, fill=False, ec="orange", ls="-", lw=1.5, alpha=0.8, zorder=12)
        circle_inner = Circle((cx, cy), r_inner, fill=False, ec="orange", ls=":", lw=1.0, alpha=0.8, zorder=12)
        self.ax.add_patch(circle_outer); self.ax.add_patch(circle_inner)
        self._ring_select_artist = [circle_outer, circle_inner]
    def _highlight_selected_ring_points(self) -> None: # ...
        if not self._ring_select_indices or self.points is None: return
        indices_to_highlight = list(self._ring_select_indices)
        valid_indices = [i for i in indices_to_highlight if 0 <= i < len(self.points)]
        if valid_indices: pts = self.points[valid_indices]; self.ax.scatter(pts[:, 1], pts[:, 0], s=42, c="#ffd34d", alpha=0.95, marker="o", lw=0.8, edgecolors="black", zorder=4)


    def _redraw(self):
        self.ax.clear()
        # --- ИЗМЕНЕНИЕ: Выбираем фон в зависимости от галочки ---
        img_to_display = self.img_arr_raw if self.show_raw_background.get() else self.img_arr_processed
        # Фоллбэк, если выбранное изображение не загружено
        if img_to_display is None:
            img_to_display = self.img_arr_processed if self.img_arr_processed is not None else self.img_arr_raw

        if img_to_display is not None:
            self.ax.imshow(img_to_display, cmap="gray", interpolation="nearest")
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
        self.ax.axis("off")

        # --- Draw Center and Radii Overlay ---
        center_drawn_yx = None
        if self.overlay and isinstance(self.overlay.get("center"), dict):
            center_data = self.overlay["center"]
            cy = float(center_data.get("y", 0))
            cx = float(center_data.get("x", 0))
            center_drawn_yx = (cy, cx)
            center_color = "red"; center_zorder = 5; center_size = 40
            if self._measure_start_idx == CENTER_AS_POINT_IDX:
                 center_color = "#FFA500"; center_zorder = 4; center_size = 50
            self.ax.scatter([cx], [cy], s=center_size, c=center_color, marker="o", zorder=center_zorder)

            dead = float(self.overlay.get("dead_radius", 0))
            sr = float(self.overlay.get("search_radius", 0))
            if dead > 0: self.ax.add_patch(Circle((cx, cy), dead, fill=False, ls="--", lw=1.5, ec="red", zorder=4))
            if sr > 0: self.ax.add_patch(Circle((cx, cy), sr, fill=False, ls=":", lw=1.0, ec="red", zorder=4))

        # --- Draw Points ---
        if self.points is not None and len(self.points) > 0:
            points_to_draw = self.points
            sizes = 22
            zorder = 3
            linewidths = 0.5
            edgecolors = "black"
            alpha = 0.9

            # --- ИЗМЕНЕНИЕ: Новая логика цветов (str типы + numeric ID) ---
            colors = []
            numeric_group_ids = set()
            max_numeric_id = -1
            if hasattr(self, 'point_types') and len(self.point_types) == len(points_to_draw):
                # Сначала собираем все числовые ID
                for pt_type in self.point_types:
                    if isinstance(pt_type, int):
                        numeric_group_ids.add(pt_type)
                        if pt_type > max_numeric_id:
                            max_numeric_id = pt_type

                # Создаем карту цветов для числовых ID
                num_numeric_groups = len(numeric_group_ids)
                cmap_N = max(max_numeric_id + 1, 1) # Используем max ID + 1 для стабильности цветов
                colormap = plt.get_cmap('viridis', cmap_N)
                numeric_color_map = {gid: colormap(gid / max(cmap_N - 1, 1)) for gid in numeric_group_ids}

                # Карта для строковых типов
                string_color_map = {
                    "unknown": "yellow",         # Желтый
                    "structural": "cyan",        # Голубой
                    "superstructural": "magenta",# Фиолетовый (ярко-розовый)
                    # "other" больше не используется как финальный тип
                }

                # Назначаем цвета
                for pt_type in self.point_types:
                    if isinstance(pt_type, str):
                        colors.append(string_color_map.get(pt_type, "gray")) # Строковый тип
                    elif isinstance(pt_type, int):
                        colors.append(numeric_color_map.get(pt_type, "gray")) # Числовой ID
                    else:
                        colors.append("gray") # Неизвестный тип данных
            else:
                # Фоллбэк
                colors = ['yellow'] * len(points_to_draw)

            self.ax.scatter(points_to_draw[:, 1], points_to_draw[:, 0],
                            s=sizes, c=colors, alpha=alpha, marker="o",
                            linewidths=linewidths, edgecolors=edgecolors, zorder=zorder)
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        # --- Highlight selected ring points ---
        self._highlight_selected_ring_points() # Желтая подсветка поверх основного цвета

        # --- Highlight measurement start point (if it's a regular point) ---
        if self._measure_start_idx is not None and self._measure_start_idx != CENTER_AS_POINT_IDX:
            if 0 <= self._measure_start_idx < len(self.points) and \
               self._measure_start_idx not in self._ring_select_indices: # Не подсвечиваем, если уже выделено кольцом
                y, x = self.points[self._measure_start_idx]
                self.ax.scatter([x], [y], s=sizes+20, c="#FFA500", # Orange, чуть больше
                                alpha=0.95, marker="o", linewidths=linewidths+0.3,
                                edgecolors=edgecolors, zorder=zorder + 1)

        # --- Draw Measurement Overlays ---
        self._draw_measurement_overlays()

        # --- Draw Ring Preview ---
        self._draw_ring_preview()

        # --- Apply Zoom ---
        self._apply_zoom()

        # --- Update Canvas ---
        self.canvas.draw_idle()