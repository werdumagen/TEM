#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Класс Модели Данных (Refactored)
--------------------------------
Инкапсулирует ВСЕ данные, связанные с точками
(координаты, значения, типы, площади, углы и т.д.).

Предоставляет API для добавления, удаления, обновления и запроса
этих данных, гарантируя их согласованность.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List, Tuple, Union


class SaedDataModel:

    def __init__(self):
        """Инициализирует все массивы данных."""
        self.points = np.zeros((0, 2), float)
        self.values = np.zeros((0,), float)  # Интенсивности в процентилях
        self.areas = np.zeros((0,), float)  # Площади пикселей
        self.angles = np.zeros((0,), float)  # Углы
        self.point_types: list[Union[str, int]] = []  # Типы (str) или ID (int)
        self.initial_group_ids: Dict[int, Optional[int]] = {}

        # --- Настройки для отрисовки ---
        self._cmap_N = 1  # Для стабильности цветов
        self._colormap = plt.get_cmap('viridis', self._cmap_N)
        self._numeric_color_map = {}
        self._string_color_map = {
            "unknown": "yellow",
            "structural": "cyan",
            "superstructural": "magenta",
        }

    def is_empty(self) -> bool:
        return len(self.points) == 0

    def _calculate_pol_from(self, center: Tuple[float, float], pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Рассчитывает полярные координаты (r, a) для точек (pts) относительно центра."""
        if len(pts) == 0:
            return np.array([]), np.array([])
        cy, cx = center
        dy = pts[:, 0] - cy
        dx = pts[:, 1] - cx
        r = np.hypot(dx, dy)
        a = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
        return r, a

    def add_point(self, y: float, x: float, value: float, center: Optional[Tuple[float, float]],
                  area: float = 0.0, point_type: Union[str, int] = "unknown",
                  initial_id: Optional[int] = None):
        """
        Добавляет новую точку и все связанные с ней данные,
        обеспечивая согласованность массивов.
        """
        self.points = np.vstack([self.points, [y, x]])
        self.values = np.append(self.values, value)
        self.areas = np.append(self.areas, area)
        self.point_types.append(point_type)

        new_idx = len(self.points) - 1
        self.initial_group_ids[new_idx] = initial_id

        # Рассчитываем угол
        new_angle = np.nan
        if center:
            _, angle_arr = self._calculate_pol_from(center, np.array([[y, x]]))
            new_angle = angle_arr[0]

        self.angles = np.append(self.angles, new_angle)

    def delete_points_by_indices(self, indices: List[int]):
        """Удаляет точки по списку индексов из всех массивов."""
        if not indices:
            return

        # Создаем маску для удаления
        mask = np.ones(len(self.points), dtype=bool)
        mask[indices] = False

        self.delete_points_by_mask(mask)

    def delete_points_by_mask(self, mask: np.ndarray):
        """Удаляет точки по булевой маске (True = сохранить, False = удалить)."""

        # 1. Фильтруем основные массивы
        self.points = self.points[mask]
        self.values = self.values[mask]
        self.areas = self.areas[mask]
        self.angles = self.angles[mask]

        # 2. Фильтруем list 'point_types'
        types_array = np.array(self.point_types, dtype=object)
        self.point_types = types_array[mask].tolist()

        # 3. Фильтруем dict 'initial_group_ids'
        old_indices_to_keep = np.where(mask)[0]
        new_initial_ids = {}
        for new_idx, old_idx in enumerate(old_indices_to_keep):
            new_initial_ids[new_idx] = self.initial_group_ids.get(old_idx)
        self.initial_group_ids = new_initial_ids

    def update_points(self, indices: List[int], new_points_yx: np.ndarray,
                      new_values: np.ndarray, center: Optional[Tuple[float, float]]):
        """
        Обновляет данные для подмножества точек (используется для "усреднения").
        """
        if len(indices) != len(new_points_yx) or len(indices) != len(new_values):
            print("Warning: Mismatch in update_points lengths.")
            return

        self.points[indices] = new_points_yx
        self.values[indices] = new_values

        # Пересчитываем углы для обновленных точек
        if center:
            _, new_angles_arr = self._calculate_pol_from(center, new_points_yx)
            self.angles[indices] = new_angles_arr
        else:
            self.angles[indices] = np.nan

        # Площади и типы не меняем

    def recalculate_angles(self, center: Optional[Tuple[float, float]]) -> np.ndarray:
        """Пересчитывает все углы на основе нового центра и сохраняет их."""
        if center and not self.is_empty():
            all_radii, self.angles = self._calculate_pol_from(center, self.points)
            return all_radii
        else:
            self.angles = np.full(len(self.points), np.nan)
            if self.is_empty():
                return np.array([])
            else:
                # Возвращаем радиусы, даже если углы NaN
                r = np.hypot(self.points[:, 1] - 0, self.points[:, 0] - 0)
                return r

    def find_nearest_point_idx(self, y: float, x: float, pix_tol: float = 8.0) -> Optional[int]:
        """Находит индекс ближайшей точки в пределах допуска."""
        if self.is_empty(): return None
        dist_sq = (self.points[:, 0] - y) ** 2 + (self.points[:, 1] - x) ** 2
        i = int(np.argmin(dist_sq))
        return i if dist_sq[i] <= pix_tol ** 2 else None

    # --- Методы для Undo/Redo ---

    def get_snapshot(self) -> Dict[str, Any]:
        """Создает "слепок" всех данных для стека Undo."""
        return {
            "points": self.points.copy(),
            "values": self.values.copy(),
            "areas": self.areas.copy(),
            "angles": self.angles.copy(),
            "point_types": list(self.point_types),
            "initial_group_ids": dict(self.initial_group_ids),
        }

    def apply_snapshot(self, snapshot: Dict[str, Any]):
        """Восстанавливает состояние модели из "слепка"."""
        self.points = snapshot.get("points", np.zeros((0, 2))).copy()
        self.values = snapshot.get("values", np.zeros((0,))).copy()
        self.areas = snapshot.get("areas", np.zeros((len(self.points),))).copy()
        self.angles = snapshot.get("angles", np.full(len(self.points), np.nan)).copy()
        self.point_types = list(snapshot.get("point_types", ["unknown"] * len(self.points)))
        self.initial_group_ids = dict(snapshot.get("initial_group_ids", {}))

        # Валидация
        self._validate_consistency()

    def _validate_consistency(self):
        """Проверяет, что все массивы имеют одинаковую длину."""
        n = len(self.points)
        if len(self.values) != n:
            self.values = np.zeros((n,), float)
            print("Warning: Model inconsistency (values) corrected.")
        if len(self.areas) != n:
            self.areas = np.zeros((n,), float)
            print("Warning: Model inconsistency (areas) corrected.")
        if len(self.angles) != n:
            self.angles = np.full((n,), np.nan)
            print("Warning: Model inconsistency (angles) corrected.")
        if len(self.point_types) != n:
            self.point_types = ["unknown"] * n
            print("Warning: Model inconsistency (point_types) corrected.")
        # initial_group_ids (dict) не требует строгой проверки длины

    # --- Методы для IO (Загрузка/Сохранение) ---

    def from_json_list(self, points_list: List[Dict[str, Any]],
                       sample_values_func, center: Optional[Tuple[float, float]]):
        """Заполняет модель данными из JSON."""
        if not points_list:
            self.__init__()  # Сбрасываем модель
            return

        yy = [float(p.get("y", 0.0)) for p in points_list]
        xx = [float(p.get("x", 0.0)) for p in points_list]
        self.points = np.column_stack([yy, xx]).astype(float)

        n = len(self.points)

        # Типы (str/int)
        raw_types = [p.get("type", "unknown") for p in points_list]
        self.point_types = []
        for t in raw_types:
            try:
                self.point_types.append(int(t))
            except (ValueError, TypeError):
                self.point_types.append(str(t))

        self.areas = np.array([float(p.get("area", 0.0)) for p in points_list], dtype=float)

        # Углы
        saved_angles = [p.get("angle") for p in points_list]
        if any(a is not None for a in saved_angles):
            self.angles = np.array([a if a is not None else np.nan for a in saved_angles], dtype=float)
        else:  # Если углов нет в JSON, инициализируем
            self.angles = np.full(n, np.nan)
            self.recalculate_angles(center)  # и пытаемся рассчитать

        # Интенсивности
        if any("intensity" in p for p in points_list):
            self.values = np.array([float(p.get("intensity", 0.0)) for p in points_list], dtype=float)
        else:
            self.values = sample_values_func(self.points)

        # ID
        self.initial_group_ids = {i: None for i in range(n)}

        self._validate_consistency()

    def to_json_list(self) -> List[Dict[str, Any]]:
        """Конвертирует данные модели в список для сохранения в JSON."""
        pts_list = []
        for i, (y, x) in enumerate(self.points):
            angle = float(self.angles[i]) if i < len(self.angles) and not np.isnan(self.angles[i]) else None

            point_data = {
                "y": float(y), "x": float(x),
                "intensity": float(self.values[i]) if i < len(self.values) else 0.0,
                "area": int(self.areas[i]) if i < len(self.areas) else 0,  # Сохраняем area как int
                "type": self.point_types[i] if i < len(self.point_types) else "unknown"
            }
            if angle is not None:
                point_data["angle"] = angle
            pts_list.append(point_data)
        return pts_list

    # --- Методы для View (Отрисовка) ---

    def _update_color_maps(self):
        """Обновляет карты цветов на основе текущих point_types."""
        numeric_group_ids = set()
        max_numeric_id = -1

        for pt_type in self.point_types:
            if isinstance(pt_type, int):
                numeric_group_ids.add(pt_type)
                if pt_type > max_numeric_id:
                    max_numeric_id = pt_type

        new_cmap_N = max(max_numeric_id + 1, 1)
        if new_cmap_N != self._cmap_N:  # Пересоздаем, только если изменилось
            self._cmap_N = new_cmap_N
            self._colormap = plt.get_cmap('viridis', self._cmap_N)
            self._numeric_color_map = {gid: self._colormap(gid / max(self._cmap_N - 1, 1))
                                       for gid in numeric_group_ids}
        elif numeric_group_ids != self._numeric_color_map.keys():
            # Если N то же, но набор ID другой (например, удалили группу)
            self._numeric_color_map = {gid: self._colormap(gid / max(self._cmap_N - 1, 1))
                                       for gid in numeric_group_ids}

    def get_colors_for_drawing(self) -> List:
        """Рассчитывает массив цветов для отрисовки."""
        if self.is_empty():
            return []

        self._update_color_maps()  # Обновляем карты при необходимости

        colors = []
        for pt_type in self.point_types:
            if isinstance(pt_type, str):
                colors.append(self._string_color_map.get(pt_type, "gray"))
            elif isinstance(pt_type, int):
                colors.append(self._numeric_color_map.get(pt_type, "gray"))
            else:
                colors.append("gray")
        return colors

    def get_point_data_for_tooltip(self, idx: int, center: Optional[Tuple[float, float]]) -> str:
        """Форматирует строку для Tooltip."""
        if idx < 0 or idx >= len(self.points):
            return "Error: Invalid Index"

        y, x = self.points[idx]

        intensity = float(self.values[idx])
        area = float(self.areas[idx])
        final_type_str = str(self.point_types[idx])
        type_label = "Type"
        if isinstance(self.point_types[idx], int):
            type_label = "Numeric ID"

        initial_group_id_str = str(self.initial_group_ids.get(idx, "N/A"))

        radius = None
        angle = float(self.angles[idx]) if not np.isnan(self.angles[idx]) else None

        if center:
            cy, cx = center
            radius = float(math.hypot(x - cx, y - cy))
            if angle is None:  # Если угол не был рассчитан, считаем на лету
                _, angle_arr = self._calculate_pol_from(center, np.array([[y, x]]))
                angle = float(angle_arr[0])

        txt_lines = []
        txt_lines.append(f"Radius: {radius:.1f} px" if radius is not None else "Radius: N/A (no center)")
        txt_lines.append(f"Angle: {angle:.1f}°" if angle is not None else "Angle: N/A")
        txt_lines.append(f"Intensity: {intensity:.1f} %")
        txt_lines.append(f"Area: {area:.1f} px²")
        txt_lines.append(f"{type_label}: {final_type_str}")
        txt_lines.append(f"Initial Group ID: {initial_group_id_str}")

        return "\n".join(txt_lines)

    def get_all_data_for_debug(self) -> List[Dict[str, Any]]:
        """Собирает все данные для сохранения в отладочный JSON."""
        data_to_save = []
        for i in range(len(self.points)):
            y, x = self.points[i]
            radius = None
            angle = float(self.angles[i]) if i < len(self.angles) and not np.isnan(self.angles[i]) else None

            if angle is not None and not self.is_empty():
                # Мы не можем пересчитать радиус, не зная центра,
                # который использовался для расчета угла.
                # Но для отладки, если угол есть, то и радиус должен быть.
                # (Логика _save_debug_data была небезопасной, она зависела от
                # текущего центра в overlay, который мог не соответствовать
                # сохраненным углам/типам.
                # Здесь мы просто сохраняем то, что есть в модели.)
                pass

            data_to_save.append({
                "index": i,
                "y": y,
                "x": x,
                "radius_px": None,  # Мы больше не храним радиус в модели
                "angle_deg": angle,
                "intensity_perc": float(self.values[i]),
                "area_px2": float(self.areas[i]),
                "final_type": self.point_types[i],
                "initial_group_id": self.initial_group_ids.get(i)
            })
        return data_to_save