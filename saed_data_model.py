#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Класс Модели Данных (Упрощенный)
--------------------------------
Инкапсулирует данные о точках (координаты, значения, площади, углы).
Классификация и группировка удалены.
"""
import numpy as np
import math
import matplotlib.pyplot as plt # Остается для _colormap, хотя он больше не используется
from typing import Optional, Dict, Any, List, Tuple, Union

class SaedDataModel:

    def __init__(self):
        """Инициализирует массивы данных."""
        self.points = np.zeros((0, 2), float)
        self.values = np.zeros((0,), float)  # Интенсивности в процентилях
        self.areas = np.zeros((0,), float)  # Площади пикселей
        self.angles = np.zeros((0,), float)  # Углы

        # --- Настройки для отрисовки (Упрощено) ---
        self._default_point_color = "cyan" # Все точки теперь одного цвета

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
                  area: float = 0.0):
        """
        Добавляет новую точку и связанные с ней данные.
        Типы и ID удалены.
        """
        self.points = np.vstack([self.points, [y, x]])
        self.values = np.append(self.values, value)
        self.areas = np.append(self.areas, area)

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
        # Типы и ID удалены

    def update_points(self, indices: List[int], new_points_yx: np.ndarray,
                      new_values: np.ndarray, center: Optional[Tuple[float, float]]):
        """
        Обновляет данные для подмножества точек (используется для "усреднения").
        Типы и ID удалены.
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
                r = np.hypot(self.points[:, 1] - 0, self.points[:, 0] - 0)
                return r

    def find_nearest_point_idx(self, y: float, x: float, pix_tol: float = 8.0) -> Optional[int]:
        """Находит индекс ближайшей точки в пределах допуска."""
        if self.is_empty(): return None
        dist_sq = (self.points[:, 0] - y) ** 2 + (self.points[:, 1] - x) ** 2
        i = int(np.argmin(dist_sq))
        return i if dist_sq[i] <= pix_tol ** 2 else None

    # --- Методы для Undo/Redo (Упрощено) ---

    # ++++++++++ НАЧАЛО ИЗМЕНЕНИЯ ++++++++++
    def get_snapshot(self) -> Dict[str, Any]:
        """Создает "слепок" всех данных для стека Undo."""
        return {
            "points": self.points.copy().tolist(), # .tolist() для JSON
            "values": self.values.copy().tolist(), # .tolist() для JSON
            "areas": self.areas.copy().tolist(),   # .tolist() для JSON
            "angles": self.angles.copy().tolist(),  # .tolist() для JSON
            # Типы и ID удалены
        }

    def apply_snapshot(self, snapshot: Dict[str, Any]):
        """Восстанавливает состояние модели из "слепка"."""
        # Конвертируем списки из JSON обратно в ndarray
        self.points = np.array(snapshot.get("points", []), dtype=float).reshape(-1, 2)
        self.values = np.array(snapshot.get("values", []), dtype=float)
        n = len(self.points)
        self.areas = np.array(snapshot.get("areas", []), dtype=float)
        self.angles = np.array(snapshot.get("angles", []), dtype=float)
        # Типы и ID удалены
        self._validate_consistency()
    # ++++++++++ КОНЕЦ ИЗМЕНЕНИЯ ++++++++++

    def _validate_consistency(self):
        """Проверяет, что все массивы имеют одинаковую длину."""
        n = len(self.points)
        if len(self.values) != n:
            # Исправляем несоответствие, если оно возникло при загрузке
            new_values = np.zeros((n,), float)
            if len(self.values) > 0:
                copy_len = min(n, len(self.values))
                new_values[:copy_len] = self.values[:copy_len]
            self.values = new_values
            print(f"Warning: Model inconsistency (values) corrected. Expected {n}, got {len(self.values)}.")
        if len(self.areas) != n:
            new_areas = np.zeros((n,), float)
            if len(self.areas) > 0:
                copy_len = min(n, len(self.areas))
                new_areas[:copy_len] = self.areas[:copy_len]
            self.areas = new_areas
            print(f"Warning: Model inconsistency (areas) corrected. Expected {n}, got {len(self.areas)}.")
        if len(self.angles) != n:
            new_angles = np.full((n,), np.nan)
            if len(self.angles) > 0:
                copy_len = min(n, len(self.angles))
                new_angles[:copy_len] = self.angles[:copy_len]
            self.angles = new_angles
            print(f"Warning: Model inconsistency (angles) corrected. Expected {n}, got {len(self.angles)}.")
        # Типы и ID удалены

    # --- Методы для IO (Упрощено) ---

    def from_json_list(self, points_list: List[Dict[str, Any]],
                       sample_values_func, center: Optional[Tuple[float, float]]):
        """Заполняет модель данными из JSON."""
        if not points_list:
            self.__init__()
            return

        yy = [float(p.get("y", 0.0)) for p in points_list]
        xx = [float(p.get("x", 0.0)) for p in points_list]
        self.points = np.column_stack([yy, xx]).astype(float)
        n = len(self.points)
        self.areas = np.array([float(p.get("area", 0.0)) for p in points_list], dtype=float)

        # Углы
        saved_angles = [p.get("angle") for p in points_list]
        if any(a is not None for a in saved_angles):
            self.angles = np.array([a if a is not None else np.nan for a in saved_angles], dtype=float)
        else:
            self.angles = np.full(n, np.nan)
            self.recalculate_angles(center)

        # Интенсивности
        if any("intensity" in p for p in points_list):
            self.values = np.array([float(p.get("intensity", 0.0)) for p in points_list], dtype=float)
        else:
            self.values = sample_values_func(self.points)

        # Типы и ID удалены
        self._validate_consistency()

    def to_json_list(self) -> List[Dict[str, Any]]:
        """Конвертирует данные модели в список для сохранения в JSON."""
        pts_list = []
        for i, (y, x) in enumerate(self.points):
            angle = float(self.angles[i]) if i < len(self.angles) and not np.isnan(self.angles[i]) else None
            point_data = {
                "y": float(y), "x": float(x),
                "intensity": float(self.values[i]) if i < len(self.values) else 0.0,
                "area": int(self.areas[i]) if i < len(self.areas) else 0,
                # Типы и ID удалены
            }
            if angle is not None:
                point_data["angle"] = angle
            pts_list.append(point_data)
        return pts_list

    # --- Методы для View (Упрощено) ---

    def get_colors_for_drawing(self) -> List[str]:
        """Возвращает массив одного цвета для всех точек."""
        return [self._default_point_color] * len(self.points)

    def get_point_data_for_tooltip(self, idx: int, center: Optional[Tuple[float, float]]) -> str:
        """Форматирует строку для Tooltip (без типа/ID)."""
        if idx < 0 or idx >= len(self.points):
            return "Error: Invalid Index"

        y, x = self.points[idx]
        intensity = float(self.values[idx])
        area = float(self.areas[idx])
        radius = None
        angle = float(self.angles[idx]) if i < len(self.angles) and not np.isnan(self.angles[idx]) else None

        if center:
            cy, cx = center
            radius = float(math.hypot(x - cx, y - cy))
            if angle is None and i < len(self.points): # Добавлена проверка i
                _, angle_arr = self._calculate_pol_from(center, np.array([[y, x]]))
                angle = float(angle_arr[0])

        txt_lines = []
        txt_lines.append(f"Radius: {radius:.1f} px" if radius is not None else "Radius: N/A")
        txt_lines.append(f"Angle: {angle:.1f}°" if angle is not None else "Angle: N/A")
        txt_lines.append(f"Intensity: {intensity:.1f} %")
        txt_lines.append(f"Area: {area:.1f} px²")
        # Типы и ID удалены

        return "\n".join(txt_lines)

    def get_all_data_for_debug(self) -> List[Dict[str, Any]]:
        """Собирает данные для отладки (без типа/ID)."""
        data_to_save = []
        for i in range(len(self.points)):
            y, x = self.points[i]
            angle = float(self.angles[i]) if i < len(self.angles) and not np.isnan(self.angles[i]) else None
            data_to_save.append({
                "index": i,
                "y": y,
                "x": x,
                "angle_deg": angle,
                "intensity_perc": float(self.values[i]),
                "area_px2": float(self.areas[i]),
                # Типы и ID удалены
            })
        return data_to_save