#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Класс Модели Данных (Упрощенный)
--------------------------------
Инкапсулирует данные о точках (координаты, значения, площади, углы).
*** ИЗМЕНЕНО: Добавлен 'source' для отслеживания происхождения точки ***
"""
import numpy as np
import math
import matplotlib.pyplot as plt  # Остается для _colormap, хотя он больше не используется
from typing import Optional, Dict, Any, List, Tuple, Union


class SaedDataModel:

    def __init__(self):
        """Инициализирует массивы данных."""
        self.points = np.zeros((0, 2), float)
        self.values = np.zeros((0,), float)  # Интенсивности в процентилях
        self.areas = np.zeros((0,), float)  # Площади пикселей
        self.angles = np.zeros((0,), float)  # Углы
        self.sources = np.zeros((0,), str)  # <<< НОВОЕ: Источник точки (manual, detected, template)

        # --- Настройки для отрисовки (Упрощено) ---
        self._default_point_color = "cyan"  # Цвет по умолчанию

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
                  area: float = 0.0, source: str = "manual"):  # <<< ИЗМЕНЕНО: Добавлен 'source'
        """
        Добавляет новую точку и связанные с ней данные.
        Типы и ID удалены.
        """
        self.points = np.vstack([self.points, [y, x]])
        self.values = np.append(self.values, value)
        self.areas = np.append(self.areas, area)
        self.sources = np.append(self.sources, source)  # <<< НОВОЕ: Сохраняем 'source'

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
        self.sources = self.sources[mask]  # <<< НОВОЕ: Фильтруем 'source'
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
        # Источник (source) при усреднении не меняем. Он остается тем же.

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
                # Фоллбэк, если нет центра (хотя это не должно происходить)
                r = np.hypot(self.points[:, 1] - 0, self.points[:, 0] - 0)
                return r

    def find_nearest_point_idx(self, y: float, x: float, pix_tol: float = 8.0) -> Optional[int]:
        """Находит индекс ближайшей точки в пределах допуска."""
        if self.is_empty(): return None
        dist_sq = (self.points[:, 0] - y) ** 2 + (self.points[:, 1] - x) ** 2
        i = int(np.argmin(dist_sq))
        return i if dist_sq[i] <= pix_tol ** 2 else None

    # --- Методы для Undo/Redo (Упрощено) ---

    def get_snapshot(self) -> Dict[str, Any]:
        """Создает "слепок" всех данных для стека Undo."""
        return {
            "points": self.points.copy(),
            "values": self.values.copy(),
            "areas": self.areas.copy(),
            "angles": self.angles.copy(),
            "sources": self.sources.copy(),  # <<< НОВОЕ: Добавляем 'sources'
            # Типы и ID удалены
        }

    def apply_snapshot(self, snapshot: Dict[str, Any]):
        """Восстанавливает состояние модели из "слепка"."""
        self.points = snapshot.get("points", np.zeros((0, 2))).copy()
        self.values = snapshot.get("values", np.zeros((0,))).copy()
        n = len(self.points)
        self.areas = snapshot.get("areas", np.zeros((n,))).copy()
        self.angles = snapshot.get("angles", np.full(n, np.nan)).copy()
        # <<< НОВОЕ: Восстанавливаем 'sources', с фоллбэком 'manual' для старых снэпшотов
        self.sources = snapshot.get("sources", np.full(n, "manual")).copy()
        # Типы и ID удалены
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
        if len(self.sources) != n:  # <<< НОВОЕ: Проверка 'sources'
            self.sources = np.full((n,), "manual")
            print("Warning: Model inconsistency (sources) corrected.")
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

        # <<< НОВОЕ: Загружаем 'source', по умолчанию 'detected', т.к. JSON обычно из temn.py
        self.sources = np.array([str(p.get("source", "detected")) for p in points_list], dtype=str)

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
                "source": str(self.sources[i]) if i < len(self.sources) else "manual",  # <<< НОВОЕ: Сохраняем 'source'
                # Типы и ID удалены
            }
            if angle is not None:
                point_data["angle"] = angle
            pts_list.append(point_data)
        return pts_list

    # --- Методы для View (Упрощено) ---

    def get_colors_for_drawing(self) -> List[str]:
        """
        *** ИЗМЕНЕНО: Возвращает цвета на основе 'source' ***
        """
        color_map = {
            "manual": "cyan",  # Добавлено вручную
            "detected": "lime",  # Найдено алгоритмом
            "template": "magenta",  # Добавлено из шаблона
            "unknown": "gray",
        }

        if len(self.sources) != len(self.points):
            # Фоллбэк, если что-то пошло не так
            return [self._default_point_color] * len(self.points)

        return [color_map.get(str(src), self._default_point_color) for src in self.sources]

    def get_point_data_for_tooltip(self, idx: int, center: Optional[Tuple[float, float]]) -> str:
        """Форматирует строку для Tooltip (без типа/ID)."""
        if idx < 0 or idx >= len(self.points):
            return "Error: Invalid Index"

        y, x = self.points[idx]
        intensity = float(self.values[idx])
        area = float(self.areas[idx])
        source = str(self.sources[idx]) if idx < len(self.sources) else "N/A"  # <<< НОВОЕ: Получаем 'source'
        radius = None
        angle = float(self.angles[idx]) if (idx < len(self.angles) and not np.isnan(self.angles[idx])) else None

        if center:
            cy, cx = center
            radius = float(math.hypot(x - cx, y - cy))
            if angle is None:
                _, angle_arr = self._calculate_pol_from(center, np.array([[y, x]]))
                angle = float(angle_arr[0])

        txt_lines = []
        txt_lines.append(f"Radius: {radius:.1f} px" if radius is not None else "Radius: N/A")
        txt_lines.append(f"Angle: {angle:.1f}°" if angle is not None else "Angle: N/A")
        txt_lines.append(f"Intensity: {intensity:.1f} %")
        txt_lines.append(f"Area: {area:.1f} px²")
        txt_lines.append(f"Source: {source}")  # <<< НОВОЕ: Добавляем 'source'
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
                "source": str(self.sources[i]) if i < len(self.sources) else "manual",  # <<< НОВОЕ: Добавляем 'source'
                # Типы и ID удалены
            })
        return data_to_save