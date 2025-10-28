#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Класс IO (Упрощенный)
---------------------
Отвечает за загрузку/сохранение JSON, изображений, сессий.
Не обрабатывает типы/ID точек.
*** ИЗМЕНЕНО: Добавлена логика выравнивания по шаблону ***
"""
import json
from pathlib import Path
from tkinter import filedialog, messagebox
import numpy as np
from typing import Tuple, Union, Optional, Dict, Any, List

# <<< НОВОЕ: Добавляем необходимые импорты >>>
import cv2
import math

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None
    print("ПРЕДУПРЕЖДЕНИЕ: scipy не найден. Выравнивание по шаблону не будет работать.")
# <<< КОНЕЦ НОВОГО >>>


try:
    from preproc import PreprocSettings, load_grayscale_with_preproc
    from percentile_utils import compute_percentile_map, map_values_to_percent
except ImportError as e:
    print(f"Ошибка импорта зависимостей: {e}")
    messagebox.showerror("Import Error", f"Failed to import dependencies: {e}")


    class PreprocSettings:
        def __init__(self, mode="raw"): self.mode = mode

        @staticmethod
        def from_json(data, fallback_mode=None): return PreprocSettings(fallback_mode or "raw")

        def to_json(self): return {"mode": self.mode}


    def load_grayscale_with_preproc(path, settings):
        raise ImportError("preproc.py not found")


    def compute_percentile_map(img):
        raise ImportError("percentile_utils.py not found")


# <<< НОВАЯ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ (вне класса) >>>
def _get_peaks_from_template_image(image_path: Path) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]]]:
    """Загружает шаблон, находит пики и возвращает (точки_yx, центр_yx)."""
    try:
        # Загружаем как Ч/Б
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Не удалось загрузить изображение шаблона: {image_path}")

        # Простое пороговое разделение для чистого бинарного изображения
        # (Шаблон - белые точки на черном фоне)
        _thresh, binary_img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

        # Ищем связанные компоненты
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

        if num_labels <= 1:
            return None, None  # Точки не найдены

        # Первый центроид (индекс 0) - это фон
        points_yx = centroids[1:, ::-1]  # Получаем все центроиды (y, x)

        # Находим центр шаблона (центр масс найденных точек)
        if len(points_yx) > 0:
            center_yx = tuple(np.mean(points_yx, axis=0))
        else:
            return None, None

        return points_yx, center_yx

    except Exception as e:
        messagebox.showerror("Ошибка шаблона", f"Не удалось обработать изображение шаблона:\n{e}")
        return None, None


# <<< НОВАЯ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ (вне класса) >>>
# <<< НОВАЯ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ (вне класса) >>>
def _find_transform_s_r(template_pts_centered_polar: np.ndarray,
                        experimental_pts_centered_polar: np.ndarray,
                        num_bins: int = 360) -> Tuple[float, float]:
    """
    Находит Масштаб (S) и Поворот (R) используя гистограмму.
    Входные массивы: [N, 2] с колонками (radius, angle_degrees).
    """

    # 1. Поиск Масштаба (S)
    # Используем медиану радиусов N ближайших к центру точек
    N_FOR_SCALE = min(20, len(template_pts_centered_polar), len(experimental_pts_centered_polar))
    if N_FOR_SCALE < 2: return 1.0, 0.0 # Недостаточно точек

    # [:, 0] это радиусы
    template_radii = np.sort(template_pts_centered_polar[:, 0])
    exp_radii = np.sort(experimental_pts_centered_polar[:, 0])

    # Пропускаем индекс 0 (может быть центральная точка с r=0)
    # И берем только те, что > 0
    valid_template_radii = template_radii[template_radii > 1e-6]
    valid_exp_radii = exp_radii[exp_radii > 1e-6]

    # Проверяем, достаточно ли точек после фильтрации
    if len(valid_template_radii) < N_FOR_SCALE or len(valid_exp_radii) < N_FOR_SCALE:
         # Если точек мало, используем все что есть (но не менее 1)
         N_FOR_SCALE = max(1, min(len(valid_template_radii), len(valid_exp_radii)))
         if N_FOR_SCALE == 0: return 1.0, 0.0 # Совсем нет точек > 0

    median_template_r = np.median(valid_template_radii[:N_FOR_SCALE])
    median_exp_r = np.median(valid_exp_radii[:N_FOR_SCALE])


    if median_template_r < 1e-6 or median_exp_r < 1e-6 or not np.isfinite(median_template_r) or not np.isfinite(median_exp_r):
        scale = 1.0
    else:
        scale = median_exp_r / median_template_r

    # 2. Поиск Поворота (R)
    # Используем гистограмму угловых разниц

    # Отмасштабированные точки шаблона (радиус, угол)
    scaled_template_polar = template_pts_centered_polar.copy()
    scaled_template_polar[:, 0] *= scale

    # Ограничим поиск только точками в общей "центральной" области
    # Возьмем 50-ю точку из эксперимента как предел радиуса
    if len(exp_radii) == 0: return scale, 0.0

    valid_exp_radii = exp_radii[exp_radii > 1e-6]
    if len(valid_exp_radii) == 0: return scale, 0.0

    max_r_idx = min(len(valid_exp_radii)-1, 50)
    max_r_from_exp = valid_exp_radii[max_r_idx]


    template_set = scaled_template_polar[(scaled_template_polar[:, 0] < (max_r_from_exp * 1.5)) & (scaled_template_polar[:, 0] > 1e-6)]
    exp_set = experimental_pts_centered_polar[(experimental_pts_centered_polar[:, 0] < (max_r_from_exp * 1.5)) & (experimental_pts_centered_polar[:, 0] > 1e-6)]

    # <<< ИСПРАВЛЕНИЕ: Получаем размер exp_set ДО цикла >>>
    exp_set_size = len(exp_set)

    if len(template_set) == 0 or exp_set_size == 0:
        return scale, 0.0

    histogram = np.zeros((num_bins,))
    bin_width = 360.0 / num_bins

    # Допуск по радиусу для сопоставления
    radius_tolerance = max(4.0, 0.05 * median_exp_r) # 4 пикселя или 5%

    if cKDTree is None: raise RuntimeError("Scipy (cKDTree) не найден.")

    # Строим k-d tree из экспериментальных точек (r, a)
    tree_exp = cKDTree(exp_set)
    tree_template = cKDTree(template_set) # <<< Строим дерево и для шаблона

    # Ищем пары для каждой точки шаблона
    # query_ball_tree находит все пары в радиусе (Евклидово)
    pairs = tree_exp.query_ball_tree(tree_template, r=radius_tolerance) # <<< Используем оба дерева

    for i, exp_indices in enumerate(pairs): # i = индекс точки шаблона
        if not exp_indices:
            continue

        # Проверяем валидность i для template_set (на всякий случай)
        if i < 0 or i >= len(template_set):
            # print(f"!!! WARNING: Invalid template index {i} from query_ball_tree pairs!")
            continue

        t_r, t_a = template_set[i]

        for j in exp_indices: # j = индекс точки эксперимента

            # <<< --- ВОТ ИСПРАВЛЕНИЕ --- >>>
            # Явно проверяем, что индекс j валиден для exp_set
            if j < 0 or j >= exp_set_size:
                # print(f"!!! WARNING: Invalid index {j} returned by query_ball_tree for exp_set size {exp_set_size}. Skipping.")
                continue # Пропускаем этот невалидный индекс
            # <<< --- КОНЕЦ ИСПРАВЛЕНИЯ --- >>>

            e_r, e_a = exp_set[j] # Теперь доступ должен быть безопасным

            # (Допуск по радиусу уже проверен k-d tree)
            angle_diff = (e_a - t_a + 360) % 360
            bin_index = int(angle_diff / bin_width)
            if 0 <= bin_index < num_bins:
                histogram[bin_index] += 1 # (radius_tolerance - abs(e_r - t_r)) # Взвешенный голос

    if np.max(histogram) == 0:
        return scale, 0.0 # Нет совпадений

    # Находим бин с макс. числом голосов
    best_bin = np.argmax(histogram)
    rotation_deg = (best_bin + 0.5) * bin_width # Центр бина

    return scale, rotation_deg


class EditorIO:

    def __init__(self, controller):
        self.controller = controller

    # ---------- Обертки для UI ----------
    def open_json_wrapper(self):
        p = filedialog.askopenfilename(
            title="Open SAED Input",
            filetypes=[("SAED Input JSON", "*saed_input.json;*.json"), ("All", "*.*")]
        )
        if p:
            try:
                self.load_input_json(Path(p), push_undo=True)
            except FileNotFoundError:
                messagebox.showerror("Error", f"File not found: {p}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load JSON:\n{e}")

    def save_points_wrapper(self):
        try:
            self._save_points()
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save points:\n{e}")

    # <<< НОВЫЙ МЕТОД-ОБЕРТКА >>>
    def fill_from_template_wrapper(self):
        """Обертка для кнопки 'Fill from Template'."""
        if self.controller.model.is_empty() and self.controller.get_center() is None:
            messagebox.showerror("Ошибка", "Сначала запустите обнаружение или загрузите точки.")
            return

        if cKDTree is None:
            messagebox.showerror("Ошибка зависимости", "SciPy (cKDTree) не найден. Выравнивание по шаблону недоступно.")
            return

        p = filedialog.askopenfilename(
            title="Выберите эталонное изображение (шаблон)",
            filetypes=[("Изображения", "*.png;*.jpg;*.jpeg;*.tif;*.tiff"), ("Все", "*.*")]
        )
        if not p:
            return

        template_path = Path(p)

        try:
            self.run_template_matching(template_path)
        except Exception as e:
            messagebox.showerror("Ошибка выравнивания", f"Произошла ошибка:\n{e}")
            import traceback
            traceback.print_exc()  # Для отладки

    # <<< НОВЫЙ ОСНОВНОЙ МЕТОД >>>
    def run_template_matching(self, template_path: Path):
        """Основная логика выравнивания по шаблону."""

        # 1. Получаем экспериментальные точки и центр
        if self.controller.model.is_empty():
            messagebox.showerror("Ошибка", "Нет экспериментальных точек для выравнивания.")
            return

        experimental_points_yx = self.controller.model.points.copy()
        experimental_center_yx = self.controller.get_center()

        if experimental_center_yx is None:
            messagebox.showerror("Ошибка", "Не установлен центр экспериментальных точек.")
            return

        exp_cy, exp_cx = experimental_center_yx

        # 2. Получаем точки шаблона и центр
        template_points_yx, template_center_yx = _get_peaks_from_template_image(template_path)

        if template_points_yx is None or template_center_yx is None:
            messagebox.showerror("Ошибка", "Не удалось найти пики на изображении шаблона.")
            return

        temp_cy, temp_cx = template_center_yx

        # 3. Конвертируем в центрированные полярные координаты

        # --- Эксперимент ---
        exp_centered_yx = experimental_points_yx - experimental_center_yx
        exp_r = np.hypot(exp_centered_yx[:, 1], exp_centered_yx[:, 0])
        exp_a = (np.degrees(np.arctan2(exp_centered_yx[:, 0], exp_centered_yx[:, 1])) + 360) % 360
        exp_polar = np.column_stack((exp_r, exp_a))

        # --- Шаблон ---
        temp_centered_yx = template_points_yx - template_center_yx
        temp_r = np.hypot(temp_centered_yx[:, 1], temp_centered_yx[:, 0])
        temp_a = (np.degrees(np.arctan2(temp_centered_yx[:, 0], temp_centered_yx[:, 1])) + 360) % 360
        temp_polar = np.column_stack((temp_r, temp_a))

        # 4. Находим Масштаб (S) и Поворот (R)
        self.controller.set_status("Выравнивание шаблона... (может занять время)")
        self.controller.update()  # Обновляем UI

        scale, rotation_deg = _find_transform_s_r(temp_polar, exp_polar)

        self.controller.set_status(f"Выравнивание найдено: Масштаб={scale:.3f}, Поворот={rotation_deg:.2f}°")

        if scale < 0.1 or scale > 10:
            messagebox.showwarning("Предупреждение",
                                   f"Необычный фактор масштабирования ({scale:.3f}). Результат может быть неточным.")

        # 5. Трансформируем *все* точки шаблона

        temp_r_all = temp_polar[:, 0]
        temp_a_all_rad = np.deg2rad(temp_polar[:, 1])

        # Применяем S и R
        scaled_r = temp_r_all * scale
        rotated_a_rad = temp_a_all_rad + np.deg2rad(rotation_deg)

        # Обратно в декартовы (центрированные)
        transformed_centered_x = scaled_r * np.cos(rotated_a_rad)
        transformed_centered_y = scaled_r * np.sin(rotated_a_rad)

        # Добавляем сдвиг (центр эксперимента)
        transformed_final_x = transformed_centered_x + exp_cx
        transformed_final_y = transformed_centered_y + exp_cy

        transformed_template_points_yx = np.column_stack((transformed_final_y, transformed_final_x))

        # 6. Находим "пропущенные" точки

        # Порог близости. 5 пикселей - разумное значение.
        proximity_threshold = 5.0

        exp_tree = cKDTree(experimental_points_yx)

        # Находим дистанцию от каждой точки шаблона до ближайшей точки эксперимента
        distances, _indices = exp_tree.query(transformed_template_points_yx)

        # Находим точки, где дистанция *больше* порога
        missed_points_mask = (distances > proximity_threshold)
        missed_points_yx = transformed_template_points_yx[missed_points_mask]

        num_missed = len(missed_points_yx)
        if num_missed == 0:
            messagebox.showinfo("Выравнивание", "Новые точки не найдены. Все точки шаблона соответствуют существующим.")
            return

        # 7. Добавляем пропущенные точки в модель

        # Получаем их интенсивность из *экспериментального* изображения
        sampled_values = self.controller.sample_intensities(missed_points_yx)

        self.controller.push_undo()

        center_for_model = self.controller.get_center()
        num_added = 0

        # Порог интенсивности (чтобы не добавлять точки в местах, где на снимке пусто)
        # Возьмем 10% процентиль
        intensity_threshold = 10.0

        for i in range(num_missed):
            y, x = missed_points_yx[i]
            value = sampled_values[i]

            # Не добавляем точки, если на экспериментальном снимке там слишком темно
            if value < intensity_threshold:
                continue

            # Используем новый 'source'
            self.controller.model.add_point(
                y=y,
                x=x,
                value=value,
                center=center_for_model,
                area=1.0,  # У точек из шаблона площадь 1
                source="template"  # <<< НОВАЯ МЕТКА
            )
            num_added += 1

        self.controller.redo.clear()
        self.controller.redraw()
        msg = f"Добавлено {num_added} новых точек (из {num_missed} кандидатов) по шаблону."
        if num_added < num_missed:
            msg += f" {num_missed - num_added} отброшено из-за низкой интенсивности."
        self.controller.set_status(msg)

    # ---------- Основная Логика IO ----------

    def load_input_json(self, path: Path, *, push_undo: bool = False, reset_view: bool = True):
        """Загружает JSON (без типов/ID)."""
        if not path.exists(): raise FileNotFoundError(f"Input JSON file not found: {path}")
        if push_undo: self.controller.push_undo()

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {path.name}: {e}")
        except Exception as e:
            raise IOError(f"Failed to read JSON file {path.name}: {e}")

        self.controller.img_arr_raw = None;
        self.controller.img_arr_processed = None
        self.controller._percent_map = None;
        self.controller._percent_lookup = None

        img_path_str = data.get("image")
        if not img_path_str: raise ValueError("JSON missing 'image' field.")
        img_p = Path(img_path_str)
        img_path = (path.parent / img_p).resolve() if not img_p.is_absolute() else img_p.resolve()
        if not img_path.exists(): raise FileNotFoundError(f"Image not found: {img_path}")
        self.controller.image_path = img_path
        fallback_mode = data.get("preproc_mode", "raw")
        self.controller._preproc_settings = PreprocSettings.from_json(data.get("preproc"), fallback_mode=fallback_mode)

        try:
            self.controller.img_arr_raw = load_grayscale_with_preproc(img_path, PreprocSettings(mode="raw"))
            self.controller.img_arr_processed = load_grayscale_with_preproc(img_path, self.controller._preproc_settings)
            p_map, uniq_vals, uniq_perc = compute_percentile_map(self.controller.img_arr_processed)
            self.controller._percent_map = p_map
            self.controller._percent_lookup = (uniq_vals, uniq_perc)
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load/process image:\n{e}")
            if self.controller.img_arr_raw is None: self.controller.img_arr_raw = np.zeros((100, 100), dtype=np.uint8)
            if self.controller.img_arr_processed is None: self.controller.img_arr_processed = self.controller.img_arr_raw.copy()

        img_h, img_w = self.controller.img_arr_processed.shape[:2]
        c = data.get("center") or {};
        r = data.get("radii") or {}
        default_cx = (img_w - 1) / 2.0;
        default_cy = (img_h - 1) / 2.0
        center_data = {"x": float(c.get("x", default_cx)), "y": float(c.get("y", default_cy))}
        self.controller.overlay = {
            "center": center_data,
            "dead_radius": float(r.get("dead", 0.0)),
            "search_radius": float(r.get("search", 0.0))
        }
        center_yx = (center_data["y"], center_data["x"])

        try:
            pts_data = data.get("points", [])
            # Модель загрузит точки без типов/ID, но С ИСТОЧНИКОМ (source)
            self.controller.model.from_json_list(pts_data, self.controller.sample_intensities, center_yx)
        except (ValueError, TypeError) as e:
            messagebox.showerror("Data Error", f"Invalid point data: {e}")
            self.controller.model.from_json_list([], self.controller.sample_intensities, center_yx)

        self.controller.ui_state.clear_tooltip()
        if reset_view: self.controller.view_cx = None; self.controller.view_cy = None
        self.controller.ensure_view_center()
        self.controller.undo.clear();
        self.controller.redo.clear()
        # УДАЛЕНО: _calculate_initial_symmetry()
        # УДАЛЕНО: _update_group_panel()
        self.controller.redraw()
        self.controller.update_zoom_hint()
        self.controller.set_status(f"Loaded: {path.name}")

    def _save_points(self) -> Path:
        """Сохраняет точки (без типов/ID) и обновляет 'saed_input.edited.json'."""
        output_dir = self.controller.get_output_dir()
        # Модель вернет список (включая 'source')
        pts_list = self.controller.model.to_json_list()

        def default_serializer(obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        spots_path = output_dir / "spots.json"
        spots_path.write_text(json.dumps({"points": pts_list}, indent=2, default=default_serializer), encoding="utf-8")

        abs_image_path = self.controller.image_path.resolve() if self.controller.image_path else None
        saed_input_edited_data = {
            "image": str(abs_image_path) if abs_image_path else None,
            "preproc_mode": self.controller._preproc_settings.mode,
            "preproc": self.controller._preproc_settings.to_json(),
            "center": self.controller.overlay.get("center"),
            "radii": {
                "dead": self.controller.overlay.get("dead_radius", 0.0),
                "search": self.controller.overlay.get("search_radius", 0.0)
            },
            "points": pts_list  # Уже включает 'source'
        }
        edited_path = output_dir / "saed_input.edited.json"
        edited_path.write_text(
            json.dumps(saed_input_edited_data, ensure_ascii=False, indent=2, default=default_serializer),
            encoding="utf-8")
        self.controller.set_status(f"Points saved to {output_dir.name}")
        return spots_path

    # --- Session Save/Load (Упрощено) ---

    def get_state(self) -> dict:
        """Возвращает состояние редактора (включая 'source' в снэпшоте)."""
        return {
            "image_path": str(self.controller.image_path.resolve()) if self.controller.image_path else None,
            "preproc_settings": self.controller._preproc_settings.to_json(),
            "overlay": self.controller.overlay,
            "zoom_val": self.controller.zoom_val,
            "view_cx": self.controller.view_cx,
            "view_cy": self.controller.view_cy,
            "show_raw_background": self.controller.show_raw_background.get(),
            # Снэпшот модели будет включать 'source'
            "data_snapshot": self.controller.model.get_snapshot(),
            "measurement": self.controller.ui_state.measurement,
            "ring_select_indices": list(self.controller.ui_state.ring_select_indices),
        }

    # --- Сохранение отладки (Упрощено) ---

    def save_debug_data(self, filename: str = "points_debug.json"):  # Изменено имя по умолчанию
        """Сохраняет отладочные данные из Модели (включая 'source')."""
        output_dir = self.controller.get_output_dir()
        if self.controller.model.is_empty():
            messagebox.showwarning("Save Debug", "No points to save.")
            return

        # Модель вернет данные (включая 'source')
        data_to_save = self.controller.model.get_all_data_for_debug()

        filepath = output_dir / filename
        try:
            def default_serializer(obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

            filepath.write_text(json.dumps(data_to_save, indent=2, default=default_serializer), encoding="utf-8")
            current_status = self.controller.status_message
            self.controller.set_status(f"{current_status} Debug data saved to {filename}")
            print(f"Debug data saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save debug data:\n{e}")
            self.controller.set_status(f"Failed to save {filename}")

    # --- Запуск Анализа (Без изменений, т.к. он просто передает данные) ---

    def start_analysis_wrapper(self):
        """Обертка для кнопки 'Start Analysis'."""
        try:
            saved_spots_path = self._save_points()
            output_dir = saved_spots_path.parent
            payload_path = output_dir / "fibo_input.json"
        except (ValueError, OSError, Exception) as e:
            messagebox.showerror("Save Error", f"Cannot proceed. Failed to save:\n{e}")
            return

        try:
            abs_image_path = self.controller.image_path.resolve() if self.controller.image_path else None
            img_to_use = self.controller.get_image_to_display()
            H, W = img_to_use.shape[:2] if img_to_use is not None else (1, 1)
            geo_center_data = {"x": (W - 1) / 2.0, "y": (H - 1) / 2.0}
            # Модель вернет точки (включая 'source')
            pts_list = self.controller.model.to_json_list()

            payload = {
                "image": str(abs_image_path) if abs_image_path else None,
                "preproc_mode": self.controller._preproc_settings.mode,
                "preproc": self.controller._preproc_settings.to_json(),
                "points": pts_list,
                "centers": {"geometric": geo_center_data, "overlay": self.controller.overlay.get("center")},
                "radii": {"dead": self.controller.overlay.get("dead_radius", 0.0),
                          "search": self.controller.overlay.get("search_radius", 0.0)},
                "spots_json": str(saved_spots_path.resolve())
            }

            def default_serializer(obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

            payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=default_serializer),
                                    encoding="utf-8")
            self.controller.set_status("Prepared data for analysis…")
        except Exception as e:
            messagebox.showerror("Data Prep Error", f"Failed to create ({payload_path.name}):\n{e}")
            return

        if self.controller.app_controller is not None:
            try:
                self.controller.app_controller.open_analysis(payload_path.resolve(), abs_image_path,
                                                             saved_spots_path.resolve())
            except Exception as e:
                print(f"Error calling controller.open_analysis: {e}")
                messagebox.showerror("Launch Error", f"Failed to switch to analysis tab.\nDetails: {e}")
        else:
            messagebox.showwarning("Standalone Mode", "Cannot switch to analysis tab.")