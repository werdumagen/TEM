#!/usr/bin/env python3
# -*- coding: utf-8 -*-
(
    "SAED Symmetry – Launcher\n"
    "========================\n"
    "Launcher for SAED point detection.\n"
    "- Added 'Centroid (Multi-Pass)' detection method (v4.7).\n"
    "- Added Hybrid detection method (Legacy -> Centroid -> Filter Blobs -> Final Proximity Filter).\n"
    "- Added separate percentile thresholds for Centroid and Legacy methods.\n"
    "- Removed automatic symmetry analysis and classification from this stage.\n"
    # --- НОВОЕ: v4.8 ---
    "- Added 'Legacy + Dual Centroid' detection method (L -> C1 -> Mask(L+C1) -> C2(Masked))"
)
from __future__ import annotations
import json, subprocess, sys, cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional
# --- Добавлены импорты ---
import math

# --- Импорт для skimage.feature ---
try:
    from skimage.feature import peak_local_max
except ImportError:
    peak_local_max = None
    print("ПРЕДУПРЕЖДЕНИЕ: scikit-image не найден или устарел. Метод 'Legacy' и 'Hybrid' не будут работать.")
# --- КОНЕЦ ---

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None
    print("ПРЕДУПРЕЖДЕНИЕ: scipy не найден. Фильтрация по близости (proximity) не будет работать.")

from percentile_utils import compute_percentile_map, map_values_to_percent
from preproc import PreprocSettings, load_grayscale_with_preproc
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ------ Функции анализа УДАЛЕНЫ ------


# -------------------------- Algorithm --------------------------
@dataclass
class CenterResult:
    cy: float
    cx: float
    method: str


# --- МЕТОД 1: Centroid ---
# Возвращает точки, labels_map И список ID сохраненных блобов
def detect_spots_by_centroid(
        arr: np.ndarray,
        user_perc: float = 99.0,
        min_area: int = 3,
        max_spots: int = 6000,
        proximity_threshold: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Detects spots using centroiding.
    Returns: (array of (y, x, intensity_perc, area), labels_map, kept_label_indices).
    """
    if cKDTree is None: raise RuntimeError("Пакет 'scipy' не найден.")

    H, W = arr.shape
    percent_map, uniq_vals, uniq_perc = compute_percentile_map(arr)
    master_blob_map = np.zeros((H, W), dtype=np.uint8)
    full_labels_map = np.zeros((H, W), dtype=np.int32)
    steps = sorted(list(set(list(range(100, int(user_perc), -5)) + [int(user_perc)])), reverse=True)
    if user_perc not in steps: steps.append(user_perc); steps.sort(reverse=True)

    kept_points_with_labels: List[Tuple[float, float, float, int, int]] = []
    kept_coords_list: List[List[float]] = []
    kept_coords_tree: Optional[cKDTree] = None
    prox_threshold_sq = proximity_threshold ** 2
    current_max_label = 0

    for th_value in steps:
        current_binary_mask = np.where(percent_map >= th_value, 255, 0).astype(np.uint8)
        mask_for_ccs = cv2.bitwise_and(current_binary_mask, current_binary_mask, mask=cv2.bitwise_not(master_blob_map))
        num_labels, labels_map_step, stats, centroids = cv2.connectedComponentsWithStats(mask_for_ccs, connectivity=8)

        if num_labels <= 1: continue

        valid_step_mask = (labels_map_step > 0)
        labels_map_step[valid_step_mask] += current_max_label
        full_labels_map[valid_step_mask] = labels_map_step[valid_step_mask]

        current_batch_points: List[Tuple[float, float, float, int, int]] = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area: continue
            cx, cy = centroids[i];
            yi, xi = int(round(cy)), int(round(cx))
            if 0 <= yi < H and 0 <= xi < W:
                # --- ИЗМЕНЕНИЕ: Убран master_blob_map[yi, xi] == 0 ---
                # Мы проверяем все точки, маскирование будет позже
                v = float(percent_map[yi, xi])
                current_batch_points.append((cy, cx, v, area, i + current_max_label))
                # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        current_max_label += (num_labels - 1)
        current_batch_points.sort(key=lambda t: -t[2])
        newly_added_coords_this_batch: List[List[float]] = []

        # --- ИЗМЕНЕНИЕ: Новая логика цикла ---
        for (y, x, v, area, label_index) in current_batch_points:
            point_coord = [y, x];

            # (1) Маскируем блоб СРАЗУ
            mask_to_add_single = (full_labels_map == label_index)
            # Проверяем, не был ли этот блоб уже полностью замаскирован
            # (хотя этого не должно случиться, если mask_for_ccs работает)
            if master_blob_map[mask_to_add_single].all():
                 continue # Блоб уже полностью в маске, пропускаем
            master_blob_map[mask_to_add_single] = 1  # Маскируем пиксели

            # (2) Проверяем близость
            min_dist_sq = float('inf')
            if kept_coords_tree is not None:
                dist, _ = kept_coords_tree.query(point_coord, k=1)
                min_dist_sq = min(min_dist_sq, dist ** 2)
            if newly_added_coords_this_batch:
                dists_sq_batch = np.sum((np.array(newly_added_coords_this_batch) - point_coord) ** 2, axis=1)
                if dists_sq_batch.size > 0: min_dist_sq = min(min_dist_sq, np.min(dists_sq_batch))

            # (3) Если слишком близко, пропускаем только добавление ТОЧКИ
            if min_dist_sq < prox_threshold_sq:
                continue # Блоб уже замаскирован, но точка не добавляется

            # (4) Если точка не отброшена, добавляем ее в список
            kept_points_with_labels.append((y, x, v, area, label_index))
            newly_added_coords_this_batch.append([y, x])
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        if newly_added_coords_this_batch:
            kept_coords_list.extend(newly_added_coords_this_batch)
            kept_coords_tree = cKDTree(kept_coords_list)

    kept_points_with_labels.sort(key=lambda t: -t[2])
    if len(kept_points_with_labels) > max_spots:
        kept_points_with_labels = kept_points_with_labels[:max_spots]

    if kept_points_with_labels:
        result_array = np.array([(p[0], p[1], p[2], p[3]) for p in kept_points_with_labels], dtype=float)
        kept_label_indices = [p[4] for p in kept_points_with_labels]
    else:
        result_array = np.zeros((0, 4), dtype=float)
        kept_label_indices = []

    return result_array, full_labels_map, kept_label_indices


# --- МЕТОД 2: Legacy ---
def detect_spots_legacy(
        arr: np.ndarray,
        user_perc: float = 99.0,
        max_spots: int = 6000,
        min_distance: int = 4,
        mask: Optional[np.ndarray] = None  # Маска ОБЛАСТЕЙ для ИСКЛЮЧЕНИЯ из поиска
) -> np.ndarray:
    """
    Detects spots using local maxima finding, optionally excluding masked areas.
    Returns array of (y, x, intensity_percentile, area=1.0).
    """
    if peak_local_max is None: raise RuntimeError("Пакет 'scikit-image' не найден.")

    H, W = arr.shape
    percent_map, uniq_vals, uniq_perc = compute_percentile_map(arr)
    threshold_abs = np.interp(user_perc, uniq_perc, uniq_vals)

    footprint_mask = None
    if mask is not None:
        if mask.shape != arr.shape:
            print("Warning: Mask shape mismatch in detect_spots_legacy. Ignoring mask.")
        else:
            footprint_mask = (mask == 0)  # Искать только там, где маска = 0

    safe_min_distance = max(1, int(min_distance))
    coordinates = peak_local_max(
        arr, min_distance=safe_min_distance, threshold_abs=threshold_abs,
        exclude_border=False, labels=(footprint_mask.astype(np.int32) if footprint_mask is not None else None),
        num_peaks=max_spots  # Ограничиваем здесь
    )

    if coordinates.shape[0] == 0: return np.zeros((0, 4), dtype=float)

    peaks_y = coordinates[:, 0];
    peaks_x = coordinates[:, 1]
    raw_intensities = arr[peaks_y, peaks_x]  # Интенсивность берем из ОРИГИНАЛА
    percentile_intensities = map_values_to_percent(raw_intensities, uniq_vals, uniq_perc)
    final_points = np.column_stack((peaks_y, peaks_x, percentile_intensities, np.ones_like(percentile_intensities)))

    sort_indices = np.argsort(final_points[:, 2])[::-1]
    final_points = final_points[sort_indices]
    if final_points.shape[0] > max_spots: final_points = final_points[:max_spots, :]

    return final_points.astype(float)


# --- НОВЫЙ МЕТОД 3 (старый Hybrid): Hybrid (Legacy -> Centroid -> Filter Blobs -> Final Proximity) ---
def detect_spots_hybrid(
        arr: np.ndarray,
        legacy_perc: float = 99.0,
        centroid_perc: float = 99.0,
        min_area: int = 3,
        max_spots: int = 6000,
        min_distance: float = 4.0,  # Общий для legacy(min_dist) и centroid(prox_thresh) + Final Filter
) -> np.ndarray:
    """
    Combines Legacy and Centroid methods with blob-based and final proximity filtering.
    1. Run Legacy (full image).
    2. Run Centroid (full image).
    3. Remove Legacy points located within the blobs of kept Centroid points.
    4. Combine remaining Legacy points and all Centroid points.
    5. Apply final proximity filter giving priority to Centroid points.
    6. Sort by intensity and trim to max_spots.
    Returns array of (y, x, intensity_percentile, area).
    """
    if peak_local_max is None: raise RuntimeError("Пакет 'scikit-image' не найден (нужен для Hybrid).")
    if cKDTree is None: raise RuntimeError("Пакет 'scipy' не найден (нужен для Hybrid).")

    print("Running Hybrid detector (Legacy -> Centroid -> Filter Blobs -> Final Proximity)...")
    H, W = arr.shape
    int_min_distance = int(round(min_distance))

    # 1. Запускаем Legacy
    print(f" Hybrid Step 1: Running Legacy (perc={legacy_perc}, dist={int_min_distance})...")
    points_legacy = detect_spots_legacy(
        arr, legacy_perc, max_spots, int_min_distance, mask=None
    )
    print(f"  Legacy found {len(points_legacy)} points.")

    # 2. Запускаем Centroid
    print(f" Hybrid Step 2: Running Centroid (perc={centroid_perc}, area={min_area}, prox={min_distance})...")
    points_centroid, labels_map, kept_centroid_label_indices = detect_spots_by_centroid(
        arr, centroid_perc, min_area, max_spots, min_distance
    )
    print(f"  Centroid found {len(points_centroid)} points.")

    # 3. Удаляем Legacy точки, попавшие в блобы Centroid
    print(" Hybrid Step 3: Removing Legacy points covered by Centroid blobs...")
    if len(points_centroid) > 0 and len(points_legacy) > 0:
        legacy_coords_yx = points_legacy[:, :2].round().astype(int)
        legacy_coords_yx[:, 0] = np.clip(legacy_coords_yx[:, 0], 0, H - 1)
        legacy_coords_yx[:, 1] = np.clip(legacy_coords_yx[:, 1], 0, W - 1)
        blob_ids_at_legacy_points = labels_map[legacy_coords_yx[:, 0], legacy_coords_yx[:, 1]]
        valid_centroid_blob_ids = set(kept_centroid_label_indices)
        mask_legacy_to_keep = np.array(
            [(blob_id == 0) or (blob_id not in valid_centroid_blob_ids) for blob_id in blob_ids_at_legacy_points])
        points_legacy_filtered = points_legacy[mask_legacy_to_keep]
        print(
            f"  Removed {len(points_legacy) - len(points_legacy_filtered)} Legacy points. {len(points_legacy_filtered)} remaining.")
    else:
        points_legacy_filtered = points_legacy
        print("  No filtering applied (no Centroid points or no Legacy points).")

    # 4. Объединяем результаты и добавляем флаг источника
    #    (0=Centroid, 1=Legacy) для приоритетной фильтрации
    combined_points_list = []
    sources = []  # 0 for Centroid, 1 for Legacy
    if len(points_centroid) > 0:
        combined_points_list.append(points_centroid)
        sources.extend([0] * len(points_centroid))
    if len(points_legacy_filtered) > 0:
        combined_points_list.append(points_legacy_filtered)
        sources.extend([1] * len(points_legacy_filtered))

    if not combined_points_list:
        return np.zeros((0, 4), dtype=float)

    combined_points = np.vstack(combined_points_list)
    sources = np.array(sources)
    print(f" Hybrid Step 4: Combined to {len(combined_points)} points.")

    # 5. Финальная фильтрация по близости с приоритетом Centroid
    print(f" Hybrid Step 5: Applying final proximity filter (dist={min_distance}, priority=Centroid)...")
    # Сортируем: сначала по источнику (Centroid=0), потом по интенсивности (убывание)
    sort_indices = np.lexsort((-combined_points[:, 2], sources))  # -V для убывания
    sorted_combined_points = combined_points[sort_indices]

    kept_points_final: List[Tuple[float, float, float, float]] = []
    kept_coords_list: List[List[float]] = []
    kept_coords_tree: Optional[cKDTree] = None
    prox_threshold_sq = min_distance ** 2

    for point_data in sorted_combined_points:
        y, x, v_perc, area = point_data
        point_coord = [y, x]
        is_too_close = False
        if kept_coords_tree is not None:
            dist, _ = kept_coords_tree.query(point_coord, k=1)
            if dist ** 2 < prox_threshold_sq:
                is_too_close = True

        if not is_too_close:
            kept_points_final.append((y, x, v_perc, area))
            kept_coords_list.append(point_coord)
            # Перестраиваем дерево (проще, но можно оптимизировать)
            kept_coords_tree = cKDTree(kept_coords_list)
            # Не обрезаем по max_spots здесь, сделаем это после финальной сортировки

    print(f"  Kept {len(kept_points_final)} points after final proximity filter.")

    if not kept_points_final:
        return np.zeros((0, 4), dtype=float)

    final_points_array = np.array(kept_points_final, dtype=float)

    # 6. Финальная сортировка по интенсивности и обрезка
    sort_indices_final = np.argsort(final_points_array[:, 2])[::-1]
    final_points_sorted = final_points_array[sort_indices_final]

    if final_points_sorted.shape[0] > max_spots:
        final_points_sorted = final_points_sorted[:max_spots, :]
        print(f"  Trimmed to {max_spots} final points.")

    return final_points_sorted


# --- КОНЕЦ МЕТОДА HYBRID ---


# <<< НОВЫЙ МЕТОД 4: Centroid (Multi-Pass) >>>
def detect_spots_centroid_multipass(
        # --- МЕТОД 3: Centroid (Multi-Pass) [Обновленная логика шагов] ---
        arr: np.ndarray,
        perc_bright: float = 99.0,
        min_area_bright: int = 2,
        perc_dim: float = 95.0,
        min_area_dim: int = 5,
        max_spots: int = 6000,
        proximity_threshold: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Detects spots using a multi-pass centroiding approach with variable min_area
    AND iterative percentile stepping (like the original Centroid method).

    Pass 1: High percentile, Low min_area (for bright, sharp spots)
    Pass 2: Low percentile, High min_area (for dim, fuzzy spots, filters noise)
    Returns: (array of (y, x, intensity_perc, area), labels_map, kept_label_indices).
    """
    if cKDTree is None: raise RuntimeError("Пакет 'scipy' не найден.")

    H, W = arr.shape
    percent_map, uniq_vals, uniq_perc = compute_percentile_map(arr)
    master_blob_map = np.zeros((H, W), dtype=np.uint8)
    full_labels_map = np.zeros((H, W), dtype=np.int32)

    # --- ИЗМЕНЕНИЕ: Генерируем шаги с шагом -5 (как в оригинальном Centroid) ---
    step_size = -5
    perc_bright_safe = max(perc_bright, perc_dim)
    perc_dim_safe = min(perc_bright, perc_dim)

    steps_map = {}

    # 1. Сначала генерируем шаги для "тусклого" прохода (они будут переписаны "яркими")
    # range(100, int(95) - 1, -5) -> [100, 95]
    steps_dim_list = sorted(list(set(list(range(100, int(perc_dim_safe) - 1, step_size)) + [int(perc_dim_safe)])),
                            reverse=True)
    for th in steps_dim_list:
        steps_map[th] = min_area_dim

    # 2. Теперь генерируем шаги для "яркого" прохода (они перезапишут верхние значения)
    # range(100, int(99) - 1, -5) -> [100]
    steps_bright_list = sorted(
        list(set(list(range(100, int(perc_bright_safe) - 1, step_size)) + [int(perc_bright_safe)])), reverse=True)
    for th in steps_bright_list:
        steps_map[th] = min_area_bright

    # Сортируем по порогу (убывание)
    final_steps_tuples = sorted(steps_map.items(), key=lambda item: -item[0])

    print(f"[Multi-Pass] Bright pass (>= {perc_bright_safe}%): min_area={min_area_bright} px")
    print(f"[Multi-Pass] Dim pass ({perc_dim_safe}% - {perc_bright_safe - 1}%): min_area={min_area_dim} px")
    print(f"[Multi-Pass] Using steps: {final_steps_tuples}")
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    kept_points_with_labels: List[Tuple[float, float, float, int, int]] = []
    kept_coords_list: List[List[float]] = []
    kept_coords_tree: Optional[cKDTree] = None
    prox_threshold_sq = proximity_threshold ** 2
    current_max_label = 0

    # --- Итерация по объединенному списку шагов ---
    for th_value, min_area_for_step in final_steps_tuples:
        # --- КОНЕЦ ---
        current_binary_mask = np.where(percent_map >= th_value, 255, 0).astype(np.uint8)
        mask_for_ccs = cv2.bitwise_and(current_binary_mask, current_binary_mask, mask=cv2.bitwise_not(master_blob_map))
        num_labels, labels_map_step, stats, centroids = cv2.connectedComponentsWithStats(mask_for_ccs, connectivity=8)

        if num_labels <= 1: continue

        valid_step_mask = (labels_map_step > 0)
        labels_map_step[valid_step_mask] += current_max_label
        full_labels_map[valid_step_mask] = labels_map_step[valid_step_mask]

        current_batch_points: List[Tuple[float, float, float, int, int]] = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # --- Используем min_area для текущего шага ---
            if area < min_area_for_step: continue
            # --- КОНЕЦ ---
            cx, cy = centroids[i];
            yi, xi = int(round(cy)), int(round(cx))
            if 0 <= yi < H and 0 <= xi < W:
                # --- ИЗМЕНЕНИЕ: Убран master_blob_map[yi, xi] == 0 ---
                v = float(percent_map[yi, xi])
                current_batch_points.append((cy, cx, v, area, i + current_max_label))
                # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        current_max_label += (num_labels - 1)
        current_batch_points.sort(key=lambda t: -t[2])
        newly_added_coords_this_batch: List[List[float]] = []

        # --- ИЗМЕНЕНИЕ: Новая логика цикла ---
        for (y, x, v, area, label_index) in current_batch_points:
            point_coord = [y, x];

            # (1) Маскируем блоб СРАЗУ
            mask_to_add_single = (full_labels_map == label_index)
            if master_blob_map[mask_to_add_single].all():
                 continue
            master_blob_map[mask_to_add_single] = 1  # Маскируем пиксели

            # (2) Проверяем близость
            min_dist_sq = float('inf')
            if kept_coords_tree is not None:
                dist, _ = kept_coords_tree.query(point_coord, k=1)
                min_dist_sq = min(min_dist_sq, dist ** 2)
            if newly_added_coords_this_batch:
                dists_sq_batch = np.sum((np.array(newly_added_coords_this_batch) - point_coord) ** 2, axis=1)
                if dists_sq_batch.size > 0: min_dist_sq = min(min_dist_sq, np.min(dists_sq_batch))

            # (3) Если слишком близко, пропускаем только добавление ТОЧКИ
            if min_dist_sq < prox_threshold_sq:
                continue

            # (4) Если точка не отброшена, добавляем ее в список
            kept_points_with_labels.append((y, x, v, area, label_index))
            newly_added_coords_this_batch.append([y, x])
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        if newly_added_coords_this_batch:
            kept_coords_list.extend(newly_added_coords_this_batch)
            kept_coords_tree = cKDTree(kept_coords_list)

    kept_points_with_labels.sort(key=lambda t: -t[2])
    if len(kept_points_with_labels) > max_spots:
        kept_points_with_labels = kept_points_with_labels[:max_spots]

    if kept_points_with_labels:
        result_array = np.array([(p[0], p[1], p[2], p[3]) for p in kept_points_with_labels], dtype=float)
        kept_label_indices = [p[4] for p in kept_points_with_labels]
    else:
        result_array = np.zeros((0, 4), dtype=float)
        kept_label_indices = []

    return result_array, full_labels_map, kept_label_indices


# <<< КОНЕЦ НОВОГО МЕТОДА 4 >>>


# +++ НОВАЯ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ +++
def _save_points_on_image(
        img_arr_u8_bgr: np.ndarray,
        points_yx: np.ndarray,
        color_bgr: Tuple[int, int, int],
        output_path: Path,
        radius: int = 3,
        thickness: int = 1
):
    """Вспомогательная функция: рисует точки (y, x) на BGR-изображении и сохраняет его."""
    img_with_points = img_arr_u8_bgr.copy()
    if points_yx is not None and len(points_yx) > 0:
        for y, x, *rest in points_yx:
            # cv2.circle принимает координаты (x, y)
            pt_xy = (int(round(x)), int(round(y)))
            cv2.circle(img_with_points, pt_xy, radius, color_bgr, thickness)
    try:
        cv2.imwrite(str(output_path), img_with_points)
        print(f"  Сохранено отладочное изображение точек: {output_path.name}")
    except Exception as e:
        print(f"  Предупреждение: Не удалось сохранить отладочное изображение {output_path.name}: {e}")


# +++ КОНЕЦ НОВОЙ ФУНКЦИИ +++


# <<< НОВЫЙ МЕТОД 5: Legacy + Dual Centroid >>>
def detect_spots_legacy_plus_dual_centroid(
        arr: np.ndarray,
        legacy_perc: float,
        legacy_min_dist: float,
        c1_perc: float,
        c1_min_area: int,
        c1_prox: float,
        c2_perc: float,
        c2_min_area: int,
        c2_prox: float,
        max_spots: int,
        debug_mask_path: Path,
        final_proximity_filter: float,
) -> np.ndarray:
    """
    Runs Legacy, filters them against C1 blobs, masks both, runs C2, combines, and filters.
    Saves debug images for L, C1 (on L mask), and C2 (on L+C1 mask).
    """
    if peak_local_max is None: raise RuntimeError("Пакет 'scikit-image' не найден (нужен для Legacy + Dual Centroid).")
    if cKDTree is None: raise RuntimeError("Пакет 'scipy' не найден (нужен для Legacy + Dual Centroid).")

    H, W = arr.shape
    int_legacy_min_dist = int(round(legacy_min_dist))

    # --- 1. Legacy Pass ---
    print(f"[L+2C] Step 1: Running Legacy (perc={legacy_perc}, dist={int_legacy_min_dist})...")
    points_legacy = detect_spots_legacy(
        arr, legacy_perc, max_spots, int_legacy_min_dist, mask=None
    )
    print(f"  Legacy found {len(points_legacy)} points.")

    # --- 2. Centroid Pass 1 (на оригинальном изображении) ---
    print(f"[L+2C] Step 2: Running Centroid 1 (perc={c1_perc}, area={c1_min_area}, prox={c1_prox})...")
    points_centroid_1, labels_map_1, kept_label_indices_1 = detect_spots_by_centroid(
        arr, c1_perc, c1_min_area, max_spots, c1_prox
    )
    print(f"  Centroid 1 found {len(points_centroid_1)} points.")

    # --- 3. Filter Legacy Points (NEW) ---
    print("[L+2C] Step 3: Filtering Legacy points covered by C1 blobs...")
    if len(points_centroid_1) > 0 and len(points_legacy) > 0:
        legacy_coords_yx = points_legacy[:, :2].round().astype(int)
        legacy_coords_yx[:, 0] = np.clip(legacy_coords_yx[:, 0], 0, H - 1)
        legacy_coords_yx[:, 1] = np.clip(legacy_coords_yx[:, 1], 0, W - 1)

        blob_ids_at_legacy_points = labels_map_1[legacy_coords_yx[:, 0], legacy_coords_yx[:, 1]]
        valid_centroid_blob_ids = set(kept_label_indices_1)

        mask_legacy_to_keep = np.array(
            [(blob_id == 0) or (blob_id not in valid_centroid_blob_ids) for blob_id in blob_ids_at_legacy_points]
        )
        points_legacy_filtered = points_legacy[mask_legacy_to_keep]
        print(
            f"  Removed {len(points_legacy) - len(points_legacy_filtered)} Legacy points. {len(points_legacy_filtered)} remaining.")
    else:
        points_legacy_filtered = points_legacy
        print("  No filtering applied (no C1 points or no Legacy points).")

    # --- 4. Mask Generation ---
    print("[L+2C] Step 4: Generating masks...")
    legacy_mask = np.zeros(arr.shape, dtype=np.uint8)
    master_mask = np.zeros(arr.shape, dtype=np.uint8)

    # 4a. Mask *filtered* Legacy points by radius
    if len(points_legacy_filtered) > 0:
        radius_to_mask = int(round(legacy_min_dist))
        if radius_to_mask > 0:
            for y, x, _, _ in points_legacy_filtered:
                cv2.circle(legacy_mask, (int(round(x)), int(round(y))), radius_to_mask, 255, -1)
        print(f"  Masked {len(points_legacy_filtered)} filtered Legacy points with radius {radius_to_mask}px.")
        master_mask[legacy_mask == 255] = 255  # Добавляем в общую маску

    # 4b. Mask Centroid 1 blobs
    if len(kept_label_indices_1) > 0:
        # --- ИЗМЕНЕНИЕ: Используем labels_map_1, а не запускаем C1 заново ---
        # (Логика C1 теперь маскирует ВСЕ блобы, а не только 'kept',
        # но для L+2C мы хотим маскировать только те, что НАШЕЛ C1)
        mask_c1_blobs = np.isin(labels_map_1, kept_label_indices_1)
        master_mask[mask_c1_blobs] = 255  # Добавляем в общую маску
        print(f"  Masked {len(kept_label_indices_1)} blobs from Centroid 1.")
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    # --- 5. Save Debug Images ---
    outdir = debug_mask_path.parent
    print(f"[L+2C] Step 5: Saving debug images to {outdir.name}...")

    # Создаем базовые изображения U8 и BGR
    arr_norm_u8 = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    arr_norm_bgr = cv2.cvtColor(arr_norm_u8, cv2.COLOR_GRAY2BGR)

    # 5a. Сохраняем *отфильтрованные* точки Legacy на ОРИГИНАЛЬНОМ изображении
    _save_points_on_image(
        arr_norm_bgr,
        points_legacy_filtered,
        (0, 255, 255),  # Желтый (BGR)
        outdir / "debug_points_legacy.png"
    )

    # 5b. Сохраняем точки C1 на изображении с МАСКОЙ LEGACY
    debug_img_bgr_L_masked = arr_norm_bgr.copy()
    debug_img_bgr_L_masked[legacy_mask == 255] = [0, 0, 0]  # Применяем только маску Legacy
    _save_points_on_image(
        debug_img_bgr_L_masked,
        points_centroid_1,
        (255, 0, 0),  # Синий (BGR)
        outdir / "debug_points_c1_on_L_mask.png"  # Новое имя
    )

    # 5c. Создаем и сохраняем изображение С ОБЩЕЙ МАСКОЙ (L+C1)
    debug_img_bgr_L_C1_masked = arr_norm_bgr.copy()
    debug_img_bgr_L_C1_masked[master_mask == 255] = [0, 0, 0]  # BGR = Black
    try:
        cv2.imwrite(str(debug_mask_path), debug_img_bgr_L_C1_masked)
        print(f"  Saved debug mask: {debug_mask_path.name}")
    except Exception as e:
        print(f"  Warning: Failed to save debug mask image: {e}")

    # --- 6. Centroid Pass 2 (на замаскированном изображении) ---
    print(f"[L+2C] Step 6: Running Centroid 2 (perc={c2_perc}, area={c2_min_area}, prox={c2_prox}) on masked image...")
    arr_masked = arr.copy()
    arr_masked[master_mask == 255] = 0  # Обнуляем замаскированные пиксели

    points_centroid_2, _, _ = detect_spots_by_centroid(
        arr_masked, c2_perc, c2_min_area, max_spots, c2_prox
    )
    print(f"  Centroid 2 found {len(points_centroid_2)} points.")

    # 6b. Сохраняем точки C2 на ОБЩЕЙ МАСКЕ (L+C1)
    _save_points_on_image(
        debug_img_bgr_L_C1_masked,  # Используем изображение с общей маской
        points_centroid_2,
        (0, 0, 255),  # Красный (BGR)
        outdir / "debug_points_c2_on_L_C1_mask.png"  # Новое имя
    )

    # --- 7. Combine Results (Приоритет: C1 > C2 > Legacy_Filtered) ---
    print("[L+2C] Step 7: Combining and applying final proximity filter...")
    combined_points_list = []
    sources = []  # 0=C1, 1=C2, 2=Legacy
    if len(points_centroid_1) > 0:
        combined_points_list.append(points_centroid_1)
        sources.extend([0] * len(points_centroid_1))
    if len(points_centroid_2) > 0:
        combined_points_list.append(points_centroid_2)
        sources.extend([1] * len(points_centroid_2))
    # --- ИСПОЛЬЗУЕМ ОТФИЛЬТРОВАННЫЕ ---
    if len(points_legacy_filtered) > 0:
        combined_points_list.append(points_legacy_filtered)
        sources.extend([2] * len(points_legacy_filtered))

    if not combined_points_list:
        return np.zeros((0, 4), dtype=float)

    combined_points = np.vstack(combined_points_list)
    sources = np.array(sources)
    print(f"  Combined to {len(combined_points)} total points before final filter.")

    # Сортируем: сначала по источнику (C1=0, C2=1, L=2), потом по интенсивности (убывание)
    sort_indices = np.lexsort((-combined_points[:, 2], sources))  # -V для убывания
    sorted_combined_points = combined_points[sort_indices]

    # --- 8. Final Proximity Filter ---
    kept_points_final: List[Tuple[float, float, float, float]] = []
    kept_coords_list: List[List[float]] = []
    kept_coords_tree: Optional[cKDTree] = None
    final_prox_threshold_sq = final_proximity_filter ** 2
    print(f"  Applying final proximity filter (dist={final_proximity_filter:.2f}px)...")

    for point_data in sorted_combined_points:
        y, x, v_perc, area = point_data
        point_coord = [y, x]
        is_too_close = False
        if kept_coords_tree is not None:
            dist, _ = kept_coords_tree.query(point_coord, k=1)
            if dist ** 2 < final_prox_threshold_sq:
                is_too_close = True

        if not is_too_close:
            kept_points_final.append((y, x, v_perc, area))
            kept_coords_list.append(point_coord)
            kept_coords_tree = cKDTree(kept_coords_list)

    print(f"  Kept {len(kept_points_final)} points after final proximity filter.")

    if not kept_points_final:
        return np.zeros((0, 4), dtype=float)

    final_points_array = np.array(kept_points_final, dtype=float)

    # --- 9. Final Sort by Intensity & Trim ---
    sort_indices_final = np.argsort(final_points_array[:, 2])[::-1]
    final_points_sorted = final_points_array[sort_indices_final]

    if final_points_sorted.shape[0] > max_spots:
        final_points_sorted = final_points_sorted[:max_spots, :]
        print(f"  Trimmed to {max_spots} final points.")

    return final_points_sorted


# <<< КОНЕЦ НОВОГО МЕТОДА 5 >>>


# --- Остальные функции (geometric_midpoint, refine_center_antipodal) без изменений ---
def geometric_midpoint(arr: np.ndarray) -> CenterResult:
    H, W = arr.shape
    return CenterResult(cy=(H - 1) / 2.0, cx=(W - 1) / 2.0, method="midpoint")


def refine_center_antipodal(center: Tuple[float, float], pts: np.ndarray, tol_ang_deg: float = 8.0,
                            tol_rel_r: float = 0.06, iters: int = 3) -> CenterResult:
    cy, cx = float(center[0]), float(center[1])
    if len(pts) < 4: return CenterResult(cy=cy, cx=cx, method="midpoint (fallback)")
    pts_yx = pts[:, :2]
    for _ in range(max(0, int(iters))):
        dy = pts_yx[:, 0] - cy;
        dx = pts_yx[:, 1] - cx;
        r = np.hypot(dx, dy)
        r_safe = np.where(r > 1e-9, r, 1e-9);
        u = np.column_stack((dx, dy)) / r_safe[:, None]
        cos_thr = -np.cos(np.deg2rad(180.0 - float(tol_ang_deg)));
        mids = []
        for i in range(len(pts_yx)):
            if r[i] < 1e-6: continue
            dots = (u @ u[i]);
            max_r_pair = np.maximum(r, r[i])
            rel_diff = np.divide(np.abs(r - r[i]), max_r_pair, out=np.zeros_like(r), where=max_r_pair > 1e-9)
            rad_ok = (rel_diff < float(tol_rel_r)) & (max_r_pair > 1e-9);
            ang_ok = (dots < cos_thr)
            valid_match = rad_ok & ang_ok & (np.arange(len(pts_yx)) != i) & (r > 1e-6)
            idx = np.where(valid_match)[0]
            if idx.size == 0: continue
            j = idx[np.argmin(np.abs(dots[idx] + 1.0))]
            yi, xi = pts_yx[i, 0], pts_yx[i, 1];
            yj, xj = pts_yx[j, 0], pts_yx[j, 1]
            mids.append(((yi + yj) / 2.0, (xi + xj) / 2.0))
        if len(mids) < 4: break
        mids = np.array(mids, dtype=float)
        cy = float(np.median(mids[:, 0]));
        cx = float(np.median(mids[:, 1]))
    return CenterResult(cy=cy, cx=cx, method="antipodal-refined")


# -------------------------- GUI --------------------------
class SAEDLauncherFrame(ttk.Frame):
    # ... (init, _get_default_output_path) ...
    def __init__(self, master: tk.Misc, controller=None):
        super().__init__(master);
        self.controller = controller
        self._scroll_canvas = None;
        self._scroll_window_id = None
        self._build_ui()

    def _get_default_output_path(self) -> str:
        # ... (код без изменений) ...
        if getattr(sys, "frozen", False):
            base_dir = Path(sys.executable).parent
        else:
            try:
                base_dir = Path.cwd()
            except OSError:
                base_dir = Path(__file__).parent
        base_name = "saed_results";
        output_path = base_dir / base_name
        if not output_path.exists(): return str(output_path)
        counter = 1
        while True:
            new_name = f"{base_name}_{counter}";
            new_path = base_dir / new_name
            if not new_path.exists(): return str(new_path)
            counter += 1;
            if counter > 999: return str(base_dir / f"{base_name}_temp_{np.random.randint(1000)}")

    def _build_ui(self):
        # ... (UI до detect_box) ...
        outer = ttk.Frame(self);
        outer.pack(fill=tk.BOTH, expand=True)
        fixed = ttk.Frame(outer, padding=(16, 16, 16, 0));
        fixed.pack(side=tk.TOP, fill=tk.X)
        fixed.grid_columnconfigure(0, weight=1)
        data_box = ttk.LabelFrame(fixed, text="Input data", padding=(12, 10, 12, 12));
        data_box.grid(row=0, column=0, sticky="nsew")
        data_box.grid_columnconfigure(1, weight=1)
        ttk.Label(data_box, text="Image:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.ent_img = ttk.Entry(data_box);
        self.ent_img.grid(row=0, column=1, columnspan=2, sticky="we", padx=6, pady=4)
        ttk.Button(data_box, text="Browse…", command=self._browse_img).grid(row=0, column=3, sticky="ew", padx=6,
                                                                            pady=4)
        ttk.Label(data_box, text="Output folder:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.ent_out = ttk.Entry(data_box);
        self.ent_out.insert(0, self._get_default_output_path());
        self.ent_out.grid(row=1, column=1, sticky="we", padx=6, pady=4)
        ttk.Button(data_box, text="Load Session…", command=self._load_session).grid(row=1, column=2, sticky="ew",
                                                                                    padx=6, pady=4)
        ttk.Button(data_box, text="Choose…", command=self._browse_out).grid(row=1, column=3, sticky="ew", padx=6,
                                                                            pady=4)
        ttk.Label(data_box, text="Center X (optional):").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.ent_cx = ttk.Entry(data_box, width=12);
        self.ent_cx.grid(row=2, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(data_box, text="Center Y:").grid(row=2, column=2, sticky="w", padx=6, pady=4)
        self.ent_cy = ttk.Entry(data_box, width=12);
        self.ent_cy.grid(row=2, column=3, sticky="w", padx=6, pady=4)
        ttk.Label(data_box, text="Leave coordinates empty for auto center. Use 'Load Session' to restore.",
                  wraplength=520, foreground="#555555").grid(row=3, column=0, columnspan=4, sticky="we", padx=6,
                                                             pady=(0, 4))

        pre_box = ttk.LabelFrame(fixed, text="Preprocessing", padding=(12, 10, 12, 12));
        pre_box.grid(row=1, column=0, sticky="nsew", pady=(10, 0));
        pre_box.grid_columnconfigure(1, weight=1)
        ttk.Label(pre_box, text="Mode:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.cmb_pre = ttk.Combobox(pre_box, values=["No processing", "NLM Denoising"], state="readonly");
        self.cmb_pre.current(0);
        self.cmb_pre.grid(row=0, column=1, sticky="w", padx=6, pady=4)
        self.spn_h_param = self._spin_param(pre_box, 1, "NLM h_param", 0.3, from_=0.01, to=2.0, increment=0.01,
                                            format_str="%.2f")
        self.cmb_pre.bind("<<ComboboxSelected>>", self._on_preproc_change)
        ttk.Label(pre_box, text="Select 'NLM Denoising' for noise reduction (uses scikit-image) and adjust 'h_param'.",
                  wraplength=520, foreground="#555555").grid(row=3, column=0, columnspan=3, sticky="we", padx=6,
                                                             pady=(2, 0))

        scroll_host = ttk.Frame(outer);
        scroll_host.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(scroll_host, borderwidth=0, highlightthickness=0);
        vscroll = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=canvas.yview)
        scrollable = ttk.Frame(canvas, padding=(16, 12, 16, 12));
        scrollable.grid_columnconfigure(0, weight=1)
        self._scroll_canvas = canvas;
        self._scroll_window_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(self._scroll_window_id, width=e.width))
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True);
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        scrollable.bind("<Enter>", self._activate_scroll);
        scrollable.bind("<Leave>", self._deactivate_scroll)
        canvas.bind("<Enter>", self._activate_scroll);
        canvas.bind("<Leave>", self._deactivate_scroll)

        detect_box = ttk.LabelFrame(scrollable, text="Detector and refinement", padding=(12, 10, 12, 12));
        detect_box.grid(row=0, column=0, sticky="nsew");
        detect_box.grid_columnconfigure(1, weight=1)

        # --- Выбор метода детекции ---
        ttk.Label(detect_box, text="Detection Method:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        # <<< ИЗМЕНЕНИЕ: Добавлен 'Legacy + Dual Centroid' >>>
        self.cmb_detect_method = ttk.Combobox(detect_box, values=["Centroid (Default)", "Legacy (Local Maxima)",
                                                                  "Hybrid (Legacy + Centroid)",
                                                                  "Centroid (Multi-Pass)",
                                                                  "Legacy + Dual Centroid"], state="readonly")
        # <<< КОНЕЦ >>>
        self.cmb_detect_method.current(0);
        self.cmb_detect_method.grid(row=0, column=1, sticky="w", padx=6, pady=4)
        self.cmb_detect_method.bind("<<ComboboxSelected>>", self._on_detect_method_change)

        # --- Min Peak Distance (ВНЕШНИЙ, используется только Legacy/Hybrid) ---
        self.spn_min_dist = self._spin_param(detect_box, 1, "Min. Peak Distance (px)", 4.0, from_=1.0, to=50.0,
                                             increment=0.5, format_str="%.1f")

        ttk.Separator(detect_box).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 6))

        # --- Рамка 1: (Centroid/Legacy/Hybrid) ---
        self.centroid_legacy_frame = ttk.Frame(detect_box, padding=0)
        self.centroid_legacy_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.centroid_legacy_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.centroid_legacy_frame, text="Peak threshold and search window",
                  font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=6,
                                                           pady=(0, 2))
        self.spn_perc_centroid = self._spin_param(self.centroid_legacy_frame, 1, "Centroid/Hybrid-C Percentile (%)",
                                                  99.0, from_=80.0, to=100.0, increment=0.1, format_str="%.1f")
        self.spn_perc_legacy = self._spin_param(self.centroid_legacy_frame, 2, "Legacy/Hybrid-L Percentile (%)", 99.0,
                                                from_=80.0, to=100.0, increment=0.1, format_str="%.1f")
        self.spn_min_area = self._spin_param(self.centroid_legacy_frame, 3, "Min. peak area (px) [Centroid/Hybrid]", 3,
                                             from_=1, to=500, increment=1)

        # --- Рамка 2: Multi-Pass ---
        self.multipass_frame = ttk.Frame(detect_box, padding=0)
        # self.multipass_frame.grid(row=3, column=0, columnspan=2, sticky="ew") # Не grid-им сразу
        self.multipass_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.multipass_frame, text="Multi-Pass Parameters", font=("TkDefaultFont", 10, "bold")).grid(row=0,
                                                                                                               column=0,
                                                                                                               columnspan=2,
                                                                                                               sticky="w",
                                                                                                               padx=6,
                                                                                                               pady=(0,
                                                                                                                     2))
        self.spn_mp_perc_bright = self._spin_param(self.multipass_frame, 1, "Bright Pass Percentile (%)", 99.0,
                                                   from_=80.0, to=100.0, increment=0.1, format_str="%.1f")
        self.spn_mp_area_bright = self._spin_param(self.multipass_frame, 2, "Bright Pass Min Area (px)", 2, from_=1,
                                                   to=500, increment=1)
        self.spn_mp_perc_dim = self._spin_param(self.multipass_frame, 3, "Dim Pass Percentile (%)", 95.0, from_=80.0,
                                                to=100.0, increment=0.1, format_str="%.1f")
        self.spn_mp_area_dim = self._spin_param(self.multipass_frame, 4, "Dim Pass Min Area (px)", 5, from_=1, to=500,
                                                increment=1)

        # <<< НОВОЕ: Рамка 3: Legacy + Dual Centroid >>>
        self.dual_centroid_frame = ttk.Frame(detect_box, padding=0)
        self.dual_centroid_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.dual_centroid_frame, text="Legacy Pass Parameters", font=("TkDefaultFont", 10, "bold")).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            padx=6,
            pady=(0,
                  2))
        self.spn_dc_perc_legacy = self._spin_param(self.dual_centroid_frame, 1, "Legacy Percentile (%)", 99.0,
                                                   from_=80.0, to=100.0, increment=0.1, format_str="%.1f")
        self.spn_dc_dist_legacy = self._spin_param(self.dual_centroid_frame, 2, "Legacy Min Dist (px)", 4.0, from_=1.0,
                                                   to=50.0, increment=0.5, format_str="%.1f")

        ttk.Separator(self.dual_centroid_frame).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 6))
        ttk.Label(self.dual_centroid_frame, text="Centroid Pass 1 Parameters (Original)",
                  font=("TkDefaultFont", 10, "bold")).grid(row=4, column=0, columnspan=2, sticky="w", padx=6,
                                                           pady=(0, 2))
        self.spn_dc_perc_c1 = self._spin_param(self.dual_centroid_frame, 5, "C1 Percentile (%)", 99.0, from_=80.0,
                                               to=100.0, increment=0.1, format_str="%.1f")
        self.spn_dc_area_c1 = self._spin_param(self.dual_centroid_frame, 6, "C1 Min Area (px)", 3, from_=1, to=500,
                                               increment=1)
        self.spn_dc_prox_c1 = self._spin_param(self.dual_centroid_frame, 7, "C1 Proximity (px)", 4.0, from_=1.0,
                                               to=50.0,
                                               increment=0.5, format_str="%.1f")

        ttk.Separator(self.dual_centroid_frame).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(4, 6))
        ttk.Label(self.dual_centroid_frame, text="Centroid Pass 2 Parameters (Masked)",
                  font=("TkDefaultFont", 10, "bold")).grid(row=9, column=0, columnspan=2, sticky="w", padx=6,
                                                           pady=(0, 2))
        self.spn_dc_perc_c2 = self._spin_param(self.dual_centroid_frame, 10, "C2 Percentile (%)", 95.0, from_=80.0,
                                                to=100.0, increment=0.1, format_str="%.1f")
        self.spn_dc_area_c2 = self._spin_param(self.dual_centroid_frame, 11, "C2 Min Area (px)", 5, from_=1, to=500,
                                               increment=1)
        self.spn_dc_prox_c2 = self._spin_param(self.dual_centroid_frame, 12, "C2 Proximity (px)", 4.0, from_=1.0,
                                               to=50.0, increment=0.5, format_str="%.1f")
        # <<< КОНЕЦ НОВОГО >>>

        # Смещаем оставшиеся элементы
        row_offset = 4  # Новая стартовая строка для общих элементов

        ttk.Separator(detect_box).grid(row=row_offset + 5, column=0, columnspan=2, sticky="ew", pady=(6, 8))
        ttk.Label(detect_box, text="Center refinement", font=("TkDefaultFont", 10, "bold")).grid(row=row_offset + 6,
                                                                                                 column=0, columnspan=2,
                                                                                                 sticky="w", padx=6,
                                                                                                 pady=(0, 2))
        self.spn_iters = self._spin_param(detect_box, row_offset + 7, "Center refinement iterations", 4, from_=0, to=10,
                                          increment=1)
        self.spn_tolang = self._spin_param(detect_box, row_offset + 8, "Antipode tolerance (°)", 8.0, from_=1.0,
                                           to=30.0, increment=0.5, format_str="%.1f")
        self.spn_tolr = self._spin_param(detect_box, row_offset + 9, "Radius tolerance (relative)", 0.06, from_=0.01,
                                         to=0.5, increment=0.01, format_str="%.2f")
        ttk.Separator(detect_box).grid(row=row_offset + 10, column=0, columnspan=2, sticky="ew", pady=(6, 8))
        ttk.Label(detect_box, text="Geometric filters", font=("TkDefaultFont", 10, "bold")).grid(row=row_offset + 11,
                                                                                                 column=0, columnspan=2,
                                                                                                 sticky="w", padx=6,
                                                                                                 pady=(0, 2))
        self.spn_dead = self._spin_param(detect_box, row_offset + 12, "Dead zone (px)", 0, from_=0, to=500, increment=1)
        self.spn_search = self._spin_param(detect_box, row_offset + 13, "Search radius (px, 0 = unlimited)", 0, from_=0,
                                           to=10000, increment=25)

        # --- spn_maxpts (общий для всех методов) ---
        self.spn_maxpts = self._spin_param(detect_box, row_offset + 14, "Maximum detected points", 6000, from_=100,
                                           to=20000, increment=100)

        self.lbl_detect_hint = ttk.Label(detect_box, text="", wraplength=520, foreground="#555555")
        self.lbl_detect_hint.grid(row=row_offset + 15, column=0, columnspan=2, sticky="we", padx=6,
                                  pady=(2, 0))  # Смещаем

        action_box = ttk.Frame(scrollable, padding=(0, 12, 0, 0));
        action_box.grid(row=1, column=0, sticky="nsew");
        action_box.grid_columnconfigure(0, weight=1)
        ttk.Label(action_box, text="Review parameters and press button to switch to interactive editing.",
                  wraplength=540, justify="left").grid(row=0, column=0, sticky="we", padx=4, pady=(0, 8))
        ttk.Button(action_box, text="Open point editor", command=self._go_editor).grid(row=1, column=0, sticky="ew",
                                                                                       padx=4, pady=(0, 12))
        filler_bg = ttk.Style().lookup("TFrame", "background") or self.winfo_toplevel().cget("background");
        bottom_filler = tk.Frame(scrollable, height=56, bg=filler_bg);
        bottom_filler.grid(row=2, column=0, sticky="ew");
        bottom_filler.grid_propagate(False)

        self._on_preproc_change(None)
        self._on_detect_method_change(None)

    # ... (методы _activate_scroll, _deactivate_scroll, _on_scroll_mousewheel, _on_preproc_change) ...
    def _activate_scroll(self, _event):
        if self._scroll_canvas is None: return
        self._scroll_canvas.bind_all("<MouseWheel>", self._on_scroll_mousewheel)
        self._scroll_canvas.bind_all("<Button-4>", self._on_scroll_mousewheel)
        self._scroll_canvas.bind_all("<Button-5>", self._on_scroll_mousewheel)

    def _deactivate_scroll(self, _event):
        if self._scroll_canvas is None: return
        self._scroll_canvas.unbind_all("<MouseWheel>")
        self._scroll_canvas.unbind_all("<Button-4>")
        self._scroll_canvas.unbind_all("<Button-5>")

    def _on_scroll_mousewheel(self, event):
        if self._scroll_canvas is None: return
        delta = 0
        if sys.platform == "win32":
            delta = -int(event.delta / 120)
        elif sys.platform == "darwin":
            delta = event.delta
        elif event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        if delta != 0: self._scroll_canvas.yview_scroll(delta, "units")

    def _on_preproc_change(self, event=None):
        if not hasattr(self, 'cmb_pre') or not hasattr(self, 'spn_h_param'): return
        try:
            mode = self.cmb_pre.get()
            self.spn_h_param.configure(state='normal' if mode == "NLM Denoising" else 'readonly')
        except tk.TclError:
            pass

    # <<< ИЗМЕНЕНИЕ: Обновлен _on_detect_method_change >>>
    def _on_detect_method_change(self, event=None):
        widgets_exist = all(hasattr(self, w) for w in [
            'cmb_detect_method', 'spn_min_area', 'lbl_detect_hint',
            'spn_perc_centroid', 'spn_perc_legacy', 'spn_min_dist',
            'centroid_legacy_frame', 'multipass_frame', 'dual_centroid_frame'  # Добавлены новые рамки
        ])
        if not widgets_exist: return

        try:
            method = self.cmb_detect_method.get()

            # Сначала прячем все рамки параметров
            self.centroid_legacy_frame.grid_forget()
            self.multipass_frame.grid_forget()
            self.dual_centroid_frame.grid_forget()

            # По умолчанию внешний spn_min_dist активен
            self.spn_min_dist.configure(state='normal')

            hint_text = ""
            if method == "Centroid (Default)":
                self.centroid_legacy_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
                self.spn_perc_centroid.configure(state='normal')
                self.spn_perc_legacy.configure(state='readonly')
                self.spn_min_area.configure(state='normal')
                # 'spn_min_dist' используется как 'proximity_threshold'
                hint_text = "Centroid: Uses connected components & area filtering. 'Min. Peak Distance' is used as Proximity Threshold."

            elif method == "Legacy (Local Maxima)":
                self.centroid_legacy_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
                self.spn_perc_centroid.configure(state='readonly')
                self.spn_perc_legacy.configure(state='normal')
                self.spn_min_area.configure(state='readonly')
                hint_text = "Legacy: Uses local maxima & distance filter. 'Min. peak area' ignored. Requires scikit-image."

            elif method == "Hybrid (Legacy + Centroid)":
                self.centroid_legacy_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
                self.spn_perc_centroid.configure(state='normal')
                self.spn_perc_legacy.configure(state='normal')
                self.spn_min_area.configure(state='normal')
                hint_text = "Hybrid: Runs Legacy (Legacy Perc), then Centroid (Centroid Perc). 'Min. Peak Distance' is used for both."

            elif method == "Centroid (Multi-Pass)":
                self.multipass_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
                # 'spn_min_dist' используется как 'proximity_threshold'
                hint_text = "Multi-Pass: Uses Bright Pass + Dim Pass. 'Min. Peak Distance' is used as Proximity Threshold."

            elif method == "Legacy + Dual Centroid":
                self.dual_centroid_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
                # Выключаем внешний spn_min_dist, т.к. все настройки внутри рамки
                self.spn_min_dist.configure(state='readonly')
                hint_text = "L+2C: Runs Legacy, filters vs C1 blobs, masks (L+C1), runs C2. Saves debug images."

            self.lbl_detect_hint.configure(text=hint_text)
        except tk.TclError:
            pass

    # --- КОНЕЦ ---

    # ... (методы _spin_param, _set_spinbox_value, _browse_img, _browse_out, _load_session) ...
    def _spin_param(self, parent, row, label, default, *, from_, to, increment, format_str=None):
        ttk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        spin = ttk.Spinbox(parent, from_=from_, to=to, increment=increment, width=10, justify="right")
        if format_str: spin.configure(format=format_str)
        self._set_spinbox_value(spin, default)
        spin.grid(row=row, column=1, sticky="w", padx=6, pady=4)
        return spin

    def _set_spinbox_value(self, spinbox: ttk.Spinbox, value):
        try:
            spinbox.set(value)
        except tk.TclError:
            try:
                current_value = spinbox.get()
                if str(current_value) != str(value): spinbox.delete(0, tk.END); spinbox.insert(0, str(value))
            except (tk.TclError, ValueError):
                print(f"Warning: Could not set spinbox value to {value}")

    def _browse_img(self):
        p = filedialog.askopenfilename(title="Select image",
                                       filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"), ("All", "*.*")])
        if p: self.ent_img.delete(0, tk.END); self.ent_img.insert(0, p)

    def _browse_out(self):
        p = filedialog.askdirectory(title="Select output folder", mustexist=False)
        if p: self.ent_out.delete(0, tk.END); self.ent_out.insert(0, p)

    def _load_session(self):
        filepath = filedialog.askopenfilename(title="Load SAED Session",
                                              filetypes=[("SAED Session", "saed_session.json"), ("All files", "*.*")])
        if not filepath or not self.controller: return
        try:
            self.controller.load_session_from_file(filepath)
            if self.controller: self.controller.set_status(f"Session loaded from {Path(filepath).name}")
        except FileNotFoundError:
            messagebox.showerror("Load Error", "Session file not found.")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load session:\n{e}")

    # <<< ИЗМЕНЕНИЕ: Обновлен get_state >>>
    def get_state(self) -> Dict[str, Any]:
        return {
            "image_path": self.ent_img.get(), "output_folder": self.ent_out.get(),
            "center_x": self.ent_cx.get(), "center_y": self.ent_cy.get(),
            "preproc_mode": self.cmb_pre.get(), "h_param": self.spn_h_param.get(),
            "detect_method": self.cmb_detect_method.get(),
            "min_dist": self.spn_min_dist.get(),
            "perc_centroid": self.spn_perc_centroid.get(),
            "perc_legacy": self.spn_perc_legacy.get(),
            "min_area": self.spn_min_area.get(),
            # --- Multi-Pass ---
            "mp_perc_bright": self.spn_mp_perc_bright.get(),
            "mp_area_bright": self.spn_mp_area_bright.get(),
            "mp_perc_dim": self.spn_mp_perc_dim.get(),
            "mp_area_dim": self.spn_mp_area_dim.get(),
            # --- НОВОЕ: Legacy + Dual Centroid ---
            "dc_perc_legacy": self.spn_dc_perc_legacy.get(),
            "dc_dist_legacy": self.spn_dc_dist_legacy.get(),
            "dc_perc_c1": self.spn_dc_perc_c1.get(),
            "dc_area_c1": self.spn_dc_area_c1.get(),
            "dc_prox_c1": self.spn_dc_prox_c1.get(),
            "dc_perc_c2": self.spn_dc_perc_c2.get(),
            "dc_area_c2": self.spn_dc_area_c2.get(),
            "dc_prox_c2": self.spn_dc_prox_c2.get(),
            # --- КОНЕЦ НОВОГО ---
            "max_pts": self.spn_maxpts.get(), "refine_iters": self.spn_iters.get(),
            "tol_angle": self.spn_tolang.get(), "tol_radius": self.spn_tolr.get(),
            "dead_zone": self.spn_dead.get(), "search_radius": self.spn_search.get(),
        }

    # <<< ИЗМЕНЕНИЕ: Обновлен set_state >>>
    def set_state(self, state: Dict[str, Any]):
        def _set_entry(widget, value):
            if value is not None and isinstance(widget, (ttk.Entry, tk.Entry)): widget.delete(0, tk.END); widget.insert(
                0, str(value))

        _set_entry(self.ent_img, state.get("image_path"));
        _set_entry(self.ent_out, state.get("output_folder"))
        _set_entry(self.ent_cx, state.get("center_x"));
        _set_entry(self.ent_cy, state.get("center_y"))
        preproc_mode = state.get("preproc_mode")
        if preproc_mode and isinstance(self.cmb_pre, ttk.Combobox):
            if preproc_mode in self.cmb_pre['values']:
                self.cmb_pre.set(preproc_mode)
            else:
                print(f"Warning: Saved preproc_mode '{preproc_mode}' not found.");
                self.cmb_pre.current(0)
        elif isinstance(self.cmb_pre, ttk.Combobox):
            self.cmb_pre.current(0)
        self._on_preproc_change(None)
        self._set_spinbox_value(self.spn_h_param, state.get("h_param", 0.3))

        detect_method = state.get("detect_method")
        default_method = "Centroid (Default)"
        all_methods = self.cmb_detect_method['values']

        # --- ИЗМЕНЕНИЕ: Устанавливаем "Legacy + Dual Centroid" как новый default, если он есть ---
        if "Legacy + Dual Centroid" in all_methods:
            default_method = "Legacy + Dual Centroid"
        elif "Centroid (Multi-Pass)" in all_methods:
            default_method = "Centroid (Multi-Pass)"

        if detect_method and isinstance(self.cmb_detect_method, ttk.Combobox):
            if detect_method in all_methods:
                self.cmb_detect_method.set(detect_method)
            else:
                print(f"Warning: Saved detect_method '{detect_method}' not found.");
                self.cmb_detect_method.set(
                    default_method)
        elif isinstance(self.cmb_detect_method, ttk.Combobox):
            self.cmb_detect_method.set(default_method)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        self._set_spinbox_value(self.spn_min_dist, state.get("min_dist", 4.0))
        self._set_spinbox_value(self.spn_perc_centroid, state.get("perc_centroid", 99.0))
        self._set_spinbox_value(self.spn_perc_legacy, state.get("perc_legacy", 99.0))
        self._set_spinbox_value(self.spn_min_area, state.get("min_area", 3))

        # --- Multi-Pass ---
        self._set_spinbox_value(self.spn_mp_perc_bright, state.get("mp_perc_bright", 99.0))
        self._set_spinbox_value(self.spn_mp_area_bright, state.get("mp_area_bright", 2))
        self._set_spinbox_value(self.spn_mp_perc_dim, state.get("mp_perc_dim", 95.0))
        self._set_spinbox_value(self.spn_mp_area_dim, state.get("mp_area_dim", 5))

        # --- НОВОЕ: Legacy + Dual Centroid ---
        self._set_spinbox_value(self.spn_dc_perc_legacy, state.get("dc_perc_legacy", 99.0))
        self._set_spinbox_value(self.spn_dc_dist_legacy, state.get("dc_dist_legacy", 4.0))
        self._set_spinbox_value(self.spn_dc_perc_c1, state.get("dc_perc_c1", 99.0))
        self._set_spinbox_value(self.spn_dc_area_c1, state.get("dc_area_c1", 3))
        self._set_spinbox_value(self.spn_dc_prox_c1, state.get("dc_prox_c1", 4.0))
        self._set_spinbox_value(self.spn_dc_perc_c2, state.get("dc_perc_c2", 95.0))
        self._set_spinbox_value(self.spn_dc_area_c2, state.get("dc_area_c2", 5))
        self._set_spinbox_value(self.spn_dc_prox_c2, state.get("dc_prox_c2", 4.0))
        # --- КОНЕЦ НОВОГО ---

        self._set_spinbox_value(self.spn_maxpts, state.get("max_pts", 6000))
        self._set_spinbox_value(self.spn_iters, state.get("refine_iters", 4))
        self._set_spinbox_value(self.spn_tolang, state.get("tol_angle", 8.0))
        self._set_spinbox_value(self.spn_tolr, state.get("tol_radius", 0.06))
        self._set_spinbox_value(self.spn_dead, state.get("dead_zone", 0))
        self._set_spinbox_value(self.spn_search, state.get("search_radius", 0))

        self._on_detect_method_change(None)  # Обновляем состояние UI детектора

    # --- КОНЕЦ ---

    # <<< ИЗМЕНЕНИЕ: Обновлен _go_editor >>>
    def _go_editor(self):
        try:
            image_path_str = self.ent_img.get();
            output_dir_str = self.ent_out.get()
            if not image_path_str: messagebox.showerror("Error", "Please select an image file."); return
            if not output_dir_str: messagebox.showerror("Error", "Please specify an output folder."); return
            image_path = Path(image_path_str).expanduser().resolve()
            outdir = Path(output_dir_str).expanduser().resolve();
            outdir.mkdir(parents=True, exist_ok=True)
            if not image_path.exists(): messagebox.showerror("Error", f"Image not found at: {image_path}"); return

            # --- Получаем ВСЕ параметры ---
            perc_centroid = float(self.spn_perc_centroid.get())
            perc_legacy = float(self.spn_perc_legacy.get())
            min_area = int(float(self.spn_min_area.get()))

            mp_perc_bright = float(self.spn_mp_perc_bright.get())
            mp_area_bright = int(float(self.spn_mp_area_bright.get()))
            mp_perc_dim = float(self.spn_mp_perc_dim.get())
            mp_area_dim = int(float(self.spn_mp_area_dim.get()))

            # --- НОВОЕ: Получаем параметры L+2C ---
            dc_perc_legacy = float(self.spn_dc_perc_legacy.get())
            dc_dist_legacy = float(self.spn_dc_dist_legacy.get())
            dc_perc_c1 = float(self.spn_dc_perc_c1.get())
            dc_area_c1 = int(float(self.spn_dc_area_c1.get()))
            dc_prox_c1 = float(self.spn_dc_prox_c1.get())
            dc_perc_c2 = float(self.spn_dc_perc_c2.get())
            dc_area_c2 = int(float(self.spn_dc_area_c2.get()))
            dc_prox_c2 = float(self.spn_dc_prox_c2.get())
            # --- КОНЕЦ НОВОГО ---

            max_pts = int(float(self.spn_maxpts.get()))
            min_dist = float(self.spn_min_dist.get())  # Внешний min_dist
            iters = int(float(self.spn_iters.get()))
            tol_ang = float(self.spn_tolang.get())
            tol_relr = float(self.spn_tolr.get())
            dead_r = float(self.spn_dead.get())
            search_r = float(self.spn_search.get())

            pre_mode = self.cmb_pre.get();
            h_param_val = float(self.spn_h_param.get())
            settings = PreprocSettings(mode="nlm",
                                       h_param=h_param_val) if pre_mode == "NLM Denoising" else PreprocSettings(
                mode="raw", h_param=h_param_val)

            try:
                arr = load_grayscale_with_preproc(image_path, settings)
            except RuntimeError as cv_err:
                messagebox.showerror("Dependency Error", str(cv_err));
                return
            except Exception as img_load_err:
                messagebox.showerror("Image Error", f"Failed to load/process image:\n{img_load_err}");
                return
            preproc_payload = settings.to_json()

            cx_txt = self.ent_cx.get().strip();
            cy_txt = self.ent_cy.get().strip()
            if cx_txt and cy_txt:
                try:
                    center0 = CenterResult(cy=float(cy_txt), cx=float(cx_txt), method="user")
                except ValueError:
                    messagebox.showwarning("Input Warning",
                                           "Invalid center coords. Using auto.");
                    center0 = geometric_midpoint(arr)
            else:
                center0 = geometric_midpoint(arr)

            # --- Вызов метода детекции с нужными параметрами ---
            detect_method_choice = self.cmb_detect_method.get()
            pts_raw = np.zeros((0, 4), dtype=float)
            log_params = {}  # Для сохранения параметров в лог

            try:
                if "Legacy (Local Maxima)" == detect_method_choice:
                    print(f"Using Legacy detector (perc={perc_legacy}, dist={min_dist})...")
                    pts_raw = detect_spots_legacy(arr, user_perc=perc_legacy, max_spots=max_pts,
                                                  min_distance=int(round(min_dist)))
                    log_params = {"perc_legacy": perc_legacy, "min_distance": min_dist}

                elif "Hybrid" in detect_method_choice:
                    print(
                        f"Using Hybrid detector (L.perc={perc_legacy}, C.perc={perc_centroid}, area={min_area}, dist={min_dist})...")
                    pts_raw = detect_spots_hybrid(arr, legacy_perc=perc_legacy, centroid_perc=perc_centroid,
                                                  min_area=min_area, max_spots=max_pts, min_distance=min_dist)
                    log_params = {"perc_legacy": perc_legacy, "perc_centroid": perc_centroid, "min_area": min_area,
                                  "min_distance": min_dist}

                elif "Multi-Pass" in detect_method_choice:
                    print(
                        f"Using Centroid (Multi-Pass) detector (Bright: {mp_perc_bright}%/{mp_area_bright}px, Dim: {mp_perc_dim}%/{mp_area_dim}px, prox={min_dist})...")
                    pts_raw, _, _ = detect_spots_centroid_multipass(
                        arr,
                        perc_bright=mp_perc_bright, min_area_bright=mp_area_bright,
                        perc_dim=mp_perc_dim, min_area_dim=mp_area_dim,
                        max_spots=max_pts, proximity_threshold=min_dist
                    )
                    log_params = {"mp_perc_bright": mp_perc_bright, "mp_area_bright": mp_area_bright,
                                  "mp_perc_dim": mp_perc_dim, "mp_area_dim": mp_area_dim,
                                  "proximity_threshold": min_dist}

                # --- НОВОЕ: Вызов Legacy + Dual Centroid ---
                elif "Legacy + Dual Centroid" in detect_method_choice:
                    print(f"Using Legacy + Dual Centroid detector...")
                    debug_mask_path = outdir / "debug_mask_L-C1.png"
                    # Определяем финальный фильтр
                    final_prox = min(dc_dist_legacy, dc_prox_c1, dc_prox_c2, min_dist)  # Включаем и внешний min_dist
                    print(f"  Final proximity filter distance set to: {final_prox:.2f}px")

                    pts_raw = detect_spots_legacy_plus_dual_centroid(
                        arr,
                        legacy_perc=dc_perc_legacy, legacy_min_dist=dc_dist_legacy,
                        c1_perc=dc_perc_c1, c1_min_area=dc_area_c1, c1_prox=dc_prox_c1,
                        c2_perc=dc_perc_c2, c2_min_area=dc_area_c2, c2_prox=dc_prox_c2,
                        max_spots=max_pts,
                        debug_mask_path=debug_mask_path,
                        final_proximity_filter=final_prox
                    )
                    log_params = {
                        "dc_perc_legacy": dc_perc_legacy, "dc_dist_legacy": dc_dist_legacy,
                        "dc_perc_c1": dc_perc_c1, "dc_area_c1": dc_area_c1, "dc_prox_c1": dc_prox_c1,
                        "dc_perc_c2": dc_perc_c2, "dc_area_c2": dc_area_c2, "dc_prox_c2": dc_prox_c2,
                        "final_proximity_filter": final_prox
                    }
                # --- КОНЕЦ НОВОГО ---

                else:  # По умолчанию Centroid
                    print(f"Using Centroid detector (perc={perc_centroid}, area={min_area}, prox={min_dist})...")
                    pts_raw, _, _ = detect_spots_by_centroid(arr, user_perc=perc_centroid, min_area=min_area,
                                                             max_spots=max_pts, proximity_threshold=min_dist)
                    log_params = {"perc_centroid": perc_centroid, "min_area": min_area, "proximity_threshold": min_dist}

            except RuntimeError as e:
                messagebox.showerror("Dependency Error", str(e));
                return
            # --- Конец вызова ---

            # --- Логика без симметрии ---
            if len(pts_raw) == 0:
                messagebox.showwarning("Detection Warning", f"No spots detected with method '{detect_method_choice}'.")
                pts_processed = np.zeros((0, 4), dtype=float);
                point_types = {};
                center = center0
            else:
                center = refine_center_antipodal((center0.cy, center0.cx), pts_raw, tol_ang_deg=tol_ang,
                                                 tol_rel_r=tol_relr, iters=iters)
                pts_processed = pts_raw.copy();
                point_types = {i: "unknown" for i in range(len(pts_processed))}
                print("Point classification disabled in launcher.")

            if (dead_r > 0 or search_r > 0) and len(pts_processed) > 0:
                dy = pts_processed[:, 0] - center.cy;
                dx = pts_processed[:, 1] - center.cx;
                r = np.hypot(dx, dy)
                mask = np.ones(len(pts_processed), dtype=bool)
                if dead_r > 0:   mask &= (r >= dead_r)
                if search_r > 0: mask &= (r <= search_r)
                indices_to_keep = np.where(mask)[0];
                pts_processed = pts_processed[indices_to_keep]
                point_types = {new_idx: "unknown" for new_idx, old_idx in enumerate(indices_to_keep)}
            else:
                point_types = {i: "unknown" for i in range(len(pts_processed))}

            points_list_for_json = []
            for i, (y, x, v, area) in enumerate(pts_processed):
                pt_type = point_types.get(i, "unknown")
                points_list_for_json.append(
                    {"y": float(y), "x": float(x), "intensity": float(v), "area": int(area), "type": pt_type})

            saed_input_data = {
                "image": str(image_path), "preproc_mode": settings.mode, "preproc": preproc_payload,
                "center": {"x": float(center.cx), "y": float(center.cy), "method": center.method},
                "radii": {"dead": float(dead_r), "search": float(search_r)}, "points": points_list_for_json
            }
            saed_input_path = outdir / "saed_input.json"
            saed_input_path.write_text(json.dumps(saed_input_data, ensure_ascii=False, indent=2), encoding="utf-8")

            if self.controller is not None:
                try:
                    self.controller.open_editor(saed_input_path)
                except Exception as exc:
                    messagebox.showerror("Error", f"Failed to open editor tab:\n{exc}")
            else:
                messagebox.showwarning("Standalone Mode", "Running standalone.")

            # --- Обновлен лог ---
            try:
                log_payload = {
                    "initial": {"x": center0.cx, "y": center0.cy, "method": center0.method},
                    "refined": {"x": center.cx, "y": center.cy, "method": center.method},
                    "dead_zone_px": dead_r, "search_radius_px": search_r,
                    "preproc_mode": settings.mode, "preproc": preproc_payload,
                    "image_size": {"H": int(arr.shape[0]), "W": int(arr.shape[1])},
                    "detection_method": detect_method_choice,
                    # Добавляем параметры конкретного метода
                    **log_params
                }
                # Очищаем None значения
                log_payload_clean = {k: v for k, v in log_payload.items() if v is not None}
                (outdir / "center_init.json").write_text(json.dumps(log_payload_clean, indent=2), encoding="utf-8")

            except Exception as log_err:
                print(f"Warning: Could not save center_init.json - {log_err}")

        except Exception as e:
            messagebox.showerror("Processing Error", f"An unexpected error occurred:\n{e}")
            import traceback;
            traceback.print_exc()
    # --- КОНЕЦ ---


# ... (класс SAEDApp и __main__ без изменений) ...
class SAEDApp(tk.Tk):
    def __init__(self):
        super().__init__();
        self.title("SAED Symmetry – Launcher");
        self.geometry("980x680");
        self.resizable(True, False)
        frame = SAEDLauncherFrame(self);
        frame.pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    SAEDApp().mainloop()