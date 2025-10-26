#!/usr/bin/env python3
# -*- coding: utf-8 -*-
(
    "SAED Symmetry – Launcher\n"
    "========================\n"
    "What's new:\n"
    "  • Preprocessing switch:\n"
    "      - 'No processing' (raw grayscale)\n"
    "      - 'NLM Denoising' (h=0.3)\n"
    "  • Automatic point classification based on symmetry analysis:\n"
    "      - Points grouped by radius (5%) and area (3%).\n"
    "      - Groups matching dominant symmetry (k) marked 'structural'.\n"
    "      - Groups matching 2*k marked 'superstructural'.\n"
    "      - Radii of structural/superstructural groups are averaged.\n"
    "  • Point types ('structural', 'superstructural', 'unknown') saved to JSON.\n"
)
from __future__ import annotations
import json, subprocess, sys, cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional
# --- Добавлены импорты ---
import math
from collections import defaultdict
from scipy.signal import find_peaks
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


# ------ Функции анализа (pol_from, symmetry_scores, cluster_rings) ------
# ... (Без изменений) ...
def pol_from(center, pts):
    cy, cx = center
    dy, dx = pts[:, 0] - cy, pts[:, 1] - cx
    r = np.hypot(dx, dy)
    a = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    return r, a

def symmetry_scores(angles, radii, ring_means, top_rings=3):
    out = {}
    if ring_means.size == 0: return out
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
        C = np.cos(phases_rad).mean(); S = np.sin(phases_rad).mean()
        out[f"{k}-fold"] = float(np.hypot(C, S))
    return out

def cluster_rings(radii, bins=100, prominence_factor=0.03, min_prominence=2):
    if len(radii) == 0: return np.array([]), np.zeros(0, dtype=int), ([], [])
    hist, edges = np.histogram(radii, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    if hist.max() > 0: prominence = max(min_prominence, hist.max() * prominence_factor)
    else: prominence = min_prominence
    pk, _ = find_peaks(hist, prominence=prominence)
    ring_centers = centers[pk]
    if len(ring_centers) == 0: return np.array([]), np.zeros_like(radii, dtype=int), (hist.tolist(), edges.tolist())
    labels = np.argmin(np.abs(radii[:, None] - ring_centers[None, :]), axis=1)
    return ring_centers, labels, (hist.tolist(), edges.tolist())
# ------ КОНЕЦ Функций анализа ------


# -------------------------- Algorithm --------------------------
@dataclass
class CenterResult:
    cy: float
    cx: float
    method: str


# --- МЕТОД 1: Centroid ---
# --- ИЗМЕНЕНИЕ: Возвращает точки, labels_map И список ID сохраненных блобов ---
def detect_spots_by_centroid(
        arr: np.ndarray,
        user_perc: float = 99.0,
        min_area: int = 3,
        max_spots: int = 6000,
        proximity_threshold: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, List[int]]: # Возвращает (массив Nx4, labels_map HxW, список label_indices)
    """
    Detects spots using centroiding.
    Returns: (array of (y, x, intensity_perc, area), labels_map, kept_label_indices).
    """
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---
    if cKDTree is None: raise RuntimeError("Пакет 'scipy' не найден.")

    H, W = arr.shape
    percent_map, uniq_vals, uniq_perc = compute_percentile_map(arr)
    master_blob_map = np.zeros((H, W), dtype=np.uint8) # Используется только для быстрой проверки занятости пикселя
    # --- ИЗМЕНЕНИЕ: Инициализация labels_map ---
    full_labels_map = np.zeros((H, W), dtype=np.int32) # Карта ID блобов для всего изображения
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---
    steps = sorted(list(set(list(range(100, int(user_perc), -5)) + [int(user_perc)])), reverse=True)
    if user_perc not in steps: steps.append(user_perc); steps.sort(reverse=True)

    kept_points_final: List[Tuple[float, float, float, int]] = []
    # --- ИЗМЕНЕНИЕ: Сохраняем label_index вместе с точкой ---
    kept_points_with_labels: List[Tuple[float, float, float, int, int]] = [] # (y, x, v, area, label_index)
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---
    kept_coords_list: List[List[float]] = []
    kept_coords_tree: Optional[cKDTree] = None
    prox_threshold_sq = proximity_threshold ** 2

    current_max_label = 0 # Отслеживаем максимальный ID блоба между шагами

    for th_value in steps:
        current_binary_mask = np.where(percent_map >= th_value, 255, 0).astype(np.uint8)
        # Находим компоненты только в НЕОБРАБОТАННЫХ областях
        mask_for_ccs = cv2.bitwise_and(current_binary_mask, current_binary_mask, mask=cv2.bitwise_not(master_blob_map))
        num_labels, labels_map_step, stats, centroids = cv2.connectedComponentsWithStats(mask_for_ccs, connectivity=8)

        if num_labels <= 1: continue

        # --- ИЗМЕНЕНИЕ: Сдвигаем ID блобов и обновляем full_labels_map ---
        valid_step_mask = (labels_map_step > 0)
        labels_map_step[valid_step_mask] += current_max_label # Сдвигаем ID
        full_labels_map[valid_step_mask] = labels_map_step[valid_step_mask] # Добавляем новые блобы в общую карту
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        current_batch_points: List[Tuple[float, float, float, int, int]] = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area: continue
            cx, cy = centroids[i]; yi, xi = int(round(cy)), int(round(cx))
            if 0 <= yi < H and 0 <= xi < W:
                # Проверяем master_blob_map, чтобы случайно не добавить точку из уже занятой области
                if master_blob_map[yi,xi] == 0:
                    v = float(percent_map[yi, xi])
                    current_batch_points.append((cy, cx, v, area, i + current_max_label)) # Используем сдвинутый ID

        current_max_label += (num_labels - 1) # Обновляем максимальный ID для следующего шага

        current_batch_points.sort(key=lambda t: -t[2])

        newly_added_coords_this_batch: List[List[float]] = []

        for (y, x, v, area, label_index) in current_batch_points:
            # Проверка на близость к УЖЕ добавленным
            point_coord = [y, x]; min_dist_sq = float('inf')
            if kept_coords_tree is not None:
                dist, _ = kept_coords_tree.query(point_coord, k=1)
                min_dist_sq = min(min_dist_sq, dist ** 2)
            # Проверка на близость к ТОЛЬКО ЧТО добавленным в этом батче
            if newly_added_coords_this_batch:
                dists_sq_batch = np.sum((np.array(newly_added_coords_this_batch) - point_coord) ** 2, axis=1)
                if dists_sq_batch.size > 0: min_dist_sq = min(min_dist_sq, np.min(dists_sq_batch))

            if min_dist_sq < prox_threshold_sq: continue

            # --- ИЗМЕНЕНИЕ: Сохраняем точку с ее label_index ---
            kept_points_with_labels.append((y, x, v, area, label_index))
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---
            newly_added_coords_this_batch.append([y, x])

            # --- Обновляем master_blob_map сразу ---
            mask_to_add_single = (full_labels_map == label_index)
            master_blob_map[mask_to_add_single] = 1
            # --- КОНЕЦ Обновления ---

        if newly_added_coords_this_batch:
            kept_coords_list.extend(newly_added_coords_this_batch)
            kept_coords_tree = cKDTree(kept_coords_list)

    # --- Фильтруем по max_spots ---
    kept_points_with_labels.sort(key=lambda t: -t[2]) # Сортируем по интенсивности
    if len(kept_points_with_labels) > max_spots:
        kept_points_with_labels = kept_points_with_labels[:max_spots]

    # --- Разделяем данные для возврата ---
    if kept_points_with_labels:
        result_array = np.array([(p[0], p[1], p[2], p[3]) for p in kept_points_with_labels], dtype=float)
        kept_label_indices = [p[4] for p in kept_points_with_labels]
    else:
        result_array = np.zeros((0, 4), dtype=float)
        kept_label_indices = []

    return result_array, full_labels_map, kept_label_indices # Возвращаем точки, карту блобов, ID сохраненных блобов


# --- МЕТОД 2: Legacy ---
def detect_spots_legacy(
        arr: np.ndarray,
        user_perc: float = 99.0,
        max_spots: int = 6000,
        min_distance: int = 4,
        mask: Optional[np.ndarray] = None # Маска ОБЛАСТЕЙ для ИСКЛЮЧЕНИЯ из поиска
) -> np.ndarray:
    """
    Detects spots using local maxima finding, optionally excluding masked areas.
    Returns array of (y, x, intensity_percentile, area=1.0).
    """
    if peak_local_max is None: raise RuntimeError("Пакет 'scikit-image' не найден.")

    H, W = arr.shape
    percent_map, uniq_vals, uniq_perc = compute_percentile_map(arr)
    threshold_abs = np.interp(user_perc, uniq_perc, uniq_vals)

    # --- Маска для peak_local_max ---
    # peak_local_max ищет пики там, где маска НЕ НОЛЬ.
    # Нам нужно инвертировать переданную маску (mask > 0 означает ИСКЛЮЧИТЬ).
    footprint_mask = None
    if mask is not None:
         if mask.shape != arr.shape:
              print("Warning: Mask shape mismatch in detect_spots_legacy. Ignoring mask.")
         else:
              footprint_mask = (mask == 0) # Искать только там, где маска = 0
    # --- КОНЕЦ ---

    safe_min_distance = max(1, int(min_distance))
    coordinates = peak_local_max(
        arr, # Ищем пики на ОРИГИНАЛЬНОМ изображении
        min_distance=safe_min_distance,
        threshold_abs=threshold_abs,
        exclude_border=False, # Не исключаем края
        labels=(footprint_mask.astype(np.int32) if footprint_mask is not None else None), # Используем маску областей
        num_peaks=max_spots # Ограничиваем количество пиков на этом этапе
    )

    if coordinates.shape[0] == 0: return np.zeros((0, 4), dtype=float)

    peaks_y = coordinates[:, 0]; peaks_x = coordinates[:, 1]
    raw_intensities = arr[peaks_y, peaks_x] # Интенсивность берем из ОРИГИНАЛА
    percentile_intensities = map_values_to_percent(raw_intensities, uniq_vals, uniq_perc)
    final_points = np.column_stack((peaks_y, peaks_x, percentile_intensities, np.ones_like(percentile_intensities)))

    # Сортировка и обрезка по max_spots уже сделаны в peak_local_max с num_peaks
    # sort_indices = np.argsort(final_points[:, 2])[::-1]
    # final_points = final_points[sort_indices]
    # if final_points.shape[0] > max_spots: final_points = final_points[:max_spots, :]

    return final_points.astype(float)


# --- НОВЫЙ МЕТОД 3: Hybrid (Legacy -> Centroid -> Filter) ---
def detect_spots_hybrid(
        arr: np.ndarray,
        user_perc: float = 99.0,
        min_area: int = 3,
        max_spots: int = 6000,
        min_distance: float = 4.0, # Общий для обоих
) -> np.ndarray:
    """
    Combines Legacy and Centroid methods with filtering.
    1. Run Legacy.
    2. Run Centroid.
    3. Remove Legacy points within Centroid blobs.
    4. Combine remaining Legacy and all Centroid points.
    Returns array of (y, x, intensity_percentile, area).
    """
    if peak_local_max is None: raise RuntimeError("Пакет 'scikit-image' не найден (нужен для Hybrid).")
    if cKDTree is None: raise RuntimeError("Пакет 'scipy' не найден (нужен для Hybrid).")

    print("Running Hybrid detector (Legacy -> Centroid -> Filter)...")
    H, W = arr.shape

    # 1. Запускаем Legacy на всем изображении
    print(" Hybrid Step 1: Running Legacy...")
    # Ограничиваем max_spots заранее, чтобы не делать лишнюю работу
    points_legacy = detect_spots_legacy(
        arr, user_perc, max_spots, int(round(min_distance)), mask=None
    )
    print(f"  Legacy found {len(points_legacy)} points.")
    if len(points_legacy) == 0: # Если Legacy ничего не нашел, запускаем только Centroid
        print("  Legacy found 0 points. Running Centroid only...")
        points_centroid, _, _ = detect_spots_by_centroid(
             arr, user_perc, min_area, max_spots, min_distance
        )
        return points_centroid

    # 2. Запускаем Centroid на всем изображении
    print(" Hybrid Step 2: Running Centroid...")
    points_centroid, labels_map, kept_centroid_label_indices = detect_spots_by_centroid(
        arr, user_perc, min_area, max_spots, min_distance
    )
    print(f"  Centroid found {len(points_centroid)} points.")

    # 3. Удаляем Legacy точки, попавшие в блобы Centroid
    print(" Hybrid Step 3: Removing Legacy points covered by Centroid blobs...")
    if len(points_centroid) > 0 and len(points_legacy) > 0:
        # Получаем целочисленные координаты точек Legacy
        legacy_coords_yx = points_legacy[:, :2].round().astype(int)
        # Убедимся, что координаты в пределах изображения
        legacy_coords_yx[:, 0] = np.clip(legacy_coords_yx[:, 0], 0, H - 1)
        legacy_coords_yx[:, 1] = np.clip(legacy_coords_yx[:, 1], 0, W - 1)

        # Получаем ID блобов, в которые попали точки Legacy
        blob_ids_at_legacy_points = labels_map[legacy_coords_yx[:, 0], legacy_coords_yx[:, 1]]

        # Создаем маску для точек Legacy: True, если точка НЕ в блобе Centroid
        # Блоб Centroid - это блоб, ID которого есть в kept_centroid_label_indices
        valid_centroid_blob_ids = set(kept_centroid_label_indices)
        mask_legacy_to_keep = np.array([
            blob_id not in valid_centroid_blob_ids for blob_id in blob_ids_at_legacy_points
        ])

        # Фильтруем точки Legacy
        points_legacy_filtered = points_legacy[mask_legacy_to_keep]
        print(f"  Removed {len(points_legacy) - len(points_legacy_filtered)} Legacy points. {len(points_legacy_filtered)} remaining.")
    else:
        points_legacy_filtered = points_legacy # Оставляем все, если Centroid ничего не нашел

    # 4. Объединяем результаты
    if len(points_centroid) > 0 and len(points_legacy_filtered) > 0:
         combined_points = np.vstack((points_centroid, points_legacy_filtered))
    elif len(points_centroid) > 0:
         combined_points = points_centroid
    elif len(points_legacy_filtered) > 0:
         combined_points = points_legacy_filtered
    else:
         return np.zeros((0, 4), dtype=float)

    print(f" Hybrid Step 4: Combined to {len(combined_points)} points.")

    # 5. Без финальной фильтрации - просто сортируем и обрезаем
    sort_indices = np.argsort(combined_points[:, 2])[::-1] # Сортируем по интенсивности V
    final_points = combined_points[sort_indices]

    if final_points.shape[0] > max_spots:
        final_points = final_points[:max_spots, :]
        print(f"  Trimmed to {max_spots} points.")

    return final_points.astype(float)
# --- КОНЕЦ НОВОГО МЕТОДА ---


# --- Остальные функции (geometric_midpoint, refine_center_antipodal, group_points_by_radius) без изменений ---
def geometric_midpoint(arr: np.ndarray) -> CenterResult:
    H, W = arr.shape
    return CenterResult(cy=(H - 1) / 2.0, cx=(W - 1) / 2.0, method="midpoint")

def refine_center_antipodal(center: Tuple[float, float], pts: np.ndarray, tol_ang_deg: float = 8.0, tol_rel_r: float = 0.06, iters: int = 3) -> CenterResult:
    cy, cx = float(center[0]), float(center[1])
    if len(pts) < 4: return CenterResult(cy=cy, cx=cx, method="midpoint (fallback)")
    pts_yx = pts[:, :2]
    for _ in range(max(0, int(iters))):
        dy = pts_yx[:, 0] - cy; dx = pts_yx[:, 1] - cx; r = np.hypot(dx, dy)
        r_safe = np.where(r > 1e-9, r, 1e-9); u = np.column_stack((dx, dy)) / r_safe[:, None]
        cos_thr = -np.cos(np.deg2rad(180.0 - float(tol_ang_deg))); mids = []
        for i in range(len(pts_yx)):
            if r[i] < 1e-6: continue
            dots = (u @ u[i]); max_r_pair = np.maximum(r, r[i])
            rel_diff = np.divide(np.abs(r - r[i]), max_r_pair, out=np.zeros_like(r), where=max_r_pair > 1e-9)
            rad_ok = (rel_diff < float(tol_rel_r)) & (max_r_pair > 1e-9); ang_ok = (dots < cos_thr)
            valid_match = rad_ok & ang_ok & (np.arange(len(pts_yx)) != i) & (r > 1e-6)
            idx = np.where(valid_match)[0]
            if idx.size == 0: continue
            j = idx[np.argmin(np.abs(dots[idx] + 1.0))]
            yi, xi = pts_yx[i, 0], pts_yx[i, 1]; yj, xj = pts_yx[j, 0], pts_yx[j, 1]
            mids.append(((yi + yj) / 2.0, (xi + xj) / 2.0))
        if len(mids) < 4: break
        mids = np.array(mids, dtype=float)
        cy = float(np.median(mids[:, 0])); cx = float(np.median(mids[:, 1]))
    return CenterResult(cy=cy, cx=cx, method="antipodal-refined")

def group_points_by_radius(pts_data: np.ndarray, center: Tuple[float, float], radius_tolerance: float = 0.05) -> Dict[int, List[int]]:
    if len(pts_data) == 0: return {}
    cy, cx = center; dy = pts_data[:, 0] - cy; dx = pts_data[:, 1] - cx; radii = np.hypot(dx, dy)
    ring_centers, ring_labels, _ = cluster_rings(radii)
    final_groups = defaultdict(list); group_counter = 0
    if len(ring_centers) > 0:
        for ring_idx in range(len(ring_centers)):
            points_in_ring_mask = (ring_labels == ring_idx); indices_in_ring = np.where(points_in_ring_mask)[0]
            if len(indices_in_ring) == 0: continue
            final_groups[group_counter].extend(indices_in_ring.tolist()); group_counter += 1
    return {k: v for k, v in final_groups.items() if v}


# -------------------------- GUI --------------------------
class SAEDLauncherFrame(ttk.Frame):
    # ... (init, _get_default_output_path) ...
    def __init__(self, master: tk.Misc, controller=None):
        super().__init__(master); self.controller = controller
        self._scroll_canvas = None; self._scroll_window_id = None
        self._build_ui()

    def _get_default_output_path(self) -> str:
        # ... (код без изменений) ...
        if getattr(sys, "frozen", False): base_dir = Path(sys.executable).parent
        else:
            try: base_dir = Path.cwd()
            except OSError: base_dir = Path(__file__).parent
        base_name = "saed_results"; output_path = base_dir / base_name
        if not output_path.exists(): return str(output_path)
        counter = 1
        while True:
            new_name = f"{base_name}_{counter}"; new_path = base_dir / new_name
            if not new_path.exists(): return str(new_path)
            counter += 1;
            if counter > 999: return str(base_dir / f"{base_name}_temp_{np.random.randint(1000)}")

    def _build_ui(self):
        # ... (UI до detect_box) ...
        outer = ttk.Frame(self); outer.pack(fill=tk.BOTH, expand=True)
        fixed = ttk.Frame(outer, padding=(16, 16, 16, 0)); fixed.pack(side=tk.TOP, fill=tk.X)
        fixed.grid_columnconfigure(0, weight=1)
        data_box = ttk.LabelFrame(fixed, text="Input data", padding=(12, 10, 12, 12)); data_box.grid(row=0, column=0, sticky="nsew")
        data_box.grid_columnconfigure(1, weight=1)
        ttk.Label(data_box, text="Image:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.ent_img = ttk.Entry(data_box); self.ent_img.grid(row=0, column=1, columnspan=2, sticky="we", padx=6, pady=4)
        ttk.Button(data_box, text="Browse…", command=self._browse_img).grid(row=0, column=3, sticky="ew", padx=6, pady=4)
        ttk.Label(data_box, text="Output folder:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.ent_out = ttk.Entry(data_box); self.ent_out.insert(0, self._get_default_output_path()); self.ent_out.grid(row=1, column=1, sticky="we", padx=6, pady=4)
        ttk.Button(data_box, text="Load Session…", command=self._load_session).grid(row=1, column=2, sticky="ew", padx=6, pady=4)
        ttk.Button(data_box, text="Choose…", command=self._browse_out).grid(row=1, column=3, sticky="ew", padx=6, pady=4)
        ttk.Label(data_box, text="Center X (optional):").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.ent_cx = ttk.Entry(data_box, width=12); self.ent_cx.grid(row=2, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(data_box, text="Center Y:").grid(row=2, column=2, sticky="w", padx=6, pady=4)
        self.ent_cy = ttk.Entry(data_box, width=12); self.ent_cy.grid(row=2, column=3, sticky="w", padx=6, pady=4)
        ttk.Label(data_box, text="Leave coordinates empty for auto center. Use 'Load Session' to restore.", wraplength=520, foreground="#555555").grid(row=3, column=0, columnspan=4, sticky="we", padx=6, pady=(0, 4))

        pre_box = ttk.LabelFrame(fixed, text="Preprocessing", padding=(12, 10, 12, 12)); pre_box.grid(row=1, column=0, sticky="nsew", pady=(10, 0)); pre_box.grid_columnconfigure(1, weight=1)
        ttk.Label(pre_box, text="Mode:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.cmb_pre = ttk.Combobox(pre_box, values=["No processing", "NLM Denoising"], state="readonly"); self.cmb_pre.current(0); self.cmb_pre.grid(row=0, column=1, sticky="w", padx=6, pady=4)
        self.spn_h_param = self._spin_param(pre_box, 1, "NLM h_param", 0.3, from_=0.01, to=2.0, increment=0.01, format_str="%.2f")
        self.cmb_pre.bind("<<ComboboxSelected>>", self._on_preproc_change)
        ttk.Label(pre_box, text="Select 'NLM Denoising' for noise reduction (uses scikit-image) and adjust 'h_param'.", wraplength=520, foreground="#555555").grid(row=3, column=0, columnspan=3, sticky="we", padx=6, pady=(2, 0))

        scroll_host = ttk.Frame(outer); scroll_host.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(scroll_host, borderwidth=0, highlightthickness=0); vscroll = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=canvas.yview)
        scrollable = ttk.Frame(canvas, padding=(16, 12, 16, 12)); scrollable.grid_columnconfigure(0, weight=1)
        self._scroll_canvas = canvas; self._scroll_window_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(self._scroll_window_id, width=e.width))
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        scrollable.bind("<Enter>", self._activate_scroll); scrollable.bind("<Leave>", self._deactivate_scroll)
        canvas.bind("<Enter>", self._activate_scroll); canvas.bind("<Leave>", self._deactivate_scroll)

        detect_box = ttk.LabelFrame(scrollable, text="Detector and refinement", padding=(12, 10, 12, 12)); detect_box.grid(row=0, column=0, sticky="nsew"); detect_box.grid_columnconfigure(1, weight=1)

        # --- Выбор метода детекции ---
        ttk.Label(detect_box, text="Detection Method:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.cmb_detect_method = ttk.Combobox(
            detect_box, values=["Centroid (Default)", "Legacy (Local Maxima)", "Hybrid (Legacy + Centroid)"], state="readonly" # Обновлен список
        )
        self.cmb_detect_method.current(0); self.cmb_detect_method.grid(row=0, column=1, sticky="w", padx=6, pady=4)
        self.cmb_detect_method.bind("<<ComboboxSelected>>", self._on_detect_method_change)

        # --- Min Peak Distance ---
        self.spn_min_dist = self._spin_param(detect_box, 1, "Min. Peak Distance (px)", 4.0, from_=1.0, to=50.0, increment=0.5, format_str="%.1f")

        ttk.Separator(detect_box).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 6))
        row_offset = 3

        ttk.Label(detect_box, text="Peak threshold and search window", font=("TkDefaultFont", 10, "bold")).grid(row=row_offset + 0, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))
        self.spn_perc = self._spin_param(detect_box, row_offset + 1, "Detection percentile (%)", 99.0, from_=80.0, to=100.0, increment=0.1, format_str="%.1f")
        self.spn_min_area = self._spin_param(detect_box, row_offset + 2, "Min. peak area (px) [Centroid/Hybrid]", 3, from_=1, to=500, increment=1)
        self.spn_maxpts = self._spin_param(detect_box, row_offset + 3, "Maximum detected points", 6000, from_=100, to=20000, increment=100)
        ttk.Separator(detect_box).grid(row=row_offset + 4, column=0, columnspan=2, sticky="ew", pady=(6, 8))
        ttk.Label(detect_box, text="Center refinement", font=("TkDefaultFont", 10, "bold")).grid(row=row_offset + 5, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))
        self.spn_iters = self._spin_param(detect_box, row_offset + 6, "Center refinement iterations", 4, from_=0, to=10, increment=1)
        self.spn_tolang = self._spin_param(detect_box, row_offset + 7, "Antipode tolerance (°)", 8.0, from_=1.0, to=30.0, increment=0.5, format_str="%.1f")
        self.spn_tolr = self._spin_param(detect_box, row_offset + 8, "Radius tolerance (relative)", 0.06, from_=0.01, to=0.5, increment=0.01, format_str="%.2f")
        ttk.Separator(detect_box).grid(row=row_offset + 9, column=0, columnspan=2, sticky="ew", pady=(6, 8))
        ttk.Label(detect_box, text="Geometric filters", font=("TkDefaultFont", 10, "bold")).grid(row=row_offset + 10, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))
        self.spn_dead = self._spin_param(detect_box, row_offset + 11, "Dead zone (px)", 0, from_=0, to=500, increment=1)
        self.spn_search = self._spin_param(detect_box, row_offset + 12, "Search radius (px, 0 = unlimited)", 0, from_=0, to=10000, increment=25)
        self.lbl_detect_hint = ttk.Label(detect_box, text="", wraplength=520, foreground="#555555")
        self.lbl_detect_hint.grid(row=row_offset + 13, column=0, columnspan=2, sticky="we", padx=6, pady=(2, 0))

        action_box = ttk.Frame(scrollable, padding=(0, 12, 0, 0)); action_box.grid(row=1, column=0, sticky="nsew"); action_box.grid_columnconfigure(0, weight=1)
        ttk.Label(action_box, text="Review parameters and press button to switch to interactive editing.", wraplength=540, justify="left").grid(row=0, column=0, sticky="we", padx=4, pady=(0, 8))
        ttk.Button(action_box, text="Open point editor", command=self._go_editor).grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 12))
        filler_bg = ttk.Style().lookup("TFrame", "background") or self.winfo_toplevel().cget("background"); bottom_filler = tk.Frame(scrollable, height=56, bg=filler_bg); bottom_filler.grid(row=2, column=0, sticky="ew"); bottom_filler.grid_propagate(False)

        self._on_preproc_change(None)
        self._on_detect_method_change(None)


    # ... (методы _activate_scroll, _deactivate_scroll, _on_scroll_mousewheel, _on_preproc_change, _on_detect_method_change) ...
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
        if sys.platform == "win32": delta = -int(event.delta / 120)
        elif sys.platform == "darwin": delta = event.delta
        elif event.num == 4: delta = -1
        elif event.num == 5: delta = 1
        if delta != 0: self._scroll_canvas.yview_scroll(delta, "units")

    def _on_preproc_change(self, event=None):
        if not hasattr(self, 'cmb_pre') or not hasattr(self, 'spn_h_param'): return
        try:
            mode = self.cmb_pre.get()
            self.spn_h_param.configure(state='normal' if mode == "NLM Denoising" else 'readonly')
        except tk.TclError: pass

    # --- Обновлен _on_detect_method_change ---
    def _on_detect_method_change(self, event=None):
        if not hasattr(self, 'cmb_detect_method') or not hasattr(self, 'spn_min_area') or not hasattr(self, 'lbl_detect_hint'): return
        try:
            method = self.cmb_detect_method.get()
            is_centroid = "Centroid" in method # True для Centroid и Hybrid
            is_legacy = "Legacy" in method     # True для Legacy и Hybrid
            # Min Area активен для Centroid и Hybrid
            self.spn_min_area.configure(state='normal' if is_centroid else 'readonly')
            # Подсказка
            hint_text = ""
            if is_centroid and not is_legacy: # Только Centroid
                 hint_text = "Centroid: Uses connected components & area filtering. Requires OpenCV."
            elif is_legacy and not is_centroid: # Только Legacy
                 hint_text = "Legacy: Uses local maxima & distance filter. 'Min. peak area' ignored. Requires scikit-image."
            else: # Hybrid
                 hint_text = "Hybrid: Runs Legacy first, then Centroid. Removes Legacy points covered by Centroid blobs. Requires OpenCV & scikit-image."
            self.lbl_detect_hint.configure(text=hint_text)
        except tk.TclError: pass
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
        try: spinbox.set(value)
        except tk.TclError:
            try:
                current_value = spinbox.get()
                if str(current_value) != str(value): spinbox.delete(0, tk.END); spinbox.insert(0, str(value))
            except (tk.TclError, ValueError): print(f"Warning: Could not set spinbox value to {value}")

    def _browse_img(self):
        p = filedialog.askopenfilename(title="Select image", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"), ("All", "*.*")])
        if p: self.ent_img.delete(0, tk.END); self.ent_img.insert(0, p)

    def _browse_out(self):
        p = filedialog.askdirectory(title="Select output folder", mustexist=False)
        if p: self.ent_out.delete(0, tk.END); self.ent_out.insert(0, p)

    def _load_session(self):
        filepath = filedialog.askopenfilename(title="Load SAED Session", filetypes=[("SAED Session", "saed_session.json"), ("All files", "*.*")])
        if not filepath or not self.controller: return
        try:
            self.controller.load_session_from_file(filepath)
            if self.controller: self.controller.set_status(f"Session loaded from {Path(filepath).name}")
        except FileNotFoundError: messagebox.showerror("Load Error", "Session file not found.")
        except Exception as e: messagebox.showerror("Load Error", f"Failed to load session:\n{e}")

    # --- Обновлен get_state/set_state ---
    def get_state(self) -> Dict[str, Any]:
        return {
            "image_path": self.ent_img.get(), "output_folder": self.ent_out.get(),
            "center_x": self.ent_cx.get(), "center_y": self.ent_cy.get(),
            "preproc_mode": self.cmb_pre.get(), "h_param": self.spn_h_param.get(),
            "detect_method": self.cmb_detect_method.get(),
            "min_dist": self.spn_min_dist.get(),
            "detect_perc": self.spn_perc.get(), "min_area": self.spn_min_area.get(),
            "max_pts": self.spn_maxpts.get(), "refine_iters": self.spn_iters.get(),
            "tol_angle": self.spn_tolang.get(), "tol_radius": self.spn_tolr.get(),
            "dead_zone": self.spn_dead.get(), "search_radius": self.spn_search.get(),
        }

    def set_state(self, state: Dict[str, Any]):
        def _set_entry(widget, value):
            if value is not None and isinstance(widget, (ttk.Entry, tk.Entry)): widget.delete(0, tk.END); widget.insert(0, str(value))

        _set_entry(self.ent_img, state.get("image_path"));
        _set_entry(self.ent_out, state.get("output_folder"))
        _set_entry(self.ent_cx, state.get("center_x"));
        _set_entry(self.ent_cy, state.get("center_y"))
        preproc_mode = state.get("preproc_mode")
        if preproc_mode and isinstance(self.cmb_pre, ttk.Combobox):
            if preproc_mode in self.cmb_pre['values']: self.cmb_pre.set(preproc_mode)
            else: print(f"Warning: Saved preproc_mode '{preproc_mode}' not found."); self.cmb_pre.current(0)
        elif isinstance(self.cmb_pre, ttk.Combobox): self.cmb_pre.current(0)
        self._on_preproc_change(None)
        self._set_spinbox_value(self.spn_h_param, state.get("h_param", 0.3))

        detect_method = state.get("detect_method")
        if detect_method and isinstance(self.cmb_detect_method, ttk.Combobox):
             if detect_method in self.cmb_detect_method['values']: self.cmb_detect_method.set(detect_method)
             else: print(f"Warning: Saved detect_method '{detect_method}' not found."); self.cmb_detect_method.current(0)
        elif isinstance(self.cmb_detect_method, ttk.Combobox): self.cmb_detect_method.current(0)
        self._on_detect_method_change(None)

        self._set_spinbox_value(self.spn_min_dist, state.get("min_dist", 4.0))
        self._set_spinbox_value(self.spn_perc, state.get("detect_perc", 99.0))
        self._set_spinbox_value(self.spn_min_area, state.get("min_area", 3))
        self._set_spinbox_value(self.spn_maxpts, state.get("max_pts", 6000))
        self._set_spinbox_value(self.spn_iters, state.get("refine_iters", 4))
        self._set_spinbox_value(self.spn_tolang, state.get("tol_angle", 8.0))
        self._set_spinbox_value(self.spn_tolr, state.get("tol_radius", 0.06))
        self._set_spinbox_value(self.spn_dead, state.get("dead_zone", 0))
        self._set_spinbox_value(self.spn_search, state.get("search_radius", 0))
    # --- КОНЕЦ ---

    # --- ИЗМЕНЕНИЕ: Обновлен _go_editor ---
    def _go_editor(self):
        try:
            image_path_str = self.ent_img.get(); output_dir_str = self.ent_out.get()
            if not image_path_str: messagebox.showerror("Error", "Please select an image file."); return
            if not output_dir_str: messagebox.showerror("Error", "Please specify an output folder."); return
            image_path = Path(image_path_str).expanduser().resolve()
            outdir = Path(output_dir_str).expanduser().resolve(); outdir.mkdir(parents=True, exist_ok=True)
            if not image_path.exists(): messagebox.showerror("Error", f"Image not found at: {image_path}"); return

            perc = float(self.spn_perc.get())
            min_area = int(float(self.spn_min_area.get()))
            max_pts = int(float(self.spn_maxpts.get()))
            min_dist = float(self.spn_min_dist.get()) # <<< Получаем Min Distance
            iters = int(float(self.spn_iters.get()))
            tol_ang = float(self.spn_tolang.get())
            tol_relr = float(self.spn_tolr.get())
            dead_r = float(self.spn_dead.get())
            search_r = float(self.spn_search.get())

            pre_mode = self.cmb_pre.get(); h_param_val = float(self.spn_h_param.get())
            settings = PreprocSettings(mode="nlm", h_param=h_param_val) if pre_mode == "NLM Denoising" else PreprocSettings(mode="raw", h_param=h_param_val)

            try: arr = load_grayscale_with_preproc(image_path, settings)
            except RuntimeError as cv_err: messagebox.showerror("Dependency Error", str(cv_err)); return
            except Exception as img_load_err: messagebox.showerror("Image Error", f"Failed to load/process image:\n{img_load_err}"); return
            preproc_payload = settings.to_json()

            cx_txt = self.ent_cx.get().strip(); cy_txt = self.ent_cy.get().strip()
            if cx_txt and cy_txt:
                try: center0 = CenterResult(cy=float(cy_txt), cx=float(cx_txt), method="user")
                except ValueError: messagebox.showwarning("Input Warning", "Invalid center coords. Using auto."); center0 = geometric_midpoint(arr)
            else: center0 = geometric_midpoint(arr)

            # --- Выбор и вызов метода детекции ---
            detect_method_choice = self.cmb_detect_method.get()
            pts_raw = np.zeros((0, 4), dtype=float) # Инициализация
            try:
                 if "Legacy" in detect_method_choice and "Hybrid" not in detect_method_choice :
                      print(f"Using Legacy detector with min_distance={min_dist}...")
                      pts_raw = detect_spots_legacy(arr, user_perc=perc, max_spots=max_pts, min_distance=int(round(min_dist)))
                 elif "Hybrid" in detect_method_choice:
                      print(f"Using Hybrid detector with min_distance={min_dist}...")
                      pts_raw = detect_spots_hybrid(arr, user_perc=perc, min_area=min_area, max_spots=max_pts, min_distance=min_dist)
                 else: # По умолчанию Centroid
                      print(f"Using Centroid detector with proximity_threshold={min_dist}...")
                      # Centroid возвращает точки и маску, маска здесь не нужна
                      pts_raw, _, _ = detect_spots_by_centroid(arr, user_perc=perc, min_area=min_area, max_spots=max_pts, proximity_threshold=min_dist)
            except RuntimeError as e: messagebox.showerror("Dependency Error", str(e)); return
            # --- Конец вызова ---

            # --- Дальнейшая логика без изменений ---
            if len(pts_raw) == 0:
                messagebox.showwarning("Detection Warning", f"No spots detected with method '{detect_method_choice}'.")
                pts_processed = np.zeros((0, 4), dtype=float); point_types = {}; center = center0; dominant_symmetry = 0
            else:
                center = refine_center_antipodal((center0.cy, center0.cx), pts_raw, tol_ang_deg=tol_ang, tol_rel_r=tol_relr, iters=iters)
                cy, cx = center.cy, center.cx; pts_r, pts_a = pol_from((cy, cx), pts_raw[:, :2])
                ring_means, _, _ = cluster_rings(pts_r[pts_r > dead_r]); sym_scores = symmetry_scores(pts_a, pts_r, ring_means)
                dominant_symmetry = 0
                if sym_scores: best_sym_str = max(sym_scores, key=sym_scores.get); dominant_symmetry = int(best_sym_str.split('-')[0])
                else: print("Could not determine dominant symmetry.")
                pts_processed = pts_raw.copy(); point_types = {i: "unknown" for i in range(len(pts_processed))}
                print("Point classification disabled in launcher.")

            if (dead_r > 0 or search_r > 0) and len(pts_processed) > 0:
                dy = pts_processed[:, 0] - center.cy; dx = pts_processed[:, 1] - center.cx; r = np.hypot(dx, dy)
                mask = np.ones(len(pts_processed), dtype=bool)
                if dead_r > 0:   mask &= (r >= dead_r)
                if search_r > 0: mask &= (r <= search_r)
                indices_to_keep = np.where(mask)[0]; pts_processed = pts_processed[indices_to_keep]
                point_types = {new_idx: "unknown" for new_idx, old_idx in enumerate(indices_to_keep)}
            else: point_types = {i: "unknown" for i in range(len(pts_processed))}

            points_list_for_json = []
            for i, (y, x, v, area) in enumerate(pts_processed):
                pt_type = point_types.get(i, "unknown")
                points_list_for_json.append({"y": float(y), "x": float(x), "intensity": float(v), "area": int(area), "type": pt_type})

            saed_input_data = {
                "image": str(image_path), "preproc_mode": settings.mode, "preproc": preproc_payload,
                "center": {"x": float(center.cx), "y": float(center.cy), "method": center.method},
                "radii": {"dead": float(dead_r), "search": float(search_r)}, "points": points_list_for_json
            }
            saed_input_path = outdir / "saed_input.json"
            saed_input_path.write_text(json.dumps(saed_input_data, ensure_ascii=False, indent=2), encoding="utf-8")

            if self.controller is not None:
                try: self.controller.open_editor(saed_input_path)
                except Exception as exc: messagebox.showerror("Error", f"Failed to open editor tab:\n{exc}")
            else: messagebox.showwarning("Standalone Mode", "Running standalone. Editor will open externally if available.")

            try:
                (outdir / "center_init.json").write_text(json.dumps({
                    "initial": {"x": center0.cx, "y": center0.cy, "method": center0.method},
                    "refined": {"x": center.cx, "y": center.cy, "method": center.method},
                    "dominant_symmetry": dominant_symmetry, "dead_zone_px": dead_r, "search_radius_px": search_r,
                    "preproc_mode": settings.mode, "preproc": preproc_payload,
                    "image_size": {"H": int(arr.shape[0]), "W": int(arr.shape[1])},
                    "detection_method": detect_method_choice,
                    "min_peak_distance_px": min_dist
                }, indent=2), encoding="utf-8")
            except Exception as log_err: print(f"Warning: Could not save center_init.json - {log_err}")

        except Exception as e:
            messagebox.showerror("Processing Error", f"An unexpected error occurred:\n{e}")
            import traceback; traceback.print_exc()
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---


# ... (класс SAEDApp и __main__ без изменений) ...
class SAEDApp(tk.Tk):
    def __init__(self):
        super().__init__(); self.title("SAED Symmetry – Launcher"); self.geometry("980x680"); self.resizable(True, False)
        frame = SAEDLauncherFrame(self); frame.pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    SAEDApp().mainloop()