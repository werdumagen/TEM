from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_percentile_map(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a percentile map (0..100) for the given array.

    The returned tuple contains:
      * the array of the same shape with values mapped to a 0..100 scale,
      * sorted unique intensity values found in ``arr``,
      * the corresponding percentile values for those intensities.
    """
    arr = np.asarray(arr, dtype=float)
    flat = arr.ravel()
    n = flat.size
    if n == 0:
        empty = np.zeros_like(arr, dtype=float)
        return empty, np.array([], dtype=float), np.array([], dtype=float)
    if n == 1:
        percent_map = np.full_like(arr, 100.0, dtype=float)
        return percent_map, np.array([float(flat[0])], dtype=float), np.array([100.0], dtype=float)

    order = np.argsort(flat, kind="mergesort")
    sorted_vals = flat[order]
    rank_positions = np.arange(1, n + 1, dtype=float)
    averaged = rank_positions.copy()

    i = 0
    while i < n:
        j = i + 1
        val = sorted_vals[i]
        while j < n and sorted_vals[j] == val:
            j += 1
        if j - i > 1:
            mean_rank = float(rank_positions[i:j].mean())
            averaged[i:j] = mean_rank
        i = j

    ranks = np.empty(n, dtype=float)
    ranks[order] = averaged
    rank_min = float(ranks.min())
    rank_max = float(ranks.max())

    if not np.isfinite(rank_min) or not np.isfinite(rank_max) or abs(rank_max - rank_min) < 1e-12:
        percent_flat = np.full(n, 100.0, dtype=float)
    else:
        percent_flat = (ranks - rank_min) / (rank_max - rank_min) * 100.0

    percent_map = percent_flat.reshape(arr.shape)

    unique_vals = []
    unique_percents = []
    i = 0
    while i < n:
        j = i + 1
        val = sorted_vals[i]
        while j < n and sorted_vals[j] == val:
            j += 1
        percent_val = float(percent_flat[order[i]])
        unique_vals.append(float(val))
        unique_percents.append(percent_val)
        i = j

    return (
        percent_map,
        np.array(unique_vals, dtype=float),
        np.array(unique_percents, dtype=float),
    )


def map_values_to_percent(
    values: np.ndarray,
    unique_vals: np.ndarray,
    unique_percents: np.ndarray,
) -> np.ndarray:
    """Map raw intensity values to the 0..100 percentile scale."""
    values = np.asarray(values, dtype=float)
    if unique_vals.size == 0:
        return np.zeros_like(values, dtype=float)
    if unique_vals.size == 1:
        return np.full_like(values, float(unique_percents[0]), dtype=float)
    mapped = np.interp(
        values,
        unique_vals,
        unique_percents,
        left=float(unique_percents[0]),
        right=float(unique_percents[-1]),
    )
    return np.clip(mapped, 0.0, 100.0)