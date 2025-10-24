#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Утилиты для анализа симметрии (извлечено из saed_editor.py)
"""
import numpy as np
from scipy.signal import find_peaks


# ------ симметрия (для отчёта) ------
def pol_from(center, pts):
    cy, cx = center
    dy, dx = pts[:, 0] - cy, pts[:, 1] - cx
    r = np.hypot(dx, dy)
    a = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    return r, a


def cluster_rings(radii):
    if len(radii) == 0:
        return np.array([]), np.zeros(0, dtype=int), ([], [])  # Return empty int array for labels
    hist, edges = np.histogram(radii, bins=60)
    centers = (edges[:-1] + edges[1:]) / 2

    if hist.max() > 0:
        prominence = max(3, hist.max() * 0.05)  # Adjust prominence based on data
    else:
        prominence = 3
    pk, _ = find_peaks(hist, prominence=prominence)

    ring_centers = centers[pk]
    if len(ring_centers) == 0:
        return np.array([]), np.zeros_like(radii, dtype=int), (hist.tolist(), edges.tolist())  # Return numpy int array

    # Assign each point to the nearest ring center
    labels = np.argmin(np.abs(radii[:, None] - ring_centers[None, :]), axis=1)
    return ring_centers, labels, (hist.tolist(), edges.tolist())  # Convert hist/edges for JSON


def symmetry_scores(angles, radii, ring_means, top_rings=3):
    out = {}
    if not ring_means or len(ring_means) == 0:  # Check if list is empty
        return out

    effective_top_rings = min(top_rings, len(ring_means))
    if effective_top_rings == 0: return out  # No rings to analyze

    idx = effective_top_rings - 1
    maxR = ring_means[idx] * 1.15  # Use 1.15 multiplier as before

    mask = radii <= maxR
    ang_sel = angles[mask]
    if len(ang_sel) == 0: return out  # No points selected

    # Calculate scores for different folds
    for k in [4, 6, 8, 10, 12]:
        period = 360.0 / k
        phases_deg = (ang_sel % period) * k
        phases_rad = np.deg2rad(phases_deg)
        C = np.cos(phases_rad).mean();
        S = np.sin(phases_rad).mean()
        out[f"{k}-fold"] = float(np.hypot(C, S))

    return out