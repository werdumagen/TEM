#!/usr/bin/env python3  # 1
# -*- coding: utf-8 -*-  # 2
(  # 3
    "SAED Symmetry – Launcher\n"  # 4
    "========================\n"  # 5
    "What's new:\n"  # 6
    "  • Preprocessing switch:\n"  # 7
    "      - 'Standard' (equalize + GaussianBlur)\n"  # 8
    "      - 'No processing' (raw grayscale)\n"  # 9
    "      - 'CLAHE' (local equalization) with configurable clipLimit and tile size.\n"  # 10
    "  • All preprocessing happens ONLY here; the image editor is untouched.\n"  # 11
    "The rest of the functionality is unchanged: manual/auto center, antipodal refinement, dead zone, search radius, launching the editor.\n"  # 12
    "\n"  # 13
    "Additionally:\n"  # 14
    "  • Save points together with their intensity in a single saed_input.json file.\n"  # 15
    "  • saed_editor launches with a single --input argument (path to saed_input.json).\n"  # 16
)  # 17
from __future__ import annotations  # 18
import json, subprocess, sys, cv2  # 19
from pathlib import Path  # 20
from dataclasses import dataclass  # 21
from typing import Tuple, Dict, Any  # 22
# 23
import numpy as np  # 24
# 25
from percentile_utils import compute_percentile_map  # 26
from preproc import PreprocSettings, load_grayscale_with_preproc  # 27
# 28
import tkinter as tk  # 29
from tkinter import ttk, filedialog, messagebox  # 30


# 31
# -------------------------- Algorithm --------------------------  # 32
@dataclass  # 33
class CenterResult:  # 34
    cy: float  # 35
    cx: float  # 36
    method: str  # 37


# 38
def detect_spots_by_centroid(  # 39
        arr: np.ndarray,  # 40
        perc: float = 99.0,  # 41
        min_area: int = 3,  # 42
        max_spots: int = 6000,  # 43
) -> np.ndarray:  # 44
    """
    Detects spots using centroiding of connected components (blobs).
    This is more robust for flat or saturated peaks than local maxima search.
    """
    H, W = arr.shape  # 45
    # 46
    # 1. Use percentile map to get a robust threshold value
    percent_map, _, _ = compute_percentile_map(arr)  # 47
    try:  # 48
        perc_val = float(np.clip(perc, 0.0, 100.0))  # 49
        th_value = float(np.percentile(percent_map, perc_val))  # 50
    except (ValueError, IndexError):  # 51
        th_value = 99.0  # Fallback
    # 52
    # 2. Create binary mask based on threshold
    # We use the percentile map as input as it's already contrast-enhanced
    binary_mask = np.where(percent_map >= th_value, 255, 0).astype(np.uint8)  # 53
    # 54
    # 3. Find all connected components ("islands" or "blobs")
    num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(  # 55
        binary_mask,  # 56
        connectivity=8  # Use 8-way connectivity
    )  # 57
    # 58
    kept = []  # 59
    # 60
    # 4. Iterate over all found labels (label 0 is the background, skip it)
    for i in range(1, num_labels):  # 61
        area = stats[i, cv2.CC_STAT_AREA]  # 62
        # 63
        # 5. Filter blobs ONLY by MINIMUM area (to remove noise)
        if area < min_area:  # 64
            continue  # Skip blobs that are too small
        # 65
        # 6. Get the centroid (cx, cy)
        cx, cy = centroids[i]  # 66
        # 67
        # 7. Get intensity at the centroid position for sorting
        yi, xi = int(round(cy)), int(round(cx))  # 68
        if 0 <= yi < H and 0 <= xi < W:  # 69
            # Use intensity from percentile map
            v = float(percent_map[yi, xi])  # 70
            kept.append((float(cy), float(cx), float(v)))  # Store as (y, x, v)
    # 71
    # 8. Sort by brightness (highest first)
    kept.sort(key=lambda t: -t[2])  # 72
    # 73
    # 9. Limit to max_spots
    if len(kept) > max_spots:  # 74
        kept = kept[:max_spots]  # 75
    # 76
    return np.array(kept, dtype=float) if kept else np.zeros((0, 3), dtype=float)  # 77


# 78
# 79
def merge_spots_by_intensity(  # 80
        pts: np.ndarray,  # 81
        radius: float,  # 82
        tol_percent: float,  # 83
        *,  # 84
        min_intensity: float | None = None,  # 85
        line_image: np.ndarray | None = None,  # 86
        percentile_map: np.ndarray | None = None,  # 87
) -> np.ndarray:  # 88
    # ... (function code remains unchanged) ... # 106
    if pts.size == 0:  # 107
        return pts  # 108
    radius = float(radius)  # 109
    tol = max(0.0, float(tol_percent) / 100.0)  # 110
    if radius <= 0.0:  # 111
        return np.asarray(pts, dtype=float)  # 112
    # 113
    pts = np.asarray(pts, dtype=float)  # 114
    if min_intensity is not None:  # 115
        mask = pts[:, 2] >= float(min_intensity)  # 116
    else:  # 117
        mask = np.ones(len(pts), dtype=bool)  # 118
    # 119
    to_merge = pts[mask]  # 120
    untouched = pts[~mask]  # 121
    if to_merge.size == 0:  # 122
        return pts  # 123
    # 124
    intensity_map: np.ndarray | None  # 125
    if percentile_map is not None:  # 126
        intensity_map = np.asarray(percentile_map, dtype=float)  # 127
    elif line_image is not None:  # 128
        percent, _, _ = compute_percentile_map(np.asarray(line_image, dtype=float))  # 129
        intensity_map = percent  # 130
    else:  # 131
        intensity_map = None  # 132
    # 133
    rad2 = radius * radius  # 134
    used = np.zeros(len(to_merge), dtype=bool)  # 135
    order = np.argsort(-to_merge[:, 2])  # start from the brightest  # 136
    merged: list[tuple[float, float, float]] = []  # 137

    # 138
    def within_tol(a: float, b: float) -> bool:  # 139
        hi = max(a, b)  # 140
        if hi == 0.0:  # 141
            return abs(a - b) == 0.0  # 142
        return abs(a - b) <= tol * hi + 1e-12  # 143

    # 144
    H = W = None  # 145
    if intensity_map is not None and intensity_map.ndim == 2:  # 146
        H, W = intensity_map.shape  # 147
    else:  # 148
        intensity_map = None  # 149

    # 150
    def clamp_round(val: float, hi: int) -> int:  # 151
        return int(min(max(round(float(val)), 0), hi))  # 152

    # 153
    def bresenham_line(y0: int, x0: int, y1: int, x1: int) -> list[tuple[int, int]]:  # 154
        points: list[tuple[int, int]] = []  # 155
        dy = abs(y1 - y0)  # 156
        dx = abs(x1 - x0)  # 157
        sy = 1 if y0 < y1 else -1  # 158
        sx = 1 if x0 < x1 else -1  # 159
        err = dx - dy  # 160
        while True:  # 161
            points.append((y0, x0))  # 162
            if y0 == y1 and x0 == x1:  # 163
                break  # 164
            e2 = err * 2  # 165
            if e2 > -dy:  # 166
                err -= dy  # 167
                x0 += sx  # 168
            if e2 < dx:  # 169
                err += dx  # 170
                y0 += sy  # 171
        return points  # 172

    # 173
    def has_intensity_dip(idx_a: int, idx_b: int) -> bool:  # 174
        if intensity_map is None or H is None or W is None:  # 175
            return False  # 176
        base_val = min(float(to_merge[idx_a, 2]), float(to_merge[idx_b, 2]))  # 177
        if base_val <= 0.0:  # 178
            return False  # 179
        y0 = clamp_round(to_merge[idx_a, 0], H - 1)  # 180
        x0 = clamp_round(to_merge[idx_a, 1], W - 1)  # 181
        y1 = clamp_round(to_merge[idx_b, 0], H - 1)  # 182
        x1 = clamp_round(to_merge[idx_b, 1], W - 1)  # 183
        pixels = bresenham_line(y0, x0, y1, x1)  # 184
        if len(pixels) <= 2:  # 185
            return False  # 186
        limit = base_val * (1.0 - tol)  # 187
        for (yy, xx) in pixels[1:-1]:  # 188
            if 0 <= yy < H and 0 <= xx < W:  # 189
                if float(intensity_map[yy, xx]) + 1e-9 < limit:  # 190
                    return True  # 191
        return False  # 192

    # 193
    for idx in order:  # 194
        if used[idx]:  # 195
            continue  # 196
        # 197
        cluster = [idx]  # 198
        sum_y = float(to_merge[idx, 0])  # 199
        sum_x = float(to_merge[idx, 1])  # 200
        intensities = [float(to_merge[idx, 2])]  # 201
        sum_v = intensities[0]  # 202
        # 203
        neighbors = []  # 204
        base_y, base_x = to_merge[idx, 0], to_merge[idx, 1]  # 205
        for j in range(len(to_merge)):  # 206
            if j == idx or used[j]:  # 207
                continue  # 208
            dy = to_merge[j, 0] - base_y  # 209
            dx = to_merge[j, 1] - base_x  # 210
            if dy * dy + dx * dx <= rad2:  # 211
                neighbors.append(j)  # 212
        # 213
        neighbors.sort(key=lambda j: abs(to_merge[j, 2] - intensities[0]))  # 214
        # 215
        for j in neighbors:  # 216
            if used[j]:  # 217
                continue  # 218
            if any(has_intensity_dip(existing, j) for existing in cluster):  # 219
                continue  # 220
            cand_v = float(to_merge[j, 2])  # 221
            new_count = len(cluster) + 1  # 222
            new_avg_v = (sum_v + cand_v) / new_count  # 223
            if all(within_tol(new_avg_v, val) for val in (*intensities, cand_v)):  # 224
                cluster.append(j)  # 225
                intensities.append(cand_v)  # 226
                sum_y += float(to_merge[j, 0])  # 227
                sum_x += float(to_merge[j, 1])  # 228
                sum_v += cand_v  # 229
        # 230
        if len(cluster) > 1:  # 231
            new_count = len(cluster)  # 232
            new_y = sum_y / new_count  # 233
            new_x = sum_x / new_count  # 234
            new_v = sum_v / new_count  # 235
            if (min_intensity is None or new_v >= min_intensity) and all(  # 236
                    within_tol(new_v, val) for val in intensities  # 237
            ):  # 238
                merged.append((new_y, new_x, new_v))  # 239
                for j in cluster:  # 240
                    used[j] = True  # 241
                continue  # 242
        # 243
        # either a single-point cluster or the resulting intensity exceeded the tolerance  # 244
        for j in cluster:  # 245
            if not used[j]:  # 246
                merged.append(tuple(to_merge[j]))  # 247
                used[j] = True  # 248
    # 249
    merged = np.array(merged, dtype=float)  # 250
    if untouched.size == 0:  # 251
        return merged  # 252
    if merged.size == 0:  # 253
        return untouched  # 254
    return np.vstack((merged, untouched))  # 255


# 256
# 257
def geometric_midpoint(arr: np.ndarray) -> CenterResult:  # 258
    H, W = arr.shape  # 259
    return CenterResult(cy=(H - 1) / 2.0, cx=(W - 1) / 2.0, method="midpoint")  # 260


def refine_center_antipodal(center: Tuple[float, float], pts: np.ndarray, tol_ang_deg: float = 8.0,
                            tol_rel_r: float = 0.06, iters: int = 3) -> CenterResult:  # 261
    cy, cx = float(center[0]), float(center[1])  # 262
    if len(pts) < 4:  # 263
        return CenterResult(cy=cy, cx=cx, method="midpoint (fallback)")  # 264
    for _ in range(max(0, int(iters))):  # 265
        dy = pts[:, 0] - cy;
        dx = pts[:, 1] - cx  # 266
        r = np.hypot(dx, dy)  # 267
        # Avoid division by zero if a point is exactly at the center
        r_safe = np.where(r > 1e-9, r, 1e-9)
        u = np.column_stack((dx, dy)) / r_safe[:, None]  # 268
        cos_thr = -np.cos(np.deg2rad(180.0 - float(tol_ang_deg)))  # 269
        mids = []  # 270
        for i in range(len(pts)):  # 271
            if r[i] < 1e-6: continue  # Skip point if it's too close to center
            dots = (u @ u[i])  # 272
            # Avoid division by zero for radius tolerance
            max_r_pair = np.maximum(r, r[i])
            # Use np.divide with where clause to handle potential zero denominators
            rel_diff = np.divide(np.abs(r - r[i]), max_r_pair, out=np.zeros_like(r), where=max_r_pair > 1e-9)
            rad_ok = (rel_diff < float(tol_rel_r)) & (max_r_pair > 1e-9)  # Ensure we don't match zero-radius points

            ang_ok = (dots < cos_thr)  # 274
            # Exclude self-comparison and points too close to center
            valid_match = rad_ok & ang_ok & (np.arange(len(pts)) != i) & (r > 1e-6)
            idx = np.where(valid_match)[0]  # 275
            if idx.size == 0: continue  # 276
            # Find the best antipodal match among valid candidates
            j = idx[np.argmin(np.abs(dots[idx] + 1.0))]  # 277
            yi, xi = pts[i, 0], pts[i, 1]  # 278
            yj, xj = pts[j, 0], pts[j, 1]  # 279
            mids.append(((yi + yj) / 2.0, (xi + xj) / 2.0))  # 280
        if len(mids) < 4: break  # 281 Not enough pairs found
        mids = np.array(mids, dtype=float)  # 282
        # Use median to be robust against outliers
        cy = float(np.median(mids[:, 0]));
        cx = float(np.median(mids[:, 1]))  # 283
    return CenterResult(cy=cy, cx=cx, method="antipodal-refined")  # 284


# 285
# -------------------------- GUI --------------------------  # 286
class SAEDLauncherFrame(ttk.Frame):  # 287
    (  # 288
        "Launcher tab suitable for both the standalone application and notebooks.\n"  # 289
    )  # 290

    # 291
    def __init__(self, master: tk.Misc, controller=None):  # 292
        super().__init__(master)  # 293
        self.controller = controller  # 294
        self._scroll_canvas = None  # 295
        self._scroll_window_id = None  # 296
        self._build_ui()  # 297

    # 298
    def _get_default_output_path(self) -> str:
        """Generates a default output path, avoiding existing directories."""
        if getattr(sys, "frozen", False):
            # For a compiled .exe, use the directory where it's located
            base_dir = Path(sys.executable).parent
        else:
            # For development, use the current working directory or script dir
            try:
                base_dir = Path.cwd()
            except OSError:
                base_dir = Path(__file__).parent

        base_name = "saed_results"
        output_path = base_dir / base_name

        if not output_path.exists():
            return str(output_path)

        # If the base path exists, find a new one by appending a number
        counter = 1
        while True:
            new_name = f"{base_name}_{counter}"
            new_path = base_dir / new_name
            if not new_path.exists():
                return str(new_path)
            counter += 1
            if counter > 999:  # Safety break
                return str(base_dir / f"{base_name}_temp_{np.random.randint(1000)}")

    def _build_ui(self):  # 299
        outer = ttk.Frame(self)  # 300
        outer.pack(fill=tk.BOTH, expand=True)  # 301
        # 302
        fixed = ttk.Frame(outer, padding=(16, 16, 16, 0))  # 303
        fixed.pack(side=tk.TOP, fill=tk.X)  # 304
        fixed.grid_columnconfigure(0, weight=1)  # 305
        # 306
        data_box = ttk.LabelFrame(fixed, text="Input data", padding=(12, 10, 12, 12))  # 307
        data_box.grid(row=0, column=0, sticky="nsew")  # 308
        # 309
        data_box.grid_columnconfigure(1, weight=1)  # Make entry widgets resizable
        # 312
        ttk.Label(data_box, text="Image:").grid(row=0, column=0, sticky="w", padx=6, pady=4)  # 313
        self.ent_img = ttk.Entry(data_box)  # 314
        self.ent_img.grid(row=0, column=1, columnspan=2, sticky="we", padx=6, pady=4)  # 315
        ttk.Button(data_box, text="Browse…", command=self._browse_img).grid(row=0, column=3, sticky="ew", padx=6,
                                                                            pady=4)  # 316
        # 317
        ttk.Label(data_box, text="Output folder:").grid(row=1, column=0, sticky="w", padx=6, pady=4)  # 318
        self.ent_out = ttk.Entry(data_box)  # 319
        self.ent_out.insert(0, self._get_default_output_path())  # 320
        self.ent_out.grid(row=1, column=1, sticky="we", padx=6, pady=4)  # 321
        # --- NEW: "Load Session" Button ---
        ttk.Button(data_box, text="Load Session…", command=self._load_session).grid(row=1, column=2, sticky="ew",
                                                                                    padx=6, pady=4)  # 322
        ttk.Button(data_box, text="Choose…", command=self._browse_out).grid(row=1, column=3, sticky="ew", padx=6,
                                                                            pady=4)  # 323
        # 324
        ttk.Label(data_box, text="Center X (optional):").grid(row=2, column=0, sticky="w", padx=6, pady=4)  # 325
        self.ent_cx = ttk.Entry(data_box, width=12)  # 326
        self.ent_cx.grid(row=2, column=1, sticky="w", padx=6, pady=4)  # 327
        ttk.Label(data_box, text="Center Y:").grid(row=2, column=2, sticky="w", padx=6, pady=4)  # 328
        self.ent_cy = ttk.Entry(data_box, width=12)  # 329
        self.ent_cy.grid(row=2, column=3, sticky="w", padx=6, pady=4)  # 330
        # 331
        ttk.Label(  # 332
            data_box,  # 333
            text="Leave the coordinates empty to let the program find the center automatically. Use 'Load Session' to restore a previous state.",
            # 334
            wraplength=520,  # 335
            foreground="#555555"  # 336
        ).grid(row=3, column=0, columnspan=4, sticky="we", padx=6, pady=(0, 4))  # 337
        # 338
        pre_box = ttk.LabelFrame(fixed, text="Preprocessing", padding=(12, 10, 12, 12))  # 339
        pre_box.grid(row=1, column=0, sticky="nsew", pady=(10, 0))  # 340
        pre_box.grid_columnconfigure(1, weight=1)  # 341
        # 342
        ttk.Label(pre_box, text="Mode:").grid(row=0, column=0, sticky="w", padx=6, pady=4)  # 343
        self.cmb_pre = ttk.Combobox(  # 344
            pre_box,  # 345
            values=["No smoothing", "Standard", "CLAHE"],  # 346
            state="readonly",  # 347
        )  # 348
        self.cmb_pre.current(0)  # 349
        self.cmb_pre.grid(row=0, column=1, sticky="w", padx=6, pady=4)  # 350
        self.cmb_pre.bind("<<ComboboxSelected>>", self._on_preproc_change)  # 351
        # 352
        ttk.Label(pre_box, text="CLAHE clipLimit / tile:").grid(row=1, column=0, sticky="w", padx=6, pady=4)  # 353
        self.spn_clip = ttk.Spinbox(pre_box, from_=0.1, to=10.0, increment=0.1, width=8, justify="right")  # 354
        self._set_spinbox_value(self.spn_clip, 1.5)  # 355
        self.spn_clip.grid(row=1, column=1, sticky="w", padx=6, pady=4)  # 356
        self.spn_tile = ttk.Spinbox(pre_box, from_=2, to=64, increment=1, width=8, justify="right")  # 357
        self._set_spinbox_value(self.spn_tile, 8)  # 358
        self.spn_tile.grid(row=1, column=2, sticky="w", padx=6, pady=4)  # 359
        # 360
        ttk.Label(  # 361
            pre_box,  # 362
            text="Select CLAHE for images with strong brightness variations. ClipLimit controls contrast, and tile size defines the local processing radius.",
            # 363
            wraplength=520,  # 364
            foreground="#555555"  # 365
        ).grid(row=2, column=0, columnspan=3, sticky="we", padx=6, pady=(2, 0))  # 366
        # 367
        scroll_host = ttk.Frame(outer)  # 368
        scroll_host.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  # 369
        # 370
        canvas = tk.Canvas(scroll_host, borderwidth=0, highlightthickness=0)  # 371
        vscroll = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=canvas.yview)  # 372
        scrollable = ttk.Frame(canvas, padding=(16, 12, 16, 12))  # 373
        scrollable.grid_columnconfigure(0, weight=1)  # 374
        # 375
        self._scroll_canvas = canvas  # 376
        self._scroll_window_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")  # 377
        canvas.configure(yscrollcommand=vscroll.set)  # 378
        # 379
        scrollable.bind(  # 380
            "<Configure>",  # 381
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))  # 382
        )  # 383
        canvas.bind(  # 384
            "<Configure>",  # 385
            lambda e: canvas.itemconfigure(self._scroll_window_id, width=e.width)  # 386
        )  # 387
        # 388
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # 389
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)  # 390
        # 391
        scrollable.bind("<Enter>", self._activate_scroll)  # 392
        scrollable.bind("<Leave>", self._deactivate_scroll)  # 393
        canvas.bind("<Enter>", self._activate_scroll)  # 394
        canvas.bind("<Leave>", self._deactivate_scroll)  # 395
        # 396
        detect_box = ttk.LabelFrame(scrollable, text="Detector and refinement", padding=(12, 10, 12, 12))  # 397
        detect_box.grid(row=0, column=0, sticky="nsew")  # 398
        detect_box.grid_columnconfigure(1, weight=1)  # 399
        # 400
        ttk.Label(  # 401
            detect_box,  # 402
            text="Peak threshold and search window",  # 403
            font=("TkDefaultFont", 10, "bold")  # 404
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))  # 405
        self.spn_perc = self._spin_param(  # 406
            detect_box, 1, "Detection percentile (%)", 99.0,  # 407
            from_=80.0, to=100.0, increment=0.1, format_str="%.1f"  # 408
        )  # 409
        self.spn_merge_perc = self._spin_param(  # 410
            detect_box, 2, "Intensity percentile for merging (%)", 95.0,  # 411
            from_=0.0, to=100.0, increment=0.5, format_str="%.1f"  # 412
        )  # 413
        self.spn_merge_rad = self._spin_param(  # 414
            detect_box, 3, "Peak merging radius (px)", 0,  # 415
            from_=0, to=50, increment=1  # 416
        )  # 417
        self.spn_merge_tol = self._spin_param(  # 418
            detect_box, 4, "Intensity similarity tolerance (%)", 10.0,  # 419
            from_=0.0, to=100.0, increment=0.5, format_str="%.1f"  # 420
        )  # 421
        # --- MODIFIED: Replaced min_sep/max_area with min_area ---
        self.spn_min_area = self._spin_param(  # 422
            detect_box, 5, "Min. peak area (px)", 3,  # 423
            from_=1, to=500, increment=1  # 424
        )  # 425
        self.spn_maxpts = self._spin_param(  # 426
            detect_box, 6, "Maximum detected points", 6000,  # 427
            from_=100, to=20000, increment=100  # 428
        )  # 429
        # 430
        ttk.Separator(detect_box).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(6, 8))  # 431
        # 432
        ttk.Label(  # 433
            detect_box,  # 434
            text="Center refinement",  # 435
            font=("TkDefaultFont", 10, "bold")  # 436
        ).grid(row=9, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))  # 437
        self.spn_iters = self._spin_param(  # 438
            detect_box, 10, "Center refinement iterations", 4,  # 439
            from_=0, to=10, increment=1  # 440
        )  # 441
        self.spn_tolang = self._spin_param(  # 442
            detect_box, 11, "Antipode tolerance (°)", 8.0,  # 443
            from_=1.0, to=30.0, increment=0.5, format_str="%.1f"  # 444
        )  # 445
        self.spn_tolr = self._spin_param(  # 446
            detect_box, 12, "Radius tolerance (relative)", 0.06,  # 447
            from_=0.01, to=0.5, increment=0.01, format_str="%.2f"  # 448
        )  # 449
        # 450
        ttk.Separator(detect_box).grid(row=13, column=0, columnspan=2, sticky="ew", pady=(6, 8))  # 451
        # 452
        ttk.Label(  # 453
            detect_box,  # 454
            text="Geometric filters",  # 455
            font=("TkDefaultFont", 10, "bold")  # 456
        ).grid(row=14, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))  # 457
        self.spn_dead = self._spin_param(  # 458
            detect_box, 15, "Dead zone (px)", 0,  # 459
            from_=0, to=500, increment=1  # 460
        )  # 461
        self.spn_search = self._spin_param(  # 462
            detect_box, 16, "Search radius (px, 0 = unlimited)", 0,  # 463
            from_=0, to=10000, increment=25  # 464
        )  # 465
        # 466
        ttk.Label(  # 467
            detect_box,  # 468
            text="Use 'Min. peak area' to filter noise. Use 'Dead zone' to filter the central beam by its position.",
            # 469
            wraplength=520,  # 470
            foreground="#555555"  # 471
        ).grid(row=17, column=0, columnspan=2, sticky="we", padx=6, pady=(2, 0))  # 472
        # 473
        action_box = ttk.Frame(scrollable, padding=(0, 12, 0, 0))  # 474
        action_box.grid(row=1, column=0, sticky="nsew")  # 475
        action_box.grid_columnconfigure(0, weight=1)  # 476
        # 477
        ttk.Label(  # 478
            action_box,  # 479
            text="Review the parameters and press the button below to switch to interactive editing of detected points.",
            # 480
            wraplength=540,  # 481
            justify="left"  # 482
        ).grid(row=0, column=0, sticky="we", padx=4, pady=(0, 8))  # 483
        # 484
        ttk.Button(action_box, text="Open point editor", command=self._go_editor).grid(  # 485
            row=1, column=0, sticky="ew", padx=4, pady=(0, 12)  # 486
        )  # 487
        # 488
        filler_bg = ttk.Style().lookup("TFrame", "background") or self.winfo_toplevel().cget("background")  # 489
        bottom_filler = tk.Frame(scrollable, height=56, bg=filler_bg)  # 490
        bottom_filler.grid(row=2, column=0, sticky="ew")  # 491
        bottom_filler.grid_propagate(False)  # 492
        # 493
        self._on_preproc_change(None)  # 494

    # 495
    def _activate_scroll(self, _event):  # 496
        if self._scroll_canvas is None:  # 497
            return  # 498
        self._scroll_canvas.bind_all("<MouseWheel>", self._on_scroll_mousewheel)  # 499
        self._scroll_canvas.bind_all("<Button-4>", self._on_scroll_mousewheel)  # 500
        self._scroll_canvas.bind_all("<Button-5>", self._on_scroll_mousewheel)  # 501

    # 502
    def _deactivate_scroll(self, _event):  # 503
        if self._scroll_canvas is None:  # 504
            return  # 505
        self._scroll_canvas.unbind_all("<MouseWheel>")  # 506
        self._scroll_canvas.unbind_all("<Button-4>")  # 507
        self._scroll_canvas.unbind_all("<Button-5>")  # 508

    # 509
    def _on_scroll_mousewheel(self, event):  # 510
        if self._scroll_canvas is None:  # 511
            return  # 512
        # Determine scroll direction and amount (platform-dependent)
        delta = 0
        if sys.platform == "win32":
            delta = -int(event.delta / 120)
        elif sys.platform == "darwin":  # macOS
            delta = event.delta
        elif event.num == 4:  # Linux scroll up
            delta = -1
        elif event.num == 5:  # Linux scroll down
            delta = 1

        if delta != 0:
            self._scroll_canvas.yview_scroll(delta, "units")

    def _on_preproc_change(self, _evt):  # 523
        mode = self.cmb_pre.get()  # 524
        clahe_enabled = (mode == "CLAHE")  # 525
        state = "normal" if clahe_enabled else "disabled"  # 526
        self.spn_clip.configure(state=state)  # 527
        self.spn_tile.configure(state=state)  # 528

    # 529
    def _spin_param(self, parent, row, label, default, *, from_, to, increment, format_str=None):  # 530
        ttk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky="w", padx=6, pady=4)  # 531
        spin = ttk.Spinbox(parent, from_=from_, to=to, increment=increment, width=10, justify="right")  # 532
        if format_str:  # 533
            spin.configure(format=format_str)  # 534
        self._set_spinbox_value(spin, default)  # 535
        spin.grid(row=row, column=1, sticky="w", padx=6, pady=4)  # 536
        return spin  # 537

    # 538
    def _set_spinbox_value(self, spinbox: ttk.Spinbox, value):  # 539
        try:  # 540
            # Try setting directly first, works for simple values
            spinbox.set(value)
        except tk.TclError:  # 542
            # Fallback: delete and insert if direct set fails (e.g., due to formatting)
            try:
                current_value = spinbox.get()
                # Only update if the value is actually different to avoid unnecessary actions
                if str(current_value) != str(value):
                    spinbox.delete(0, tk.END)
                    spinbox.insert(0, str(value))
            except (tk.TclError, ValueError):
                # Handle cases where get() might fail or value cannot be stringified easily
                print(f"Warning: Could not set spinbox value to {value}")

    # 545
    def _browse_img(self):  # 546
        p = filedialog.askopenfilename(title="Select image",
                                       filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),
                                                  ("All", "*.*")])  # 547
        if p: self.ent_img.delete(0, tk.END); self.ent_img.insert(0, p)  # 548

    # 549
    def _browse_out(self):  # 550
        p = filedialog.askdirectory(title="Select output folder", mustexist=False)  # Allow creating new folders
        if p: self.ent_out.delete(0, tk.END); self.ent_out.insert(0, p)  # 552

    # 553
    # --- NEW: Load Session Method ---
    def _load_session(self):
        """Asks user for a session file and tells the controller to load it."""
        filepath = filedialog.askopenfilename(
            title="Load SAED Session",
            filetypes=[("SAED Session", "saed_session.json"), ("All files", "*.*")]
        )
        if not filepath or not self.controller:
            return
        try:
            self.controller.load_session_from_file(filepath)
            if self.controller:
                self.controller.set_status(f"Session loaded from {Path(filepath).name}")
        except FileNotFoundError:
            messagebox.showerror("Load Error", "Session file not found.")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load session:\n{e}")

    # --- NEW: get_state and set_state Methods ---
    def get_state(self) -> Dict[str, Any]:
        """Returns a serializable dictionary of the launcher's settings."""
        return {
            "image_path": self.ent_img.get(),
            "output_folder": self.ent_out.get(),
            "center_x": self.ent_cx.get(),
            "center_y": self.ent_cy.get(),
            "preproc_mode": self.cmb_pre.get(),
            "clahe_clip": self.spn_clip.get(),
            "clahe_tiles": self.spn_tile.get(),
            "detect_perc": self.spn_perc.get(),
            "merge_perc": self.spn_merge_perc.get(),
            "merge_radius": self.spn_merge_rad.get(),
            "merge_tol": self.spn_merge_tol.get(),
            "min_area": self.spn_min_area.get(),  # <-- MODIFIED
            "max_pts": self.spn_maxpts.get(),
            "refine_iters": self.spn_iters.get(),
            "tol_angle": self.spn_tolang.get(),
            "tol_radius": self.spn_tolr.get(),
            "dead_zone": self.spn_dead.get(),
            "search_radius": self.spn_search.get(),
        }

    def set_state(self, state: Dict[str, Any]):
        """Restores the launcher's settings from a dictionary."""

        def _set_entry(widget, value):
            if value is not None and isinstance(widget, (ttk.Entry, tk.Entry)):
                widget.delete(0, tk.END)
                widget.insert(0, str(value))

        _set_entry(self.ent_img, state.get("image_path"))
        _set_entry(self.ent_out, state.get("output_folder"))
        _set_entry(self.ent_cx, state.get("center_x"))
        _set_entry(self.ent_cy, state.get("center_y"))

        # Set Combobox value safely
        preproc_mode = state.get("preproc_mode")
        if preproc_mode and isinstance(self.cmb_pre, ttk.Combobox):
            if preproc_mode in self.cmb_pre['values']:
                self.cmb_pre.set(preproc_mode)
            else:
                print(f"Warning: Saved preproc_mode '{preproc_mode}' not found in options. Using default.")
                self.cmb_pre.current(0)  # Fallback to first option
        elif isinstance(self.cmb_pre, ttk.Combobox):
            self.cmb_pre.current(0)  # Default if not in state

        self._on_preproc_change(None)  # Update UI based on new mode

        # Set Spinbox values safely, providing defaults
        self._set_spinbox_value(self.spn_clip, state.get("clahe_clip", 1.5))
        self._set_spinbox_value(self.spn_tile, state.get("clahe_tiles", 8))
        self._set_spinbox_value(self.spn_perc, state.get("detect_perc", 99.0))
        self._set_spinbox_value(self.spn_merge_perc, state.get("merge_perc", 95.0))
        self._set_spinbox_value(self.spn_merge_rad, state.get("merge_radius", 0))
        self._set_spinbox_value(self.spn_merge_tol, state.get("merge_tol", 10.0))
        self._set_spinbox_value(self.spn_min_area, state.get("min_area", 3))  # <-- MODIFIED
        self._set_spinbox_value(self.spn_maxpts, state.get("max_pts", 6000))
        self._set_spinbox_value(self.spn_iters, state.get("refine_iters", 4))
        self._set_spinbox_value(self.spn_tolang, state.get("tol_angle", 8.0))
        self._set_spinbox_value(self.spn_tolr, state.get("tol_radius", 0.06))
        self._set_spinbox_value(self.spn_dead, state.get("dead_zone", 0))
        self._set_spinbox_value(self.spn_search, state.get("search_radius", 0))

    def _go_editor(self):
        try:
            # --- Ensure paths are absolute ---
            image_path_str = self.ent_img.get()
            output_dir_str = self.ent_out.get()

            if not image_path_str:
                messagebox.showerror("Error", "Please select an image file.");
                return
            if not output_dir_str:
                messagebox.showerror("Error", "Please specify an output folder.");
                return

            image_path = Path(image_path_str).expanduser().resolve()  # Get absolute path
            outdir = Path(output_dir_str).expanduser().resolve();  # Get absolute path
            outdir.mkdir(parents=True, exist_ok=True)

            if not image_path.exists():
                messagebox.showerror("Error", f"Image not found at: {image_path}");
                return

            # --- Get parameters (as before) ---
            perc = float(self.spn_perc.get())
            merge_apply_perc = float(self.spn_merge_perc.get())
            merge_radius = float(self.spn_merge_rad.get())
            merge_tol = float(self.spn_merge_tol.get())
            # --- MODIFIED: Get new area parameters ---
            min_area = int(float(self.spn_min_area.get()))
            max_pts = int(float(self.spn_maxpts.get()))
            iters = int(float(self.spn_iters.get()))
            tol_ang = float(self.spn_tolang.get())
            tol_relr = float(self.spn_tolr.get())
            dead_r = float(self.spn_dead.get())
            search_r = float(self.spn_search.get())

            # --- preprocessing ---
            pre_mode = self.cmb_pre.get()
            if pre_mode == "No smoothing":
                settings = PreprocSettings(mode="raw")
            elif pre_mode == "Standard":
                settings = PreprocSettings(mode="standard")
            else:  # CLAHE
                clip = float(self.spn_clip.get())
                tiles = int(float(self.spn_tile.get()))
                settings = PreprocSettings(mode="clahe", clahe_clip=clip, clahe_tiles=tiles)

            # --- Load image ---
            try:
                arr = load_grayscale_with_preproc(image_path, settings)
            except RuntimeError as cv_err:  # Catch OpenCV missing specifically
                messagebox.showerror("Dependency Error", str(cv_err))
                return  # Stop processing if required dependency is missing
            except Exception as img_load_err:
                messagebox.showerror("Image Error", f"Failed to load or process image:\n{img_load_err}")
                return  # Stop processing

            mode = settings.mode
            preproc_payload = settings.to_json()

            # --- Center calculation ---
            cx_txt = self.ent_cx.get().strip();
            cy_txt = self.ent_cy.get().strip()
            if cx_txt and cy_txt:
                try:
                    center0 = CenterResult(cy=float(cy_txt), cx=float(cx_txt), method="user")
                except ValueError:
                    messagebox.showwarning("Input Warning", "Invalid center coordinates. Using automatic center.")
                    center0 = geometric_midpoint(arr)
            else:
                center0 = geometric_midpoint(arr)

            # --- Peak detection and refinement ---
            # --- MODIFIED: Call new function ---
            pts = detect_spots_by_centroid(
                arr, perc=perc, min_area=min_area, max_spots=max_pts
            )

            if len(pts) == 0:
                print("Warning: No spots detected initially.")  # Use print for non-critical warning
                # Optionally try lower percentile if no spots found
                lower_perc = max(85.0, perc - 5.0)  # Example fallback
                print(f"Retrying spot detection with percentile {lower_perc:.1f}%...")
                # --- MODIFIED: Call new function in fallback ---
                pts = detect_spots_by_centroid(
                    arr, perc=lower_perc, min_area=min_area, max_spots=max_pts
                )
                if len(pts) == 0:
                    messagebox.showwarning("Detection Warning",
                                           "No spots detected even with lower threshold. Proceeding without points.")

            # Refine center even if few points, refine_center_antipodal handles low point counts
            center = refine_center_antipodal((center0.cy, center0.cx), pts, tol_ang_deg=tol_ang, tol_rel_r=tol_relr,
                                             iters=iters)

            # --- Geometric filters ---
            if (dead_r > 0 or search_r > 0) and len(pts) > 0:
                dy = pts[:, 0] - center.cy;
                dx = pts[:, 1] - center.cx;
                r = np.hypot(dx, dy)
                mask = np.ones(len(pts), dtype=bool)
                if dead_r > 0:   mask &= (r >= dead_r)
                if search_r > 0: mask &= (r <= search_r)
                pts = pts[mask]

            # --- Merging ---
            if len(pts) > 0 and merge_radius > 0:
                merge_threshold = None
                if merge_apply_perc > 0.0:
                    # Ensure percentile calculation doesn't fail if pts[:, 2] is constant or empty
                    if len(np.unique(pts[:, 2])) > 1:
                        try:
                            merge_threshold = float(np.percentile(pts[:, 2], merge_apply_perc))
                        except IndexError:  # Handle empty pts array after filtering
                            pass
                    elif len(pts) > 0:
                        merge_threshold = float(pts[0, 2] * (merge_apply_perc / 100.0))  # Fallback if constant

                pts = merge_spots_by_intensity(
                    pts,
                    radius=merge_radius,
                    tol_percent=merge_tol,
                    min_intensity=merge_threshold,
                    line_image=arr,  # Pass original preprocessed image for line check
                )

            # --- Create saed_input.json in the output directory ---
            points_list = [{"y": float(y), "x": float(x), "intensity": float(v)} for (y, x, v) in pts.tolist()]
            saed_input_data = {
                "image": str(image_path),  # Save absolute path
                "preproc_mode": mode,
                "preproc": preproc_payload,
                "center": {"x": float(center.cx), "y": float(center.cy), "method": center.method},
                "radii": {"dead": float(dead_r), "search": float(search_r)},
                "points": points_list
            }
            saed_input_path = outdir / "saed_input.json"  # Define path in output dir
            saed_input_path.write_text(json.dumps(saed_input_data, ensure_ascii=False, indent=2), encoding="utf-8")

            # --- Launch the editor via controller ---
            if self.controller is not None:
                try:
                    self.controller.open_editor(saed_input_path)  # Pass the correct path
                except Exception as exc:
                    messagebox.showerror("Error", f"Failed to open the editor tab:\n{exc}")
            else:
                # Fallback: Launch external editor (ensure path handling is correct here too if used)
                # ... (external launch code - needs similar path care) ...
                messagebox.showwarning("Standalone Mode",
                                       "Running in standalone mode. Editor will open externally if available.")

            # --- Save center log (optional service info) ---
            try:
                (outdir / "center_init.json").write_text(json.dumps({
                    "initial": {"x": center0.cx, "y": center0.cy, "method": center0.method},
                    "refined": {"x": center.cx, "y": center.cy, "method": center.method},  # Use refined method name
                    "dead_zone_px": dead_r,
                    "search_radius_px": search_r,
                    "preproc_mode": mode,
                    "preproc": preproc_payload,
                    "image_size": {"H": int(arr.shape[0]), "W": int(arr.shape[1])}
                }, indent=2), encoding="utf-8")
            except Exception as log_err:
                print(f"Warning: Could not save center_init.json - {log_err}")


        except Exception as e:
            messagebox.showerror("Processing Error", f"An unexpected error occurred during processing:\n{e}")


class SAEDApp(tk.Tk):  # 767
    # 768
    (  # 769
        "Backwards-compatible standalone application using the tab frame.\n"  # 770
    )  # 771

    # 772
    def __init__(self):  # 773
        super().__init__()  # 774
        # 775
        self.title("SAED Symmetry – Launcher")  # 776
        # 777
        self.geometry("980x680")  # 778
        # 779
        self.resizable(True, False)  # 780
        # 781
        frame = SAEDLauncherFrame(self)  # 782
        # 783
        frame.pack(fill=tk.BOTH, expand=True)  # 784


# 785
if __name__ == "__main__":  # 786
    SAEDApp().mainloop()  # 787