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
        max_area: int = 500,  # 43
        max_spots: int = 6000,  # 44
) -> np.ndarray:  # 45
    """
    Detects spots using centroiding of connected components (blobs).
    This is more robust for flat or saturated peaks than local maxima search.
    """
    H, W = arr.shape  # 46
    # 47
    # 1. Use percentile map to get a robust threshold value
    percent_map, _, _ = compute_percentile_map(arr)  # 48
    try:  # 49
        perc_val = float(np.clip(perc, 0.0, 100.0))  # 50
        th_value = float(np.percentile(percent_map, perc_val))  # 51
    except (ValueError, IndexError):  # 52
        th_value = 99.0  # Fallback
    # 53
    # 2. Create binary mask based on threshold
    # We use the percentile map as input as it's already contrast-enhanced
    binary_mask = np.where(percent_map >= th_value, 255, 0).astype(np.uint8)  # 54
    # 55
    # 3. Find all connected components ("islands" or "blobs")
    num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(  # 56
        binary_mask,  # 57
        connectivity=8  # Use 8-way connectivity
    )  # 58
    # 59
    kept = []  # 60
    # 61
    # 4. Iterate over all found labels (label 0 is the background, skip it)
    for i in range(1, num_labels):  # 62
        area = stats[i, cv2.CC_STAT_AREA]  # 63
        # 64
        # 5. Filter blobs by their area
        if not (min_area <= area <= max_area):  # 65
            continue  # Skip blobs that are too small (noise) or too large (center beam)
        # 66
        # 6. Get the centroid (cx, cy)
        cx, cy = centroids[i]  # 67
        # 68
        # 7. Get intensity at the centroid position for sorting
        yi, xi = int(round(cy)), int(round(cx))  # 69
        if 0 <= yi < H and 0 <= xi < W:  # 70
            # Use intensity from percentile map
            v = float(percent_map[yi, xi])  # 71
            kept.append((float(cy), float(cx), float(v)))  # Store as (y, x, v)
    # 72
    # 8. Sort by brightness (highest first)
    kept.sort(key=lambda t: -t[2])  # 73
    # 74
    # 9. Limit to max_spots
    if len(kept) > max_spots:  # 75
        kept = kept[:max_spots]  # 76
    # 77
    return np.array(kept, dtype=float) if kept else np.zeros((0, 3), dtype=float)  # 78


# 79
# 80
def merge_spots_by_intensity(  # 81
        pts: np.ndarray,  # 82
        radius: float,  # 83
        tol_percent: float,  # 84
        *,  # 85
        min_intensity: float | None = None,  # 86
        line_image: np.ndarray | None = None,  # 87
        percentile_map: np.ndarray | None = None,  # 88
) -> np.ndarray:  # 89
    # ... (function code remains unchanged) ... # 107
    if pts.size == 0:  # 108
        return pts  # 109
    radius = float(radius)  # 110
    tol = max(0.0, float(tol_percent) / 100.0)  # 111
    if radius <= 0.0:  # 112
        return np.asarray(pts, dtype=float)  # 113
    # 114
    pts = np.asarray(pts, dtype=float)  # 115
    if min_intensity is not None:  # 116
        mask = pts[:, 2] >= float(min_intensity)  # 117
    else:  # 118
        mask = np.ones(len(pts), dtype=bool)  # 119
    # 120
    to_merge = pts[mask]  # 121
    untouched = pts[~mask]  # 122
    if to_merge.size == 0:  # 123
        return pts  # 124
    # 125
    intensity_map: np.ndarray | None  # 126
    if percentile_map is not None:  # 127
        intensity_map = np.asarray(percentile_map, dtype=float)  # 128
    elif line_image is not None:  # 129
        percent, _, _ = compute_percentile_map(np.asarray(line_image, dtype=float))  # 130
        intensity_map = percent  # 131
    else:  # 132
        intensity_map = None  # 133
    # 134
    rad2 = radius * radius  # 135
    used = np.zeros(len(to_merge), dtype=bool)  # 136
    order = np.argsort(-to_merge[:, 2])  # start from the brightest  # 137
    merged: list[tuple[float, float, float]] = []  # 138

    # 139
    def within_tol(a: float, b: float) -> bool:  # 140
        hi = max(a, b)  # 141
        if hi == 0.0:  # 142
            return abs(a - b) == 0.0  # 143
        return abs(a - b) <= tol * hi + 1e-12  # 144

    # 145
    H = W = None  # 146
    if intensity_map is not None and intensity_map.ndim == 2:  # 147
        H, W = intensity_map.shape  # 148
    else:  # 149
        intensity_map = None  # 150

    # 151
    def clamp_round(val: float, hi: int) -> int:  # 152
        return int(min(max(round(float(val)), 0), hi))  # 153

    # 154
    def bresenham_line(y0: int, x0: int, y1: int, x1: int) -> list[tuple[int, int]]:  # 155
        points: list[tuple[int, int]] = []  # 156
        dy = abs(y1 - y0)  # 157
        dx = abs(x1 - x0)  # 158
        sy = 1 if y0 < y1 else -1  # 159
        sx = 1 if x0 < x1 else -1  # 160
        err = dx - dy  # 161
        while True:  # 162
            points.append((y0, x0))  # 163
            if y0 == y1 and x0 == x1:  # 164
                break  # 165
            e2 = err * 2  # 166
            if e2 > -dy:  # 167
                err -= dy  # 168
                x0 += sx  # 169
            if e2 < dx:  # 170
                err += dx  # 171
                y0 += sy  # 172
        return points  # 173

    # 174
    def has_intensity_dip(idx_a: int, idx_b: int) -> bool:  # 175
        if intensity_map is None or H is None or W is None:  # 176
            return False  # 177
        base_val = min(float(to_merge[idx_a, 2]), float(to_merge[idx_b, 2]))  # 178
        if base_val <= 0.0:  # 179
            return False  # 180
        y0 = clamp_round(to_merge[idx_a, 0], H - 1)  # 181
        x0 = clamp_round(to_merge[idx_a, 1], W - 1)  # 182
        y1 = clamp_round(to_merge[idx_b, 0], H - 1)  # 183
        x1 = clamp_round(to_merge[idx_b, 1], W - 1)  # 184
        pixels = bresenham_line(y0, x0, y1, x1)  # 185
        if len(pixels) <= 2:  # 186
            return False  # 187
        limit = base_val * (1.0 - tol)  # 188
        for (yy, xx) in pixels[1:-1]:  # 189
            if 0 <= yy < H and 0 <= xx < W:  # 190
                if float(intensity_map[yy, xx]) + 1e-9 < limit:  # 191
                    return True  # 192
        return False  # 193

    # 194
    for idx in order:  # 195
        if used[idx]:  # 196
            continue  # 197
        # 198
        cluster = [idx]  # 199
        sum_y = float(to_merge[idx, 0])  # 200
        sum_x = float(to_merge[idx, 1])  # 201
        intensities = [float(to_merge[idx, 2])]  # 202
        sum_v = intensities[0]  # 203
        # 204
        neighbors = []  # 205
        base_y, base_x = to_merge[idx, 0], to_merge[idx, 1]  # 206
        for j in range(len(to_merge)):  # 207
            if j == idx or used[j]:  # 208
                continue  # 209
            dy = to_merge[j, 0] - base_y  # 210
            dx = to_merge[j, 1] - base_x  # 211
            if dy * dy + dx * dx <= rad2:  # 212
                neighbors.append(j)  # 213
        # 214
        neighbors.sort(key=lambda j: abs(to_merge[j, 2] - intensities[0]))  # 215
        # 216
        for j in neighbors:  # 217
            if used[j]:  # 218
                continue  # 219
            if any(has_intensity_dip(existing, j) for existing in cluster):  # 220
                continue  # 221
            cand_v = float(to_merge[j, 2])  # 222
            new_count = len(cluster) + 1  # 223
            new_avg_v = (sum_v + cand_v) / new_count  # 224
            if all(within_tol(new_avg_v, val) for val in (*intensities, cand_v)):  # 225
                cluster.append(j)  # 226
                intensities.append(cand_v)  # 227
                sum_y += float(to_merge[j, 0])  # 228
                sum_x += float(to_merge[j, 1])  # 229
                sum_v += cand_v  # 230
        # 231
        if len(cluster) > 1:  # 232
            new_count = len(cluster)  # 233
            new_y = sum_y / new_count  # 234
            new_x = sum_x / new_count  # 235
            new_v = sum_v / new_count  # 236
            if (min_intensity is None or new_v >= min_intensity) and all(  # 237
                    within_tol(new_v, val) for val in intensities  # 238
            ):  # 239
                merged.append((new_y, new_x, new_v))  # 240
                for j in cluster:  # 241
                    used[j] = True  # 242
                continue  # 243
        # 244
        # either a single-point cluster or the resulting intensity exceeded the tolerance  # 245
        for j in cluster:  # 246
            if not used[j]:  # 247
                merged.append(tuple(to_merge[j]))  # 248
                used[j] = True  # 249
    # 250
    merged = np.array(merged, dtype=float)  # 251
    if untouched.size == 0:  # 252
        return merged  # 253
    if merged.size == 0:  # 254
        return untouched  # 255
    return np.vstack((merged, untouched))  # 256


# 257
# 258
def geometric_midpoint(arr: np.ndarray) -> CenterResult:  # 259
    H, W = arr.shape  # 260
    return CenterResult(cy=(H - 1) / 2.0, cx=(W - 1) / 2.0, method="midpoint")  # 261


def refine_center_antipodal(center: Tuple[float, float], pts: np.ndarray, tol_ang_deg: float = 8.0,
                            tol_rel_r: float = 0.06, iters: int = 3) -> CenterResult:  # 262
    cy, cx = float(center[0]), float(center[1])  # 263
    if len(pts) < 4:  # 264
        return CenterResult(cy=cy, cx=cx, method="midpoint (fallback)")  # 265
    for _ in range(max(0, int(iters))):  # 266
        dy = pts[:, 0] - cy;
        dx = pts[:, 1] - cx  # 267
        r = np.hypot(dx, dy)  # 268
        # Avoid division by zero if a point is exactly at the center
        r_safe = np.where(r > 1e-9, r, 1e-9)
        u = np.column_stack((dx, dy)) / r_safe[:, None]  # 269
        cos_thr = -np.cos(np.deg2rad(180.0 - float(tol_ang_deg)))  # 270
        mids = []  # 271
        for i in range(len(pts)):  # 272
            if r[i] < 1e-6: continue  # Skip point if it's too close to center
            dots = (u @ u[i])  # 273
            # Avoid division by zero for radius tolerance
            max_r_pair = np.maximum(r, r[i])
            # Use np.divide with where clause to handle potential zero denominators
            rel_diff = np.divide(np.abs(r - r[i]), max_r_pair, out=np.zeros_like(r), where=max_r_pair > 1e-9)
            rad_ok = (rel_diff < float(tol_rel_r)) & (max_r_pair > 1e-9)  # Ensure we don't match zero-radius points

            ang_ok = (dots < cos_thr)  # 275
            # Exclude self-comparison and points too close to center
            valid_match = rad_ok & ang_ok & (np.arange(len(pts)) != i) & (r > 1e-6)
            idx = np.where(valid_match)[0]  # 276
            if idx.size == 0: continue  # 277
            # Find the best antipodal match among valid candidates
            j = idx[np.argmin(np.abs(dots[idx] + 1.0))]  # 278
            yi, xi = pts[i, 0], pts[i, 1]  # 279
            yj, xj = pts[j, 0], pts[j, 1]  # 280
            mids.append(((yi + yj) / 2.0, (xi + xj) / 2.0))  # 281
        if len(mids) < 4: break  # 282 Not enough pairs found
        mids = np.array(mids, dtype=float)  # 283
        # Use median to be robust against outliers
        cy = float(np.median(mids[:, 0]));
        cx = float(np.median(mids[:, 1]))  # 284
    return CenterResult(cy=cy, cx=cx, method="antipodal-refined")  # 285


# 286
# -------------------------- GUI --------------------------  # 287
class SAEDLauncherFrame(ttk.Frame):  # 288
    (  # 289
        "Launcher tab suitable for both the standalone application and notebooks.\n"  # 290
    )  # 291

    # 292
    def __init__(self, master: tk.Misc, controller=None):  # 293
        super().__init__(master)  # 294
        self.controller = controller  # 295
        self._scroll_canvas = None  # 296
        self._scroll_window_id = None  # 297
        self._build_ui()  # 298

    # 299
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

    def _build_ui(self):  # 300
        outer = ttk.Frame(self)  # 301
        outer.pack(fill=tk.BOTH, expand=True)  # 302
        # 303
        fixed = ttk.Frame(outer, padding=(16, 16, 16, 0))  # 304
        fixed.pack(side=tk.TOP, fill=tk.X)  # 305
        fixed.grid_columnconfigure(0, weight=1)  # 306
        # 307
        data_box = ttk.LabelFrame(fixed, text="Input data", padding=(12, 10, 12, 12))  # 308
        data_box.grid(row=0, column=0, sticky="nsew")  # 309
        # 310
        data_box.grid_columnconfigure(1, weight=1)  # Make entry widgets resizable
        # 313
        ttk.Label(data_box, text="Image:").grid(row=0, column=0, sticky="w", padx=6, pady=4)  # 314
        self.ent_img = ttk.Entry(data_box)  # 315
        self.ent_img.grid(row=0, column=1, columnspan=2, sticky="we", padx=6, pady=4)  # 316
        ttk.Button(data_box, text="Browse…", command=self._browse_img).grid(row=0, column=3, sticky="ew", padx=6,
                                                                            pady=4)  # 317
        # 318
        ttk.Label(data_box, text="Output folder:").grid(row=1, column=0, sticky="w", padx=6, pady=4)  # 319
        self.ent_out = ttk.Entry(data_box)  # 320
        self.ent_out.insert(0, self._get_default_output_path())  # 321
        self.ent_out.grid(row=1, column=1, sticky="we", padx=6, pady=4)  # 322
        # --- NEW: "Load Session" Button ---
        ttk.Button(data_box, text="Load Session…", command=self._load_session).grid(row=1, column=2, sticky="ew",
                                                                                    padx=6, pady=4)  # 323
        ttk.Button(data_box, text="Choose…", command=self._browse_out).grid(row=1, column=3, sticky="ew", padx=6,
                                                                            pady=4)  # 324
        # 325
        ttk.Label(data_box, text="Center X (optional):").grid(row=2, column=0, sticky="w", padx=6, pady=4)  # 326
        self.ent_cx = ttk.Entry(data_box, width=12)  # 327
        self.ent_cx.grid(row=2, column=1, sticky="w", padx=6, pady=4)  # 328
        ttk.Label(data_box, text="Center Y:").grid(row=2, column=2, sticky="w", padx=6, pady=4)  # 329
        self.ent_cy = ttk.Entry(data_box, width=12)  # 330
        self.ent_cy.grid(row=2, column=3, sticky="w", padx=6, pady=4)  # 331
        # 332
        ttk.Label(  # 333
            data_box,  # 334
            text="Leave the coordinates empty to let the program find the center automatically. Use 'Load Session' to restore a previous state.",
            # 335
            wraplength=520,  # 336
            foreground="#555555"  # 337
        ).grid(row=3, column=0, columnspan=4, sticky="we", padx=6, pady=(0, 4))  # 338
        # 339
        pre_box = ttk.LabelFrame(fixed, text="Preprocessing", padding=(12, 10, 12, 12))  # 340
        pre_box.grid(row=1, column=0, sticky="nsew", pady=(10, 0))  # 341
        pre_box.grid_columnconfigure(1, weight=1)  # 342
        # 343
        ttk.Label(pre_box, text="Mode:").grid(row=0, column=0, sticky="w", padx=6, pady=4)  # 344
        self.cmb_pre = ttk.Combobox(  # 345
            pre_box,  # 346
            values=["No smoothing", "Standard", "CLAHE"],  # 347
            state="readonly",  # 348
        )  # 349
        self.cmb_pre.current(0)  # 350
        self.cmb_pre.grid(row=0, column=1, sticky="w", padx=6, pady=4)  # 351
        self.cmb_pre.bind("<<ComboboxSelected>>", self._on_preproc_change)  # 352
        # 353
        ttk.Label(pre_box, text="CLAHE clipLimit / tile:").grid(row=1, column=0, sticky="w", padx=6, pady=4)  # 354
        self.spn_clip = ttk.Spinbox(pre_box, from_=0.1, to=10.0, increment=0.1, width=8, justify="right")  # 355
        self._set_spinbox_value(self.spn_clip, 1.5)  # 356
        self.spn_clip.grid(row=1, column=1, sticky="w", padx=6, pady=4)  # 357
        self.spn_tile = ttk.Spinbox(pre_box, from_=2, to=64, increment=1, width=8, justify="right")  # 358
        self._set_spinbox_value(self.spn_tile, 8)  # 359
        self.spn_tile.grid(row=1, column=2, sticky="w", padx=6, pady=4)  # 360
        # 361
        ttk.Label(  # 362
            pre_box,  # 363
            text="Select CLAHE for images with strong brightness variations. ClipLimit controls contrast, and tile size defines the local processing radius.",
            # 364
            wraplength=520,  # 365
            foreground="#555555"  # 366
        ).grid(row=2, column=0, columnspan=3, sticky="we", padx=6, pady=(2, 0))  # 367
        # 368
        scroll_host = ttk.Frame(outer)  # 369
        scroll_host.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  # 370
        # 371
        canvas = tk.Canvas(scroll_host, borderwidth=0, highlightthickness=0)  # 372
        vscroll = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=canvas.yview)  # 373
        scrollable = ttk.Frame(canvas, padding=(16, 12, 16, 12))  # 374
        scrollable.grid_columnconfigure(0, weight=1)  # 375
        # 376
        self._scroll_canvas = canvas  # 377
        self._scroll_window_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")  # 378
        canvas.configure(yscrollcommand=vscroll.set)  # 379
        # 380
        scrollable.bind(  # 381
            "<Configure>",  # 382
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))  # 383
        )  # 384
        canvas.bind(  # 385
            "<Configure>",  # 386
            lambda e: canvas.itemconfigure(self._scroll_window_id, width=e.width)  # 387
        )  # 388
        # 389
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # 390
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)  # 391
        # 392
        scrollable.bind("<Enter>", self._activate_scroll)  # 393
        scrollable.bind("<Leave>", self._deactivate_scroll)  # 394
        canvas.bind("<Enter>", self._activate_scroll)  # 395
        canvas.bind("<Leave>", self._deactivate_scroll)  # 396
        # 397
        detect_box = ttk.LabelFrame(scrollable, text="Detector and refinement", padding=(12, 10, 12, 12))  # 398
        detect_box.grid(row=0, column=0, sticky="nsew")  # 399
        detect_box.grid_columnconfigure(1, weight=1)  # 400
        # 401
        ttk.Label(  # 402
            detect_box,  # 403
            text="Peak threshold and search window",  # 404
            font=("TkDefaultFont", 10, "bold")  # 405
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))  # 406
        self.spn_perc = self._spin_param(  # 407
            detect_box, 1, "Detection percentile (%)", 99.0,  # 408
            from_=80.0, to=100.0, increment=0.1, format_str="%.1f"  # 409
        )  # 410
        self.spn_merge_perc = self._spin_param(  # 411
            detect_box, 2, "Intensity percentile for merging (%)", 95.0,  # 412
            from_=0.0, to=100.0, increment=0.5, format_str="%.1f"  # 413
        )  # 414
        self.spn_merge_rad = self._spin_param(  # 415
            detect_box, 3, "Peak merging radius (px)", 0,  # 416
            from_=0, to=50, increment=1  # 417
        )  # 418
        self.spn_merge_tol = self._spin_param(  # 419
            detect_box, 4, "Intensity similarity tolerance (%)", 10.0,  # 420
            from_=0.0, to=100.0, increment=0.5, format_str="%.1f"  # 421
        )  # 422
        # --- MODIFIED: Replaced min_sep with min_area and max_area ---
        self.spn_min_area = self._spin_param(  # 423
            detect_box, 5, "Min. peak area (px)", 3,  # 424
            from_=1, to=500, increment=1  # 425
        )  # 426
        self.spn_max_area = self._spin_param(  # 427
            detect_box, 6, "Max. peak area (px)", 500,  # 428
            from_=10, to=10000, increment=10  # 429
        )  # 430
        self.spn_maxpts = self._spin_param(  # 431
            detect_box, 7, "Maximum detected points", 6000,  # 432
            from_=100, to=20000, increment=100  # 433
        )  # 434
        # 435
        ttk.Separator(detect_box).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(6, 8))  # 436
        # 437
        ttk.Label(  # 438
            detect_box,  # 439
            text="Center refinement",  # 440
            font=("TkDefaultFont", 10, "bold")  # 441
        ).grid(row=9, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))  # 442
        self.spn_iters = self._spin_param(  # 443
            detect_box, 10, "Center refinement iterations", 4,  # 444
            from_=0, to=10, increment=1  # 445
        )  # 446
        self.spn_tolang = self._spin_param(  # 447
            detect_box, 11, "Antipode tolerance (°)", 8.0,  # 448
            from_=1.0, to=30.0, increment=0.5, format_str="%.1f"  # 449
        )  # 450
        self.spn_tolr = self._spin_param(  # 451
            detect_box, 12, "Radius tolerance (relative)", 0.06,  # 452
            from_=0.01, to=0.5, increment=0.01, format_str="%.2f"  # 453
        )  # 454
        # 455
        ttk.Separator(detect_box).grid(row=13, column=0, columnspan=2, sticky="ew", pady=(6, 8))  # 456
        # 457
        ttk.Label(  # 458
            detect_box,  # 459
            text="Geometric filters",  # 460
            font=("TkDefaultFont", 10, "bold")  # 461
        ).grid(row=14, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))  # 462
        self.spn_dead = self._spin_param(  # 463
            detect_box, 15, "Dead zone (px)", 0,  # 464
            from_=0, to=500, increment=1  # 465
        )  # 466
        self.spn_search = self._spin_param(  # 467
            detect_box, 16, "Search radius (px, 0 = unlimited)", 0,  # 468
            from_=0, to=10000, increment=25  # 469
        )  # 470
        # 471
        ttk.Label(  # 472
            detect_box,  # 473
            text="Use Min/Max area to filter noise and the central beam. Dead zone filters by radius *after* center refinement.",
            # 474
            wraplength=520,  # 475
            foreground="#555555"  # 476
        ).grid(row=17, column=0, columnspan=2, sticky="we", padx=6, pady=(2, 0))  # 477
        # 478
        action_box = ttk.Frame(scrollable, padding=(0, 12, 0, 0))  # 479
        action_box.grid(row=1, column=0, sticky="nsew")  # 480
        action_box.grid_columnconfigure(0, weight=1)  # 481
        # 482
        ttk.Label(  # 483
            action_box,  # 484
            text="Review the parameters and press the button below to switch to interactive editing of detected points.",
            # 485
            wraplength=540,  # 486
            justify="left"  # 487
        ).grid(row=0, column=0, sticky="we", padx=4, pady=(0, 8))  # 488
        # 489
        ttk.Button(action_box, text="Open point editor", command=self._go_editor).grid(  # 490
            row=1, column=0, sticky="ew", padx=4, pady=(0, 12)  # 491
        )  # 492
        # 493
        filler_bg = ttk.Style().lookup("TFrame", "background") or self.winfo_toplevel().cget("background")  # 494
        bottom_filler = tk.Frame(scrollable, height=56, bg=filler_bg)  # 495
        bottom_filler.grid(row=2, column=0, sticky="ew")  # 496
        bottom_filler.grid_propagate(False)  # 497
        # 498
        self._on_preproc_change(None)  # 499

    # 500
    def _activate_scroll(self, _event):  # 501
        if self._scroll_canvas is None:  # 502
            return  # 503
        self._scroll_canvas.bind_all("<MouseWheel>", self._on_scroll_mousewheel)  # 504
        self._scroll_canvas.bind_all("<Button-4>", self._on_scroll_mousewheel)  # 505
        self._scroll_canvas.bind_all("<Button-5>", self._on_scroll_mousewheel)  # 506

    # 507
    def _deactivate_scroll(self, _event):  # 508
        if self._scroll_canvas is None:  # 509
            return  # 510
        self._scroll_canvas.unbind_all("<MouseWheel>")  # 511
        self._scroll_canvas.unbind_all("<Button-4>")  # 512
        self._scroll_canvas.unbind_all("<Button-5>")  # 513

    # 514
    def _on_scroll_mousewheel(self, event):  # 515
        if self._scroll_canvas is None:  # 516
            return  # 517
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

    def _on_preproc_change(self, _evt):  # 528
        mode = self.cmb_pre.get()  # 529
        clahe_enabled = (mode == "CLAHE")  # 530
        state = "normal" if clahe_enabled else "disabled"  # 531
        self.spn_clip.configure(state=state)  # 532
        self.spn_tile.configure(state=state)  # 533

    # 534
    def _spin_param(self, parent, row, label, default, *, from_, to, increment, format_str=None):  # 535
        ttk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky="w", padx=6, pady=4)  # 536
        spin = ttk.Spinbox(parent, from_=from_, to=to, increment=increment, width=10, justify="right")  # 537
        if format_str:  # 538
            spin.configure(format=format_str)  # 539
        self._set_spinbox_value(spin, default)  # 540
        spin.grid(row=row, column=1, sticky="w", padx=6, pady=4)  # 541
        return spin  # 542

    # 543
    def _set_spinbox_value(self, spinbox: ttk.Spinbox, value):  # 544
        try:  # 545
            # Try setting directly first, works for simple values
            spinbox.set(value)
        except tk.TclError:  # 547
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

    # 550
    def _browse_img(self):  # 551
        p = filedialog.askopenfilename(title="Select image",
                                       filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),
                                                  ("All", "*.*")])  # 552
        if p: self.ent_img.delete(0, tk.END); self.ent_img.insert(0, p)  # 553

    # 554
    def _browse_out(self):  # 555
        p = filedialog.askdirectory(title="Select output folder", mustexist=False)  # Allow creating new folders
        if p: self.ent_out.delete(0, tk.END); self.ent_out.insert(0, p)  # 557

    # 558
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
            "max_area": self.spn_max_area.get(),  # <-- NEW
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
        self._set_spinbox_value(self.spn_max_area, state.get("max_area", 500))  # <-- NEW
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
            max_area = int(float(self.spn_max_area.get()))
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
                arr, perc=perc, min_area=min_area, max_area=max_area, max_spots=max_pts
            )

            if len(pts) == 0:
                print("Warning: No spots detected initially.")  # Use print for non-critical warning
                # Optionally try lower percentile if no spots found
                lower_perc = max(85.0, perc - 5.0)  # Example fallback
                print(f"Retrying spot detection with percentile {lower_perc:.1f}%...")
                # --- MODIFIED: Call new function in fallback ---
                pts = detect_spots_by_centroid(
                    arr, perc=lower_perc, min_area=min_area, max_area=max_area, max_spots=max_pts
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


class SAEDApp(tk.Tk):  # 772
    # 773
    (  # 774
        "Backwards-compatible standalone application using the tab frame.\n"  # 775
    )  # 776

    # 777
    def __init__(self):  # 778
        super().__init__()  # 779
        # 780
        self.title("SAED Symmetry – Launcher")  # 781
        # 782
        self.geometry("980x680")  # 783
        # 784
        self.resizable(True, False)  # 785
        # 786
        frame = SAEDLauncherFrame(self)  # 787
        # 788
        frame.pack(fill=tk.BOTH, expand=True)  # 789


# 790
if __name__ == "__main__":  # 791
    SAEDApp().mainloop()  # 792