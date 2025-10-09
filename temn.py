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
import json, subprocess, sys  # 19
from pathlib import Path  # 20
from dataclasses import dataclass  # 21
from typing import Tuple  # 22
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
def detect_spots(  # 39
    arr: np.ndarray,  # 40
    perc: float = 99.0,  # 41
    win: int = 7,  # 42
    min_sep: int = 5,  # 43
    max_spots: int = 6000,  # 44
) -> np.ndarray:  # 45
    H, W = arr.shape  # 46
    rad = max(1, win // 2)  # 47
    percent_map, _, _ = compute_percentile_map(arr)  # 48
    perc = float(np.clip(perc, 0.0, 100.0))  # 49
    th = float(np.percentile(percent_map, perc))  # 50
    cand = []  # 51
    for y in range(rad, H-rad):  # 52
        for x in range(rad, W-rad):  # 53
            v = percent_map[y, x]  # 54
            if v < th: continue  # 55
            patch = percent_map[y-rad:y+rad+1, x-rad:x+rad+1]  # 56
            if v >= patch.max():  # 57
                cand.append((float(y), float(x), float(v)))  # 58
    cand.sort(key=lambda t: -t[2])  # 59
    kept = []  # 60
    msep2 = float(min_sep)**2  # 61
    for (y, x, v) in cand:  # 62
        if all((y - y0) ** 2 + (x - x0) ** 2 >= msep2 for (y0, x0, _) in kept):  # 63
            kept.append((y, x, v))  # 64
        if len(kept) >= max_spots:  # 65
            break  # 66
    return np.array(kept, dtype=float) if kept else np.zeros((0, 3), dtype=float)  # 67
# 68
# 69
def merge_spots_by_intensity(  # 70
    pts: np.ndarray,  # 71
    radius: float,  # 72
    tol_percent: float,  # 73
    *,  # 74
    min_intensity: float | None = None,  # 75
    line_image: np.ndarray | None = None,  # 76
    percentile_map: np.ndarray | None = None,  # 77
) -> np.ndarray:  # 78
    (  # 79
        "Merge nearby points with similar intensity.\n"  # 80
        "\n"  # 81
        "Point intensities (the third coordinate) are expected in the 0..100 percentile scale.\n"  # 82
        "\n"  # 83
        "radius: neighbour search radius in pixels.\n"  # 84
        "tol_percent: relative intensity tolerance, given in percent of the brighter\n"  # 85
        "             of the compared points.\n"  # 86
        "min_intensity: if provided, merging applies only to points whose intensity\n"  # 87
        "               is not below this threshold (also in percent). The rest are\n"  # 88
        "               returned unchanged.\n"  # 89
        "line_image: original image (before converting to percentiles), used to\n"  # 90
        "            detect intensity dips along the line between points. Ignored if\n"  # 91
        "            ``percentile_map`` is provided.\n"  # 92
        "percentile_map: intensity map in the 0..100 scale matching ``line_image``.\n"  # 93
        "                If not provided, it will be computed from ``line_image``.\n"  # 94
        "                If neither is provided, the line check is disabled.\n"  # 95
    )  # 96
    if pts.size == 0:  # 97
        return pts  # 98
    radius = float(radius)  # 99
    tol = max(0.0, float(tol_percent) / 100.0)  # 100
    if radius <= 0.0:  # 101
        return np.asarray(pts, dtype=float)  # 102
# 103
    pts = np.asarray(pts, dtype=float)  # 104
    if min_intensity is not None:  # 105
        mask = pts[:, 2] >= float(min_intensity)  # 106
    else:  # 107
        mask = np.ones(len(pts), dtype=bool)  # 108
# 109
    to_merge = pts[mask]  # 110
    untouched = pts[~mask]  # 111
    if to_merge.size == 0:  # 112
        return pts  # 113
# 114
    intensity_map: np.ndarray | None  # 115
    if percentile_map is not None:  # 116
        intensity_map = np.asarray(percentile_map, dtype=float)  # 117
    elif line_image is not None:  # 118
        percent, _, _ = compute_percentile_map(np.asarray(line_image, dtype=float))  # 119
        intensity_map = percent  # 120
    else:  # 121
        intensity_map = None  # 122
# 123
    rad2 = radius * radius  # 124
    used = np.zeros(len(to_merge), dtype=bool)  # 125
    order = np.argsort(-to_merge[:, 2])  # start from the brightest  # 126
    merged: list[tuple[float, float, float]] = []  # 127
# 128
    def within_tol(a: float, b: float) -> bool:  # 129
        hi = max(a, b)  # 130
        if hi == 0.0:  # 131
            return abs(a - b) == 0.0  # 132
        return abs(a - b) <= tol * hi + 1e-12  # 133
# 134
    H = W = None  # 135
    if intensity_map is not None and intensity_map.ndim == 2:  # 136
        H, W = intensity_map.shape  # 137
    else:  # 138
        intensity_map = None  # 139
# 140
    def clamp_round(val: float, hi: int) -> int:  # 141
        return int(min(max(round(float(val)), 0), hi))  # 142
# 143
    def bresenham_line(y0: int, x0: int, y1: int, x1: int) -> list[tuple[int, int]]:  # 144
        points: list[tuple[int, int]] = []  # 145
        dy = abs(y1 - y0)  # 146
        dx = abs(x1 - x0)  # 147
        sy = 1 if y0 < y1 else -1  # 148
        sx = 1 if x0 < x1 else -1  # 149
        err = dx - dy  # 150
        while True:  # 151
            points.append((y0, x0))  # 152
            if y0 == y1 and x0 == x1:  # 153
                break  # 154
            e2 = err * 2  # 155
            if e2 > -dy:  # 156
                err -= dy  # 157
                x0 += sx  # 158
            if e2 < dx:  # 159
                err += dx  # 160
                y0 += sy  # 161
        return points  # 162
# 163
    def has_intensity_dip(idx_a: int, idx_b: int) -> bool:  # 164
        if intensity_map is None or H is None or W is None:  # 165
            return False  # 166
        base_val = min(float(to_merge[idx_a, 2]), float(to_merge[idx_b, 2]))  # 167
        if base_val <= 0.0:  # 168
            return False  # 169
        y0 = clamp_round(to_merge[idx_a, 0], H - 1)  # 170
        x0 = clamp_round(to_merge[idx_a, 1], W - 1)  # 171
        y1 = clamp_round(to_merge[idx_b, 0], H - 1)  # 172
        x1 = clamp_round(to_merge[idx_b, 1], W - 1)  # 173
        pixels = bresenham_line(y0, x0, y1, x1)  # 174
        if len(pixels) <= 2:  # 175
            return False  # 176
        limit = base_val * (1.0 - tol)  # 177
        for (yy, xx) in pixels[1:-1]:  # 178
            if 0 <= yy < H and 0 <= xx < W:  # 179
                if float(intensity_map[yy, xx]) + 1e-9 < limit:  # 180
                    return True  # 181
        return False  # 182
# 183
    for idx in order:  # 184
        if used[idx]:  # 185
            continue  # 186
# 187
        cluster = [idx]  # 188
        sum_y = float(to_merge[idx, 0])  # 189
        sum_x = float(to_merge[idx, 1])  # 190
        intensities = [float(to_merge[idx, 2])]  # 191
        sum_v = intensities[0]  # 192
# 193
        neighbors = []  # 194
        base_y, base_x = to_merge[idx, 0], to_merge[idx, 1]  # 195
        for j in range(len(to_merge)):  # 196
            if j == idx or used[j]:  # 197
                continue  # 198
            dy = to_merge[j, 0] - base_y  # 199
            dx = to_merge[j, 1] - base_x  # 200
            if dy * dy + dx * dx <= rad2:  # 201
                neighbors.append(j)  # 202
# 203
        neighbors.sort(key=lambda j: abs(to_merge[j, 2] - intensities[0]))  # 204
# 205
        for j in neighbors:  # 206
            if used[j]:  # 207
                continue  # 208
            if any(has_intensity_dip(existing, j) for existing in cluster):  # 209
                continue  # 210
            cand_v = float(to_merge[j, 2])  # 211
            new_count = len(cluster) + 1  # 212
            new_avg_v = (sum_v + cand_v) / new_count  # 213
            if all(within_tol(new_avg_v, val) for val in (*intensities, cand_v)):  # 214
                cluster.append(j)  # 215
                intensities.append(cand_v)  # 216
                sum_y += float(to_merge[j, 0])  # 217
                sum_x += float(to_merge[j, 1])  # 218
                sum_v += cand_v  # 219
# 220
        if len(cluster) > 1:  # 221
            new_count = len(cluster)  # 222
            new_y = sum_y / new_count  # 223
            new_x = sum_x / new_count  # 224
            new_v = sum_v / new_count  # 225
            if (min_intensity is None or new_v >= min_intensity) and all(  # 226
                within_tol(new_v, val) for val in intensities  # 227
            ):  # 228
                merged.append((new_y, new_x, new_v))  # 229
                for j in cluster:  # 230
                    used[j] = True  # 231
                continue  # 232
# 233
        # either a single-point cluster or the resulting intensity exceeded the tolerance  # 234
        for j in cluster:  # 235
            if not used[j]:  # 236
                merged.append(tuple(to_merge[j]))  # 237
                used[j] = True  # 238
# 239
    merged = np.array(merged, dtype=float)  # 240
    if untouched.size == 0:  # 241
        return merged  # 242
    if merged.size == 0:  # 243
        return untouched  # 244
    return np.vstack((merged, untouched))  # 245
# 246
# 247
def geometric_midpoint(arr: np.ndarray) -> CenterResult:  # 248
    H, W = arr.shape  # 249
    return CenterResult(cy=(H - 1) / 2.0, cx=(W - 1) / 2.0, method="midpoint")  # 250
def refine_center_antipodal(center: Tuple[float,float], pts: np.ndarray, tol_ang_deg: float=8.0, tol_rel_r: float=0.06, iters: int=3) -> CenterResult:  # 251
    cy, cx = float(center[0]), float(center[1])  # 252
    if len(pts) < 4:  # 253
        return CenterResult(cy=cy, cx=cx, method="midpoint (fallback)")  # 254
    for _ in range(max(0, int(iters))):  # 255
        dy = pts[:,0]-cy; dx = pts[:,1]-cx  # 256
        r = np.hypot(dx, dy)  # 257
        u = np.column_stack((dx, dy)) / (r[:,None]+1e-9)  # 258
        cos_thr = -np.cos(np.deg2rad(180.0 - float(tol_ang_deg)))  # 259
        mids = []  # 260
        for i in range(len(pts)):  # 261
            dots = (u @ u[i])  # 262
            rad_ok = (np.abs(r - r[i]) / np.maximum(r, r[i]) < float(tol_rel_r))  # 263
            ang_ok = (dots < cos_thr)  # 264
            idx = np.where(rad_ok & ang_ok)[0]  # 265
            if idx.size == 0: continue  # 266
            j = idx[np.argmin(np.abs(dots[idx] + 1.0))]  # 267
            yi, xi = pts[i,0], pts[i,1]  # 268
            yj, xj = pts[j,0], pts[j,1]  # 269
            mids.append(((yi+yj)/2.0, (xi+xj)/2.0))  # 270
        if len(mids) < 4: break  # 271
        mids = np.array(mids, dtype=float)  # 272
        cy = float(np.median(mids[:,0])); cx = float(np.median(mids[:,1]))  # 273
    return CenterResult(cy=cy, cx=cx, method="antipodal-refined")  # 274
# 275
# -------------------------- GUI --------------------------  # 276
class SAEDLauncherFrame(ttk.Frame):  # 277
    (  # 278
        "Launcher tab suitable for both the standalone application and notebooks.\n"  # 279
    )  # 280
# 281
    def __init__(self, master: tk.Misc, controller=None):  # 282
        super().__init__(master)  # 283
        self.controller = controller  # 284
        self._scroll_canvas = None  # 285
        self._scroll_window_id = None  # 286
        self._build_ui()  # 287
# 288
    def _build_ui(self):  # 289
        outer = ttk.Frame(self)  # 290
        outer.pack(fill=tk.BOTH, expand=True)  # 291
# 292
        fixed = ttk.Frame(outer, padding=(16, 16, 16, 0))  # 293
        fixed.pack(side=tk.TOP, fill=tk.X)  # 294
        fixed.grid_columnconfigure(0, weight=1)  # 295
# 296
        data_box = ttk.LabelFrame(fixed, text="Input data", padding=(12, 10, 12, 12))  # 297
        data_box.grid(row=0, column=0, sticky="nsew")  # 298
        for col in (0, 1, 2, 3):  # 299
            weight = 1 if col == 1 else 0  # 300
            data_box.grid_columnconfigure(col, weight=weight)  # 301
# 302
        ttk.Label(data_box, text="Image:").grid(row=0, column=0, sticky="w", padx=6, pady=4)  # 303
        self.ent_img = ttk.Entry(data_box)  # 304
        self.ent_img.grid(row=0, column=1, columnspan=2, sticky="we", padx=6, pady=4)  # 305
        ttk.Button(data_box, text="Browse…", command=self._browse_img).grid(row=0, column=3, sticky="ew", padx=6, pady=4)  # 306
# 307
        ttk.Label(data_box, text="Output folder:").grid(row=1, column=0, sticky="w", padx=6, pady=4)  # 308
        self.ent_out = ttk.Entry(data_box)  # 309
        self.ent_out.insert(0, "saed_results")  # 310
        self.ent_out.grid(row=1, column=1, columnspan=2, sticky="we", padx=6, pady=4)  # 311
        ttk.Button(data_box, text="Choose…", command=self._browse_out).grid(row=1, column=3, sticky="ew", padx=6, pady=4)  # 312
# 313
        ttk.Label(data_box, text="Center X (optional):").grid(row=2, column=0, sticky="w", padx=6, pady=4)  # 314
        self.ent_cx = ttk.Entry(data_box, width=12)  # 315
        self.ent_cx.grid(row=2, column=1, sticky="w", padx=6, pady=4)  # 316
        ttk.Label(data_box, text="Center Y:").grid(row=2, column=2, sticky="w", padx=6, pady=4)  # 317
        self.ent_cy = ttk.Entry(data_box, width=12)  # 318
        self.ent_cy.grid(row=2, column=3, sticky="w", padx=6, pady=4)  # 319
# 320
        ttk.Label(  # 321
            data_box,  # 322
            text="Leave the coordinates empty to let the program find the center automatically.",  # 323
            wraplength=520,  # 324
            foreground="#555555"  # 325
        ).grid(row=3, column=0, columnspan=4, sticky="we", padx=6, pady=(0, 4))  # 326
# 327
        pre_box = ttk.LabelFrame(fixed, text="Preprocessing", padding=(12, 10, 12, 12))  # 328
        pre_box.grid(row=1, column=0, sticky="nsew", pady=(10, 0))  # 329
        pre_box.grid_columnconfigure(1, weight=1)  # 330
# 331
        ttk.Label(pre_box, text="Mode:").grid(row=0, column=0, sticky="w", padx=6, pady=4)  # 332
        self.cmb_pre = ttk.Combobox(  # 333
            pre_box,  # 334
            values=["No smoothing","Standard", "CLAHE"],  # 335
            state="readonly",  # 336
        )  # 337
        self.cmb_pre.current(0)  # 338
        self.cmb_pre.grid(row=0, column=1, sticky="w", padx=6, pady=4)  # 339
        self.cmb_pre.bind("<<ComboboxSelected>>", self._on_preproc_change)  # 340
# 341
        ttk.Label(pre_box, text="CLAHE clipLimit / tile:").grid(row=1, column=0, sticky="w", padx=6, pady=4)  # 342
        self.spn_clip = ttk.Spinbox(pre_box, from_=0.1, to=10.0, increment=0.1, width=8, justify="right")  # 343
        self._set_spinbox_value(self.spn_clip, 1.5)  # 344
        self.spn_clip.grid(row=1, column=1, sticky="w", padx=6, pady=4)  # 345
        self.spn_tile = ttk.Spinbox(pre_box, from_=2, to=64, increment=1, width=8, justify="right")  # 346
        self._set_spinbox_value(self.spn_tile, 8)  # 347
        self.spn_tile.grid(row=1, column=2, sticky="w", padx=6, pady=4)  # 348
# 349
        ttk.Label(  # 350
            pre_box,  # 351
            text="Select CLAHE for images with strong brightness variations. ClipLimit controls contrast, and tile size defines the local processing radius.",  # 352
            wraplength=520,  # 353
            foreground="#555555"  # 354
        ).grid(row=2, column=0, columnspan=3, sticky="we", padx=6, pady=(2, 0))  # 355
# 356
        scroll_host = ttk.Frame(outer)  # 357
        scroll_host.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  # 358
# 359
        canvas = tk.Canvas(scroll_host, borderwidth=0, highlightthickness=0)  # 360
        vscroll = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=canvas.yview)  # 361
        scrollable = ttk.Frame(canvas, padding=(16, 12, 16, 12))  # 362
        scrollable.grid_columnconfigure(0, weight=1)  # 363
# 364
        self._scroll_canvas = canvas  # 365
        self._scroll_window_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")  # 366
        canvas.configure(yscrollcommand=vscroll.set)  # 367
# 368
        scrollable.bind(  # 369
            "<Configure>",  # 370
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))  # 371
        )  # 372
        canvas.bind(  # 373
            "<Configure>",  # 374
            lambda e: canvas.itemconfigure(self._scroll_window_id, width=e.width)  # 375
        )  # 376
# 377
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # 378
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)  # 379
# 380
        scrollable.bind("<Enter>", self._activate_scroll)  # 381
        scrollable.bind("<Leave>", self._deactivate_scroll)  # 382
        canvas.bind("<Enter>", self._activate_scroll)  # 383
        canvas.bind("<Leave>", self._deactivate_scroll)  # 384
# 385
        detect_box = ttk.LabelFrame(scrollable, text="Detector and refinement", padding=(12, 10, 12, 12))  # 386
        detect_box.grid(row=0, column=0, sticky="nsew")  # 387
        detect_box.grid_columnconfigure(1, weight=1)  # 388
# 389
        ttk.Label(  # 390
            detect_box,  # 391
            text="Peak threshold and search window",  # 392
            font=("TkDefaultFont", 10, "bold")  # 393
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))  # 394
        self.spn_perc = self._spin_param(  # 395
            detect_box, 1, "Detection percentile (%)", 99.0,  # 396
            from_=80.0, to=100.0, increment=0.1, format_str="%.1f"  # 397
        )  # 398
        self.spn_merge_perc = self._spin_param(  # 399
            detect_box, 2, "Intensity percentile for merging (%)", 95.0,  # 400
            from_=0.0, to=100.0, increment=0.5, format_str="%.1f"  # 401
        )  # 402
        self.spn_merge_rad = self._spin_param(  # 403
            detect_box, 3, "Peak merging radius (px)", 0,  # 404
            from_=0, to=50, increment=1  # 405
        )  # 406
        self.spn_merge_tol = self._spin_param(  # 407
            detect_box, 4, "Intensity similarity tolerance (%)", 10.0,  # 408
            from_=0.0, to=100.0, increment=0.5, format_str="%.1f"  # 409
        )  # 410
        self.spn_minsep = self._spin_param(  # 411
            detect_box, 5, "Min. distance between peaks (px)", 5,  # 412
            from_=1, to=50, increment=1  # 413
        )  # 414
        self.spn_maxpts = self._spin_param(  # 415
            detect_box, 6, "Maximum detected points", 6000,  # 416
            from_=100, to=20000, increment=100  # 417
        )  # 418
# 419
        ttk.Separator(detect_box).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(6, 8))  # 420
# 421
        ttk.Label(  # 422
            detect_box,  # 423
            text="Center refinement",  # 424
            font=("TkDefaultFont", 10, "bold")  # 425
        ).grid(row=8, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))  # 426
        self.spn_iters = self._spin_param(  # 427
            detect_box, 9, "Center refinement iterations", 4,  # 428
            from_=0, to=10, increment=1  # 429
        )  # 430
        self.spn_tolang = self._spin_param(  # 431
            detect_box, 10, "Antipode tolerance (°)", 8.0,  # 432
            from_=1.0, to=30.0, increment=0.5, format_str="%.1f"  # 433
        )  # 434
        self.spn_tolr = self._spin_param(  # 435
            detect_box, 11, "Radius tolerance (relative)", 0.06,  # 436
            from_=0.01, to=0.5, increment=0.01, format_str="%.2f"  # 437
        )  # 438
# 439
        ttk.Separator(detect_box).grid(row=12, column=0, columnspan=2, sticky="ew", pady=(6, 8))  # 440
# 441
        ttk.Label(  # 442
            detect_box,  # 443
            text="Geometric filters",  # 444
            font=("TkDefaultFont", 10, "bold")  # 445
        ).grid(row=13, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))  # 446
        self.spn_dead = self._spin_param(  # 447
            detect_box, 14, "Dead zone (px)", 0,  # 448
            from_=0, to=500, increment=1  # 449
        )  # 450
        self.spn_search = self._spin_param(  # 451
            detect_box, 15, "Search radius (px, 0 = unlimited)", 0,  # 452
            from_=0, to=10000, increment=25  # 453
        )  # 454
# 455
        ttk.Label(  # 456
            detect_box,  # 457
            text="Use the dead zone to filter out the overexposed center. The search radius limits outer rings"  # 458
                 " and speeds up processing.",  # 459
            wraplength=520,  # 460
            foreground="#555555"  # 461
        ).grid(row=16, column=0, columnspan=2, sticky="we", padx=6, pady=(2, 0))  # 462
# 463
        action_box = ttk.Frame(scrollable, padding=(0, 12, 0, 0))  # 464
        action_box.grid(row=1, column=0, sticky="nsew")  # 465
        action_box.grid_columnconfigure(0, weight=1)  # 466
# 467
        ttk.Label(  # 468
            action_box,  # 469
            text="Review the parameters and press the button below to switch to interactive editing of detected points.",  # 470
            wraplength=540,  # 471
            justify="left"  # 472
        ).grid(row=0, column=0, sticky="we", padx=4, pady=(0, 8))  # 473
# 474
        ttk.Button(action_box, text="Open point editor", command=self._go_editor).grid(  # 475
            row=1, column=0, sticky="ew", padx=4, pady=(0, 12)  # 476
        )  # 477
# 478
        filler_bg = ttk.Style().lookup("TFrame", "background") or self.winfo_toplevel().cget("background")  # 479
        bottom_filler = tk.Frame(scrollable, height=56, bg=filler_bg)  # 480
        bottom_filler.grid(row=2, column=0, sticky="ew")  # 481
        bottom_filler.grid_propagate(False)  # 482
# 483
        self._on_preproc_change(None)  # 484
# 485
    def _activate_scroll(self, _event):  # 486
        if self._scroll_canvas is None:  # 487
            return  # 488
        self._scroll_canvas.bind_all("<MouseWheel>", self._on_scroll_mousewheel)  # 489
        self._scroll_canvas.bind_all("<Button-4>", self._on_scroll_mousewheel)  # 490
        self._scroll_canvas.bind_all("<Button-5>", self._on_scroll_mousewheel)  # 491
# 492
    def _deactivate_scroll(self, _event):  # 493
        if self._scroll_canvas is None:  # 494
            return  # 495
        self._scroll_canvas.unbind_all("<MouseWheel>")  # 496
        self._scroll_canvas.unbind_all("<Button-4>")  # 497
        self._scroll_canvas.unbind_all("<Button-5>")  # 498
# 499
    def _on_scroll_mousewheel(self, event):  # 500
        if self._scroll_canvas is None:  # 501
            return  # 502
        if getattr(event, "delta", 0):  # 503
            self._scroll_canvas.yview_scroll(int(-event.delta / 120), "units")  # 504
        elif getattr(event, "num", None) == 4:  # 505
            self._scroll_canvas.yview_scroll(-1, "units")  # 506
        elif getattr(event, "num", None) == 5:  # 507
            self._scroll_canvas.yview_scroll(1, "units")  # 508
# 509
    def _on_preproc_change(self, _evt):  # 510
        mode = self.cmb_pre.get()  # 511
        clahe_enabled = (mode == "CLAHE")  # 512
        state = "normal" if clahe_enabled else "disabled"  # 513
        self.spn_clip.configure(state=state)  # 514
        self.spn_tile.configure(state=state)  # 515
# 516
    def _spin_param(self, parent, row, label, default, *, from_, to, increment, format_str=None):  # 517
        ttk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky="w", padx=6, pady=4)  # 518
        spin = ttk.Spinbox(parent, from_=from_, to=to, increment=increment, width=10, justify="right")  # 519
        if format_str:  # 520
            spin.configure(format=format_str)  # 521
        self._set_spinbox_value(spin, default)  # 522
        spin.grid(row=row, column=1, sticky="w", padx=6, pady=4)  # 523
        return spin  # 524
# 525
    def _set_spinbox_value(self, spinbox: ttk.Spinbox, value):  # 526
        try:  # 527
            spinbox.set(value)  # 528
        except tk.TclError:  # 529
            spinbox.delete(0, tk.END)  # 530
            spinbox.insert(0, str(value))  # 531
# 532
    def _browse_img(self):  # 533
        p = filedialog.askopenfilename(title="Select image", filetypes=[("Images","*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),("All","*.*")])  # 534
        if p: self.ent_img.delete(0, tk.END); self.ent_img.insert(0, p)  # 535
# 536
    def _browse_out(self):  # 537
        p = filedialog.askdirectory(title="Select output folder")  # 538
        if p: self.ent_out.delete(0, tk.END); self.ent_out.insert(0, p)  # 539
# 540
    def _go_editor(self):  # 541
        try:  # 542
            image_path = Path(self.ent_img.get()).expanduser()  # 543
            if not image_path.exists():  # 544
                messagebox.showerror("Error", "Image not found"); return  # 545
            outdir = Path(self.ent_out.get()).expanduser(); outdir.mkdir(parents=True, exist_ok=True)  # 546
# 547
            perc = float(self.spn_perc.get())  # 548
            merge_apply_perc = float(self.spn_merge_perc.get())  # 549
            merge_radius = float(self.spn_merge_rad.get())  # 550
            merge_tol = float(self.spn_merge_tol.get())  # 551
            min_sep = int(float(self.spn_minsep.get()))  # 552
            max_pts = int(float(self.spn_maxpts.get()))  # 553
            iters = int(float(self.spn_iters.get()))  # 554
            tol_ang = float(self.spn_tolang.get())  # 555
            tol_relr = float(self.spn_tolr.get())  # 556
            dead_r = float(self.spn_dead.get())  # 557
            search_r = float(self.spn_search.get())  # 558
# 559
            # --- preprocessing ---  # 560
            pre_mode = self.cmb_pre.get()  # 561
            if pre_mode == "No smoothing":  # 562
                settings = PreprocSettings(mode="raw")  # 563
            elif pre_mode == "Standard":  # 564
                settings = PreprocSettings(mode="standard")  # 565
            else:  # CLAHE  # 566
                clip = float(self.spn_clip.get())  # 567
                tiles = int(float(self.spn_tile.get()))  # 568
                settings = PreprocSettings(mode="clahe", clahe_clip=clip, clahe_tiles=tiles)  # 569
# 570
            arr = load_grayscale_with_preproc(image_path, settings)  # 571
            mode = settings.mode  # 572
            preproc_payload = settings.to_json()  # 573
# 574
            # Center: manual takes priority, otherwise geometric  # 575
            cx_txt = self.ent_cx.get().strip(); cy_txt = self.ent_cy.get().strip()  # 576
            if cx_txt and cy_txt:  # 577
                center0 = CenterResult(cy=float(cy_txt), cx=float(cx_txt), method="user")  # 578
            else:  # 579
                center0 = geometric_midpoint(arr)  # 580
# 581
            # Peak detection and center refinement  # 582
            loc_win = 7  # base size of the local maximum window  # 583
            pts = detect_spots(arr, perc=perc, win=loc_win, min_sep=min_sep, max_spots=max_pts)  # 584
            if len(pts) < 120:  # 585
                pts = detect_spots(arr, perc=max(97.5, perc - 1.0), win=loc_win, min_sep=min_sep, max_spots=max_pts)  # 586
# 587
            center = refine_center_antipodal((center0.cy, center0.cx), pts, tol_ang_deg=tol_ang, tol_rel_r=tol_relr, iters=iters)  # 588
# 589
            # Geometric filters  # 590
            if (dead_r > 0 or search_r > 0) and len(pts):  # 591
                dy = pts[:, 0] - center.cy;  # 592
                dx = pts[:, 1] - center.cx;  # 593
                r = np.hypot(dx, dy)  # 594
                mask = np.ones(len(pts), dtype=bool)  # 595
                if dead_r > 0:   mask &= (r >= dead_r)  # 596
                if search_r > 0: mask &= (r <= search_r)  # 597
                pts = pts[mask]  # 598
# 599
            if len(pts) and merge_radius > 0:  # 600
                merge_threshold = None  # 601
                if merge_apply_perc > 0.0:  # 602
                    merge_threshold = float(np.percentile(pts[:, 2], merge_apply_perc))  # 603
                pts = merge_spots_by_intensity(  # 604
                    pts,  # 605
                    radius=merge_radius,  # 606
                    tol_percent=merge_tol,  # 607
                    min_intensity=merge_threshold,  # 608
                    line_image=arr,  # 609
                )  # 610
# 611
            # --- saed_input.json (single file for the editor) ---  # 612
            # pts: ndarray [y, x, v]; store v as intensity  # 613
            points = [{"y": float(y), "x": float(x), "intensity": float(v)} for (y, x, v) in pts.tolist()]  # 614
            saed_input = {  # 615
                "image": str(image_path),  # 616
                "preproc_mode": mode,  # 617
                "preproc": preproc_payload,  # 618
                "center": {"x": float(center.cx), "y": float(center.cy), "method": center.method},  # 619
                "radii": {"dead": float(dead_r), "search": float(search_r)},  # 620
                "points": points  # 621
            }  # 622
            saed_input_path = image_path.with_name("saed_input.json")  # 623
            saed_input_path.write_text(json.dumps(saed_input, ensure_ascii=False, indent=2), encoding="utf-8")  # 624
# 625
            # launch the editor with a single input  # 626
            if self.controller is not None:  # 627
                try:  # 628
                    self.controller.open_editor(saed_input_path)  # 629
                except Exception as exc:  # 630
                    messagebox.showerror("Error", f"Failed to open the editor: {exc}")  # 631
            else:  # 632
                if getattr(sys, "frozen", False):  # 633
                    editor = Path(sys.executable).with_name("saed_editor.exe")  # 634
                    cmd = [str(editor), "--input", str(saed_input_path)]  # 635
                else:  # 636
                    editor_py = Path(__file__).with_name("saed_editor.py")  # 637
                    cmd = [sys.executable, str(editor_py), "--input", str(saed_input_path)]  # 638
                subprocess.Popen(cmd, shell=False)  # 639
# 640
            # center log (service info)  # 641
            (outdir/"center_init.json").write_text(json.dumps({  # 642
                "initial": {"x": center0.cx, "y": center0.cy, "method": center0.method},  # 643
                "refined": {"x": center.cx,  "y": center.cy,  "method": "antipodal-refined"},  # 644
                "dead_zone_px": dead_r,  # 645
                "search_radius_px": search_r,  # 646
                "preproc_mode": mode,  # 647
                "preproc": preproc_payload,  # 648
                "image_size": {"H": int(arr.shape[0]), "W": int(arr.shape[1])}  # 649
            }, indent=2), encoding="utf-8")  # 650
# 651
# 652
        except Exception as e:  # 653
# 654
            messagebox.showerror("Error", str(e))  # 655
# 656
        class SAEDApp(tk.Tk):  # 657
# 658
            (  # 659
                "Backwards-compatible standalone application using the tab frame.\n"  # 660
            )  # 661
# 662
            def __init__(self):  # 663
                super().__init__()  # 664
# 665
                self.title("SAED Symmetry – Launcher")  # 666
# 667
                self.geometry("980x680")  # 668
# 669
                self.resizable(True, False)  # 670
# 671
                frame = SAEDLauncherFrame(self)  # 672
# 673
                frame.pack(fill=tk.BOTH, expand=True)  # 674
# 675
        if __name__ == "__main__":  # 676
            SAEDApp().mainloop()  # 677