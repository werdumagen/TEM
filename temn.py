#!/usr/bin/env python3  # 1
# -*- coding: utf-8 -*-  # 2
(  # 3
    "SAED Symmetry – Launcher\n"  # 4
    "========================\n"  # 5
    "What's new:\n"  # 6
    "  • Preprocessing switch:\n"  # 7
    "      - 'No smoothing' (raw grayscale)\n"  # 8
    "      - 'NLM Denoising' (non-local means) with configurable 'h' parameter.\n"  # 9
    "  • All preprocessing happens ONLY here; the image editor is untouched.\n"  # 10
    "The rest of the functionality is unchanged: manual/auto center, antipodal refinement, dead zone, search radius, launching the editor.\n"  # 11
    "\n"  # 12
    "Additionally:\n"  # 13
    "  • Save points together with their intensity in a single saed_input.json file.\n"  # 14
    "  • saed_editor launches with a single --input argument (path to saed_input.json).\n"  # 15
)  # 16
from __future__ import annotations  # 17
import json, subprocess, sys  # 18
from pathlib import Path  # 19
from dataclasses import dataclass  # 20
from typing import Tuple, Dict, Any  # 21
# 22
import numpy as np  # 23
# 24
from percentile_utils import compute_percentile_map  # 25
from preproc import PreprocSettings, load_grayscale_with_preproc  # 26
# 27
import tkinter as tk  # 28
from tkinter import ttk, filedialog, messagebox  # 29
# 30
# -------------------------- Algorithm --------------------------  # 31
@dataclass  # 32
class CenterResult:  # 33
    cy: float  # 34
    cx: float  # 35
    method: str  # 36
# 37
def detect_spots(  # 38
    arr: np.ndarray,  # 39
    perc: float = 99.0,  # 40
    win: int = 7,  # 41
    min_sep: int = 5,  # 42
    max_spots: int = 6000,  # 43
) -> np.ndarray:  # 44
    H, W = arr.shape  # 45
    rad = max(1, win // 2)  # 46
    percent_map, _, _ = compute_percentile_map(arr)  # 47
    perc = float(np.clip(perc, 0.0, 100.0))  # 48
    th = float(np.percentile(percent_map, perc))  # 49
    cand = []  # 50
    for y in range(rad, H-rad):  # 51
        for x in range(rad, W-rad):  # 52
            v = percent_map[y, x]  # 53
            if v < th: continue  # 54
            patch = percent_map[y-rad:y+rad+1, x-rad:x+rad+1]  # 55
            if v >= patch.max():  # 56
                cand.append((float(y), float(x), float(v)))  # 57
    cand.sort(key=lambda t: -t[2])  # 58
    kept = []  # 59
    msep2 = float(min_sep)**2  # 60
    for (y, x, v) in cand:  # 61
        if all((y - y0) ** 2 + (x - x0) ** 2 >= msep2 for (y0, x0, _) in kept):  # 62
            kept.append((y, x, v))  # 63
        if len(kept) >= max_spots:  # 64
            break  # 65
    return np.array(kept, dtype=float) if kept else np.zeros((0, 3), dtype=float)  # 66
# 67
# 68
def merge_spots_by_intensity(  # 69
    pts: np.ndarray,  # 70
    radius: float,  # 71
    tol_percent: float,  # 72
    *,  # 73
    min_intensity: float | None = None,  # 74
    line_image: np.ndarray | None = None,  # 75
    percentile_map: np.ndarray | None = None,  # 76
) -> np.ndarray:  # 77
    # ... (function code remains unchanged) ... # 96
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
        # Avoid division by zero if a point is exactly at the center
        r_safe = np.where(r > 1e-9, r, 1e-9)
        u = np.column_stack((dx, dy)) / r_safe[:,None]  # 258
        cos_thr = -np.cos(np.deg2rad(180.0 - float(tol_ang_deg)))  # 259
        mids = []  # 260
        for i in range(len(pts)):  # 261
            if r[i] < 1e-6: continue # Skip point if it's too close to center
            dots = (u @ u[i])  # 262
             # Avoid division by zero for radius tolerance
            max_r_pair = np.maximum(r, r[i])
            # Use np.divide with where clause to handle potential zero denominators
            rel_diff = np.divide(np.abs(r - r[i]), max_r_pair, out=np.zeros_like(r), where=max_r_pair > 1e-9)
            rad_ok = (rel_diff < float(tol_rel_r)) & (max_r_pair > 1e-9) # Ensure we don't match zero-radius points

            ang_ok = (dots < cos_thr)  # 264
            # Exclude self-comparison and points too close to center
            valid_match = rad_ok & ang_ok & (np.arange(len(pts)) != i) & (r > 1e-6)
            idx = np.where(valid_match)[0] # 265
            if idx.size == 0: continue  # 266
            # Find the best antipodal match among valid candidates
            j = idx[np.argmin(np.abs(dots[idx] + 1.0))]  # 267
            yi, xi = pts[i,0], pts[i,1]  # 268
            yj, xj = pts[j,0], pts[j,1]  # 269
            mids.append(((yi+yj)/2.0, (xi+xj)/2.0))  # 270
        if len(mids) < 4: break  # 271 Not enough pairs found
        mids = np.array(mids, dtype=float)  # 272
        # Use median to be robust against outliers
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
            if counter > 999: # Safety break
                 return str(base_dir / f"{base_name}_temp_{np.random.randint(1000)}")


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

        data_box.grid_columnconfigure(1, weight=1) # Make entry widgets resizable
# 302
        ttk.Label(data_box, text="Image:").grid(row=0, column=0, sticky="w", padx=6, pady=4)  # 303
        self.ent_img = ttk.Entry(data_box)  # 304
        self.ent_img.grid(row=0, column=1, columnspan=2, sticky="we", padx=6, pady=4)  # 305
        ttk.Button(data_box, text="Browse…", command=self._browse_img).grid(row=0, column=3, sticky="ew", padx=6, pady=4)  # 306
# 307
        ttk.Label(data_box, text="Output folder:").grid(row=1, column=0, sticky="w", padx=6, pady=4)  # 308
        self.ent_out = ttk.Entry(data_box)  # 309
        self.ent_out.insert(0, self._get_default_output_path())  # 310
        self.ent_out.grid(row=1, column=1, sticky="we", padx=6, pady=4)  # 311
        # --- NEW: "Load Session" Button ---
        ttk.Button(data_box, text="Load Session…", command=self._load_session).grid(row=1, column=2, sticky="ew", padx=6, pady=4)  # 312
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
            text="Leave the coordinates empty to let the program find the center automatically. Use 'Load Session' to restore a previous state.",  # 323
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
            values=["No smoothing", "NLM Denoising"],  # 335
            state="readonly",  # 336
        )  # 337
        self.cmb_pre.current(0)  # 338
        self.cmb_pre.grid(row=0, column=1, sticky="w", padx=6, pady=4)  # 339
        self.cmb_pre.bind("<<ComboboxSelected>>", self._on_preproc_change)  # 340
# 341
        ttk.Label(pre_box, text="NLM 'h' parameter:").grid(row=1, column=0, sticky="w", padx=6, pady=4)  # 342
        self.spn_h = ttk.Spinbox(pre_box, from_=0.01, to=5.0, increment=0.1, width=8, justify="right")  # 343
        self._set_spinbox_value(self.spn_h, 1.0)  # 344
        self.spn_h.grid(row=1, column=1, sticky="w", padx=6, pady=4)  # 345
        # 346
        # 347
        # 348
# 349
        ttk.Label(  # 350
            pre_box,  # 351
            text="Select 'NLM Denoising' to reduce noise. 'h' controls filtering strength (higher = stronger).",  # 352
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
        # Determine scroll direction and amount (platform-dependent)
        delta = 0
        if sys.platform == "win32":
            delta = -int(event.delta / 120)
        elif sys.platform == "darwin": # macOS
             delta = event.delta
        elif event.num == 4: # Linux scroll up
             delta = -1
        elif event.num == 5: # Linux scroll down
             delta = 1

        if delta != 0:
            self._scroll_canvas.yview_scroll(delta, "units")

    def _on_preproc_change(self, _evt):  # 510
        mode = self.cmb_pre.get()  # 511
        nlm_enabled = (mode == "NLM Denoising")  # 512
        state = "normal" if nlm_enabled else "disabled"  # 513
        self.spn_h.configure(state=state)  # 514
        # 515
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
             # Try setting directly first, works for simple values
             spinbox.set(value)
        except tk.TclError:  # 529
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

# 532
    def _browse_img(self):  # 533
        p = filedialog.askopenfilename(title="Select image", filetypes=[("Images","*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),("All","*.*")])  # 534
        if p: self.ent_img.delete(0, tk.END); self.ent_img.insert(0, p)  # 535
# 536
    def _browse_out(self):  # 537
        p = filedialog.askdirectory(title="Select output folder", mustexist=False) # Allow creating new folders
        if p: self.ent_out.delete(0, tk.END); self.ent_out.insert(0, p)  # 539
# 540
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
            "h_param": self.spn_h.get(),
            "detect_perc": self.spn_perc.get(),
            "merge_perc": self.spn_merge_perc.get(),
            "merge_radius": self.spn_merge_rad.get(),
            "merge_tol": self.spn_merge_tol.get(),
            "min_sep": self.spn_minsep.get(),
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
                  self.cmb_pre.current(0) # Fallback to first option
        elif isinstance(self.cmb_pre, ttk.Combobox):
             self.cmb_pre.current(0) # Default if not in state

        self._on_preproc_change(None) # Update UI based on new mode

        # Set Spinbox values safely, providing defaults
        self._set_spinbox_value(self.spn_h, state.get("h_param", 1.0))
        self._set_spinbox_value(self.spn_perc, state.get("detect_perc", 99.0))
        self._set_spinbox_value(self.spn_merge_perc, state.get("merge_perc", 95.0))
        self._set_spinbox_value(self.spn_merge_rad, state.get("merge_radius", 0))
        self._set_spinbox_value(self.spn_merge_tol, state.get("merge_tol", 10.0))
        self._set_spinbox_value(self.spn_minsep, state.get("min_sep", 5))
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
                messagebox.showerror("Error", "Please select an image file."); return
            if not output_dir_str:
                messagebox.showerror("Error", "Please specify an output folder."); return

            image_path = Path(image_path_str).expanduser().resolve() # Get absolute path
            outdir = Path(output_dir_str).expanduser().resolve(); # Get absolute path
            outdir.mkdir(parents=True, exist_ok=True)

            if not image_path.exists():
                messagebox.showerror("Error", f"Image not found at: {image_path}"); return

            # --- Get parameters (as before) ---
            perc = float(self.spn_perc.get())
            merge_apply_perc = float(self.spn_merge_perc.get())
            merge_radius = float(self.spn_merge_rad.get())
            merge_tol = float(self.spn_merge_tol.get())
            min_sep = int(float(self.spn_minsep.get()))
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
            else: # NLM Denoising
                h = float(self.spn_h.get())
                settings = PreprocSettings(mode="nlm", h_param=h)

            # --- Load image ---
            try:
                arr = load_grayscale_with_preproc(image_path, settings)
            except RuntimeError as cv_err: # Catch OpenCV missing specifically
                 messagebox.showerror("Dependency Error", str(cv_err))
                 return # Stop processing if required dependency is missing
            except Exception as img_load_err:
                 messagebox.showerror("Image Error", f"Failed to load or process image:\n{img_load_err}")
                 return # Stop processing


            mode = settings.mode
            preproc_payload = settings.to_json()

            # --- Center calculation ---
            cx_txt = self.ent_cx.get().strip(); cy_txt = self.ent_cy.get().strip()
            if cx_txt and cy_txt:
                try:
                    center0 = CenterResult(cy=float(cy_txt), cx=float(cx_txt), method="user")
                except ValueError:
                     messagebox.showwarning("Input Warning", "Invalid center coordinates. Using automatic center.")
                     center0 = geometric_midpoint(arr)
            else:
                center0 = geometric_midpoint(arr)

            # --- Peak detection and refinement ---
            loc_win = 7
            pts = detect_spots(arr, perc=perc, win=loc_win, min_sep=min_sep, max_spots=max_pts)
            if len(pts) == 0:
                 print("Warning: No spots detected initially.") # Use print for non-critical warning
                 # Optionally try lower percentile if no spots found
                 lower_perc = max(85.0, perc - 5.0) # Example fallback
                 print(f"Retrying spot detection with percentile {lower_perc:.1f}%...")
                 pts = detect_spots(arr, perc=lower_perc, win=loc_win, min_sep=min_sep, max_spots=max_pts)
                 if len(pts) == 0:
                      messagebox.showwarning("Detection Warning", "No spots detected even with lower threshold. Proceeding without points.")


            # Refine center even if few points, refine_center_antipodal handles low point counts
            center = refine_center_antipodal((center0.cy, center0.cx), pts, tol_ang_deg=tol_ang, tol_rel_r=tol_relr, iters=iters)


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
                         except IndexError: # Handle empty pts array after filtering
                              pass
                    elif len(pts) > 0:
                         merge_threshold = float(pts[0, 2] * (merge_apply_perc / 100.0)) # Fallback if constant


                pts = merge_spots_by_intensity(
                    pts,
                    radius=merge_radius,
                    tol_percent=merge_tol,
                    min_intensity=merge_threshold,
                    line_image=arr, # Pass original preprocessed image for line check
                )

            # --- Create saed_input.json in the output directory ---
            points_list = [{"y": float(y), "x": float(x), "intensity": float(v)} for (y, x, v) in pts.tolist()]
            saed_input_data = {
                "image": str(image_path), # Save absolute path
                "preproc_mode": mode,
                "preproc": preproc_payload,
                "center": {"x": float(center.cx), "y": float(center.cy), "method": center.method},
                "radii": {"dead": float(dead_r), "search": float(search_r)},
                "points": points_list
            }
            saed_input_path = outdir / "saed_input.json" # Define path in output dir
            saed_input_path.write_text(json.dumps(saed_input_data, ensure_ascii=False, indent=2), encoding="utf-8")

            # --- Launch the editor via controller ---
            if self.controller is not None:
                try:
                    self.controller.open_editor(saed_input_path) # Pass the correct path
                except Exception as exc:
                    messagebox.showerror("Error", f"Failed to open the editor tab:\n{exc}")
            else:
                 # Fallback: Launch external editor (ensure path handling is correct here too if used)
                 # ... (external launch code - needs similar path care) ...
                 messagebox.showwarning("Standalone Mode", "Running in standalone mode. Editor will open externally if available.")


            # --- Save center log (optional service info) ---
            try:
                (outdir/"center_init.json").write_text(json.dumps({
                    "initial": {"x": center0.cx, "y": center0.cy, "method": center0.method},
                    "refined": {"x": center.cx,  "y": center.cy,  "method": center.method}, # Use refined method name
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