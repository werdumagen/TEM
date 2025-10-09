#!/usr/bin/env python3  # 1
# -*- coding: utf-8 -*-  # 2
"""  # 3
SAED Editor + Analysis  # 4
----------------------  # 5
• Middle mouse button (MMB) tooltip on a point: shows x, y, intensity; closes with Esc and any actions.  # 6
  # 7
Removed according to the spec:  # 8
• Moving regular points.  # 9
• Any mentions/functions of panning or moving the window with the middle mouse button.  # 10
  # 11
The remaining editor functionality is preserved.  # 12
"""  # 13
import sys, json, subprocess  # 14
from pathlib import Path  # 15
from typing import Optional  # 16
import tkinter as tk  # 17
from tkinter import ttk, filedialog, messagebox  # 18
import numpy as np  # 19
from percentile_utils import compute_percentile_map, map_values_to_percent  # 20
from preproc import PreprocSettings, load_grayscale_with_preproc  # 21
import matplotlib  # 22
matplotlib.use("TkAgg")  # 23
import matplotlib.pyplot as plt  # 24
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # 25
from matplotlib.patches import Circle  # 26
from scipy.signal import find_peaks  # 27
  # 28
# ------ симметрия (для отчёта) ------  # 29
def pol_from(center, pts):  # 30
    cy, cx = center  # 31
    dy, dx = pts[:, 0] - cy, pts[:, 1] - cx  # 32
    r = np.hypot(dx, dy)  # 33
    a = (np.degrees(np.arctan2(dy, dx)) + 360) % 360  # 34
    return r, a  # 35
  # 36
def cluster_rings(radii):  # 37
    if len(radii) == 0:  # 38
        return np.array([]), np.array([]), ([], [])  # 39
    hist, edges = np.histogram(radii, bins=60)  # 40
    centers = (edges[:-1] + edges[1:]) / 2  # 41
    pk, _ = find_peaks(hist, prominence=3)  # 42
    ring_centers = centers[pk]  # 43
    if len(ring_centers) == 0:  # 44
        return np.array([]), np.zeros_like(radii, int), (hist, edges)  # 45
    labels = np.argmin(np.abs(radii[:, None] - ring_centers[None, :]), axis=1)  # 46
    return ring_centers, labels, (hist, edges)  # 47
  # 48
def symmetry_scores(angles, radii, ring_means, top_rings=3):  # 49
    out = {}  # 50
    if not ring_means:  # 51
        return out  # 52
    idx = min(top_rings - 1, len(ring_means) - 1)  # 53
    maxR = ring_means[idx] * 1.15  # 54
    ang_sel = angles[radii <= maxR]  # 55
    for k in [4, 6, 8, 10, 12]:  # 56
        period = 360.0 / k  # 57
        phases = np.deg2rad((ang_sel % period) * k)  # 58
        C = np.cos(phases).mean(); S = np.sin(phases).mean()  # 59
        out[f"{k}-fold"] = float(np.hypot(C, S))  # 60
    return out  # 61
  # 62
class PointEditor(tk.Frame):  # 63
    def __init__(self, master: tk.Misc, controller=None,  # 64
                 input_json: str | None = None, auto_load: bool = True):  # 65
        super().__init__(master)  # 66
        self.controller = controller  # 67
        # данные  # 68
        self.points = np.zeros((0, 2), float)   # [y, x]  # 69
        self.values = np.zeros((0,), float)     # интенсивности (параллельно points)  # 70
        self.rect_start = None  # 71
        self.rect_artist = None  # 72
        self.overlay = None  # {center:{x,y}, dead_radius, search_radius}  # 73
        self.image_path: Optional[Path] = None  # 74
        self.img_arr: Optional[np.ndarray] = None  # 75
        self._percent_map: Optional[np.ndarray] = None  # 76
        self._percent_lookup: Optional[tuple[np.ndarray, np.ndarray]] = None  # 77
        self._preproc_settings: PreprocSettings = PreprocSettings(mode="raw")  # 78
  # 79
        # Undo/Redo  # 80
        self._undo = []  # 81
        self._redo = []  # 82
        self._history_cap = 300  # 83
  # 84
        # Перетаскивание центра  # 85
        self.center_dragging = False  # 86
        self._center_hit_radius = 10.0  # пикселей  # 87
  # 88
        # Масштаб (ползунок)  # 89
        self.zoom_val = 0  # 0..100  # 90
        self.view_cx = None  # 91
        self.view_cy = None  # 92
  # 93
        # Tooltip (подсказка на MMB по точке)  # 94
        self._tooltip = None          # matplotlib.text.Annotation  # 95
        self._tooltip_idx = None  # 96
  # 97
        # Измерение расстояний между точками  # 98
        self._measure_active = False  # 99
        self._measure_start_idx: Optional[int] = None  # 100
        self._measure_start_point: Optional[tuple[float, float]] = None  # 101
        self._measure_preview_end: Optional[tuple[float, float]] = None  # 102
        self._measure_preview_artist = None  # 103
        self._measure_line_artist = None  # 104
        self._measure_annotation = None  # 105
        self._measurement: Optional[dict[str, object]] = None  # 106
  # 107
        # Объединение точек по радиусу от выбранной  # 108
        self._merge_seed_idx: Optional[int] = None  # 109
        self._merge_seed_origin: Optional[tuple[float, float]] = None  # 110
        self._last_cursor_pos: Optional[tuple[float, float]] = None  # 111
  # 112
        self._build_ui()  # 113
  # 114
        # первичная загрузка  # 115
        if auto_load and input_json:  # 116
            self.load_input_json(Path(input_json), push_undo=False)  # 117
        else:  # 118
            self._ensure_view_center()  # 119
            self._redraw()  # 120
  # 121
    # ---------- UI ----------  # 122
    def _build_ui(self):  # 123
        self.columnconfigure(1, weight=1)  # 124
        self.rowconfigure(0, weight=1)  # 125
  # 126
        side_panel = ttk.Frame(self, padding=(16, 16, 12, 16))  # 127
        side_panel.grid(row=0, column=0, sticky="ns")  # 128
        side_panel.columnconfigure(0, weight=1)  # 129
  # 130
        controls = ttk.Frame(side_panel)  # 131
        controls.pack(side=tk.TOP, fill=tk.X)  # 132
  # 133
        header_row = ttk.Frame(controls)  # 134
        header_row.pack(fill=tk.X)  # 135
  # 136
        history_group = ttk.Frame(header_row)  # 137
        history_group.pack(side=tk.LEFT)  # 138
        ttk.Button(history_group, text="◀", width=3, command=self._undo_btn, style="Toolbutton").pack(  # 139
            side=tk.LEFT, padx=(0, 4)  # 140
        )  # 141
        ttk.Button(history_group, text="▶", width=3, command=self._redo_btn, style="Toolbutton").pack(  # 142
            side=tk.LEFT, padx=(0, 4)  # 143
        )  # 144
  # 145
        help_button = ttk.Button(header_row, text="?", width=3, command=self._toggle_help, style="Toolbutton")  # 146
        help_button.pack(side=tk.RIGHT)  # 147
  # 148
        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 10))  # 149
  # 150
        zoom_group = ttk.LabelFrame(controls, text="Scale", padding=(12, 8, 12, 10))  # 151
        zoom_group.pack(fill=tk.X)  # 152
        self.zoom_var = tk.DoubleVar(value=self.zoom_val)  # 153
        self.zoom_scale = ttk.Scale(zoom_group, from_=0, to=100, variable=self.zoom_var, command=self._on_zoom_change)  # 154
        self.zoom_scale.pack(fill=tk.X, padx=4, pady=(0, 6))  # 155
        self.zoom_hint = ttk.Label(zoom_group, anchor="w")  # 156
        self.zoom_hint.pack(fill=tk.X, padx=4)  # 157
        self.zoom_scale.set(self.zoom_val)  # 158
  # 159
        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 10))  # 160
  # 161
        file_group = ttk.Frame(controls)  # 162
        file_group.pack(fill=tk.X)  # 163
        ttk.Button(file_group, text="Open JSON…", command=self._open_json).pack(side=tk.LEFT, padx=(0, 6))  # 164
        ttk.Button(file_group, text="Save", command=self._save_points).pack(side=tk.LEFT, padx=(0, 6))  # 165
  # 166
        analysis_group = ttk.Frame(controls)  # 167
        analysis_group.pack(fill=tk.X, pady=(8, 0))  # 168
        ttk.Button(analysis_group, text="Start analysis", command=self._start_analysis).pack(side=tk.LEFT, padx=(0, 6))  # 169
  # 170
        self.help_panel = ttk.LabelFrame(side_panel, text="Hints", padding=(16, 12, 16, 12))  # 171
        help_text = (  # 172
            "Left mouse button on empty area — add a point\n"  # 173
            "Left mouse button on the center — drag the center and recalculate filters\n"  # 174
            "Right mouse button on a point — delete\n"  # 175
            "Middle mouse button on a point — show coordinates and intensity\n"  # 176
            "Left mouse button on a point — select; move the cursor and press Enter to merge nearby points\n"  # 177
            "Hold left mouse button from point to point — measure distance\n"  # 178
            "Shift + drag — rectangular range deletion"  # 179
        )  # 180
        ttk.Label(self.help_panel, text=help_text, justify="left", wraplength=780).pack(fill=tk.X)  # 181
        self._help_visible = False  # 182
  # 183
        self._side_spacer = ttk.Frame(side_panel)  # 184
        self._side_spacer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  # 185
  # 186
        status_frame = ttk.Frame(side_panel, padding=(0, 0, 0, 0))  # 187
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))  # 188
        self.status_label = ttk.Label(status_frame, anchor="w", justify="left")  # 189
        self.status_label.pack(fill=tk.X)  # 190
  # 191
        def _status_wrap(event, label=self.status_label):  # 192
            if not label.winfo_exists():  # 193
                return  # 194
            new_wrap = max(int(event.width) - 8, 120)  # 195
            current_wrap = int(label.cget("wraplength") or 0)  # 196
            if new_wrap != current_wrap:  # 197
                label.configure(wraplength=new_wrap)  # 198
  # 199
        self.status_label.bind("<Configure>", _status_wrap)  # 200
  # 201
        canvas_frame = ttk.Frame(self, padding=(0, 16, 16, 16))  # 202
        canvas_frame.grid(row=0, column=1, sticky="nsew")  # 203
        canvas_frame.rowconfigure(0, weight=1)  # 204
        canvas_frame.columnconfigure(0, weight=1)  # 205
  # 206
        self.fig = plt.Figure(figsize=(9.4, 6.4))  # 207
        self.ax = self.fig.add_subplot(111)  # 208
        self.ax.axis("off")  # 209
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)  # 210
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")  # 211
        self.canvas.mpl_connect("button_press_event", self._on_down)  # 212
        self.canvas.mpl_connect("button_release_event", self._on_up)  # 213
        self.canvas.mpl_connect("motion_notify_event", self._on_move)  # 214
        self.canvas.mpl_connect("key_press_event", self._on_key)  # 215
  # 216
        self._default_status = "Mode: point editor"  # 217
        self._status_message = ""  # 218
        self._update_zoom_hint()  # 219
        self._set_status(self._default_status)  # 220
  # 221
    def _toggle_help(self):  # 222
        self._help_visible = not self._help_visible  # 223
        if self._help_visible:  # 224
            self.help_panel.pack(side=tk.TOP, fill=tk.X, pady=(12, 8), before=self._side_spacer)  # 225
            self._set_status("Detailed hints expanded")  # 226
        else:  # 227
            self.help_panel.pack_forget()  # 228
  # 229
    def _update_zoom_hint(self):  # 230
        if hasattr(self, "zoom_hint"):  # 231
            value = int(round(self.zoom_var.get())) if hasattr(self, "zoom_var") else self.zoom_val  # 232
            self.zoom_hint.configure(text=f"Current zoom: {value}% (0 = full frame)")  # 233
  # 234
    def _set_status(self, text: str):  # 235
        self._status_message = text  # 236
        if hasattr(self, "status_label"):  # 237
            self.status_label.configure(text=text)  # 238
        if self.controller is not None and hasattr(self.controller, "set_status"):  # 239
            try:  # 240
                self.controller.set_status(text)  # 241
            except Exception:  # 242
                pass  # 243
  # 244
    # ---------- IO ----------  # 245
    def _open_json(self):  # 246
        p = filedialog.askopenfilename(filetypes=[("SAED Input JSON","*saed_input.json;*.json"), ("All","*.*")])  # 247
        if p:  # 248
            self.load_input_json(Path(p), push_undo=True)  # 249
  # 250
    def load_input_json(self, path: Path, *, push_undo: bool = False, reset_view: bool = True):  # 251
        """Public JSON loading method, also used by the tab controller."""  # 252
        if push_undo:  # 253
            self._push_undo()  # 254
        self._load_input_json(path)  # 255
        self._clear_tooltip()  # 256
        if reset_view:  # 257
            self.view_cx = None  # 258
            self.view_cy = None  # 259
        self._ensure_view_center()  # 260
        self._redraw()  # 261
        self._update_zoom_hint()  # 262
        self._set_status(f"Loaded: {path.name}")  # 263
  # 264
    def _load_input_json(self, path: Path):  # 265
        try:  # 266
            data = json.loads(path.read_text(encoding="utf-8"))  # 267
        except Exception as e:  # 268
            messagebox.showerror("Error", f"Failed to read JSON:\n{e}")  # 269
            return  # 270
  # 271
        self._percent_map = None  # 272
        self._percent_lookup = None  # 273
  # 274
        # загрузка изображения по пути из JSON  # 275
        img_path = data.get("image")  # 276
        if not img_path:  # 277
            messagebox.showerror("Error", "The JSON is missing the 'image' field.")  # 278
            return  # 279
        self.image_path = Path(img_path)  # 280
        fallback_mode = data.get("preproc_mode")  # 281
        if not isinstance(fallback_mode, str):  # 282
            fallback_mode = None  # 283
        self._preproc_settings = PreprocSettings.from_json(  # 284
            data.get("preproc"), fallback_mode=fallback_mode  # 285
        )  # 286
        try:  # 287
            self.img_arr = load_grayscale_with_preproc(self.image_path, self._preproc_settings)  # 288
        except Exception as e:  # 289
            messagebox.showerror("Error", f"Failed to prepare the image:\n{e}")  # 290
            return  # 291
  # 292
        self._percent_map, uniq_vals, uniq_perc = compute_percentile_map(self.img_arr)  # 293
        self._percent_lookup = (uniq_vals, uniq_perc)  # 294
  # 295
        # overlay: центр и радиусы  # 296
        c = data.get("center") or {}  # 297
        r = data.get("radii") or {}  # 298
        self.overlay = {  # 299
            "center": {"x": float(c.get("x", (self.img_arr.shape[1]-1)/2.0)),  # 300
                       "y": float(c.get("y", (self.img_arr.shape[0]-1)/2.0))},  # 301
            "dead_radius": float(r.get("dead") or 0.0),  # 302
            "search_radius": float(r.get("search") or 0.0),  # 303
        }  # 304
  # 305
        # точки: y,x,intensity (если интенсивности нет, рассчитать)  # 306
        pts = data.get("points", [])  # 307
        if pts:  # 308
            yy = [float(p.get("y")) for p in pts]  # 309
            xx = [float(p.get("x")) for p in pts]  # 310
            self.points = np.column_stack([yy, xx]).astype(float)  # 311
            if self._percent_map is not None:  # 312
                self.values = self._sample_intensities(self.points)  # 313
            elif any("intensity" in p for p in pts):  # 314
                vv = np.array([float(p.get("intensity", 0.0)) for p in pts], dtype=float)  # 315
                if self._percent_lookup is not None:  # 316
                    self.values = map_values_to_percent(vv, *self._percent_lookup)  # 317
                else:  # 318
                    self.values = vv  # 319
            else:  # 320
                self.values = np.zeros((len(self.points),), float)  # 321
        else:  # 322
            self.points = np.zeros((0,2), float)  # 323
            self.values = np.zeros((0,), float)  # 324
  # 325
    def _save_points(self) -> Path:  # 326
        """  # 327
        Saves points (including intensities) next to the original input.json:  # 328
        - spots.json  — list of points (y,x,intensity)  # 329
        - saed_input.edited.json — original input JSON with updated points  # 330
        """  # 331
        base = self.image_path.with_name("spots.json") if self.image_path else Path("spots.json")  # 332
        pts = []  # 333
        # пересэмплируем интенсивности в текущих координатах  # 334
        if self._percent_map is not None or self.img_arr is not None:  # 335
            vals = self._sample_intensities(self.points)  # 336
        else:  # 337
            vals = self.values  # 338
        for (y, x), v in zip(self.points, vals):  # 339
            pts.append({"y": float(y), "x": float(x), "intensity": float(v)})  # 340
        base.write_text(json.dumps({"points": pts}, indent=2), encoding="utf-8")  # 341
  # 342
        # также пересохраним обновлённый saed_input  # 343
        si = {  # 344
            "image": str(self.image_path) if self.image_path else None,  # 345
            "preproc_mode": self._preproc_settings.mode,  # 346
            "preproc": self._preproc_settings.to_json(),  # 347
            "center": (self.overlay.get("center") if self.overlay else None),  # 348
            "radii": {  # 349
                "dead": float(self.overlay.get("dead_radius") or 0.0) if self.overlay else 0.0,  # 350
                "search": float(self.overlay.get("search_radius") or 0.0) if self.overlay else 0.0  # 351
            },  # 352
            "points": pts  # 353
        }  # 354
        edited = self.image_path.with_name("saed_input.edited.json") if self.image_path else Path(  # 355
            "saed_input.edited.json")  # 356
        edited.write_text(json.dumps(si, ensure_ascii=False, indent=2), encoding="utf-8")  # 357
        self._set_status(f"Points saved: {base.name}")  # 358
        return base  # 359
  # 360
    # ---------- Helpers ----------  # 361
    def _sample_intensities(self, pts_yx: np.ndarray) -> np.ndarray:  # 362
        if len(pts_yx) == 0:  # 363
            return np.zeros((0,), float)  # 364
  # 365
        if self._percent_map is not None:  # 366
            src = self._percent_map  # 367
            H, W = src.shape[:2]  # 368
            out = []  # 369
            for y, x in pts_yx:  # 370
                yi = int(round(y)); xi = int(round(x))  # 371
                yi = max(0, min(H - 1, yi)); xi = max(0, min(W - 1, xi))  # 372
                out.append(float(src[yi, xi]))  # 373
            return np.array(out, float)  # 374
  # 375
        if self.img_arr is None:  # 376
            return np.zeros((len(pts_yx),), float)  # 377
  # 378
        H, W = self.img_arr.shape[:2]  # 379
        raw = []  # 380
        for y, x in pts_yx:  # 381
            yi = int(round(y)); xi = int(round(x))  # 382
            yi = max(0, min(H - 1, yi)); xi = max(0, min(W - 1, xi))  # 383
            raw.append(float(self.img_arr[yi, xi]))  # 384
        raw = np.array(raw, float)  # 385
        if self._percent_lookup is not None:  # 386
            return map_values_to_percent(raw, *self._percent_lookup)  # 387
        return raw  # 388
  # 389
    def _img_xy(self, e):  # 390
        return None if (e.xdata is None or e.ydata is None) else (e.ydata, e.xdata)  # 391
  # 392
    def _near_idx(self, y, x, pix_tol=8):  # 393
        if len(self.points) == 0: return None  # 394
        d2 = (self.points[:,0]-y)**2 + (self.points[:,1]-x)**2  # 395
        i = int(np.argmin(d2))  # 396
        return i if d2[i]**0.5 <= pix_tol else None  # 397
  # 398
    def _center_hit(self, y, x):  # 399
        if not (self.overlay and self.overlay.get("center")): return False  # 400
        cy = float(self.overlay["center"].get("y", 0.0))  # 401
        cx = float(self.overlay["center"].get("x", 0.0))  # 402
        return ((y - cy)**2 + (x - cx)**2) ** 0.5 <= self._center_hit_radius  # 403
  # 404
    def _apply_center_filters(self):  # 405
        """Removes points that end up in the dead zone or outside the search radius after moving the center."""  # 406
        if not (self.overlay and self.overlay.get("center")): return  # 407
        cy = float(self.overlay["center"].get("y", 0.0))  # 408
        cx = float(self.overlay["center"].get("x", 0.0))  # 409
        dead = float(self.overlay.get("dead_radius") or 0.0)  # 410
        sr   = float(self.overlay.get("search_radius") or 0.0)  # 411
        if len(self.points) == 0 or (dead <= 0 and sr <= 0): return  # 412
        r = np.hypot(self.points[:,1]-cx, self.points[:,0]-cy)  # 413
        mask = np.ones(len(self.points), dtype=bool)  # 414
        if dead > 0: mask &= (r >= dead)  # 415
        if sr   > 0: mask &= (r <= sr)  # 416
        if self._merge_seed_idx is not None:  # 417
            if self._merge_seed_idx >= len(mask) or not mask[self._merge_seed_idx]:  # 418
                self._clear_merge_seed()  # 419
            else:  # 420
                new_idx = int(np.count_nonzero(mask[: self._merge_seed_idx + 1]) - 1)  # 421
                self._merge_seed_idx = new_idx  # 422
        self.points = self.points[mask]  # 423
        if len(self.values)==len(mask):  # 424
            self.values = self.values[mask]  # 425
        else:  # 426
            self.values = self._sample_intensities(self.points)  # 427
        if self._merge_seed_idx is not None and self._merge_seed_idx < len(self.points):  # 428
            self._merge_seed_origin = (  # 429
                float(self.points[self._merge_seed_idx, 0]),  # 430
                float(self.points[self._merge_seed_idx, 1]),  # 431
            )  # 432
  # 433
    # ---------- Объединение точек ----------  # 434
    def _clear_merge_seed(self, *, keep_status: bool = False) -> bool:  # 435
        cleared = self._merge_seed_idx is not None  # 436
        self._merge_seed_idx = None  # 437
        self._merge_seed_origin = None  # 438
        if cleared and not keep_status:  # 439
            self._set_status(self._default_status)  # 440
        return cleared  # 441
  # 442
    def _select_merge_seed(self, idx: int) -> None:  # 443
        if idx < 0 or idx >= len(self.points):  # 444
            self._clear_merge_seed()  # 445
            return  # 446
        self._merge_seed_idx = int(idx)  # 447
        y, x = self.points[idx]  # 448
        self._merge_seed_origin = (float(y), float(x))  # 449
        self._set_status(  # 450
            "A point is selected for merging. Move the cursor and press Enter to set the radius."  # 451
        )  # 452
  # 453
    def _merge_selected_with_radius(self) -> bool:  # 454
        if self._merge_seed_idx is None:  # 455
            return False  # 456
        if len(self.points) == 0:  # 457
            self._clear_merge_seed()  # 458
            return False  # 459
  # 460
        idx = int(self._merge_seed_idx)  # 461
        if idx < 0 or idx >= len(self.points):  # 462
            self._clear_merge_seed()  # 463
            self._set_status("The selected point is unavailable. Choose the point again.")  # 464
            return False  # 465
  # 466
        if self._last_cursor_pos is None:  # 467
            self._set_status("Move the cursor inside the image to set the merge radius.")  # 468
            return False  # 469
  # 470
        base_cur_y, base_cur_x = map(float, self.points[idx])  # 471
        cur_y, cur_x = self._last_cursor_pos  # 472
        radius = float(np.hypot(cur_x - base_cur_x, cur_y - base_cur_y))  # 473
        if radius <= 0.0:  # 474
            self._set_status("Radius is too small. Move the cursor and press Enter again.")  # 475
            return False  # 476
  # 477
        origin = self._merge_seed_origin or (base_cur_y, base_cur_x)  # 478
        distances = np.hypot(self.points[:, 1] - origin[1], self.points[:, 0] - origin[0])  # 479
        candidate_indices = [int(i) for i, dist in enumerate(distances) if dist <= radius + 1e-6]  # 480
        if idx not in candidate_indices:  # 481
            candidate_indices.append(idx)  # 482
            candidate_indices.sort()  # 483
  # 484
        if len(candidate_indices) <= 1:  # 485
            self._set_status("No other points found within the selected radius.")  # 486
            return False  # 487
  # 488
        self._push_undo()  # 489
        self._cancel_measurement_preview()  # 490
        self._clear_measurement_result()  # 491
  # 492
        if len(self.values) != len(self.points):  # 493
            self.values = self._sample_intensities(self.points)  # 494
  # 495
        old_points = self.points.copy()  # 496
        old_values = self.values.copy()  # 497
        use_values = len(old_values) == len(old_points)  # 498
  # 499
        local_points = old_points[candidate_indices]  # 500
        local_values = old_values[candidate_indices] if use_values else None  # 501
  # 502
        base_subset_idx = candidate_indices.index(idx)  # 503
        new_point = local_points[base_subset_idx].astype(float)  # 504
        new_value = float(local_values[base_subset_idx]) if use_values else None  # 505
  # 506
        order = [i for i in range(len(candidate_indices)) if i != base_subset_idx]  # 507
        order.sort(  # 508
            key=lambda local_idx: float(  # 509
                np.hypot(  # 510
                    local_points[local_idx, 1] - origin[1],  # 511
                    local_points[local_idx, 0] - origin[0],  # 512
                )  # 513
            )  # 514
        )  # 515
  # 516
        for local_idx in order:  # 517
            new_point = (new_point + local_points[local_idx]) / 2.0  # 518
            if use_values and new_value is not None and local_values is not None:  # 519
                new_value = (new_value + float(local_values[local_idx])) / 2.0  # 520
  # 521
        candidate_set = set(candidate_indices)  # 522
        candidate_set.discard(idx)  # 523
        new_points_list: list[list[float]] = []  # 524
        new_values_list: list[float] = []  # 525
        inserted = False  # 526
        new_index = None  # 527
  # 528
        for old_idx, pt in enumerate(old_points):  # 529
            if old_idx == idx:  # 530
                new_points_list.append([float(new_point[0]), float(new_point[1])])  # 531
                if use_values and new_value is not None:  # 532
                    new_values_list.append(float(new_value))  # 533
                inserted = True  # 534
                new_index = len(new_points_list) - 1  # 535
                continue  # 536
            if old_idx in candidate_set:  # 537
                continue  # 538
            new_points_list.append([float(pt[0]), float(pt[1])])  # 539
            if use_values:  # 540
                new_values_list.append(float(old_values[old_idx]))  # 541
  # 542
        if not inserted:  # 543
            new_index = len(new_points_list)  # 544
            new_points_list.append([float(new_point[0]), float(new_point[1])])  # 545
            if use_values and new_value is not None:  # 546
                new_values_list.append(float(new_value))  # 547
  # 548
        if new_points_list:  # 549
            self.points = np.array(new_points_list, dtype=float)  # 550
        else:  # 551
            self.points = np.zeros((0, 2), dtype=float)  # 552
  # 553
        if use_values:  # 554
            self.values = np.array(new_values_list, dtype=float)  # 555
        else:  # 556
            self.values = self._sample_intensities(self.points)  # 557
  # 558
        self._merge_seed_idx = new_index if new_index is not None else None  # 559
        if self._merge_seed_idx is not None:  # 560
            self._merge_seed_origin = (  # 561
                float(self.points[self._merge_seed_idx, 0]),  # 562
                float(self.points[self._merge_seed_idx, 1]),  # 563
            )  # 564
        else:  # 565
            self._merge_seed_origin = None  # 566
  # 567
        self._last_cursor_pos = None  # 568
  # 569
        merged_count = len(candidate_indices)  # 570
        self._set_status(  # 571
            f"Merged {merged_count} points within a radius of {radius:.1f} px."  # 572
        )  # 573
        return True  # 574
  # 575
    # ---------- Tooltip и измерения ----------  # 576
    def _remove_measure_preview_artist(self) -> bool:  # 577
        if self._measure_preview_artist is not None:  # 578
            try:  # 579
                self._measure_preview_artist.remove()  # 580
            except Exception:  # 581
                pass  # 582
            self._measure_preview_artist = None  # 583
            return True  # 584
        return False  # 585
  # 586
    def _remove_measurement_artists(self) -> bool:  # 587
        removed = False  # 588
        if self._measure_line_artist is not None:  # 589
            try:  # 590
                self._measure_line_artist.remove()  # 591
            except Exception:  # 592
                pass  # 593
            self._measure_line_artist = None  # 594
            removed = True  # 595
        if self._measure_annotation is not None:  # 596
            try:  # 597
                self._measure_annotation.remove()  # 598
            except Exception:  # 599
                pass  # 600
            self._measure_annotation = None  # 601
            removed = True  # 602
        return removed  # 603
  # 604
    def _cancel_measurement_preview(self) -> bool:  # 605
        removed = self._remove_measure_preview_artist()  # 606
        has_state = (  # 607
            self._measure_active  # 608
            or self._measure_start_point is not None  # 609
            or self._measure_preview_end is not None  # 610
        )  # 611
        removed = removed or has_state  # 612
        self._measure_active = False  # 613
        self._measure_start_idx = None  # 614
        self._measure_start_point = None  # 615
        self._measure_preview_end = None  # 616
        return removed  # 617
  # 618
    def _clear_measurement_result(self) -> bool:  # 619
        removed = self._measurement is not None  # 620
        removed = self._remove_measurement_artists() or removed  # 621
        self._measurement = None  # 622
        return removed  # 623
  # 624
    def _start_measurement(self, idx: int) -> None:  # 625
        if idx < 0 or idx >= len(self.points):  # 626
            return  # 627
        self._measure_active = True  # 628
        self._measure_start_idx = idx  # 629
        y0, x0 = self.points[idx]  # 630
        self._measure_start_point = (float(y0), float(x0))  # 631
        self._measure_preview_end = None  # 632
        self._remove_measure_preview_artist()  # 633
  # 634
    def _update_measurement_preview(self, pos: Optional[tuple[float, float]]) -> None:  # 635
        if not self._measure_active or self._measure_start_point is None:  # 636
            return  # 637
        if pos is None:  # 638
            self._measure_preview_end = None  # 639
            if self._remove_measure_preview_artist():  # 640
                if hasattr(self, "canvas"):  # 641
                    self.canvas.draw_idle()  # 642
            return  # 643
        y1, x1 = pos  # 644
        self._measure_preview_end = (float(y1), float(x1))  # 645
        y0, x0 = self._measure_start_point  # 646
        current_xlim = self.ax.get_xlim()  # 647
        current_ylim = self.ax.get_ylim()  # 648
        if self._measure_preview_artist is None:  # 649
            (line,) = self.ax.plot(  # 650
                [x0, x1],  # 651
                [y0, y1],  # 652
                color="#ffcc33",  # 653
                linewidth=1.6,  # 654
                linestyle="--",  # 655
                alpha=0.9,  # 656
                scalex=False,  # 657
                scaley=False,  # 658
            )  # 659
            self._measure_preview_artist = line  # 660
        else:  # 661
            self._measure_preview_artist.set_data([x0, x1], [y0, y1])  # 662
        self.ax.set_xlim(current_xlim)  # 663
        self.ax.set_ylim(current_ylim)  # 664
        if hasattr(self, "canvas"):  # 665
            self.canvas.draw_idle()  # 666
  # 667
    def _finalize_measurement(self, end_idx: Optional[int]) -> None:  # 668
        if not self._measure_active or self._measure_start_point is None:  # 669
            self._cancel_measurement_preview()  # 670
            return  # 671
        if end_idx is None or end_idx == self._measure_start_idx or end_idx < 0 or end_idx >= len(self.points):  # 672
            if self._cancel_measurement_preview() and hasattr(self, "canvas"):  # 673
                self.canvas.draw_idle()  # 674
            return  # 675
        start_y, start_x = self._measure_start_point  # 676
        end_y, end_x = map(float, self.points[end_idx])  # 677
        length = float(np.hypot(end_x - start_x, end_y - start_y))  # 678
        self._measurement = {  # 679
            "start": (start_y, start_x),  # 680
            "end": (end_y, end_x),  # 681
            "length": length,  # 682
        }  # 683
        self._cancel_measurement_preview()  # 684
        self._redraw()  # 685
  # 686
    def _draw_measurement_overlays(self) -> None:  # 687
        self._measure_line_artist = None  # 688
        self._measure_annotation = None  # 689
        self._measure_preview_artist = None  # 690
  # 691
        if self._measurement is not None:  # 692
            start_y, start_x = self._measurement["start"]  # 693
            end_y, end_x = self._measurement["end"]  # 694
            length = float(self._measurement.get("length", 0.0))  # 695
            (line,) = self.ax.plot(  # 696
                [start_x, end_x],  # 697
                [start_y, end_y],  # 698
                color="#ffcc33",  # 699
                linewidth=1.8,  # 700
                alpha=0.95,  # 701
                scalex=False,  # 702
                scaley=False,  # 703
            )  # 704
            self._measure_line_artist = line  # 705
            mid_x = (start_x + end_x) / 2.0  # 706
            mid_y = (start_y + end_y) / 2.0  # 707
            txt = f"L = {length:.1f} px"  # 708
            self._measure_annotation = self.ax.annotate(  # 709
                txt,  # 710
                xy=(mid_x, mid_y),  # 711
                xytext=(0, -14),  # 712
                textcoords="offset points",  # 713
                ha="center",  # 714
                bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9),  # 715
                fontsize=9,  # 716
            )  # 717
  # 718
        if (  # 719
            self._measure_active  # 720
            and self._measure_start_point is not None  # 721
            and self._measure_preview_end is not None  # 722
        ):  # 723
            y0, x0 = self._measure_start_point  # 724
            y1, x1 = self._measure_preview_end  # 725
            (pline,) = self.ax.plot(  # 726
                [x0, x1],  # 727
                [y0, y1],  # 728
                color="#ffcc33",  # 729
                linewidth=1.6,  # 730
                linestyle="--",  # 731
                alpha=0.9,  # 732
                scalex=False,  # 733
                scaley=False,  # 734
            )  # 735
            self._measure_preview_artist = pline  # 736
  # 737
    def _clear_tooltip(self, *, keep_measure: bool = False, keep_preview: bool = False):  # 738
        removed = False  # 739
        if self._tooltip is not None:  # 740
            try:  # 741
                self._tooltip.remove()  # 742
            except Exception:  # 743
                pass  # 744
            self._tooltip = None  # 745
            self._tooltip_idx = None  # 746
            removed = True  # 747
        if not keep_preview:  # 748
            if self._cancel_measurement_preview():  # 749
                removed = True  # 750
        if not keep_measure:  # 751
            if self._clear_measurement_result():  # 752
                removed = True  # 753
        if removed and hasattr(self, "canvas"):  # 754
            self.canvas.draw_idle()  # 755
  # 756
    def _show_tooltip_for_idx(self, idx):  # 757
        if idx is None or idx < 0 or idx >= len(self.points):  # 758
            return  # 759
        y, x = self.points[idx]  # 760
        # intensity по текущему изображению/values  # 761
        inten = float(self.values[idx]) if idx < len(self.values) else 0.0  # 762
        if self._percent_map is not None:  # 763
            H, W = self._percent_map.shape[:2]  # 764
            yi = max(0, min(H - 1, int(round(y))))  # 765
            xi = max(0, min(W - 1, int(round(x))))  # 766
            inten = float(self._percent_map[yi, xi])  # 767
        elif self.img_arr is not None:  # 768
            H, W = self.img_arr.shape[:2]  # 769
            yi = max(0, min(H - 1, int(round(y))))  # 770
            xi = max(0, min(W - 1, int(round(x))))  # 771
            raw_val = float(self.img_arr[yi, xi])  # 772
            if self._percent_lookup is not None:  # 773
                inten = float(map_values_to_percent(np.array([raw_val], dtype=float), *self._percent_lookup)[0])  # 774
            else:  # 775
                inten = raw_val  # 776
  # 777
        # удалить предыдущую подсказку  # 778
        self._clear_tooltip()  # 779
  # 780
        # создать аннотацию возле точки  # 781
        txt = f"x={x:.1f}, y={y:.1f}, I={inten:.1f}%"  # 782
        self._tooltip = self.ax.annotate(  # 783
            txt, xy=(x, y), xytext=(10, 10), textcoords="offset points",  # 784
            bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9),  # 785
            fontsize=9  # 786
        )  # 787
        self._tooltip_idx = idx  # 788
        self.canvas.draw_idle()  # 789
  # 790
    # ---------- Undo/Redo ----------  # 791
    def _make_snapshot(self):  # 792
        center = None  # 793
        if self.overlay and isinstance(self.overlay, dict) and self.overlay.get("center"):  # 794
            c = self.overlay["center"]  # 795
            if c is not None and "x" in c and "y" in c:  # 796
                center = {"x": float(c["x"]), "y": float(c["y"])}  # 797
        return {  # 798
            "points": self.points.copy(),  # 799
            "values": self.values.copy(),  # 800
            "center": center,  # 801
            "view_cx": self.view_cx, "view_cy": self.view_cy  # 802
        }  # 803
  # 804
    def _push_undo(self):  # 805
        self._undo.append(self._make_snapshot())  # 806
        if len(self._undo) > self._history_cap:  # 807
            self._undo.pop(0)  # 808
        self._redo.clear()  # 809
  # 810
    def _apply_snapshot(self, snap):  # 811
        self.center_dragging = False  # 812
        if self.rect_artist is not None:  # 813
            try: self.rect_artist.remove()  # 814
            except Exception: pass  # 815
            self.rect_artist = None  # 816
        self.rect_start = None  # 817
        self._clear_merge_seed(keep_status=True)  # 818
        self._last_cursor_pos = None  # 819
  # 820
        self.points = snap["points"].copy()  # 821
        self.values = snap["values"].copy()  # 822
        if snap["center"] is None:  # 823
            if self.overlay and "center" in self.overlay: self.overlay.pop("center")  # 824
        else:  # 825
            if self.overlay is None: self.overlay = {}  # 826
            self.overlay["center"] = {"x": float(snap["center"]["x"]), "y": float(snap["center"]["y"])}  # 827
        self.view_cx = snap.get("view_cx", self.view_cx)  # 828
        self.view_cy = snap.get("view_cy", self.view_cy)  # 829
        self._ensure_view_center()  # 830
  # 831
    def _undo_btn(self):  # 832
        self._clear_tooltip()  # 833
        if not self._undo: return  # 834
        self._redo.append(self._make_snapshot())  # 835
        if len(self._redo) > self._history_cap: self._redo.pop(0)  # 836
        snap = self._undo.pop(-1)  # 837
        self._apply_snapshot(snap)  # 838
        self._redraw()  # 839
  # 840
    def _redo_btn(self):  # 841
        self._clear_tooltip()  # 842
        if not self._redo: return  # 843
        self._undo.append(self._make_snapshot())  # 844
        if len(self._redo) > self._history_cap: self._undo.pop(0)  # 845
        snap = self._redo.pop(-1)  # 846
        self._apply_snapshot(snap)  # 847
        self._redraw()  # 848
  # 849
    # ---------- View-center helpers ----------  # 850
    def _ensure_view_center(self):  # 851
        if self.img_arr is None:  # 852
            return  # 853
        H, W = self.img_arr.shape[:2]  # 854
        if self.overlay and self.overlay.get("center"):  # 855
            cx = float(self.overlay["center"].get("x", (W-1)/2.0))  # 856
            cy = float(self.overlay["center"].get("y", (H-1)/2.0))  # 857
        else:  # 858
            cx = (W-1)/2.0; cy = (H-1)/2.0  # 859
        if self.view_cx is None: self.view_cx = cx  # 860
        if self.view_cy is None: self.view_cy = cy  # 861
  # 862
    # ---------- Zoom ----------  # 863
    def _apply_zoom(self):  # 864
        if self.img_arr is None:  # 865
            return  # 866
        H, W = self.img_arr.shape[:2]  # 867
        self._ensure_view_center()  # 868
  # 869
        if self.zoom_val <= 0:  # 870
            self.ax.set_xlim(-0.5, W - 0.5)  # 871
            self.ax.set_ylim(H - 0.5, -0.5)  # 872
            return  # 873
  # 874
        min_dim = min(H, W)  # 875
        L = int(round(min_dim - (min_dim - 50) * (self.zoom_val / 100.0)))  # 876
        L = max(50, min_dim if L < 50 else L)  # 877
        half = L / 2.0  # 878
  # 879
        cx = float(self.view_cx); cy = float(self.view_cy)  # 880
        x0 = max(-0.5, cx - half); x1 = min(W - 0.5, cx + half)  # 881
        y0 = max(-0.5, cy - half); y1 = min(H - 0.5, cy + half)  # 882
  # 883
        # если окно "урезано" границами — подправим  # 884
        if (x1 - x0) < L:  # 885
            if x0 <= -0.5: x1 = x0 + L  # 886
            elif x1 >= (W - 0.5): x0 = x1 - L  # 887
        if (y1 - y0) < L:  # 888
            if y0 <= -0.5: y1 = y0 + L  # 889
            elif y1 >= (H - 0.5): y0 = y1 - L  # 890
  # 891
        self.ax.set_xlim(x0, x1)  # 892
        self.ax.set_ylim(y1, y0)  # 893
  # 894
    def _on_zoom_change(self, val):  # 895
        try:  # 896
            self.zoom_val = int(float(val))  # 897
        except Exception:  # 898
            self.zoom_val = 0  # 899
        if hasattr(self, "zoom_var"):  # 900
            current = int(round(self.zoom_var.get()))  # 901
            if current != self.zoom_val:  # 902
                self.zoom_var.set(self.zoom_val)  # 903
        self._update_zoom_hint()  # 904
        self._clear_tooltip()  # 905
        self._redraw()  # 906
  # 907
    # ---------- Mouse / Keyboard events ----------  # 908
    def _on_key(self, e):  # 909
        # закрыть tooltip по Esc  # 910
        if e.key == "escape":  # 911
            self._clear_tooltip()  # 912
            if self._merge_seed_idx is not None:  # 913
                self._clear_merge_seed()  # 914
        elif e.key in {"enter", "return"}:  # 915
            if self._merge_selected_with_radius():  # 916
                self._redraw()  # 917
  # 918
    def _on_down(self, e):  # 919
        pos = self._img_xy(e)  # 920
        # любое действие закрывает tooltip  # 921
        self._clear_tooltip()  # 922
  # 923
        if pos is not None:  # 924
            self._last_cursor_pos = (float(pos[0]), float(pos[1]))  # 925
        else:  # 926
            self._last_cursor_pos = None  # 927
  # 928
        # Средняя кнопка: только подсказка, если попали в точку  # 929
        if e.button == 2:  # 930
            if pos is None:  # 931
                return  # 932
            y, x = pos  # 933
            idx = self._near_idx(y, x, pix_tol=8)  # 934
            if idx is not None:  # 935
                self._show_tooltip_for_idx(idx)  # 936
            return  # не считаем это добавлением/редактированием  # 937
  # 938
        if e.button == 1 and not (e.key and "shift" in e.key):  # 939
            if pos is None:  # 940
                return  # 941
            y, x = pos  # 942
            # Перетаскивание центра — оставить  # 943
            if self._center_hit(y, x):  # 944
                self._clear_merge_seed()  # 945
                self._push_undo()  # 946
                self.center_dragging = True  # 947
                self._redo.clear()  # 948
                return  # 949
  # 950
            # Добавление новой точки по клику в пустое место (без последующего перетаскивания!)  # 951
            i = self._near_idx(y, x)  # 952
            if i is None:  # 953
                self._clear_merge_seed()  # 954
                self._push_undo()  # 955
                self.points = np.vstack([self.points, [y, x]])  # 956
                sampled = self._sample_intensities(np.array([[y, x]]))[0]  # 957
                self.values = np.append(self.values, sampled)  # 958
                self._redo.clear()  # 959
            else:  # 960
                self._select_merge_seed(i)  # 961
                self._start_measurement(i)  # 962
            # если кликнули по существующей точке — перемещение запрещено (используем для измерения)  # 963
  # 964
        elif e.button == 3:  # 965
            if pos is None:  # 966
                return  # 967
            y, x = pos  # 968
            i = self._near_idx(y, x)  # 969
            if i is not None:  # 970
                self._push_undo()  # 971
                self.points = np.delete(self.points, i, axis=0)  # 972
                self.values = np.delete(self.values, i, axis=0)  # 973
                if self._merge_seed_idx is not None:  # 974
                    if i == self._merge_seed_idx:  # 975
                        self._clear_merge_seed()  # 976
                    elif i < self._merge_seed_idx:  # 977
                        self._merge_seed_idx -= 1  # 978
                        if 0 <= self._merge_seed_idx < len(self.points):  # 979
                            self._merge_seed_origin = (  # 980
                                float(self.points[self._merge_seed_idx, 0]),  # 981
                                float(self.points[self._merge_seed_idx, 1]),  # 982
                            )  # 983
                self._redo.clear()  # 984
  # 985
        elif e.button == 1 and e.key and "shift" in e.key:  # 986
            if pos is not None:  # 987
                self._last_cursor_pos = (float(pos[0]), float(pos[1]))  # 988
            self._push_undo()  # 989
            self.rect_start = pos  # 990
            self._redo.clear()  # 991
  # 992
        self._redraw()  # 993
  # 994
    def _on_move(self, e):  # 995
        pos = self._img_xy(e)  # 996
        # любое движение закрывает tooltip  # 997
        keep = self._measure_active  # 998
        self._clear_tooltip(keep_measure=keep, keep_preview=keep)  # 999
  # 1000
        if pos is not None:  # 1001
            self._last_cursor_pos = (float(pos[0]), float(pos[1]))  # 1002
        else:  # 1003
            self._last_cursor_pos = None  # 1004
  # 1005
        # Перетаскивание центра — оставить  # 1006
        if self.center_dragging and pos is not None:  # 1007
            y, x = pos  # 1008
            if self.overlay is None:  # 1009
                self.overlay = {}  # 1010
            self.overlay["center"] = {"x": float(x), "y": float(y)}  # 1011
            self._redraw()  # 1012
            return  # 1013
  # 1014
        if self._measure_active:  # 1015
            self._update_measurement_preview(pos)  # 1016
            return  # 1017
  # 1018
        # Прямоугольник выделения для удаления  # 1019
        if self.rect_start and e.xdata is not None and e.ydata is not None:  # 1020
            y0, x0 = self.rect_start  # 1021
            y1, x1 = e.ydata, e.xdata  # 1022
            self._redraw()  # 1023
            self.rect_artist = self.ax.add_patch(  # 1024
                plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", ls="--", lw=1.5)  # 1025
            )  # 1026
            self.canvas.draw_idle()  # 1027
            return  # 1028
  # 1029
    def _on_up(self, e):  # 1030
        # любое действие закрывает tooltip  # 1031
        keep = self._measure_active  # 1032
        self._clear_tooltip(keep_measure=keep, keep_preview=keep)  # 1033
  # 1034
        # Завершение перетаскивания центра  # 1035
        if self.center_dragging:  # 1036
            self.center_dragging = False  # 1037
            self._apply_center_filters()  # 1038
            if self.overlay and self.overlay.get("center"):  # 1039
                self.view_cx = float(self.overlay["center"]["x"])  # 1040
                self.view_cy = float(self.overlay["center"]["y"])  # 1041
            self._redraw()  # 1042
            return  # 1043
  # 1044
        if self._measure_active:  # 1045
            pos = self._img_xy(e)  # 1046
            end_idx = None  # 1047
            if pos is not None:  # 1048
                end_idx = self._near_idx(pos[0], pos[1])  # 1049
            self._finalize_measurement(end_idx)  # 1050
            return  # 1051
  # 1052
        # Завершение прямоугольника удаления  # 1053
        if self.rect_start:  # 1054
            y0, x0 = self.rect_start; y1, x1 = e.ydata, e.xdata  # 1055
            if y1 is not None and x1 is not None:  # 1056
                ymin, ymax = sorted([y0, y1]); xmin, xmax = sorted([x0, x1])  # 1057
                mask = ~((self.points[:,0] >= ymin) & (self.points[:,0] <= ymax) &  # 1058
                         (self.points[:,1] >= xmin) & (self.points[:,1] <= xmax))  # 1059
                if self._merge_seed_idx is not None:  # 1060
                    if self._merge_seed_idx >= len(mask) or not mask[self._merge_seed_idx]:  # 1061
                        self._clear_merge_seed()  # 1062
                    else:  # 1063
                        new_idx = int(np.count_nonzero(mask[: self._merge_seed_idx + 1]) - 1)  # 1064
                        self._merge_seed_idx = new_idx  # 1065
                self.points = self.points[mask]  # 1066
                if len(self.values) == len(mask):  # 1067
                    self.values = self.values[mask]  # 1068
                else:  # 1069
                    self.values = self._sample_intensities(self.points)  # 1070
                if self._merge_seed_idx is not None and self._merge_seed_idx < len(self.points):  # 1071
                    self._merge_seed_origin = (  # 1072
                        float(self.points[self._merge_seed_idx, 0]),  # 1073
                        float(self.points[self._merge_seed_idx, 1]),  # 1074
                    )  # 1075
            self.rect_start = None  # 1076
            if self.rect_artist is not None:  # 1077
                self.rect_artist.remove(); self.rect_artist = None  # 1078
            self._redraw()  # 1079
  # 1080
    # ---------- Draw ----------  # 1081
    def _redraw(self):  # 1082
        self.ax.clear()  # 1083
        if self.img_arr is not None:  # 1084
            self.ax.imshow(self.img_arr, cmap="gray", interpolation="nearest")  # 1085
        self.ax.axis("off")  # 1086
  # 1087
        if self.overlay and self.overlay.get("center"):  # 1088
            cy = float(self.overlay["center"].get("y", 0))  # 1089
            cx = float(self.overlay["center"].get("x", 0))  # 1090
            self.ax.scatter([cx], [cy], s=36, c="red", marker="o")  # 1091
            dead = float(self.overlay.get("dead_radius") or 0)  # 1092
            sr   = float(self.overlay.get("search_radius") or 0)  # 1093
            for R in [dead, sr]:  # 1094
                if R > 0:  # 1095
                    self.ax.add_patch(Circle((cx, cy), R, fill=False, ls="--", lw=2.0, ec="red"))  # 1096
  # 1097
        if len(self.points):  # 1098
            if (  # 1099
                self._merge_seed_idx is not None  # 1100
                and 0 <= self._merge_seed_idx < len(self.points)  # 1101
            ):  # 1102
                mask = np.ones(len(self.points), dtype=bool)  # 1103
                mask[self._merge_seed_idx] = False  # 1104
                if np.any(mask):  # 1105
                    self.ax.scatter(  # 1106
                        self.points[mask, 1],  # 1107
                        self.points[mask, 0],  # 1108
                        s=22,  # 1109
                        alpha=0.9,  # 1110
                        marker="o",  # 1111
                        linewidths=0.5,  # 1112
                        edgecolors="black",  # 1113
                    )  # 1114
                seed_y = float(self.points[self._merge_seed_idx, 0])  # 1115
                seed_x = float(self.points[self._merge_seed_idx, 1])  # 1116
                self.ax.scatter(  # 1117
                    [seed_x],  # 1118
                    [seed_y],  # 1119
                    s=38,  # 1120
                    alpha=0.95,  # 1121
                    marker="o",  # 1122
                    linewidths=0.8,  # 1123
                    edgecolors="black",  # 1124
                    c="#ffd34d",  # 1125
                )  # 1126
            else:  # 1127
                self.ax.scatter(  # 1128
                    self.points[:, 1],  # 1129
                    self.points[:, 0],  # 1130
                    s=22,  # 1131
                    alpha=0.9,  # 1132
                    marker="o",  # 1133
                    linewidths=0.5,  # 1134
                    edgecolors="black",  # 1135
                )  # 1136
  # 1137
        self._draw_measurement_overlays()  # 1138
        self._apply_zoom()  # 1139
        self.canvas.draw_idle()  # 1140
  # 1141
    # ---------- Анализ ----------  # 1142
    def _start_analysis(self):  # 1143
        saved = self._save_points()  # сохраняет текущие точки (и интенсивности) и возвращает путь к spots.json  # 1144
        self._set_status("Preparing data for analysis…")  # 1145
  # 1146
        # payload для fibonachi_analysis  # 1147
        payload_path = None  # 1148
        try:  # 1149
            if self.img_arr is not None:  # 1150
                geo_cy = (self.img_arr.shape[0] - 1) / 2.0  # 1151
                geo_cx = (self.img_arr.shape[1] - 1) / 2.0  # 1152
            else:  # 1153
                geo_cy = geo_cx = None  # 1154
  # 1155
            # актуальные точки (после правок) + интенсивности из текущего изображения  # 1156
            points_list = []  # 1157
            if len(self.points):  # 1158
                if self._percent_map is not None:  # 1159
                    H, W = self._percent_map.shape[:2]  # 1160
                elif self.img_arr is not None:  # 1161
                    H, W = self.img_arr.shape[:2]  # 1162
                else:  # 1163
                    H = W = None  # 1164
                for idx, (y, x) in enumerate(self.points.tolist()):  # 1165
                    inten = None  # 1166
                    if H is not None:  # 1167
                        yi = max(0, min(H - 1, int(round(y))))  # 1168
                        xi = max(0, min(W - 1, int(round(x))))  # 1169
                        if self._percent_map is not None:  # 1170
                            inten = float(self._percent_map[yi, xi])  # 1171
                        elif self.img_arr is not None:  # 1172
                            raw_val = float(self.img_arr[yi, xi])  # 1173
                            if self._percent_lookup is not None:  # 1174
                                inten = float(  # 1175
                                    map_values_to_percent(  # 1176
                                        np.array([raw_val], dtype=float), *self._percent_lookup  # 1177
                                    )[0]  # 1178
                                )  # 1179
                            else:  # 1180
                                inten = raw_val  # 1181
                    if inten is None and idx < len(self.values):  # 1182
                        inten = float(self.values[idx])  # 1183
                    points_list.append({"x": float(x), "y": float(y), "intensity": inten})  # 1184
  # 1185
            # актуальный центр из overlay  # 1186
            overlay_center = None  # 1187
            if self.overlay and isinstance(self.overlay, dict) and self.overlay.get("center") is not None:  # 1188
                c = self.overlay["center"]  # 1189
                overlay_center = {"x": float(c.get("x")), "y": float(c.get("y"))}  # 1190
  # 1191
            # радиусы из overlay  # 1192
            dead_val = None  # 1193
            search_val = None  # 1194
            if self.overlay:  # 1195
                if self.overlay.get("dead_radius") is not None:  # 1196
                    dead_val = float(self.overlay.get("dead_radius"))  # 1197
                if self.overlay.get("search_radius") is not None:  # 1198
                    search_val = float(self.overlay.get("search_radius"))  # 1199
  # 1200
            payload = {  # 1201
                "image": str(self.image_path) if self.image_path else None,  # 1202
                "preproc_mode": self._preproc_settings.mode,  # 1203
                "preproc": self._preproc_settings.to_json(),  # 1204
                "points": points_list,  # 1205
                "centers": {  # 1206
                    "geometric": {"x": float(geo_cx), "y": float(geo_cy)} if geo_cx is not None else None,  # 1207
                    "overlay": overlay_center  # 1208
                },  # 1209
                "radii": {  # 1210
                    "dead": dead_val,  # 1211
                    "search": search_val  # 1212
                },  # 1213
                "spots_json": str(saved) if saved else None  # 1214
            }  # 1215
  # 1216
            payload_path = (self.image_path.with_name("fibo_input.json") if self.image_path  # 1217
                            else Path("fibo_input.json"))  # 1218
            payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")  # 1219
  # 1220
  # 1221
        except Exception as e:  # 1222
  # 1223
            payload_path = None  # 1224
  # 1225
            messagebox.showerror("Data preparation error",  # 1226
  # 1227
                                 f"Failed to prepare data for fibonachi_analysis:\n{e}")  # 1228
  # 1229
        used_controller = False  # 1230
  # 1231
        if self.controller is not None and payload_path is not None:  # 1232
  # 1233
            try:  # 1234
  # 1235
                self.controller.open_analysis(payload_path, self.image_path, saved)  # 1236
                self._set_status("Analysis opened in tab")  # 1237
  # 1238
                used_controller = True  # 1239
  # 1240
            except Exception as e:  # 1241
  # 1242
                messagebox.showerror("Launch error",  # 1243
  # 1244
                                     f"Failed to switch to the analyzer:\n{e}")  # 1245
  # 1246
        if not used_controller:  # 1247
  # 1248
            # запуск fibonachi_analysis.exe (совместимость со старыми ключами сохранена)  # 1249
  # 1250
            try:  # 1251
  # 1252
                if getattr(sys, "frozen", False):  # 1253
  # 1254
                    fibexe = Path(sys.executable).with_name("fibonachi_analysis.exe")  # 1255
  # 1256
                else:  # 1257
  # 1258
                    fibexe = Path(__file__).with_name("fibonachi_analysis.exe")  # 1259
  # 1260
                cmd = [str(fibexe)]  # 1261
  # 1262
                if payload_path is not None:  # 1263
                    cmd += ["--payload", str(payload_path)]  # 1264
  # 1265
                if self.image_path is not None:  # 1266
                    cmd += ["--image", str(self.image_path)]  # 1267
  # 1268
                if saved is not None:  # 1269
                    cmd += ["--points", str(saved)]  # 1270
  # 1271
                subprocess.Popen(cmd, shell=False)  # 1272
                self._set_status("External analysis started")  # 1273
  # 1274
            except Exception as e:  # 1275
  # 1276
                messagebox.showerror("Launch error",  # 1277
  # 1278
                                     f"Failed to launch fibonachi_analysis.exe:\n{e}")  # 1279
  # 1280
        # быстрый отчёт (как раньше)  # 1281
        if self.img_arr is None or len(self.points) == 0:  # 1282
            messagebox.showinfo("Analysis", "No image or points available for analysis.")  # 1283
            return  # 1284
  # 1285
        cy, cx = (self.img_arr.shape[0] - 1) / 2.0, (self.img_arr.shape[1] - 1) / 2.0  # 1286
        radii, angles = pol_from((cy, cx), self.points)  # 1287
        rc, labels, _ = cluster_rings(radii)  # 1288
        ring_means = [np.mean(radii[labels == i]) for i in np.unique(labels)] if len(rc) else []  # 1289
        sym = symmetry_scores(angles, radii, ring_means)  # 1290
  # 1291
        lines = ["SAED Symmetry Analysis", "=======================", "",  # 1292
                 f"File: {self.image_path}",  # 1293
                 f"Saved: {saved.name if saved else '-'}",  # 1294
                 f"Points: {len(self.points)}", "", "Symmetries:"]  # 1295
        for k, v in sorted(sym.items(), key=lambda kv: -kv[1]):  # 1296
            lines.append(f"  {k:>7}: {v:.3f}")  # 1297
        self._show_report("\n".join(lines))  # 1298
  # 1299
    def _show_report(self, text: str):  # 1300
        win = tk.Toplevel(self)  # 1301
        win.title("SAED Report")  # 1302
        txt = tk.Text(win, wrap="word")  # 1303
        txt.pack(fill=tk.BOTH, expand=True)  # 1304
        txt.insert("1.0", text)  # 1305
        txt.config(state=tk.DISABLED)  # 1306
        self._set_status("Symmetry report generated")  # 1307
  # 1308
class PointEditorApp(tk.Tk):  # 1309
    """Standalone wrapper embedding the editor into the root window."""  # 1310
  # 1311
    def __init__(self, input_json: str | None = None):  # 1312
        super().__init__()  # 1313
        self.title("SAED Editor + Analysis")  # 1314
        self.geometry("1100x800")  # 1315
        self.resizable(True, True)  # 1316
        self.editor = PointEditor(self, input_json=input_json)  # 1317
        self.editor.pack(fill=tk.BOTH, expand=True)  # 1318
  # 1319
  # 1320
# -------- CLI ---------  # 1321
def _parse_args(argv):  # 1322
    import argparse  # 1323
  # 1324
    p = argparse.ArgumentParser()  # 1325
    p.add_argument("--input", type=str, required=False, help="Path to saed_input.json")  # 1326
    return p.parse_args(argv)  # 1327
  # 1328
  # 1329
if __name__ == "__main__":  # 1330
    args = _parse_args(sys.argv[1:])  # 1331
    root = PointEditorApp(args.input)  # 1332
    root.mainloop()  # 1333