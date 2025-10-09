#!/usr/bin/env python3  # 1
# -*- coding: utf-8 -*-  # 2
# fibonachi_analysis.py  # 3
#  # 4
# Mode 1 (LMB): search and edit S/L, Fibonacci "words", average L/S along the chain.  # 5
# Mode 2 (RMB): without S/L — segment names "n-(n-1)" and neighboring segment ratios, average ratio.  # 6
#  # 7
# New:  # 8
# - Clicking a list row in LMB mode highlights the selected "word".  # 9
# - Clicking a list row in RMB mode highlights the two neighboring segments that form the ratio.  # 10
#  # 11
# General:  # 12
# - First click (LMB or RMB) is an anchor; a yellow "rubber band" stretches to the cursor.  # 13
# - Second click with the same button type collects points near the segment (≤ tolerance) and analyzes them.  # 14
# - Save the image, clear the selection, auto-load fibo_input.json, show center/radii if they exist.  # 15
# - ESC resets both modes (anchors, rubber bands).  # 16
# - In the "S/L sequence" field you can invert the selected letters: 'i' / 'sh' / 'Sh'.  # 17
# 18
from __future__ import annotations  # 19
# 20
import sys, json, math  # 21
from pathlib import Path  # 22
from typing import Optional, Tuple, List, Dict  # 23
# 24
import numpy as np  # 25
from PIL import Image  # 26
import tkinter as tk  # 27
from tkinter import filedialog, messagebox  # 28
# 29
import matplotlib  # 30
matplotlib.use('TkAgg')  # 31
import matplotlib.pyplot as plt  # 32
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # 33
from matplotlib.patches import Circle  # 34
# 35
from preproc import PreprocSettings, load_grayscale_with_preproc  # 36
# 37
# ---------------- загрузка ----------------  # 38
# 39
def _parse_cli(argv=None):  # 40
    import argparse  # 41
    p = argparse.ArgumentParser(description="fibonachi_analysis — load input data")  # 42
    p.add_argument("--payload", type=str, default=None, help="Path to fibo_input.json")  # 43
    p.add_argument("--image", type=str, default=None, help="Path to the image (fallback)")  # 44
    p.add_argument("--points", type=str, default=None, help="Path to JSON with points (fallback)")  # 45
    return p.parse_args(argv)  # 46
# 47
# 48
def _candidate_dirs(extra_image: Optional[Path]) -> List[Path]:  # 49
    cands: List[Path] = []  # 50
    try:  # 51
        cands.append(Path.cwd())  # 52
    except Exception:  # 53
        pass  # 54
    if getattr(sys, "frozen", False):  # 55
        try:  # 56
            cands.append(Path(sys.executable).resolve().parent)  # 57
        except Exception:  # 58
            pass  # 59
        try:  # 60
            cands.append(Path(getattr(sys, "_MEIPASS")))  # 61
        except Exception:  # 62
            pass  # 63
    else:  # 64
        try:  # 65
            cands.append(Path(__file__).resolve().parent)  # 66
        except Exception:  # 67
            pass  # 68
    if extra_image:  # 69
        try:  # 70
            cands.append(extra_image.resolve().parent)  # 71
        except Exception:  # 72
            pass  # 73
    uniq, seen = [], set()  # 74
    for d in cands:  # 75
        rp = str(d.resolve())  # 76
        if rp not in seen:  # 77
            uniq.append(d); seen.add(rp)  # 78
    return uniq  # 79
# 80
# 81
def _autofind_json(extra_image: Optional[Path]) -> Optional[Path]:  # 82
    pats = ["fibo_input.json", "*fibo*input*.json", "*.fibo.json", "*.json"]  # 83
    for base in _candidate_dirs(extra_image):  # 84
        for pat in pats:  # 85
            try:  # 86
                for p in base.glob(pat):  # 87
                    name = p.name.lower()  # 88
                    if "fibo" in name and "input" in name:  # 89
                        return p.resolve()  # 90
                    if pat == "*.json":  # 91
                        try:  # 92
                            obj = json.loads(p.read_text(encoding="utf-8"))  # 93
                            if isinstance(obj, dict) and "image" in obj and "points" in obj:  # 94
                                return p.resolve()  # 95
                        except Exception:  # 96
                            pass  # 97
            except Exception:  # 98
                continue  # 99
    return None  # 100
# 101
# 102
def find_default_json(base_dir: Path) -> Optional[Path]:  # 103
    cand = base_dir / "fibo_input.json"  # 104
    if cand.exists():  # 105
        try:  # 106
            d = json.loads(cand.read_text(encoding="utf-8"))  # 107
            if "image" in d and "points" in d:  # 108
                return cand.resolve()  # 109
        except Exception:  # 110
            pass  # 111
    for p in base_dir.glob("*.json"):  # 112
        try:  # 113
            d = json.loads(p.read_text(encoding="utf-8"))  # 114
            if "image" in d and "points" in d:  # 115
                return p.resolve()  # 116
        except Exception:  # 117
            pass  # 118
    return _autofind_json(None)  # 119
# 120
# 121
def load_input(json_path: Path):  # 122
    d = json.loads(json_path.read_text(encoding='utf-8'))  # 123
    img = Path(d['image']) if d.get('image') else None  # 124
    if not img:  # 125
        raise RuntimeError("JSON does not contain the key 'image'.")  # 126
    pts = np.array([[float(p['y']), float(p['x'])] for p in d.get('points', [])], float)  # 127
    center = None; dead = 0.0; srch = 0.0  # 128
    if isinstance(d.get('centers'), dict):  # 129
        c = d['centers'].get('overlay') or d['centers'].get('geometric')  # 130
        if c and 'x' in c and 'y' in c:  # 131
            center = (float(c['y']), float(c['x']))  # 132
    if isinstance(d.get('radii'), dict):  # 133
        if d['radii'].get('dead') is not None:  dead = float(d['radii']['dead'])  # 134
        if d['radii'].get('search') is not None: srch = float(d['radii']['search'])  # 135
    fallback_mode = d.get('preproc_mode')  # 136
    if not isinstance(fallback_mode, str):  # 137
        fallback_mode = None  # 138
    preproc = PreprocSettings.from_json(d.get('preproc'), fallback_mode=fallback_mode)  # 139
    return img, pts, center, dead, srch, preproc  # 140
# 141
# ---------------- анализ S/L ----------------  # 142
# 143
def cluster_lengths(lengths: np.ndarray):  # 144
    """k=2 clustering of lengths into S/L; returns (labels, Slen, Llen, sidx, lidx)."""  # 145
    if lengths.size == 0:  # 146
        return np.array([], dtype=int), float("nan"), float("nan"), 0, 1  # 147
    c0, c1 = float(lengths.min()), float(lengths.max())  # 148
    if c0 == c1:  # 149
        lab = np.zeros(len(lengths), dtype=int)  # 150
        return lab, c0, float("nan"), 0, 1  # 151
    lab = np.zeros(len(lengths), dtype=int)  # 152
    for _ in range(60):  # 153
        d0 = np.abs(lengths - c0)  # 154
        d1 = np.abs(lengths - c1)  # 155
        lab = (d1 < d0).astype(int)  # 156
        nc0 = float(lengths[lab == 0].mean()) if np.any(lab == 0) else c0  # 157
        nc1 = float(lengths[lab == 1].mean()) if np.any(lab == 1) else c1  # <-- фикс: lab, не lаб  # 158
        if abs(nc0 - c0) < 1e-6 and abs(nc1 - c1) < 1e-6:  # 159
            c0, c1 = nc0, nc1; break  # 160
        c0, c1 = nc0, nc1  # 161
    # гарантируем S < L  # 162
    m0 = float(lengths[lab == 0].mean()) if np.any(lab == 0) else float("nan")  # 163
    m1 = float(lengths[lab == 1].mean()) if np.any(lab == 1) else float("nan")  # 164
    if (not math.isnan(m0)) and (not math.isnan(m1)) and m0 > m1:  # 165
        lab = 1 - lab  # 166
        m0, m1 = m1, m0  # 167
    return lab, m0, m1, 0, 1  # 168
# 169
def fib_list_upto(n: int) -> List[int]:  # 170
    """Fibonacci numbers up to n (inclusive), starting with 1,1,2,3,..."""  # 171
    if n <= 0: return []  # 172
    seq = [1, 1]  # 173
    while seq[-1] < n:  # 174
        seq.append(seq[-1] + seq[-2])  # 175
    return [k for k in seq if k <= n]  # 176
# 177
def gen_fibonacci_words(max_len: int, start: str = "L") -> List[str]:  # 178
    """Generate prefixes of the "fibo-word": L->LS, S->L, up to max_len."""  # 179
    if max_len <= 0: return []  # 180
    words = ["L" if start.upper() == "L" else "S"]  # 181
    while len(words[-1]) <= max_len:  # 182
        nxt = "".join(("LS" if ch == "L" else "L") for ch in words[-1])  # 183
        if len(nxt) > max_len: break  # 184
        words.append(nxt)  # 185
    return words  # 186
# 187
# ---------------- GUI ----------------  # 188
# 189
class FibonacciAnalysisFrame(tk.Frame):  # 190
    def __init__(self, master: tk.Misc, controller=None, auto_load: bool = True):  # 191
        super().__init__(master)  # 192
        self.controller = controller  # 193
# 194
        # Данные  # 195
        self.img_path: Optional[Path] = None  # 196
        self.points: Optional[np.ndarray] = None      # (N,2) [y,x]  # 197
        self.center: Optional[Tuple[float, float]] = None  # 198
        self.dead: float = 0.0  # 199
        self.srch: float = 0.0  # 200
        self.preproc: PreprocSettings = PreprocSettings(mode="raw")  # 201
        self.img_arr: Optional[np.ndarray] = None  # 202
# 203
        # Общие параметры  # 204
        self.pick_tol = 10.0  # 205
        self.max_dist_line = 12.0  # 206
# 207
        # Режим 1 (ЛКМ): S/L  # 208
        self.selected_idx: List[int] = []  # 209
        self.anchor_idx: Optional[int] = None  # 210
        self.rubber_line = None  # 211
        # Текущее состояние S/L  # 212
        self.curr_chain: Optional[np.ndarray] = None  # 213
        self.curr_seg: Optional[np.ndarray] = None  # 214
        self.curr_labels: Optional[List[str]] = None  # 215
        self.curr_ratio: float = float('nan')  # 216
# 217
        # Режим 2 (ПКМ): отношения сегментов  # 218
        self.ratio_anchor_idx: Optional[int] = None  # 219
        self.rubber_line_ratio = None  # 220
        self.ratio_selected_idx: List[int] = []  # 221
# 222
        # Режим 3 (СКМ): построение многоугольников  # 223
        self.polygon_current_idx: List[int] = []  # 224
        self.polygons_idx: List[List[int]] = []  # 225
        self.polygon_rubber_line = None  # 226
        self._polygon_history: List[Tuple[List[int], List[List[int]]]] = []  # 227
        self._polygon_redo: List[Tuple[List[int], List[List[int]]]] = []  # 228
# 229
        # Последний активный режим анализа (для перерисовки поверх полигонов)  # 230
        self._last_analysis_mode: Optional[str] = None  # 231
# 232
        # Маппинг строк Listbox -> объект подсветки  # 233
        # для S/L: ('sl', i0, n) ; для отношений: ('ratio', k) где k — индекс отношения (сегменты k-1 и k)  # 234
        self.list_index_map: Dict[int, Tuple] = {}  # 235
# 236
        # --- верхняя панель ---  # 237
        self.rowconfigure(0, weight=1)  # 238
        self.columnconfigure(0, weight=1)  # 239
# 240
        container = tk.Frame(self)  # 241
        container.grid(row=0, column=0, sticky="nsew")  # 242
        container.rowconfigure(0, weight=1)  # 243
        container.columnconfigure(0, weight=1)  # 244
        container.columnconfigure(1, weight=0)  # 245
# 246
        left = tk.Frame(container)  # 247
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=12)  # 248
        left.rowconfigure(0, weight=1)  # 249
        left.columnconfigure(0, weight=1)  # 250
# 251
        right = tk.Frame(container, width=470)  # 252
        right.grid(row=0, column=1, sticky="ns", pady=12)  # 253
        right.columnconfigure(0, weight=1)  # 254
# 255
        controls = tk.Frame(right)  # 256
        controls.grid(row=0, column=0, sticky="ew", padx=6, pady=(0, 10))  # 257
        controls.columnconfigure(0, weight=1)  # 258
        controls.columnconfigure(1, weight=1)  # 259
# 260
        tk.Button(controls, text='Open JSON…', command=self.open_json).grid(row=0, column=0, sticky='ew', padx=4, pady=2)  # 261
        tk.Button(controls, text='Start analysis (LMB chain)', command=self.run_analysis).grid(row=0, column=1, sticky='ew', padx=4, pady=2)  # 262
        tk.Button(controls, text='Save PNG', command=self.save_png).grid(row=1, column=0, sticky='ew', padx=4, pady=2)  # 263
        tk.Button(controls, text='Clear selection', command=self.clear_selection).grid(row=1, column=1, sticky='ew', padx=4, pady=2)  # 264
        tk.Label(controls, text='Selection thickness (px):').grid(row=2, column=0, sticky='w', padx=4, pady=(8, 2))  # 265
        self.entBand = tk.Spinbox(controls, from_=2, to=100, width=6, command=self._onBand)  # 266
        self.entBand.delete(0, 'end'); self.entBand.insert(0, str(int(self.max_dist_line)))  # 267
        self.entBand.grid(row=2, column=1, sticky='ew', padx=4, pady=(8, 2))  # 268
# 269
        self.status = tk.Label(right, text='', anchor='w')  # 270
        self.status.grid(row=1, column=0, sticky='ew', padx=6, pady=(0, 10))  # 271
# 272
        # заголовок списка (динамический)  # 273
        self.lst_header = tk.Label(right, text='Found words (Fibonacci subsegments)')  # 274
        self.lst_header.grid(row=2, column=0, sticky='w', padx=6, pady=(0, 2))  # 275
        self.lst = tk.Listbox(right, width=66, height=22)  # 276
        self.lst.grid(row=3, column=0, sticky='nsew', padx=6)  # 277
        right.rowconfigure(3, weight=1)  # 278
        self.lst.bind('<<ListboxSelect>>', self._on_list_select)  # 279
# 280
        # подписи со средними  # 281
        self.lbl_ratio = tk.Label(right, text='Average L/S along chain: —')  # 282
        self.lbl_ratio.grid(row=4, column=0, sticky='w', padx=6, pady=(6, 4))  # 283
        self.lbl_ratio_neigh = tk.Label(right, text='Average neighboring segment ratio: —')  # 284
        self.lbl_ratio_neigh.grid(row=5, column=0, sticky='w', padx=6, pady=(2, 8))  # 285
# 286
        # последовательность S/L (полная) + бинды инверсии  # 287
        tk.Label(right, text='S/L sequence (full):').grid(row=6, column=0, sticky='w', padx=6, pady=(4, 2))  # 288
        self.txt_sl = tk.Text(right, height=6, wrap='word')  # 289
        self.txt_sl.grid(row=7, column=0, sticky='ew', padx=6, pady=(0, 4))  # 290
        self.txt_sl.bind('<KeyPress>', self._on_sl_keypress)  # 291
# 292
        # префиксы «фиб-слова»  # 293
        tk.Label(right, text='Prefixes of "fib-words" (L→LS, S→L)').grid(row=8, column=0, sticky='w', padx=6, pady=(8, 2))  # 294
        self.txt_words = tk.Text(right, height=10, state='disabled')  # 295
        self.txt_words.grid(row=9, column=0, sticky='ew', padx=6, pady=(0, 8))  # 296
# 297
        self.fig = plt.Figure(figsize=(9.6, 6.6)); self.ax = self.fig.add_subplot(111)  # 298
        self.ax.axis('off')  # 299
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)  # 300
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')  # 301
        # обработчики мыши и клавиш  # 302
        self.canvas.mpl_connect('button_press_event', self._on_click)  # 303
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)  # 304
        self.bind('<Escape>', lambda e: self.clear_selection())  # 305
        self.bind_all('<Return>', self._on_enter_key)  # 306
        self.bind_all('<z>', self._on_polygon_undo)  # 307
        self.bind_all('<z>', self._on_polygon_undo)  # 308
        self.bind_all('<y>', self._on_polygon_redo)  # 309
        self.bind_all('<y>', self._on_polygon_redo)  # 310
# 311
        self._reset_polygon_history()  # 312
# 313
        # автозагрузка  # 314
        if auto_load:  # 315
            base = Path(getattr(sys, '_MEIPASS', Path(__file__).parent)) if getattr(sys, 'frozen', False) else Path(  # 316
                __file__).parent  # 317
            auto = find_default_json(base)  # 318
            if auto:  # 319
                try:  # 320
                    self.load_json(auto)  # 321
                    self.status.config(text=f'Loaded: {auto.name}')  # 322
                except Exception as e:  # 323
                    messagebox.showerror('Load error', str(e))  # 324
            else:  # 325
                self.status.config(text='JSON not found. Select a file manually.')  # 326
        else:  # 327
            self.status.config(text='JSON not loaded. Use "Open JSON…".')  # 328
    # ---------------- callbacks ----------------  # 329
# 330
    def _onBand(self):  # 331
        try:  # 332
            self.max_dist_line = max(2.0, float(self.entBand.get()))  # 333
        except Exception:  # 334
            pass  # 335
# 336
    def _on_click(self, event):  # 337
        """LMB — S/L mode; RMB — segment ratio mode."""  # 338
        if self.points is None or event.xdata is None or event.ydata is None:  # 339
            return  # 340
        x, y = float(event.xdata), float(event.ydata)  # 341
# 342
        # найти ближайший пик  # 343
        d2 = (self.points[:,1] - x)**2 + (self.points[:,0] - y)**2  # 344
        j = int(np.argmin(d2))  # 345
        if math.sqrt(d2[j]) > self.pick_tol:  # 346
            return  # 347
# 348
        if event.button == 1:  # 349
            # --- РЕЖИМ 1 (S/L) ---  # 350
            if self.anchor_idx is None:  # 351
                self.anchor_idx = j  # 352
                self._clear_rubber()  # 353
                self.draw_base(); self._draw_anchor(self.anchor_idx)  # 354
            else:  # 355
                i0, i1 = self.anchor_idx, j  # 356
                self.selected_idx = self._collect_points_along_segment(i0, i1, self.max_dist_line)  # 357
                self.anchor_idx = None  # 358
                self._clear_rubber()  # 359
                self.draw_base(); self._draw_selection(self.selected_idx)  # 360
                self.run_analysis()  # посчитать S/L и слова  # 361
        elif event.button == 3:  # 362
            # --- РЕЖИМ 2 (отношения) ---  # 363
            if self.ratio_anchor_idx is None:  # 364
                self.ratio_anchor_idx = j  # 365
                self._clear_rubber_ratio()  # 366
                self.draw_base(); self._draw_anchor(self.ratio_anchor_idx)  # 367
            else:  # 368
                i0, i1 = self.ratio_anchor_idx, j  # 369
                self.ratio_selected_idx = self._collect_points_along_segment(i0, i1, self.max_dist_line)  # 370
                self.ratio_anchor_idx = None  # 371
                self._clear_rubber_ratio()  # 372
                self.draw_base();  # 373
                self._draw_selection(self.ratio_selected_idx)  # 374
                self.run_ratio_analysis()  # посчитать отношения соседних сегментов  # 375
        elif event.button == 2:  # 376
            # --- РЕЖИМ 3 (многоугольники) ---  # 377
            self._handle_polygon_click(j)  # 378
# 379
    def _on_motion(self, event):  # 380
        if event.xdata is None or event.ydata is None or self.points is None:  # 381
            return  # 382
            # ЛКМ-резинка  # 383
        if self.anchor_idx is not None:  # 384
            ax = self.points[self.anchor_idx, 1]  # 385
            ay = self.points[self.anchor_idx, 0]  # 386
            bx = float(event.xdata); by = float(event.ydata)  # 387
            if self.rubber_line is None:  # 388
                (self.rubber_line,) = self.ax.plot([ax, bx], [ay, by], color='yellow', lw=2.0, alpha=0.9)  # 389
            else:  # 390
                self.rubber_line.set_data([ax, bx], [ay, by])  # 391
            self.canvas.draw_idle()  # 392
        # ПКМ-резинка  # 393
        if self.ratio_anchor_idx is not None:  # 394
            ax = self.points[self.ratio_anchor_idx, 1]  # 395
            ay = self.points[self.ratio_anchor_idx, 0]  # 396
            bx = float(event.xdata); by = float(event.ydata)  # 397
        if self.rubber_line_ratio is None:  # 398
            (self.rubber_line_ratio,) = self.ax.plot([ax, bx], [ay, by], color='yellow', lw=2.0, alpha=0.9)  # 399
        else:  # 400
            self.rubber_line_ratio.set_data([ax, bx], [ay, by])  # 401
        self.canvas.draw_idle()  # 402
        # СКМ-резинка (последнее ребро полигона)  # 403
        if self.polygon_current_idx:  # 404
            last_idx = self.polygon_current_idx[-1]  # 405
            ax = self.points[last_idx, 1]  # 406
            ay = self.points[last_idx, 0]  # 407
            bx = float(event.xdata);  # 408
            by = float(event.ydata)  # 409
            if self.polygon_rubber_line is None:  # 410
                (self.polygon_rubber_line,) = self.ax.plot([ax, bx], [ay, by], color='orange', lw=2.0, alpha=0.8,  # 411
                                                           zorder=3.4)  # 412
            else:  # 413
                self.polygon_rubber_line.set_data([ax, bx], [ay, by])  # 414
            self.canvas.draw_idle()  # 415
# 416
    def _on_list_select(self, event):  # 417
        """Highlight on list row click (both modes)."""  # 418
        if not self.list_index_map:  # 419
            return  # 420
        sel = self.lst.curselection()  # 421
        if not sel:  # 422
            return  # 423
        row = sel[0]  # 424
        meta = self.list_index_map.get(row)  # 425
        if not meta:  # 426
            return  # 427
        kind = meta[0]  # 428
# 429
        # Всегда перерисуем базу и цепочку, затем подсветим нужный объект  # 430
        self.draw_base()  # 431
        if kind == 'sl':  # 432
            # meta: ('sl', i0, n)  # 433
            i0, n = meta[1], meta[2]  # 434
            if self.curr_chain is None or self.curr_labels is None:  # 435
                return  # 436
            # показать всю выбранную ЛКМ-цепочку  # 437
            self._draw_selection(self.selected_idx)  # 438
            # подсветить выбранное окно  # 439
            self._highlight_word(self.curr_chain, self.curr_labels, i0, n)  # 440
        elif kind == 'ratio':  # 441
            # meta: ('ratio', k)  # 442
            k = meta[1]  # отношение между сегментами k-1 и k (индексация seg)  # 443
            if len(self.ratio_selected_idx) < 3:  # 444
                return  # 445
            chain = self.points[self.ratio_selected_idx].copy()  # 446
            self._draw_selection(self.ratio_selected_idx)  # 447
            # подсветим два сегмента: k-1 и k  # 448
            self._highlight_ratio_pair(chain, k-1, k)  # 449
        self.canvas.draw_idle()  # 450
# 451
    # ---------------- файлы/рисование ----------------#  # 452
    def open_json(self):  # 453
        p = filedialog.askopenfilename(filetypes=[('JSON', '*.json'), ('All', '*.*')])  # 454
        if not p: return  # 455
        self.load_json(Path(p))  # 456
# 457
    def load_json(self, json_path: Path):  # 458
        self.img_path, self.points, self.center, self.dead, self.srch, self.preproc = load_input(json_path)  # 459
        self.img_arr = None  # 460
        if self.img_path is not None:  # 461
            try:  # 462
                self.img_arr = load_grayscale_with_preproc(self.img_path, self.preproc)  # 463
            except Exception as exc:  # 464
                messagebox.showerror('Error', f'Failed to prepare the image:\n{exc}')  # 465
                self.img_arr = None  # 466
        self.clear_selection(redraw=False)  # 467
        self.draw_base()  # 468
        self.status.config(text=f'Loaded: {json_path.name}')  # 469
# 470
    def save_png(self):  # 471
        if self.img_path is None:  # 472
            messagebox.showinfo('Save', 'No data loaded.'); return  # 473
        p = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])  # 474
        if not p: return  # 475
        self.fig.savefig(p, dpi=150)  # 476
        self.status.config(text=f'Saved: {Path(p).name}')  # 477
# 478
    def draw_base(self):  # 479
        self.ax.clear()  # 480
        if self.img_arr is not None:  # 481
            self.ax.imshow(self.img_arr, cmap='gray', interpolation='nearest')  # 482
        elif self.img_path:  # 483
            im = Image.open(self.img_path).convert('L')  # 484
            self.ax.imshow(np.array(im), cmap='gray', interpolation='nearest')  # 485
        if self.points is not None and len(self.points):  # 486
            self.ax.scatter(self.points[:, 1], self.points[:, 0],  # 487
                            s=24, c='cyan', edgecolors='black', linewidths=0.4, label='peaks')  # 488
        if self.center is not None:  # 489
            cy, cx = self.center  # 490
            self.ax.scatter([cx], [cy], s=40, c='red', marker='o', label='center')  # 491
            if self.dead and self.dead > 0:  # 492
                self.ax.add_patch(Circle((cx, cy), self.dead, fill=False, ec='red', ls='--', lw=1.5))  # 493
            if self.srch and self.srch > 0:  # 494
                self.ax.add_patch(Circle((cx, cy), self.srch, fill=False, ec='red', ls=':', lw=1.0))  # 495
        self.ax.axis('off')  # 496
        self._draw_polygons()  # 497
        self.canvas.draw_idle()  # 498
# 499
    def _draw_anchor(self, idx: int):  # 500
        y, x = self.points[idx]  # 501
        self.ax.scatter([x], [y], s=52, c='yellow', edgecolors='k', linewidths=0.6, zorder=4)  # 502
        self.canvas.draw_idle()  # 503
# 504
    def _draw_selection(self, idxs: List[int]):  # 505
        if not idxs: return  # 506
        sel = self.points[idxs]  # 507
        self.ax.scatter(sel[:,1], sel[:,0], s=36, c='magenta', edgecolors='k', linewidths=0.6, zorder=3)  # 508
        for i in range(len(sel)-1):  # 509
            y1,x1 = sel[i]; y2,x2 = sel[i+1]  # 510
            self.ax.plot([x1,x2],[y1,y2], color='yellow', lw=1.8, ls='--', zorder=2)  # 511
            self.ax.text(x1, y1, str(i+1), color='magenta', fontsize=8, ha='right', va='bottom')  # 512
        yN, xN = sel[-1]  # 513
        self.ax.text(xN, yN, str(len(sel)), color='magenta', fontsize=8, ha='right', va='bottom')  # 514
        self.canvas.draw_idle()  # 515
# 516
    def _clear_rubber(self):  # 517
        if self.rubber_line is not None:  # 518
            try: self.rubber_line.remove()  # 519
            except Exception: pass  # 520
            self.rubber_line = None  # 521
            self.canvas.draw_idle()  # 522
# 523
    def _clear_rubber_ratio(self):  # 524
        if self.rubber_line_ratio is not None:  # 525
            try: self.rubber_line_ratio.remove()  # 526
            except Exception: pass  # 527
            self.rubber_line_ratio = None  # 528
            self.canvas.draw_idle()  # 529
# 530
    def _clear_polygon_rubber(self):  # 531
        if self.polygon_rubber_line is not None:  # 532
            try: self.polygon_rubber_line.remove()  # 533
            except Exception: pass  # 534
            self.polygon_rubber_line = None  # 535
            self.canvas.draw_idle()  # 536
# 537
    def clear_selection(self, redraw: bool=True):  # 538
        # Режим S/L  # 539
        self.selected_idx.clear()  # 540
        self.anchor_idx = None  # 541
        self._clear_rubber()  # 542
        self.curr_chain = None  # 543
        self.curr_seg = None  # 544
        self.curr_labels = None  # 545
        self.curr_ratio = float('nan')  # 546
        # Режим отношений  # 547
        self.ratio_selected_idx.clear()  # 548
        self.ratio_anchor_idx = None  # 549
        self._clear_rubber_ratio()  # 550
        # Режим многоугольников  # 551
        self.polygon_current_idx.clear()  # 552
        self.polygons_idx.clear()  # 553
        self._clear_polygon_rubber()  # 554
        self._reset_polygon_history()  # 555
        self._last_analysis_mode = None  # 556
        # UI  # 557
        self.lst.delete(0, tk.END)  # 558
        self.list_index_map.clear()  # 559
        self.lst_header.config(text='Found words (Fibonacci subsegments)')  # 560
        self.lbl_ratio.config(text='Average L/S along chain: —')  # 561
        self.lbl_ratio_neigh.config(text='Average neighboring segment ratio: —')  # 562
        self.txt_sl.delete('1.0', tk.END)  # 563
        self.txt_words.configure(state='normal');  # 564
        self.txt_words.delete('1.0', tk.END);  # 565
        self.txt_words.configure(state='disabled')  # 566
        if redraw: self.draw_base()  # 567
# 568
    def _capture_polygon_state(self) -> Tuple[List[int], List[List[int]]]:  # 569
        return (  # 570
            list(self.polygon_current_idx),  # 571
            [list(poly) for poly in self.polygons_idx],  # 572
        )  # 573
# 574
    def _reset_polygon_history(self) -> None:  # 575
        self._polygon_history = [self._capture_polygon_state()]  # 576
        self._polygon_redo.clear()  # 577
# 578
    def _record_polygon_state(self) -> None:  # 579
        state = self._capture_polygon_state()  # 580
        if self._polygon_history and state == self._polygon_history[-1]:  # 581
            return  # 582
        self._polygon_history.append(state)  # 583
        self._polygon_redo.clear()  # 584
# 585
    def _restore_polygon_state(self, state: Tuple[List[int], List[List[int]]]) -> None:  # 586
        current, finished = state  # 587
        self.polygon_current_idx = list(current)  # 588
        self.polygons_idx = [list(poly) for poly in finished]  # 589
        self._clear_polygon_rubber()  # 590
# 591
    def _on_polygon_undo(self, event):  # 592
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)):  # 593
            return  # 594
        if len(self._polygon_history) <= 1:  # 595
            return "break"  # 596
        state = self._polygon_history.pop()  # 597
        self._polygon_redo.append(state)  # 598
        prev_state = self._polygon_history[-1]  # 599
        self._restore_polygon_state(prev_state)  # 600
        self._refresh_canvas_after_polygon()  # 601
        self.status.config(text='Last polygon construction action undone.')  # 602
        return "break"  # 603
# 604
    def _on_polygon_redo(self, event):  # 605
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)):  # 606
            return  # 607
        if not self._polygon_redo:  # 608
            return "break"  # 609
        state = self._polygon_redo.pop()  # 610
        self._polygon_history.append(state)  # 611
        self._restore_polygon_state(state)  # 612
        self._refresh_canvas_after_polygon()  # 613
        self.status.config(text='Polygon construction action redone.')  # 614
        return "break"  # 615
# 616
    # ---------------- геометрия ----------------  # 617
# 618
    def _collect_points_along_segment(self, i0: int, i1: int, max_dist: float) -> List[int]:  # 619
        """Indices of points within max_dist from segment p0->p1, ordered by projection."""  # 620
        p0 = self.points[i0][[1,0]]  # (x,y)  # 621
        p1 = self.points[i1][[1,0]]  # (x,y)  # 622
        v = p1 - p0  # 623
        vv = float(np.dot(v, v))  # 624
        if vv == 0: return [i0]  # 625
        idx = []  # 626
        for k, (y, x) in enumerate(self.points):  # 627
            w = np.array([x, y]) - p0  # 628
            t = float(np.dot(w, v) / vv)  # 629
            if 0.0 <= t <= 1.0:  # 630
                proj = p0 + t * v  # 631
                dist = float(np.hypot(x - proj[0], y - proj[1]))  # 632
                if dist <= max_dist:  # 633
                    idx.append((t, k))  # 634
        idx.sort(key=lambda z: z[0])  # 635
        chain = [k for t,k in idx]  # 636
        # гарантировать крайние точки  # 637
        if chain and chain[0] != i0:  # 638
            if i0 in chain: chain.remove(i0)  # 639
            chain.insert(0, i0)  # 640
        if chain and chain[-1] != i1:  # 641
            if i1 in chain: chain.remove(i1)  # 642
            chain.append(i1)  # 643
        # уникальность с сохранением порядка  # 644
        seen=set(); out=[]  # 645
        for k in chain:  # 646
            if k not in seen:  # 647
                out.append(k); seen.add(k)  # 648
        return out  # 649
# 650
    # ---------------- редактирование SL ----------------  # 651
# 652
    def _on_sl_keypress(self, event):  # 653
        """Invert selected L/S with keys: 'i', 'sh', 'Sh'."""  # 654
        if event.char not in ('i', 'I', 'ш', 'Ш'):  # 655
            return  # 656
        try:  # 657
            start = self.txt_sl.index("sel.first")  # 658
            end   = self.txt_sl.index("sel.last")  # 659
        except tk.TclError:  # 660
            return "break"  # 661
        segment = self.txt_sl.get(start, end)  # 662
        flipped = ''.join('S' if ch == 'L' else ('L' if ch == 'S' else ch) for ch in segment)  # 663
        self.txt_sl.delete(start, end)  # 664
        self.txt_sl.insert(start, flipped)  # 665
        # пересчитать слова по отредактированному SL  # 666
        self._recompute_words_from_manual_SL()  # 667
        return "break"  # 668
# 669
    def _set_sl_text(self, s: str):  # 670
        self.txt_sl.delete('1.0', tk.END)  # 671
        self.txt_sl.insert('1.0', s)  # 672
# 673
    def _get_sl_text_letters(self) -> List[str]:  # 674
        raw = self.txt_sl.get('1.0', tk.END)  # 675
        return [ch for ch in raw if ch in ('L','S')]  # 676
# 677
    # ---------------- анализ: Режим 1 (S/L) ----------------  # 678
# 679
    def run_analysis(self):  # 680
        if self.points is None or len(self.points) < 2:  # 681
            messagebox.showinfo('Analysis', 'Not enough points (need ≥ 2).'); return  # 682
        if len(self.selected_idx) < 2:  # 683
            messagebox.showinfo('Analysis', 'Select two points first (LMB).'); return  # 684
# 685
        chain = self.points[self.selected_idx].copy()  # 686
        seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)  # 687
# 688
        labels, Slen, Llen, sidx, lidx = cluster_lengths(seg)  # 689
        SL = ['S' if labels[i] == sidx else 'L' for i in range(len(seg))]  # 690
        ratio = (Llen / Slen) if (Slen and not math.isnan(Slen)) else float('nan')  # 691
# 692
        self.curr_chain = chain  # 693
        self.curr_seg = seg  # 694
        self.curr_labels = SL[:]  # 695
        self.curr_ratio = ratio  # 696
# 697
        self._set_sl_text(''.join(SL))  # 698
        self._recompute_words_and_redraw()  # 699
        self._last_analysis_mode = 'sl'  # 700
# 701
    def _recompute_words_from_manual_SL(self):  # 702
        if self.curr_chain is None or self.curr_seg is None:  # 703
            return  # 704
        SL = self._get_sl_text_letters()  # 705
        m = len(self.curr_seg)  # 706
        if len(SL) < m: SL = SL + ['S']*(m-len(SL))  # 707
        if len(SL) > m: SL = SL[:m]  # 708
        self.curr_labels = SL  # 709
        self._recompute_words_and_redraw()  # 710
# 711
    def _highlight_word(self, chain: np.ndarray, SL: List[str], i0: int, n: int):  # 712
        # Highlight a window with length n LETTERS (=> n SEGMENTS) starting from segment i0.  # 713
        # Draw exactly n segments: k = i0 .. i0 + n - 1.  # 714
        # Place the number n at the end point of the window: chain[i0 + n].  # 715
        for k in range(i0, i0 + n):  # 716
            y1, x1 = chain[k]; y2, x2 = chain[k + 1]  # 717
            self.ax.plot([x1, x2], [y1, y2], color='lime', lw=3.2)  # 718
            my, mx = (y1 + y2) / 2, (x1 + x2) / 2  # 719
            self.ax.text(mx, my, SL[k], color='red', fontsize=9, ha='center', va='center')  # 720
        yN, xN = chain[i0 + n]  # 721
        self.ax.text(xN, yN, str(n), color='white', fontsize=8, ha='right', va='bottom')  # 722
# 723
    def _recompute_words_and_redraw(self):  # 724
        chain = self.curr_chain  # 725
        SL = self.curr_labels  # 726
        ratio = self.curr_ratio  # 727
# 728
        # найти слова по строгому правилу (L,S) = (F_{k-1}, F_{k-2}), n>=3  # 729
        found: List[Tuple[int, int, str, int, int]] = []  # 730
        fibNs = [n for n in fib_list_upto(len(SL)) if n >= 3]  # 731
        for n in fibNs:  # 732
            fibs = fib_list_upto(n)  # 733
            k = len(fibs) - 1  # 734
            exp1 = (fibs[k - 1], fibs[k - 2]) if k >= 2 else (1, 0)  # 735
            for i in range(0, len(SL) - n + 1):  # 736
                sub = SL[i:i + n]  # 737
                Lc, Sc = sub.count('L'), sub.count('S')  # 738
                if (Lc, Sc) == exp1:  # 739
                    found.append((n, i, ''.join(sub), Lc, Sc))  # 740
# 741
        # перерисовка  # 742
        self.draw_base()  # 743
        self._draw_selection(self.selected_idx)  # 744
        if found:  # 745
            found.sort(key=lambda z: (-z[0], z[1]))  # 746
            n, i0, word, Lc, Sc = found[0]  # 747
            self._highlight_word(chain, SL, i0, n)  # 748
            self.ax.text(0.01, 0.02,  # 749
                         f'Chain: L/S≈{ratio:.3f}  | Best word: n={n}, L={Lc}, S={Sc}',  # 750
                         transform=self.ax.transAxes, color='lime', fontsize=10,  # 751
                         ha='left', va='bottom')  # 752
        else:  # 753
            self.ax.text(0.01, 0.02, f'Chain: L/S≈{ratio:.3f}. No matches (n≥3) found.',  # 754
                         transform=self.ax.transAxes, color='orange', fontsize=10,  # 755
                         ha='left', va='bottom')  # 756
        self.canvas.draw_idle()  # 757
# 758
        # заполнить список + карту строк для подсветки  # 759
        self.lst.delete(0, tk.END)  # 760
        self.list_index_map.clear()  # 761
        self.lst_header.config(text='Found words (Fibonacci subsegments)')  # 762
        groups: Dict[int, List[Tuple[int,int,str,int,int]]] = {}  # 763
        for entry in found:  # 764
            groups.setdefault(entry[0], []).append(entry)  # 765
        row = 0  # 766
        if groups:  # 767
            for n in sorted(groups.keys()):  # 768
                self.lst.insert(tk.END, f'— n={n} —'); row += 1  # заголовок группы (без подсветки)  # 769
                for (n_, i0, word, Lc, Sc) in groups[n]:  # 770
                    self.lst.insert(tk.END, f'  i={i0}  word={word}  L={Lc} S={Sc}')  # 771
                    self.list_index_map[row] = ('sl', i0, n)  # строка -> окно (i0, n)  # 772
                    row += 1  # 773
                self.lst.insert(tk.END, ''); row += 1  # 774
        else:  # 775
            self.lst.insert(tk.END, 'No matches (n≥3)')  # 776
# 777
        # подписи  # 778
        if math.isfinite(ratio): self.lbl_ratio.config(text=f'Average L/S along chain: {ratio:.3f}')  # 779
        else:                    self.lbl_ratio.config(text='Average L/S along chain: —')  # 780
        self.lbl_ratio_neigh.config(text='Average neighboring segment ratio: —')  # 781
# 782
        # референс-префиксы  # 783
        self.txt_words.configure(state='normal')  # 784
        self.txt_words.delete('1.0', tk.END)  # 785
        max_len_ref = max(groups.keys(), default=min(len(SL), 34))  # 786
        for w in gen_fibonacci_words(max_len=max_len_ref, start='L'):  # 787
            self.txt_words.insert(tk.END, f'len={len(w)} → {w}\n')  # 788
        self.txt_words.configure(state='disabled')  # 789
# 790
        self.status.config(text=f'Selected points (LMB): {len(chain)}. Segments: {len(SL)}. '  # 791
                                f'Words (n≥3): {sum(len(v) for v in groups.values())}.')  # 792
# 793
    # ---------------- анализ: Режим 2 (отношения сегментов) ----------------  # 794
# 795
    def _highlight_ratio_pair(self, chain: np.ndarray, seg_a: int, seg_b: int):  # 796
        """Highlight two neighboring segments by their indices (0..M-1)."""  # 797
        M = len(chain) - 1  # 798
        if not (0 <= seg_a < M and 0 <= seg_b < M):  # 799
            return  # 800
        for k in (seg_a, seg_b):  # 801
            y1, x1 = chain[k]; y2, x2 = chain[k + 1]  # 802
            self.ax.plot([x1, x2], [y1, y2], color='lime', lw=3.2)  # 803
            my, mx = (y1 + y2) / 2, (x1 + x2) / 2  # 804
            self.ax.text(mx, my, f'{k+2}-{k+1}', color='red', fontsize=9, ha='center', va='center')  # 805
# 806
    def run_ratio_analysis(self):  # 807
        """Analysis for the chain selected with RMB: segment names 'n-(n-1)', neighboring ratios, mean value."""  # 808
        if self.points is None or len(self.points) < 2:  # 809
            messagebox.showinfo('Analysis', 'Not enough points.'); return  # 810
        if len(self.ratio_selected_idx) < 3:  # 811
            messagebox.showinfo('Analysis', 'Need ≥ 3 points (RMB).'); return  # 812
# 813
        chain = self.points[self.ratio_selected_idx].copy()  # 814
        seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)  # s1(2-1), s2(3-2), ...  # 815
# 816
        # подписи сегментов "n-(n-1)" и рисование  # 817
        self.draw_base()  # 818
        self._draw_selection(self.ratio_selected_idx)  # 819
        for i in range(len(chain)-1):  # 820
            y1,x1 = chain[i]; y2,x2 = chain[i+1]  # 821
            my, mx = (y1+y2)/2, (x1+x2)/2  # 822
            label = f"{i+2}-{i+1}"  # 823
            self.ax.text(mx, my, label, color='yellow', fontsize=9, ha='center', va='center')  # 824
        self.canvas.draw_idle()  # 825
# 826
        # отношения соседних сегментов: r_i = s_{i+1}/s_{i}  # 827
        ratios = []  # 828
        for i in range(1, len(seg)):  # 829
            if seg[i-1] > 0:  # 830
                ratios.append(seg[i]/seg[i-1])  # 831
            else:  # 832
                ratios.append(float('nan'))  # 833
# 834
        # список (режим отношений) + карта строк для подсветки  # 835
        self.lst.delete(0, tk.END)  # 836
        self.list_index_map.clear()  # 837
        self.lst_header.config(text='Neighboring segment ratios (RMB mode)')  # 838
        row = 0  # 839
        if len(ratios) == 0:  # 840
            self.lst.insert(tk.END, 'Not enough segments for ratios.')  # 841
        else:  # 842
            for i, r in enumerate(ratios, start=2):  # 843
                self.lst.insert(tk.END, f'  ({i + 1}-{i}) / ({i}-{i - 1})  ≈  {r:.6g}')  # 844
                k = i - 1  # 845
                self.list_index_map[row] = ('ratio', k)  # подсветим сегменты k-1 и k  # 846
                row += 1  # 847
# 848
        # среднее арифметическое по конечным значениям  # 849
        finite = [r for r in ratios if math.isfinite(r)]  # 850
        mean_ratio = float(np.mean(finite)) if finite else float('nan')  # 851
        if math.isfinite(mean_ratio):  # 852
            self.lbl_ratio_neigh.config(text=f'Average neighboring segment ratio: {mean_ratio:.6g}')  # 853
        else:  # 854
            self.lbl_ratio_neigh.config(text='Average neighboring segment ratio: —')  # 855
# 856
        # не трогаем S/L-поля (оставляем как были)  # 857
        self.status.config(text=f'Selected points (RMB): {len(chain)}. Segments: {len(seg)}. Ratios: {len(ratios)}.')  # 858
        self._last_analysis_mode = 'ratio'  # 859
# 860
    # ---------------- анализ: Режим 3 (полигоны) ----------------  # 861
# 862
    def _draw_polygons(self):  # 863
        """Draw finished polygons and the current one."""  # 864
        if self.points is None or (not self.polygons_idx and not self.polygon_current_idx):  # 865
            return  # 866
# 867
        # Завершённые полигоны  # 868
        for num, idxs in enumerate(self.polygons_idx, start=1):  # 869
            if len(idxs) < 3:  # 870
                continue  # 871
            pts = self.points[idxs]  # 872
            xs = pts[:, 1]  # 873
            ys = pts[:, 0]  # 874
            self.ax.fill(xs, ys, facecolor='deepskyblue', alpha=0.25,  # 875
                         edgecolor='blue', linewidth=1.4, zorder=1.5)  # 876
            cx = float(np.mean(xs))  # 877
            cy = float(np.mean(ys))  # 878
            self.ax.text(cx, cy, f'P{num}', color='navy', fontsize=9,  # 879
                         ha='center', va='center', zorder=1.6)  # 880
# 881
        # Текущий строящийся полигон  # 882
        if self.polygon_current_idx:  # 883
            pts_cur = self.points[self.polygon_current_idx]  # 884
            xs = pts_cur[:, 1]  # 885
            ys = pts_cur[:, 0]  # 886
            self.ax.plot(xs, ys, color='orange', lw=2.2, zorder=3.1)  # 887
            self.ax.scatter(xs, ys, s=46, c='orange', edgecolors='k', linewidths=0.6, zorder=3.2)  # 888
            first_x, first_y = xs[0], ys[0]  # 889
            self.ax.scatter([first_x], [first_y], s=70, facecolors='none', edgecolors='orange',  # 890
                            linewidths=1.5, zorder=3.3)  # 891
            if len(xs) >= 2:  # 892
                self.ax.plot([xs[-1], first_x], [ys[-1], first_y], color='orange', lw=1.2, ls=':', zorder=3.0)  # 893
            for idx, (xv, yv) in enumerate(zip(xs, ys), start=1):  # 894
                self.ax.text(xv, yv, str(idx), color='orange', fontsize=8,  # 895
                             ha='right', va='bottom', zorder=3.4)  # 896
# 897
    def _handle_polygon_click(self, point_idx: int):  # 898
        """Handle a middle mouse click to build a polygon."""  # 899
        if point_idx < 0 or point_idx >= len(self.points):  # 900
            return  # 901
# 902
        if not self.polygon_current_idx:  # 903
            self.polygon_current_idx.append(point_idx)  # 904
            self._refresh_canvas_after_polygon()  # 905
            self.status.config(text=f'Polygon construction: first vertex selected (#{point_idx + 1}).')  # 906
            self._record_polygon_state()  # 907
            return  # 908
# 909
        first_idx = self.polygon_current_idx[0]  # 910
        if point_idx == first_idx:  # 911
            if len(self.polygon_current_idx) < 3:  # 912
                self.status.config(text='Polygon requires ≥ 3 unique points.')  # 913
                return  # 914
            # Завершить полигон  # 915
            self.polygons_idx.append(self.polygon_current_idx.copy())  # 916
            poly_num = len(self.polygons_idx)  # 917
            vertex_count = len(self.polygon_current_idx)  # 918
            self.polygon_current_idx.clear()  # 919
            self._clear_polygon_rubber()  # 920
            self._refresh_canvas_after_polygon()  # 921
            self.status.config(text=f'Polygon #{poly_num} closed. Vertices: {vertex_count}.')  # 922
            self._record_polygon_state()  # 923
            return  # 924
# 925
        if point_idx in self.polygon_current_idx:  # 926
            self.status.config(text='Vertex already added. Choose another point or close the polygon.')  # 927
            return  # 928
# 929
        self.polygon_current_idx.append(point_idx)  # 930
        self._refresh_canvas_after_polygon()  # 931
        self.status.config(text=f'Polygon construction: total vertices {len(self.polygon_current_idx)}.')  # 932
        self._record_polygon_state()  # 933
# 934
    def _refresh_canvas_after_polygon(self):  # 935
        """Redraw the image, taking current modes and polygons into account."""  # 936
        if self._last_analysis_mode == 'sl' and self.curr_chain is not None:  # 937
            self._recompute_words_and_redraw()  # 938
        elif self._last_analysis_mode == 'ratio' and len(self.ratio_selected_idx) >= 3:  # 939
            self.run_ratio_analysis()  # 940
        else:  # 941
            self.draw_base()  # 942
        self.polygon_rubber_line = None  # 943
# 944
    def _polygon_area(self, idxs: List[int]) -> float:  # 945
        if self.points is None or len(idxs) < 3:  # 946
            return 0.0  # 947
        pts = self.points[idxs]  # 948
        xs = pts[:, 1]  # 949
        ys = pts[:, 0]  # 950
        shifted_x = np.roll(xs, -1)  # 951
        shifted_y = np.roll(ys, -1)  # 952
        area = 0.5 * abs(float(np.dot(xs, shifted_y) - np.dot(ys, shifted_x)))  # 953
        return area  # 954
# 955
    def _on_enter_key(self, event):  # 956
        """Compute the areas of the selected polygons and their ratios when Enter is pressed."""  # 957
        # Не мешаем виджетам ввода обрабатывать Enter  # 958
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)):  # 959
            return  # 960
# 961
        if self.polygon_current_idx:  # 962
            messagebox.showinfo('Polygons', 'Finish the current polygon (return to the first point).')  # 963
            return  # 964
        if len(self.polygons_idx) < 2:  # 965
            messagebox.showinfo('Polygons', 'Need at least two completed polygons.')  # 966
            return  # 967
# 968
        areas = [self._polygon_area(poly) for poly in self.polygons_idx]  # 969
        self.lst.delete(0, tk.END)  # 970
        self.list_index_map.clear()  # 971
        self.lst_header.config(text='Polygons (MMB) — areas and ratios')  # 972
# 973
        for i, area in enumerate(areas, start=1):  # 974
            self.lst.insert(tk.END, f'Polygon area {i}: {area:.6g}')  # 975
# 976
        self.lst.insert(tk.END, '')  # 977
        lines_added = False  # 978
        for idx in range(len(areas), 1, -1):  # 979
            prev_area = areas[idx - 2]  # 980
            curr_area = areas[idx - 1]  # 981
            if prev_area == 0:  # 982
                ratio_text = 'undefined (previous area = 0)'  # 983
            else:  # 984
                ratio = math.sqrt(curr_area / prev_area)  # 985
                ratio_text = f'{ratio:.6g}'  # 986
            self.lst.insert(tk.END, f'Size ratio {idx} and {idx - 1}: {ratio_text}')  # 987
            lines_added = True  # 988
# 989
        if not lines_added:  # 990
            self.lst.insert(tk.END, 'Not enough polygons for ratios.')  # 991
# 992
        self.status.config(text=f'Areas computed: {len(areas)}. See the list on the right.')  # 993
# 994
class App(tk.Tk):  # 995
    """Standalone wrapper compatible with the previous CLI."""  # 996
# 997
    def __init__(self):  # 998
        super().__init__()  # 999
        self.title('fibonachi_analysis')  # 1000
        self.geometry('1520x980')  # 1001
        self.resizable(True, True)  # 1002
        frame = FibonacciAnalysisFrame(self)  # 1003
        frame.pack(fill=tk.BOTH, expand=True)  # 1004
        self.frame = frame  # 1005
# 1006 .
# 1007
# ---- запуск ----  # 1008
if __name__ == '__main__':  # 1009
    app = App()  # 1010
    app.mainloop()  # 1011