#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fibonachi_analysis.py

Режим 1 (ЛКМ): поиск и редактирование S/L, фиб-«слов», среднее L/S по цепочке.
Режим 2 (ПКМ): без S/L — имена сегментов "n-(n-1)" и отношения соседних сегментов, среднее отношение.

Новое:
- Клик по строке списка в режиме ЛКМ подсвечивает выбранное "слово".
- Клик по строке списка в режиме ПКМ подсвечивает два соседних сегмента, участвующих в отношении.

Общее:
- Первый клик (ЛКМ или ПКМ) — якорь; тянется жёлтая «резинка» до курсора.
- Второй клик тем же типом кнопки — собираем точки рядом с отрезком (≤ допуск) и анализируем.
- Сохранение картинки, очистка выбора, автозагрузка fibo_input.json, показ центра/радиусов если есть.
- ESC — сброс обоих режимов (якоря, резинки).
- В поле «Последовательность S/L» можно инвертировать выделенные буквы: 'i' / 'ш' / 'Ш'.
"""
from __future__ import annotations

import sys, json, math
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle

# ---------------- загрузка ----------------

def _parse_cli(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="fibonachi_analysis — загрузка входных данных")
    p.add_argument("--payload", type=str, default=None, help="Путь к fibo_input.json")
    p.add_argument("--image", type=str, default=None, help="Путь к изображению (fallback)")
    p.add_argument("--points", type=str, default=None, help="Путь к JSON с точками (fallback)")
    return p.parse_args(argv)


def _candidate_dirs(extra_image: Optional[Path]) -> List[Path]:
    cands: List[Path] = []
    try:
        cands.append(Path.cwd())
    except Exception:
        pass
    if getattr(sys, "frozen", False):
        try:
            cands.append(Path(sys.executable).resolve().parent)
        except Exception:
            pass
        try:
            cands.append(Path(getattr(sys, "_MEIPASS")))
        except Exception:
            pass
    else:
        try:
            cands.append(Path(__file__).resolve().parent)
        except Exception:
            pass
    if extra_image:
        try:
            cands.append(extra_image.resolve().parent)
        except Exception:
            pass
    uniq, seen = [], set()
    for d in cands:
        rp = str(d.resolve())
        if rp not in seen:
            uniq.append(d); seen.add(rp)
    return uniq


def _autofind_json(extra_image: Optional[Path]) -> Optional[Path]:
    pats = ["fibo_input.json", "*fibo*input*.json", "*.fibo.json", "*.json"]
    for base in _candidate_dirs(extra_image):
        for pat in pats:
            try:
                for p in base.glob(pat):
                    name = p.name.lower()
                    if "fibo" in name and "input" in name:
                        return p.resolve()
                    if pat == "*.json":
                        try:
                            obj = json.loads(p.read_text(encoding="utf-8"))
                            if isinstance(obj, dict) and "image" in obj and "points" in obj:
                                return p.resolve()
                        except Exception:
                            pass
            except Exception:
                continue
    return None


def find_default_json(base_dir: Path) -> Optional[Path]:
    cand = base_dir / "fibo_input.json"
    if cand.exists():
        try:
            d = json.loads(cand.read_text(encoding="utf-8"))
            if "image" in d and "points" in d:
                return cand.resolve()
        except Exception:
            pass
    for p in base_dir.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if "image" in d and "points" in d:
                return p.resolve()
        except Exception:
            pass
    return _autofind_json(None)


def load_input(json_path: Path):
    d = json.loads(json_path.read_text(encoding='utf-8'))
    img = Path(d['image']) if d.get('image') else None
    if not img:
        raise RuntimeError("JSON не содержит ключ 'image'.")
    pts = np.array([[float(p['y']), float(p['x'])] for p in d.get('points', [])], float)
    center = None; dead = 0.0; srch = 0.0
    if isinstance(d.get('centers'), dict):
        c = d['centers'].get('overlay') or d['centers'].get('geometric')
        if c and 'x' in c and 'y' in c:
            center = (float(c['y']), float(c['x']))
    if isinstance(d.get('radii'), dict):
        if d['radii'].get('dead') is not None:  dead = float(d['radii']['dead'])
        if d['radii'].get('search') is not None: srch = float(d['radii']['search'])
    return img, pts, center, dead, srch

# ---------------- анализ S/L ----------------

def cluster_lengths(lengths: np.ndarray):
    """k=2 кластеризация длин на S/L; возвращает (labels, Slen, Llen, sidx, lidx)."""
    if lengths.size == 0:
        return np.array([], dtype=int), float("nan"), float("nan"), 0, 1
    c0, c1 = float(lengths.min()), float(lengths.max())
    if c0 == c1:
        lab = np.zeros(len(lengths), dtype=int)
        return lab, c0, float("nan"), 0, 1
    lab = np.zeros(len(lengths), dtype=int)
    for _ in range(60):
        d0 = np.abs(lengths - c0)
        d1 = np.abs(lengths - c1)
        lab = (d1 < d0).astype(int)
        nc0 = float(lengths[lab == 0].mean()) if np.any(lab == 0) else c0
        nc1 = float(lengths[lab == 1].mean()) if np.any(lab == 1) else c1  # <-- фикс: lab, не lаб
        if abs(nc0 - c0) < 1e-6 and abs(nc1 - c1) < 1e-6:
            c0, c1 = nc0, nc1; break
        c0, c1 = nc0, nc1
    # гарантируем S < L
    m0 = float(lengths[lab == 0].mean()) if np.any(lab == 0) else float("nan")
    m1 = float(lengths[lab == 1].mean()) if np.any(lab == 1) else float("nan")
    if (not math.isnan(m0)) and (not math.isnan(m1)) and m0 > m1:
        lab = 1 - lab
        m0, m1 = m1, m0
    return lab, m0, m1, 0, 1

def fib_list_upto(n: int) -> List[int]:
    """Фибоначчи до n (вкл.), начиная с 1,1,2,3,..."""
    if n <= 0: return []
    seq = [1, 1]
    while seq[-1] < n:
        seq.append(seq[-1] + seq[-2])
    return [k for k in seq if k <= n]

def gen_fibonacci_words(max_len: int, start: str = "L") -> List[str]:
    """Генерация префиксов «фиб-слова»: L->LS, S->L, до max_len."""
    if max_len <= 0: return []
    words = ["L" if start.upper() == "L" else "S"]
    while len(words[-1]) <= max_len:
        nxt = "".join(("LS" if ch == "L" else "L") for ch in words[-1])
        if len(nxt) > max_len: break
        words.append(nxt)
    return words

# ---------------- GUI ----------------

class FibonacciAnalysisFrame(tk.Frame):
    def __init__(self, master: tk.Misc, controller=None, auto_load: bool = True):
        super().__init__(master)
        self.controller = controller

        # Данные
        self.img_path: Optional[Path] = None
        self.points: Optional[np.ndarray] = None      # (N,2) [y,x]
        self.center: Optional[Tuple[float, float]] = None
        self.dead: float = 0.0
        self.srch: float = 0.0

        # Общие параметры
        self.pick_tol = 10.0
        self.max_dist_line = 12.0

        # Режим 1 (ЛКМ): S/L
        self.selected_idx: List[int] = []
        self.anchor_idx: Optional[int] = None
        self.rubber_line = None
        # Текущее состояние S/L
        self.curr_chain: Optional[np.ndarray] = None
        self.curr_seg: Optional[np.ndarray] = None
        self.curr_labels: Optional[List[str]] = None
        self.curr_ratio: float = float('nan')

        # Режим 2 (ПКМ): отношения сегментов
        self.ratio_anchor_idx: Optional[int] = None
        self.rubber_line_ratio = None
        self.ratio_selected_idx: List[int] = []

        # Маппинг строк Listbox -> объект подсветки
        # для S/L: ('sl', i0, n) ; для отношений: ('ratio', k) где k — индекс отношения (сегменты k-1 и k)
        self.list_index_map: Dict[int, Tuple] = {}

        # --- верхняя панель ---
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        container = tk.Frame(self)
        container.grid(row=0, column=0, sticky="nsew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=0)

        left = tk.Frame(container)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=12)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        right = tk.Frame(container, width=470)
        right.grid(row=0, column=1, sticky="ns", pady=12)
        right.columnconfigure(0, weight=1)

        controls = tk.Frame(right)
        controls.grid(row=0, column=0, sticky="ew", padx=6, pady=(0, 10))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        tk.Button(controls, text='Открыть JSON…', command=self.open_json).grid(row=0, column=0, sticky='ew', padx=4, pady=2)
        tk.Button(controls, text='Начать анализ (ЛКМ-цепочка)', command=self.run_analysis).grid(row=0, column=1, sticky='ew', padx=4, pady=2)
        tk.Button(controls, text='Сохранить PNG', command=self.save_png).grid(row=1, column=0, sticky='ew', padx=4, pady=2)
        tk.Button(controls, text='Очистить выбор', command=self.clear_selection).grid(row=1, column=1, sticky='ew', padx=4, pady=2)
        tk.Label(controls, text='Толщина отбора (px):').grid(row=2, column=0, sticky='w', padx=4, pady=(8, 2))
        self.entBand = tk.Spinbox(controls, from_=2, to=100, width=6, command=self._onBand)
        self.entBand.delete(0, 'end'); self.entBand.insert(0, str(int(self.max_dist_line)))
        self.entBand.grid(row=2, column=1, sticky='ew', padx=4, pady=(8, 2))

        self.status = tk.Label(right, text='', anchor='w')
        self.status.grid(row=1, column=0, sticky='ew', padx=6, pady=(0, 10))

        # заголовок списка (динамический)
        self.lst_header = tk.Label(right, text='Найденные слова (подотрезки Фибоначчи)')
        self.lst_header.grid(row=2, column=0, sticky='w', padx=6, pady=(0, 2))
        self.lst = tk.Listbox(right, width=66, height=22)
        self.lst.grid(row=3, column=0, sticky='nsew', padx=6)
        right.rowconfigure(3, weight=1)
        self.lst.bind('<<ListboxSelect>>', self._on_list_select)

        # подписи со средними
        self.lbl_ratio = tk.Label(right, text='Среднее L/S по цепочке: —')
        self.lbl_ratio.grid(row=4, column=0, sticky='w', padx=6, pady=(6, 4))
        self.lbl_ratio_neigh = tk.Label(right, text='Среднее отношение соседних сегментов: —')
        self.lbl_ratio_neigh.grid(row=5, column=0, sticky='w', padx=6, pady=(2, 8))

        # последовательность S/L (полная) + бинды инверсии
        tk.Label(right, text='Последовательность S/L (полная):').grid(row=6, column=0, sticky='w', padx=6, pady=(4, 2))
        self.txt_sl = tk.Text(right, height=6, wrap='word')
        self.txt_sl.grid(row=7, column=0, sticky='ew', padx=6, pady=(0, 4))
        self.txt_sl.bind('<KeyPress>', self._on_sl_keypress)

        # префиксы «фиб-слова»
        tk.Label(right, text='Префиксы "фиб-слова" (L→LS, S→L)').grid(row=8, column=0, sticky='w', padx=6, pady=(8, 2))
        self.txt_words = tk.Text(right, height=10, state='disabled')
        self.txt_words.grid(row=9, column=0, sticky='ew', padx=6, pady=(0, 8))

        self.fig = plt.Figure(figsize=(9.6, 6.6)); self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        # обработчики мыши и клавиш
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.bind('<Escape>', lambda e: self.clear_selection())

        # автозагрузка
        if auto_load:
            base = Path(getattr(sys, '_MEIPASS', Path(__file__).parent)) if getattr(sys, 'frozen', False) else Path(
                __file__).parent
            auto = find_default_json(base)
            if auto:
                try:
                    self.load_json(auto)
                    self.status.config(text=f'Загружен: {auto.name}')
                except Exception as e:
                    messagebox.showerror('Ошибка загрузки', str(e))
            else:
                self.status.config(text='JSON не найден. Выберите файл вручную.')
        else:
            self.status.config(text='JSON не загружен. Используйте «Открыть JSON…».')
    # ---------------- callbacks ----------------

    def _onBand(self):
        try:
            self.max_dist_line = max(2.0, float(self.entBand.get()))
        except Exception:
            pass

    def _on_click(self, event):
        """ЛКМ — режим S/L; ПКМ — режим отношений сегментов."""
        if self.points is None or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        # найти ближайший пик
        d2 = (self.points[:,1] - x)**2 + (self.points[:,0] - y)**2
        j = int(np.argmin(d2))
        if math.sqrt(d2[j]) > self.pick_tol:
            return

        if event.button == 1:
            # --- РЕЖИМ 1 (S/L) ---
            if self.anchor_idx is None:
                self.anchor_idx = j
                self._clear_rubber()
                self.draw_base(); self._draw_anchor(self.anchor_idx)
            else:
                i0, i1 = self.anchor_idx, j
                self.selected_idx = self._collect_points_along_segment(i0, i1, self.max_dist_line)
                self.anchor_idx = None
                self._clear_rubber()
                self.draw_base(); self._draw_selection(self.selected_idx)
                self.run_analysis()  # посчитать S/L и слова
        elif event.button == 3:
            # --- РЕЖИМ 2 (отношения) ---
            if self.ratio_anchor_idx is None:
                self.ratio_anchor_idx = j
                self._clear_rubber_ratio()
                self.draw_base(); self._draw_anchor(self.ratio_anchor_idx)
            else:
                i0, i1 = self.ratio_anchor_idx, j
                self.ratio_selected_idx = self._collect_points_along_segment(i0, i1, self.max_dist_line)
                self.ratio_anchor_idx = None
                self._clear_rubber_ratio()
                self.draw_base(); self._draw_selection(self.ratio_selected_idx)
                self.run_ratio_analysis()  # посчитать отношения соседних сегментов

    def _on_motion(self, event):
        if event.xdata is None or event.ydata is None or self.points is None:
            return
        # ЛКМ-резинка
        if self.anchor_idx is not None:
            ax = self.points[self.anchor_idx, 1]
            ay = self.points[self.anchor_idx, 0]
            bx = float(event.xdata); by = float(event.ydata)
            if self.rubber_line is None:
                (self.rubber_line,) = self.ax.plot([ax, bx], [ay, by], color='yellow', lw=2.0, alpha=0.9)
            else:
                self.rubber_line.set_data([ax, bx], [ay, by])
            self.canvas.draw_idle()
        # ПКМ-резинка
        if self.ratio_anchor_idx is not None:
            ax = self.points[self.ratio_anchor_idx, 1]
            ay = self.points[self.ratio_anchor_idx, 0]
            bx = float(event.xdata); by = float(event.ydata)
            if self.rubber_line_ratio is None:
                (self.rubber_line_ratio,) = self.ax.plot([ax, bx], [ay, by], color='yellow', lw=2.0, alpha=0.9)
            else:
                self.rubber_line_ratio.set_data([ax, bx], [ay, by])
            self.canvas.draw_idle()

    def _on_list_select(self, event):
        """Подсветка при клике по строке списка (оба режима)."""
        if not self.list_index_map:
            return
        sel = self.lst.curselection()
        if not sel:
            return
        row = sel[0]
        meta = self.list_index_map.get(row)
        if not meta:
            return
        kind = meta[0]

        # Всегда перерисуем базу и цепочку, затем подсветим нужный объект
        self.draw_base()
        if kind == 'sl':
            # meta: ('sl', i0, n)
            i0, n = meta[1], meta[2]
            if self.curr_chain is None or self.curr_labels is None:
                return
            # показать всю выбранную ЛКМ-цепочку
            self._draw_selection(self.selected_idx)
            # подсветить выбранное окно
            self._highlight_word(self.curr_chain, self.curr_labels, i0, n)
        elif kind == 'ratio':
            # meta: ('ratio', k)
            k = meta[1]  # отношение между сегментами k-1 и k (индексация seg)
            if len(self.ratio_selected_idx) < 3:
                return
            chain = self.points[self.ratio_selected_idx].copy()
            self._draw_selection(self.ratio_selected_idx)
            # подсветим два сегмента: k-1 и k
            self._highlight_ratio_pair(chain, k-1, k)
        self.canvas.draw_idle()

    # ---------------- файлы/рисование ----------------

    def open_json(self):
        p = filedialog.askopenfilename(filetypes=[('JSON', '*.json'), ('All', '*.*')])
        if not p: return
        self.load_json(Path(p))

    def load_json(self, json_path: Path):
        self.img_path, self.points, self.center, self.dead, self.srch = load_input(json_path)
        self.clear_selection(redraw=False)
        self.draw_base()
        self.status.config(text=f'Загружен: {json_path.name}')

    def save_png(self):
        if self.img_path is None:
            messagebox.showinfo('Сохранить', 'Нет загруженных данных.'); return
        p = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])
        if not p: return
        self.fig.savefig(p, dpi=150)
        self.status.config(text=f'Сохранено: {Path(p).name}')

    def draw_base(self):
        self.ax.clear()
        if self.img_path:
            im = Image.open(self.img_path).convert('L')
            self.ax.imshow(np.array(im), cmap='gray', interpolation='nearest')
        if self.points is not None and len(self.points):
            self.ax.scatter(self.points[:, 1], self.points[:, 0],
                            s=24, c='cyan', edgecolors='black', linewidths=0.4, label='peaks')
        if self.center is not None:
            cy, cx = self.center
            self.ax.scatter([cx], [cy], s=40, c='red', marker='o', label='center')
            if self.dead and self.dead > 0:
                self.ax.add_patch(Circle((cx, cy), self.dead, fill=False, ec='red', ls='--', lw=1.5))
            if self.srch and self.srch > 0:
                self.ax.add_patch(Circle((cx, cy), self.srch, fill=False, ec='red', ls=':', lw=1.0))
        self.ax.axis('off')
        self.canvas.draw_idle()

    def _draw_anchor(self, idx: int):
        y, x = self.points[idx]
        self.ax.scatter([x], [y], s=52, c='yellow', edgecolors='k', linewidths=0.6, zorder=4)
        self.canvas.draw_idle()

    def _draw_selection(self, idxs: List[int]):
        if not idxs: return
        sel = self.points[idxs]
        self.ax.scatter(sel[:,1], sel[:,0], s=36, c='magenta', edgecolors='k', linewidths=0.6, zorder=3)
        for i in range(len(sel)-1):
            y1,x1 = sel[i]; y2,x2 = sel[i+1]
            self.ax.plot([x1,x2],[y1,y2], color='yellow', lw=1.8, ls='--', zorder=2)
            self.ax.text(x1, y1, str(i+1), color='magenta', fontsize=8, ha='right', va='bottom')
        yN, xN = sel[-1]
        self.ax.text(xN, yN, str(len(sel)), color='magenta', fontsize=8, ha='right', va='bottom')
        self.canvas.draw_idle()

    def _clear_rubber(self):
        if self.rubber_line is not None:
            try: self.rubber_line.remove()
            except Exception: pass
            self.rubber_line = None
            self.canvas.draw_idle()

    def _clear_rubber_ratio(self):
        if self.rubber_line_ratio is not None:
            try: self.rubber_line_ratio.remove()
            except Exception: pass
            self.rubber_line_ratio = None
            self.canvas.draw_idle()

    def clear_selection(self, redraw: bool=True):
        # Режим S/L
        self.selected_idx.clear()
        self.anchor_idx = None
        self._clear_rubber()
        self.curr_chain = None
        self.curr_seg = None
        self.curr_labels = None
        self.curr_ratio = float('nan')
        # Режим отношений
        self.ratio_selected_idx.clear()
        self.ratio_anchor_idx = None
        self._clear_rubber_ratio()
        # UI
        self.lst.delete(0, tk.END)
        self.list_index_map.clear()
        self.lst_header.config(text='Найденные слова (подотрезки Фибоначчи)')
        self.lbl_ratio.config(text='Среднее L/S по цепочке: —')
        self.lbl_ratio_neigh.config(text='Среднее отношение соседних сегментов: —')
        self.txt_sl.delete('1.0', tk.END)
        self.txt_words.configure(state='normal'); self.txt_words.delete('1.0', tk.END); self.txt_words.configure(state='disabled')
        if redraw: self.draw_base()

    # ---------------- геометрия ----------------

    def _collect_points_along_segment(self, i0: int, i1: int, max_dist: float) -> List[int]:
        """Индексы точек в пределах max_dist от отрезка p0->p1, упорядоченные по проекции."""
        p0 = self.points[i0][[1,0]]  # (x,y)
        p1 = self.points[i1][[1,0]]  # (x,y)
        v = p1 - p0
        vv = float(np.dot(v, v))
        if vv == 0: return [i0]
        idx = []
        for k, (y, x) in enumerate(self.points):
            w = np.array([x, y]) - p0
            t = float(np.dot(w, v) / vv)
            if 0.0 <= t <= 1.0:
                proj = p0 + t * v
                dist = float(np.hypot(x - proj[0], y - proj[1]))
                if dist <= max_dist:
                    idx.append((t, k))
        idx.sort(key=lambda z: z[0])
        chain = [k for t,k in idx]
        # гарантировать крайние точки
        if chain and chain[0] != i0:
            if i0 in chain: chain.remove(i0)
            chain.insert(0, i0)
        if chain and chain[-1] != i1:
            if i1 in chain: chain.remove(i1)
            chain.append(i1)
        # уникальность с сохранением порядка
        seen=set(); out=[]
        for k in chain:
            if k not in seen:
                out.append(k); seen.add(k)
        return out

    # ---------------- редактирование SL ----------------

    def _on_sl_keypress(self, event):
        """Инверсия выделенных L/S по клавишам: 'i', 'ш', 'Ш'."""
        if event.char not in ('i', 'I', 'ш', 'Ш'):
            return
        try:
            start = self.txt_sl.index("sel.first")
            end   = self.txt_sl.index("sel.last")
        except tk.TclError:
            return "break"
        segment = self.txt_sl.get(start, end)
        flipped = ''.join('S' if ch == 'L' else ('L' if ch == 'S' else ch) for ch in segment)
        self.txt_sl.delete(start, end)
        self.txt_sl.insert(start, flipped)
        # пересчитать слова по отредактированному SL
        self._recompute_words_from_manual_SL()
        return "break"

    def _set_sl_text(self, s: str):
        self.txt_sl.delete('1.0', tk.END)
        self.txt_sl.insert('1.0', s)

    def _get_sl_text_letters(self) -> List[str]:
        raw = self.txt_sl.get('1.0', tk.END)
        return [ch for ch in raw if ch in ('L','S')]

    # ---------------- анализ: Режим 1 (S/L) ----------------

    def run_analysis(self):
        if self.points is None or len(self.points) < 2:
            messagebox.showinfo('Анализ', 'Недостаточно точек (нужно ≥ 2).'); return
        if len(self.selected_idx) < 2:
            messagebox.showinfo('Анализ', 'Сначала выберите две точки (ЛКМ).'); return

        chain = self.points[self.selected_idx].copy()
        seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)

        labels, Slen, Llen, sidx, lidx = cluster_lengths(seg)
        SL = ['S' if labels[i] == sidx else 'L' for i in range(len(seg))]
        ratio = (Llen / Slen) if (Slen and not math.isnan(Slen)) else float('nan')

        self.curr_chain = chain
        self.curr_seg = seg
        self.curr_labels = SL[:]
        self.curr_ratio = ratio

        self._set_sl_text(''.join(SL))
        self._recompute_words_and_redraw()

    def _recompute_words_from_manual_SL(self):
        if self.curr_chain is None or self.curr_seg is None:
            return
        SL = self._get_sl_text_letters()
        m = len(self.curr_seg)
        if len(SL) < m: SL = SL + ['S']*(m-len(SL))
        if len(SL) > m: SL = SL[:m]
        self.curr_labels = SL
        self._recompute_words_and_redraw()

    def _highlight_word(self, chain: np.ndarray, SL: List[str], i0: int, n: int):
        """
        Подсветить окно длиной n БУКВ (=> n СЕГМЕНТОВ), начинающееся с сегмента i0.
        Рисуем РОВНО n отрезков: k = i0 .. i0+n-1.
        Номер n ставим у конечной точки окна: chain[i0 + n].
        """
        for k in range(i0, i0 + n):
            y1, x1 = chain[k]; y2, x2 = chain[k + 1]
            self.ax.plot([x1, x2], [y1, y2], color='lime', lw=3.2)
            my, mx = (y1 + y2) / 2, (x1 + x2) / 2
            self.ax.text(mx, my, SL[k], color='red', fontsize=9, ha='center', va='center')
        yN, xN = chain[i0 + n]
        self.ax.text(xN, yN, str(n), color='white', fontsize=8, ha='right', va='bottom')

    def _recompute_words_and_redraw(self):
        chain = self.curr_chain
        SL = self.curr_labels
        ratio = self.curr_ratio

        # найти слова по строгому правилу (L,S) = (F_{k-1}, F_{k-2}), n>=3
        found: List[Tuple[int, int, str, int, int]] = []
        fibNs = [n for n in fib_list_upto(len(SL)) if n >= 3]
        for n in fibNs:
            fibs = fib_list_upto(n)
            k = len(fibs) - 1
            exp1 = (fibs[k - 1], fibs[k - 2]) if k >= 2 else (1, 0)
            for i in range(0, len(SL) - n + 1):
                sub = SL[i:i + n]
                Lc, Sc = sub.count('L'), sub.count('S')
                if (Lc, Sc) == exp1:
                    found.append((n, i, ''.join(sub), Lc, Sc))

        # перерисовка
        self.draw_base()
        self._draw_selection(self.selected_idx)
        if found:
            found.sort(key=lambda z: (-z[0], z[1]))
            n, i0, word, Lc, Sc = found[0]
            self._highlight_word(chain, SL, i0, n)
            self.ax.text(0.01, 0.02,
                         f'Цепочка: L/S≈{ratio:.3f}  | Лучшее слово: n={n}, L={Lc}, S={Sc}',
                         transform=self.ax.transAxes, color='lime', fontsize=10,
                         ha='left', va='bottom')
        else:
            self.ax.text(0.01, 0.02, f'Цепочка: L/S≈{ratio:.3f}. Совпадений (n≥3) не найдено.',
                         transform=self.ax.transAxes, color='orange', fontsize=10,
                         ha='left', va='bottom')
        self.canvas.draw_idle()

        # заполнить список + карту строк для подсветки
        self.lst.delete(0, tk.END)
        self.list_index_map.clear()
        self.lst_header.config(text='Найденные слова (подотрезки Фибоначчи)')
        groups: Dict[int, List[Tuple[int,int,str,int,int]]] = {}
        for entry in found:
            groups.setdefault(entry[0], []).append(entry)
        row = 0
        if groups:
            for n in sorted(groups.keys()):
                self.lst.insert(tk.END, f'— n={n} —'); row += 1  # заголовок группы (без подсветки)
                for (n_, i0, word, Lc, Sc) in groups[n]:
                    self.lst.insert(tk.END, f'  i={i0}  word={word}  L={Lc} S={Sc}')
                    self.list_index_map[row] = ('sl', i0, n)  # строка -> окно (i0, n)
                    row += 1
                self.lst.insert(tk.END, ''); row += 1
        else:
            self.lst.insert(tk.END, 'нет совпадений (n≥3)')

        # подписи
        if math.isfinite(ratio): self.lbl_ratio.config(text=f'Среднее L/S по цепочке: {ratio:.3f}')
        else:                    self.lbl_ratio.config(text='Среднее L/S по цепочке: —')
        self.lbl_ratio_neigh.config(text='Среднее отношение соседних сегментов: —')

        # референс-префиксы
        self.txt_words.configure(state='normal')
        self.txt_words.delete('1.0', tk.END)
        max_len_ref = max(groups.keys(), default=min(len(SL), 34))
        for w in gen_fibonacci_words(max_len=max_len_ref, start='L'):
            self.txt_words.insert(tk.END, f'len={len(w)} → {w}\n')
        self.txt_words.configure(state='disabled')

        self.status.config(text=f'Выбрано точек (ЛКМ): {len(chain)}. Сегментов: {len(SL)}. '
                                f'Слов (n≥3): {sum(len(v) for v in groups.values())}.')

    # ---------------- анализ: Режим 2 (отношения сегментов) ----------------

    def _highlight_ratio_pair(self, chain: np.ndarray, seg_a: int, seg_b: int):
        """Подсветить два соседних сегмента по их индексам (0..M-1)."""
        M = len(chain) - 1
        if not (0 <= seg_a < M and 0 <= seg_b < M):
            return
        for k in (seg_a, seg_b):
            y1, x1 = chain[k]; y2, x2 = chain[k + 1]
            self.ax.plot([x1, x2], [y1, y2], color='lime', lw=3.2)
            my, mx = (y1 + y2) / 2, (x1 + x2) / 2
            self.ax.text(mx, my, f'{k+2}-{k+1}', color='red', fontsize=9, ha='center', va='center')

    def run_ratio_analysis(self):
        """Анализ для цепочки, выбранной ПКМ: названия сегментов 'n-(n-1)', отношения соседних, среднее."""
        if self.points is None or len(self.points) < 2:
            messagebox.showinfo('Анализ', 'Недостаточно точек.'); return
        if len(self.ratio_selected_idx) < 3:
            messagebox.showinfo('Анализ', 'Нужно ≥ 3 точки (ПКМ).'); return

        chain = self.points[self.ratio_selected_idx].copy()
        seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)  # s1(2-1), s2(3-2), ...

        # подписи сегментов "n-(n-1)" и рисование
        self.draw_base()
        self._draw_selection(self.ratio_selected_idx)
        for i in range(len(chain)-1):
            y1,x1 = chain[i]; y2,x2 = chain[i+1]
            my, mx = (y1+y2)/2, (x1+x2)/2
            label = f"{i+2}-{i+1}"
            self.ax.text(mx, my, label, color='yellow', fontsize=9, ha='center', va='center')
        self.canvas.draw_idle()

        # отношения соседних сегментов: r_i = s_{i+1}/s_{i}
        ratios = []
        for i in range(1, len(seg)):
            if seg[i-1] > 0:
                ratios.append(seg[i]/seg[i-1])
            else:
                ratios.append(float('nan'))

        # список (режим отношений) + карта строк для подсветки
        self.lst.delete(0, tk.END)
        self.list_index_map.clear()
        self.lst_header.config(text='Отношения соседних сегментов (ПКМ-режим)')
        row = 0
        if len(ratios) == 0:
            self.lst.insert(tk.END, 'Недостаточно сегментов для отношений.')
        else:
            for i, r in enumerate(ratios, start=2):
                self.lst.insert(tk.END, f'  ({i+1}-{i}) / ({i}-{i-1})  ≈  {r:.6g}')
                k = i-1
                self.list_index_map[row] = ('ratio', k)  # подсветим сегменты k-1 и k
                row += 1

        # среднее арифметическое по конечным значениям
        finite = [r for r in ratios if math.isfinite(r)]
        mean_ratio = float(np.mean(finite)) if finite else float('nan')
        if math.isfinite(mean_ratio):
            self.lbl_ratio_neigh.config(text=f'Среднее отношение соседних сегментов: {mean_ratio:.6g}')
        else:
            self.lbl_ratio_neigh.config(text='Среднее отношение соседних сегментов: —')

        # не трогаем S/L-поля (оставляем как были)
        self.status.config(text=f'Выбрано точек (ПКМ): {len(chain)}. Сегментов: {len(seg)}. Отношений: {len(ratios)}.')

class App(tk.Tk):
    """Standalone-обёртка, совместимая с предыдущим CLI."""

    def __init__(self):
        super().__init__()
        self.title('fibonachi_analysis')
        self.geometry('1520x980')
        self.resizable(True, True)
        frame = FibonacciAnalysisFrame(self)
        frame.pack(fill=tk.BOTH, expand=True)
        self.frame = frame


# ---- запуск ----
if __name__ == '__main__':
    app = App()
    app.mainloop()