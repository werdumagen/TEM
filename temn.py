#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAED Symmetry – Launcher
========================
Новое:
  • Переключатель предобработки:
      - «Стандартная» (equalize + GaussianBlur)
      - «Без сглаживания» (сырой grayscale)
      - «CLAHE» (локальное выравнивание) с настраиваемым clipLimit и размером тайла.
  • Весь препроцесс выполняется ТОЛЬКО здесь, редактор изображение не трогает.
Остальной функционал без изменений: ручной/автоцентр, уточнение по антиподам, мёртвая зона, радиус поиска, запуск редактора.

Дополнено:
  • Сохранение точек вместе с их интенсивностью в единый файл saed_input.json.
  • saed_editor запускается с одним аргументом --input (путь к saed_input.json).
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# opencv нужен только если выбран режим CLAHE
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # сообщим пользователю при выборе CLAHE

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# -------------------------- Алгоритм --------------------------
@dataclass
class CenterResult:
    cy: float
    cx: float
    method: str

def load_grayscale_with_preproc(path: Path, mode: str,
                                clahe_clip: float = 1.5,
                                clahe_tiles: int = 8) -> np.ndarray:
    """
    mode: 'standard' | 'raw' | 'clahe'
    """
    if mode == 'clahe':
        if cv2 is None:
            raise RuntimeError("Для режима CLAHE требуется пакет opencv-python (или opencv-python-headless).")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("Не удалось прочитать изображение через OpenCV.")
        clip = max(0.1, float(clahe_clip))
        tiles = max(2, int(clahe_tiles))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
        img = clahe.apply(img)
        return img.astype(np.float32)

    # PIL пути (standard/raw)
    pil = Image.open(path).convert("L")
    if mode == 'standard':
        pil = ImageOps.equalize(pil)
        pil = pil.filter(ImageFilter.GaussianBlur(radius=0.8))
    elif mode == 'raw':
        pass  # только grayscale
    else:
        raise ValueError(f"Unknown preproc mode: {mode}")
    return np.array(pil, dtype=np.float32)

def detect_spots(arr: np.ndarray, perc: float=99.0, win: int=7, min_sep: int=5, max_spots: int=6000) -> np.ndarray:
    H, W = arr.shape
    rad = max(1, win // 2)
    th = float(np.percentile(arr, perc))
    cand = []
    for y in range(rad, H-rad):
        for x in range(rad, W-rad):
            v = arr[y, x]
            if v < th: continue
            patch = arr[y-rad:y+rad+1, x-rad:x+rad+1]
            if v >= patch.max():
                cand.append((float(y), float(x), float(v)))
    cand.sort(key=lambda t: -t[2])
    kept = []
    msep2 = float(min_sep)**2
    for (y, x, v) in cand:
        if all((y-y0)**2 + (x-x0)**2 >= msep2 for (y0, x0, _) in kept):
            kept.append((y, x, v))
        if len(kept) >= max_spots:
            break
    return np.array(kept, dtype=float) if kept else np.zeros((0,3), dtype=float)

def geometric_midpoint(arr: np.ndarray) -> CenterResult:
    H, W = arr.shape
    return CenterResult(cy=(H-1)/2.0, cx=(W-1)/2.0, method="midpoint")

def refine_center_antipodal(center: Tuple[float,float], pts: np.ndarray, tol_ang_deg: float=8.0, tol_rel_r: float=0.06, iters: int=3) -> CenterResult:
    cy, cx = float(center[0]), float(center[1])
    if len(pts) < 4:
        return CenterResult(cy=cy, cx=cx, method="midpoint (fallback)")
    for _ in range(max(0, int(iters))):
        dy = pts[:,0]-cy; dx = pts[:,1]-cx
        r = np.hypot(dx, dy)
        u = np.column_stack((dx, dy)) / (r[:,None]+1e-9)
        cos_thr = -np.cos(np.deg2rad(180.0 - float(tol_ang_deg)))
        mids = []
        for i in range(len(pts)):
            dots = (u @ u[i])
            rad_ok = (np.abs(r - r[i]) / np.maximum(r, r[i]) < float(tol_rel_r))
            ang_ok = (dots < cos_thr)
            idx = np.where(rad_ok & ang_ok)[0]
            if idx.size == 0: continue
            j = idx[np.argmin(np.abs(dots[idx] + 1.0))]
            yi, xi = pts[i,0], pts[i,1]
            yj, xj = pts[j,0], pts[j,1]
            mids.append(((yi+yj)/2.0, (xi+xj)/2.0))
        if len(mids) < 4: break
        mids = np.array(mids, dtype=float)
        cy = float(np.median(mids[:,0])); cx = float(np.median(mids[:,1]))
    return CenterResult(cy=cy, cx=cx, method="antipodal-refined")

# -------------------------- GUI --------------------------
class SAEDApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SAED Symmetry – Launcher")
        self.geometry("980x680")
        self.resizable(True, False)
        self._build_ui()

    def _build_ui(self):
        pad = {"padx":8, "pady":6}
        frm = ttk.Frame(self); frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Изображение:").grid(row=0, column=0, sticky="w", **pad)
        self.ent_img = ttk.Entry(frm, width=68); self.ent_img.grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(frm, text="Обзор…", command=self._browse_img).grid(row=0, column=2, **pad)

        ttk.Label(frm, text="Папка вывода:").grid(row=1, column=0, sticky="w", **pad)
        self.ent_out = ttk.Entry(frm, width=68); self.ent_out.insert(0, "saed_results")
        self.ent_out.grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(frm, text="Выбрать…", command=self._browse_out).grid(row=1, column=2, **pad)

        # Центр (вручную — опционально)
        ttk.Label(frm, text="Центр X (пусто = авто)").grid(row=2, column=0, sticky="w", **pad)
        self.ent_cx = ttk.Entry(frm, width=12); self.ent_cx.grid(row=2, column=1, sticky="w", **pad)
        ttk.Label(frm, text="Центр Y").grid(row=2, column=2, sticky="e", **pad)
        self.ent_cy = ttk.Entry(frm, width=12); self.ent_cy.grid(row=2, column=3, sticky="w", **pad)

        row = 3
        # ---- предобработка ----
        ttk.Label(frm, text="Предобработка:").grid(row=row, column=0, sticky="w", **pad)
        self.cmb_pre = ttk.Combobox(frm, values=["Стандартная", "Без сглаживания", "CLAHE"], state="readonly", width=20)
        self.cmb_pre.current(0)
        self.cmb_pre.grid(row=row, column=1, sticky="w", **pad)
        self.cmb_pre.bind("<<ComboboxSelected>>", self._on_preproc_change)
        ttk.Label(frm, text="CLAHE clipLimit / tile:").grid(row=row, column=2, sticky="e", **pad)
        self.ent_clip = ttk.Entry(frm, width=8); self.ent_clip.insert(0, "1.5"); self.ent_clip.grid(row=row, column=3, sticky="w", **pad)
        self.ent_tile = ttk.Entry(frm, width=8); self.ent_tile.insert(0, "8");  self.ent_tile.grid(row=row, column=4, **pad)
        row += 1

        # ---- параметры детектора/уточнения ----
        self.ent_perc   = self._param(frm, row,   "Процентиль детекции (perc)",  "99.0"); row+=1
        self.ent_win    = self._param(frm, row,   "Окно лок. максимума (odd)",   "7");    row+=1
        self.ent_minsep = self._param(frm, row,   "Мин. шаг между пиками (px)",  "5");    row+=1
        self.ent_maxpts = self._param(frm, row,   "Макс. число пиков",           "6000"); row+=1

        self.ent_iters  = self._param(frm, row,   "Итерации уточнения центра",   "4");    row+=1
        self.ent_tolang = self._param(frm, row,   "Допуск антиподов (°)",        "8.0");  row+=1
        self.ent_tolr   = self._param(frm, row,   "Допуск по радиусу (отн.)",    "0.06"); row+=1

        self.ent_dead   = self._param(frm, row,   "Мёртвая зона (px)",           "0");    row+=1
        self.ent_search = self._param(frm, row,   "Радиус поиска (px, 0=без лимита)", "0"); row+=1

        ttk.Button(frm, text="Перейти в редактор точек", command=self._go_editor).grid(row=row, column=0, columnspan=5, sticky="we", **pad)

        for c in range(5): frm.grid_columnconfigure(c, weight=1)
        self._on_preproc_change(None)

    def _on_preproc_change(self, _evt):
        mode = self.cmb_pre.get()
        clahe_enabled = (mode == "CLAHE")
        state = "normal" if clahe_enabled else "disabled"
        self.ent_clip.configure(state=state)
        self.ent_tile.configure(state=state)

    def _param(self, parent, row, label, default):
        pad = {"padx":8, "pady":6}
        ttk.Label(parent, text=label+":").grid(row=row, column=0, sticky="w", **pad)
        e = ttk.Entry(parent, width=14); e.insert(0, str(default))
        e.grid(row=row, column=1, sticky="w", **pad)
        return e

    def _browse_img(self):
        p = filedialog.askopenfilename(title="Выберите изображение", filetypes=[("Images","*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),("All","*.*")])
        if p: self.ent_img.delete(0, tk.END); self.ent_img.insert(0, p)

    def _browse_out(self):
        p = filedialog.askdirectory(title="Выберите папку вывода")
        if p: self.ent_out.delete(0, tk.END); self.ent_out.insert(0, p)

    def _go_editor(self):
        try:
            image_path = Path(self.ent_img.get()).expanduser()
            if not image_path.exists():
                messagebox.showerror("Ошибка", "Изображение не найдено"); return
            outdir = Path(self.ent_out.get()).expanduser(); outdir.mkdir(parents=True, exist_ok=True)

            perc     = float(self.ent_perc.get())
            win      = int(self.ent_win.get())
            min_sep  = int(self.ent_minsep.get())
            max_pts  = int(self.ent_maxpts.get())
            iters    = int(self.ent_iters.get())
            tol_ang  = float(self.ent_tolang.get())
            tol_relr = float(self.ent_tolr.get())
            dead_r   = float(self.ent_dead.get())
            search_r = float(self.ent_search.get())

            # --- предобработка ---
            pre_mode = self.cmb_pre.get()
            if pre_mode == "Стандартная":
                mode = "standard"
                arr = load_grayscale_with_preproc(image_path, mode)
            elif pre_mode == "Без сглаживания":
                mode = "raw"
                arr = load_grayscale_with_preproc(image_path, mode)
            else:  # CLAHE
                mode = "clahe"
                clip = float(self.ent_clip.get())
                tiles = int(self.ent_tile.get())
                arr = load_grayscale_with_preproc(image_path, mode, clahe_clip=clip, clahe_tiles=tiles)

            # Центр: ручной приоритет, иначе геометрический
            cx_txt = self.ent_cx.get().strip(); cy_txt = self.ent_cy.get().strip()
            if cx_txt and cy_txt:
                center0 = CenterResult(cy=float(cy_txt), cx=float(cx_txt), method="user")
            else:
                center0 = geometric_midpoint(arr)

            # Детекция пиков и уточнение центра
            pts = detect_spots(arr, perc=perc, win=win, min_sep=min_sep, max_spots=max_pts)
            if len(pts) < 120:
                pts = detect_spots(arr, perc=max(97.5, perc-1.0), win=win, min_sep=min_sep, max_spots=max_pts)

            center = refine_center_antipodal((center0.cy, center0.cx), pts, tol_ang_deg=tol_ang, tol_rel_r=tol_relr, iters=iters)

            # Геометрические фильтры
            if (dead_r > 0 or search_r > 0) and len(pts):
                dy = pts[:,0]-center.cy; dx = pts[:,1]-center.cx; r = np.hypot(dx, dy)
                mask = np.ones(len(pts), dtype=bool)
                if dead_r > 0:   mask &= (r >= dead_r)
                if search_r > 0: mask &= (r <= search_r)
                pts = pts[mask]

            # --- saed_input.json (единый файл для редактора) ---
            # pts: ndarray [y, x, v]; сохраняем v как интенсивность
            points = [{"y": float(y), "x": float(x), "intensity": float(v)} for (y, x, v) in pts.tolist()]
            saed_input = {
                "image": str(image_path),
                "preproc_mode": mode,
                "center": {"x": float(center.cx), "y": float(center.cy), "method": center.method},
                "radii": {"dead": float(dead_r), "search": float(search_r)},
                "points": points
            }
            saed_input_path = image_path.with_name("saed_input.json")
            saed_input_path.write_text(json.dumps(saed_input, ensure_ascii=False, indent=2), encoding="utf-8")

            # запуск редактора с одним входом
            if getattr(sys, "frozen", False):
                editor = Path(sys.executable).with_name("saed_editor.exe")
                cmd = [str(editor), "--input", str(saed_input_path)]
            else:
                editor_py = Path(__file__).with_name("saed_editor.py")
                cmd = [sys.executable, str(editor_py), "--input", str(saed_input_path)]
            subprocess.Popen(cmd, shell=False)

            # лог центра (служебный)
            (outdir/"center_init.json").write_text(json.dumps({
                "initial": {"x": center0.cx, "y": center0.cy, "method": center0.method},
                "refined": {"x": center.cx,  "y": center.cy,  "method": "antipodal-refined"},
                "dead_zone_px": dead_r,
                "search_radius_px": search_r,
                "preproc_mode": mode,
                "image_size": {"H": int(arr.shape[0]), "W": int(arr.shape[1])}
            }, indent=2), encoding="utf-8")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

if __name__ == "__main__":
    SAEDApp().mainloop()
