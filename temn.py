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
class SAEDLauncherFrame(ttk.Frame):
    """Вкладка-лаунчер, пригодная как для standalone-приложения, так и для notebook."""

    def __init__(self, master: tk.Misc, controller=None):
        super().__init__(master)
        self.controller = controller
        self._scroll_canvas = None
        self._scroll_window_id = None
        self._build_ui()

    def _build_ui(self):
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)

        fixed = ttk.Frame(outer, padding=(16, 16, 16, 0))
        fixed.pack(side=tk.TOP, fill=tk.X)
        fixed.grid_columnconfigure(0, weight=1)

        data_box = ttk.LabelFrame(fixed, text="Исходные данные", padding=(12, 10, 12, 12))
        data_box.grid(row=0, column=0, sticky="nsew")
        for col in (0, 1, 2, 3):
            weight = 1 if col == 1 else 0
            data_box.grid_columnconfigure(col, weight=weight)

        ttk.Label(data_box, text="Изображение:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.ent_img = ttk.Entry(data_box)
        self.ent_img.grid(row=0, column=1, columnspan=2, sticky="we", padx=6, pady=4)
        ttk.Button(data_box, text="Обзор…", command=self._browse_img).grid(row=0, column=3, sticky="ew", padx=6, pady=4)

        ttk.Label(data_box, text="Папка вывода:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.ent_out = ttk.Entry(data_box)
        self.ent_out.insert(0, "saed_results")
        self.ent_out.grid(row=1, column=1, columnspan=2, sticky="we", padx=6, pady=4)
        ttk.Button(data_box, text="Выбрать…", command=self._browse_out).grid(row=1, column=3, sticky="ew", padx=6, pady=4)

        ttk.Label(data_box, text="Центр X (опционально):").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.ent_cx = ttk.Entry(data_box, width=12)
        self.ent_cx.grid(row=2, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(data_box, text="Центр Y:").grid(row=2, column=2, sticky="w", padx=6, pady=4)
        self.ent_cy = ttk.Entry(data_box, width=12)
        self.ent_cy.grid(row=2, column=3, sticky="w", padx=6, pady=4)

        ttk.Label(
            data_box,
            text="Если координаты оставить пустыми, программа найдёт центр автоматически.",
            wraplength=520,
            foreground="#555555"
        ).grid(row=3, column=0, columnspan=4, sticky="we", padx=6, pady=(0, 4))

        pre_box = ttk.LabelFrame(fixed, text="Предобработка", padding=(12, 10, 12, 12))
        pre_box.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        pre_box.grid_columnconfigure(1, weight=1)

        ttk.Label(pre_box, text="Режим:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.cmb_pre = ttk.Combobox(
            pre_box,
            values=["Стандартная", "Без сглаживания", "CLAHE"],
            state="readonly",
        )
        self.cmb_pre.current(0)
        self.cmb_pre.grid(row=0, column=1, sticky="w", padx=6, pady=4)
        self.cmb_pre.bind("<<ComboboxSelected>>", self._on_preproc_change)

        ttk.Label(pre_box, text="CLAHE clipLimit / tile:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.spn_clip = ttk.Spinbox(pre_box, from_=0.1, to=10.0, increment=0.1, width=8, justify="right")
        self._set_spinbox_value(self.spn_clip, 1.5)
        self.spn_clip.grid(row=1, column=1, sticky="w", padx=6, pady=4)
        self.spn_tile = ttk.Spinbox(pre_box, from_=2, to=64, increment=1, width=8, justify="right")
        self._set_spinbox_value(self.spn_tile, 8)
        self.spn_tile.grid(row=1, column=2, sticky="w", padx=6, pady=4)

        ttk.Label(
            pre_box,
            text="Выберите CLAHE для снимков с сильными перепадами яркости. ClipLimit управляет контрастом, размер тайла — локаль"
                 "ным радиусом обработки.",
            wraplength=520,
            foreground="#555555"
        ).grid(row=2, column=0, columnspan=3, sticky="we", padx=6, pady=(2, 0))

        scroll_host = ttk.Frame(outer)
        scroll_host.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(scroll_host, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=canvas.yview)
        scrollable = ttk.Frame(canvas, padding=(16, 12, 16, 12))
        scrollable.grid_columnconfigure(0, weight=1)

        self._scroll_canvas = canvas
        self._scroll_window_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)

        scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(self._scroll_window_id, width=e.width)
        )

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        scrollable.bind("<Enter>", self._activate_scroll)
        scrollable.bind("<Leave>", self._deactivate_scroll)
        canvas.bind("<Enter>", self._activate_scroll)
        canvas.bind("<Leave>", self._deactivate_scroll)

        detect_box = ttk.LabelFrame(scrollable, text="Детектор и уточнение", padding=(12, 10, 12, 12))
        detect_box.grid(row=0, column=0, sticky="nsew")
        detect_box.grid_columnconfigure(1, weight=1)

        ttk.Label(
            detect_box,
            text="Порог и окно поиска пиков",
            font=("TkDefaultFont", 10, "bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))
        self.spn_perc = self._spin_param(
            detect_box, 1, "Процентиль детекции (%)", 99.0,
            from_=80.0, to=100.0, increment=0.1, format_str="%.1f"
        )
        self.spn_win = self._spin_param(
            detect_box, 2, "Размер окна локального максимума", 7,
            from_=3, to=21, increment=2
        )
        self.spn_minsep = self._spin_param(
            detect_box, 3, "Мин. расстояние между пиками (px)", 5,
            from_=1, to=50, increment=1
        )
        self.spn_maxpts = self._spin_param(
            detect_box, 4, "Максимум обнаруженных точек", 6000,
            from_=100, to=20000, increment=100
        )

        ttk.Separator(detect_box).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 8))

        ttk.Label(
            detect_box,
            text="Уточнение центра",
            font=("TkDefaultFont", 10, "bold")
        ).grid(row=6, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))
        self.spn_iters = self._spin_param(
            detect_box, 7, "Итерации уточнения центра", 4,
            from_=0, to=10, increment=1
        )
        self.spn_tolang = self._spin_param(
            detect_box, 8, "Допуск антиподов (°)", 8.0,
            from_=1.0, to=30.0, increment=0.5, format_str="%.1f"
        )
        self.spn_tolr = self._spin_param(
            detect_box, 9, "Допуск по радиусу (отн.)", 0.06,
            from_=0.01, to=0.5, increment=0.01, format_str="%.2f"
        )

        ttk.Separator(detect_box).grid(row=10, column=0, columnspan=2, sticky="ew", pady=(6, 8))

        ttk.Label(
            detect_box,
            text="Геометрические фильтры",
            font=("TkDefaultFont", 10, "bold")
        ).grid(row=11, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))
        self.spn_dead = self._spin_param(
            detect_box, 12, "Мёртвая зона (px)", 0,
            from_=0, to=500, increment=1
        )
        self.spn_search = self._spin_param(
            detect_box, 13, "Радиус поиска (px, 0 = без лимита)", 0,
            from_=0, to=10000, increment=25
        )

        ttk.Label(
            detect_box,
            text="Используйте мёртвую зону, чтобы отфильтровать пересвеченный центр. Радиус поиска ограничивает внешние кольца"
                 " и ускоряет обработку.",
            wraplength=520,
            foreground="#555555"
        ).grid(row=14, column=0, columnspan=2, sticky="we", padx=6, pady=(2, 0))

        action_box = ttk.Frame(scrollable, padding=(0, 12, 0, 0))
        action_box.grid(row=1, column=0, sticky="nsew")
        action_box.grid_columnconfigure(0, weight=1)

        ttk.Label(
            action_box,
            text="Проверьте параметры и нажмите кнопку ниже, чтобы перейти к интерактивному редактированию найденных точек.",
            wraplength=540,
            justify="left"
        ).grid(row=0, column=0, sticky="we", padx=4, pady=(0, 8))

        ttk.Button(action_box, text="Перейти в редактор точек", command=self._go_editor).grid(
            row=1, column=0, sticky="ew", padx=4, pady=(0, 12)
        )

        filler_bg = ttk.Style().lookup("TFrame", "background") or self.winfo_toplevel().cget("background")
        bottom_filler = tk.Frame(scrollable, height=56, bg=filler_bg)
        bottom_filler.grid(row=2, column=0, sticky="ew")
        bottom_filler.grid_propagate(False)

        self._on_preproc_change(None)

    def _activate_scroll(self, _event):
        if self._scroll_canvas is None:
            return
        self._scroll_canvas.bind_all("<MouseWheel>", self._on_scroll_mousewheel)
        self._scroll_canvas.bind_all("<Button-4>", self._on_scroll_mousewheel)
        self._scroll_canvas.bind_all("<Button-5>", self._on_scroll_mousewheel)

    def _deactivate_scroll(self, _event):
        if self._scroll_canvas is None:
            return
        self._scroll_canvas.unbind_all("<MouseWheel>")
        self._scroll_canvas.unbind_all("<Button-4>")
        self._scroll_canvas.unbind_all("<Button-5>")

    def _on_scroll_mousewheel(self, event):
        if self._scroll_canvas is None:
            return
        if getattr(event, "delta", 0):
            self._scroll_canvas.yview_scroll(int(-event.delta / 120), "units")
        elif getattr(event, "num", None) == 4:
            self._scroll_canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self._scroll_canvas.yview_scroll(1, "units")

    def _on_preproc_change(self, _evt):
        mode = self.cmb_pre.get()
        clahe_enabled = (mode == "CLAHE")
        state = "normal" if clahe_enabled else "disabled"
        self.spn_clip.configure(state=state)
        self.spn_tile.configure(state=state)

    def _spin_param(self, parent, row, label, default, *, from_, to, increment, format_str=None):
        ttk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        spin = ttk.Spinbox(parent, from_=from_, to=to, increment=increment, width=10, justify="right")
        if format_str:
            spin.configure(format=format_str)
        self._set_spinbox_value(spin, default)
        spin.grid(row=row, column=1, sticky="w", padx=6, pady=4)
        return spin

    def _set_spinbox_value(self, spinbox: ttk.Spinbox, value):
        try:
            spinbox.set(value)
        except tk.TclError:
            spinbox.delete(0, tk.END)
            spinbox.insert(0, str(value))

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

            perc = float(self.spn_perc.get())
            win = int(float(self.spn_win.get()))
            min_sep = int(float(self.spn_minsep.get()))
            max_pts = int(float(self.spn_maxpts.get()))
            iters = int(float(self.spn_iters.get()))
            tol_ang = float(self.spn_tolang.get())
            tol_relr = float(self.spn_tolr.get())
            dead_r = float(self.spn_dead.get())
            search_r = float(self.spn_search.get())

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
                clip = float(self.spn_clip.get())
                tiles = int(float(self.spn_tile.get()))
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
            if self.controller is not None:
                try:
                    self.controller.open_editor(saed_input_path)
                except Exception as exc:
                    messagebox.showerror("Ошибка", f"Не удалось открыть редактор: {exc}")
            else:
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

        class SAEDApp(tk.Tk):

            """Backwards-compatible standalone приложение, использующее фрейм вкладки."""

            def __init__(self):
                super().__init__()

                self.title("SAED Symmetry – Launcher")

                self.geometry("980x680")

                self.resizable(True, False)

                frame = SAEDLauncherFrame(self)

                frame.pack(fill=tk.BOTH, expand=True)

        if __name__ == "__main__":
            SAEDApp().mainloop()
