#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAED Editor + Analysis
----------------------
• Подсказка по средней кнопке мыши (MMB) на точке: показывает x, y, intensity; закрывается по Esc и при любых действиях.

Удалено по ТЗ:
• Перемещение обычных точек.
• Любые упоминания/функции панорамирования или перемещения окна средней кнопкой мыши.

Остальной функционал редактора сохранён.
"""
import sys, json, subprocess
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
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
        return np.array([]), np.array([]), ([], [])
    hist, edges = np.histogram(radii, bins=60)
    centers = (edges[:-1] + edges[1:]) / 2
    pk, _ = find_peaks(hist, prominence=3)
    ring_centers = centers[pk]
    if len(ring_centers) == 0:
        return np.array([]), np.zeros_like(radii, int), (hist, edges)
    labels = np.argmin(np.abs(radii[:, None] - ring_centers[None, :]), axis=1)
    return ring_centers, labels, (hist, edges)

def symmetry_scores(angles, radii, ring_means, top_rings=3):
    out = {}
    if not ring_means:
        return out
    idx = min(top_rings - 1, len(ring_means) - 1)
    maxR = ring_means[idx] * 1.15
    ang_sel = angles[radii <= maxR]
    for k in [4, 6, 8, 10, 12]:
        period = 360.0 / k
        phases = np.deg2rad((ang_sel % period) * k)
        C = np.cos(phases).mean(); S = np.sin(phases).mean()
        out[f"{k}-fold"] = float(np.hypot(C, S))
    return out

class PointEditor(tk.Tk):
    def __init__(self, input_json: str | None = None):
        super().__init__()
        self.title("SAED Editor + Analysis")
        self.geometry("1100x800")

        # данные
        self.points = np.zeros((0, 2), float)   # [y, x]
        self.values = np.zeros((0,), float)     # интенсивности (параллельно points)
        self.rect_start = None
        self.rect_artist = None
        self.overlay = None  # {center:{x,y}, dead_radius, search_radius}
        self.image_path = None
        self.img_arr = None

        # Undo/Redo
        self._undo = []
        self._redo = []
        self._history_cap = 300

        # Перетаскивание центра
        self.center_dragging = False
        self._center_hit_radius = 10.0  # пикселей

        # Масштаб (ползунок)
        self.zoom_val = 0  # 0..100
        self.view_cx = None
        self.view_cy = None

        # Tooltip (подсказка на MMB по точке)
        self._tooltip = None          # matplotlib.text.Annotation
        self._tooltip_idx = None

        self._build_ui()

        # первичная загрузка
        if input_json:
            self._load_input_json(Path(input_json))
        self._ensure_view_center()
        self._redraw()

    # ---------- UI ----------
    def _build_ui(self):
        top = tk.Frame(self); top.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top, text="◀", width=2, height=1, command=self._undo_btn).pack(side=tk.LEFT, padx=(6,2), pady=6)
        tk.Button(top, text="▶", width=2, height=1, command=self._redo_btn).pack(side=tk.LEFT, padx=(2,10), pady=6)

        self.zoom_scale = tk.Scale(
            top, from_=0, to=100, orient=tk.HORIZONTAL, length=240,
            label="Масштаб (0=весь кадр, 100=50×50)", command=self._on_zoom_change
        )
        self.zoom_scale.set(self.zoom_val)
        self.zoom_scale.pack(side=tk.LEFT, padx=(0,10), pady=6)

        tk.Button(top, text="Открыть JSON…", command=self._open_json).pack(side=tk.LEFT, padx=6, pady=6)
        tk.Button(top, text="Сохранить", command=self._save_points).pack(side=tk.LEFT, padx=6, pady=6)
        tk.Button(top, text="Начать анализ", command=self._start_analysis).pack(side=tk.LEFT, padx=10, pady=6)

        tk.Label(
            top,
            text="ЛКМ по центру: тянуть центр • Отпускание ЛКМ: кадр центрируется • ЛКМ по пустому месту: добавить точку • ПКМ по точке: удалить • СКМ по точке: подсказка • Shift+Drag: прямоугольник-удаление"
        ).pack(side=tk.LEFT, padx=12)

        self.fig = plt.Figure(figsize=(9.4, 6.4)); self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_down)
        self.canvas.mpl_connect("button_release_event", self._on_up)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("key_press_event", self._on_key)

    # ---------- IO ----------
    def _open_json(self):
        p = filedialog.askopenfilename(filetypes=[("SAED Input JSON","*saed_input.json;*.json"), ("All","*.*")])
        if p:
            self._push_undo()
            self._load_input_json(Path(p))
            self._clear_tooltip()
            self.view_cx = self.view_cy = None
            self._ensure_view_center()
            self._redraw()

    def _load_input_json(self, path: Path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать JSON:\n{e}")
            return

        # загрузка изображения по пути из JSON
        img_path = data.get("image")
        if not img_path:
            messagebox.showerror("Ошибка", "В JSON отсутствует поле 'image'.")
            return
        self.image_path = Path(img_path)
        img = Image.open(self.image_path).convert("L")
        self.img_arr = np.array(img, float)

        # overlay: центр и радиусы
        c = data.get("center") or {}
        r = data.get("radii") or {}
        self.overlay = {
            "center": {"x": float(c.get("x", (self.img_arr.shape[1]-1)/2.0)),
                       "y": float(c.get("y", (self.img_arr.shape[0]-1)/2.0))},
            "dead_radius": float(r.get("dead") or 0.0),
            "search_radius": float(r.get("search") or 0.0),
        }

        # точки: y,x,intensity (если интенсивности нет, рассчитать)
        pts = data.get("points", [])
        if pts:
            yy = [float(p.get("y")) for p in pts]
            xx = [float(p.get("x")) for p in pts]
            self.points = np.column_stack([yy, xx]).astype(float)
            if any("intensity" in p for p in pts):
                vv = [float(p.get("intensity", 0.0)) for p in pts]
                self.values = np.array(vv, float)
            else:
                self.values = self._sample_intensities(self.points)
        else:
            self.points = np.zeros((0,2), float)
            self.values = np.zeros((0,), float)

    def _save_points(self) -> Path:
        """
        Сохраняет точки (включая интенсивности) рядом с исходным input.json:
        - spots.json  — список точек (y,x,intensity)
        - saed_input.edited.json — исходный входной JSON с обновлёнными точками
        """
        base = self.image_path.with_name("spots.json") if self.image_path else Path("spots.json")
        pts = []
        # пересэмплируем интенсивности в текущих координатах
        vals = self._sample_intensities(self.points) if self.img_arr is not None else self.values
        for (y, x), v in zip(self.points, vals):
            pts.append({"y": float(y), "x": float(x), "intensity": float(v)})
        base.write_text(json.dumps({"points": pts}, indent=2), encoding="utf-8")

        # также пересохраним обновлённый saed_input
        si = {
            "image": str(self.image_path) if self.image_path else None,
            "center": (self.overlay.get("center") if self.overlay else None),
            "radii": {
                "dead": float(self.overlay.get("dead_radius") or 0.0) if self.overlay else 0.0,
                "search": float(self.overlay.get("search_radius") or 0.0) if self.overlay else 0.0
            },
            "points": pts
        }
        edited = self.image_path.with_name("saed_input.edited.json") if self.image_path else Path("saed_input.edited.json")
        edited.write_text(json.dumps(si, ensure_ascii=False, indent=2), encoding="utf-8")
        return base

    # ---------- Helpers ----------
    def _sample_intensities(self, pts_yx: np.ndarray) -> np.ndarray:
        if self.img_arr is None or len(pts_yx) == 0:
            return np.zeros((len(pts_yx),), float)
        H, W = self.img_arr.shape[:2]
        out = []
        for y, x in pts_yx:
            yi = int(round(y)); xi = int(round(x))
            yi = max(0, min(H-1, yi)); xi = max(0, min(W-1, xi))
            out.append(float(self.img_arr[yi, xi]))
        return np.array(out, float)

    def _img_xy(self, e):
        return None if (e.xdata is None or e.ydata is None) else (e.ydata, e.xdata)

    def _near_idx(self, y, x, pix_tol=8):
        if len(self.points) == 0: return None
        d2 = (self.points[:,0]-y)**2 + (self.points[:,1]-x)**2
        i = int(np.argmin(d2))
        return i if d2[i]**0.5 <= pix_tol else None

    def _center_hit(self, y, x):
        if not (self.overlay and self.overlay.get("center")): return False
        cy = float(self.overlay["center"].get("y", 0.0))
        cx = float(self.overlay["center"].get("x", 0.0))
        return ((y - cy)**2 + (x - cx)**2) ** 0.5 <= self._center_hit_radius

    def _apply_center_filters(self):
        """Удаляет точки, оказавшиеся в мёртвой зоне или вне радиуса поиска после перемещения центра."""
        if not (self.overlay and self.overlay.get("center")): return
        cy = float(self.overlay["center"].get("y", 0.0))
        cx = float(self.overlay["center"].get("x", 0.0))
        dead = float(self.overlay.get("dead_radius") or 0.0)
        sr   = float(self.overlay.get("search_radius") or 0.0)
        if len(self.points) == 0 or (dead <= 0 and sr <= 0): return
        r = np.hypot(self.points[:,1]-cx, self.points[:,0]-cy)
        mask = np.ones(len(self.points), dtype=bool)
        if dead > 0: mask &= (r >= dead)
        if sr   > 0: mask &= (r <= sr)
        self.points = self.points[mask]
        self.values = self.values[mask] if len(self.values)==len(mask) else self._sample_intensities(self.points)

    # ---------- Tooltip ----------
    def _clear_tooltip(self):
        if self._tooltip is not None:
            try:
                self._tooltip.remove()
            except Exception:
                pass
            self._tooltip = None
            self._tooltip_idx = None
            self.canvas.draw_idle()

    def _show_tooltip_for_idx(self, idx):
        if idx is None or idx < 0 or idx >= len(self.points):
            return
        y, x = self.points[idx]
        # intensity по текущему изображению/values
        if self.img_arr is not None:
            H, W = self.img_arr.shape[:2]
            yi = max(0, min(H-1, int(round(y))))
            xi = max(0, min(W-1, int(round(x))))
            inten = float(self.img_arr[yi, xi])
        else:
            inten = float(self.values[idx]) if idx < len(self.values) else 0.0

        # удалить предыдущую подсказку
        self._clear_tooltip()

        # создать аннотацию возле точки
        txt = f"x={x:.1f}, y={y:.1f}, I={inten:.1f}"
        self._tooltip = self.ax.annotate(
            txt, xy=(x, y), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9),
            fontsize=9
        )
        self._tooltip_idx = idx
        self.canvas.draw_idle()

    # ---------- Undo/Redo ----------
    def _make_snapshot(self):
        center = None
        if self.overlay and isinstance(self.overlay, dict) and self.overlay.get("center"):
            c = self.overlay["center"]
            if c is not None and "x" in c and "y" in c:
                center = {"x": float(c["x"]), "y": float(c["y"])}
        return {
            "points": self.points.copy(),
            "values": self.values.copy(),
            "center": center,
            "view_cx": self.view_cx, "view_cy": self.view_cy
        }

    def _push_undo(self):
        self._undo.append(self._make_snapshot())
        if len(self._undo) > self._history_cap:
            self._undo.pop(0)
        self._redo.clear()

    def _apply_snapshot(self, snap):
        self.center_dragging = False
        if self.rect_artist is not None:
            try: self.rect_artist.remove()
            except Exception: pass
            self.rect_artist = None
        self.rect_start = None

        self.points = snap["points"].copy()
        self.values = snap["values"].copy()
        if snap["center"] is None:
            if self.overlay and "center" in self.overlay: self.overlay.pop("center")
        else:
            if self.overlay is None: self.overlay = {}
            self.overlay["center"] = {"x": float(snap["center"]["x"]), "y": float(snap["center"]["y"])}
        self.view_cx = snap.get("view_cx", self.view_cx)
        self.view_cy = snap.get("view_cy", self.view_cy)
        self._ensure_view_center()

    def _undo_btn(self):
        self._clear_tooltip()
        if not self._undo: return
        self._redo.append(self._make_snapshot())
        if len(self._redo) > self._history_cap: self._redo.pop(0)
        snap = self._undo.pop(-1)
        self._apply_snapshot(snap)
        self._redraw()

    def _redo_btn(self):
        self._clear_tooltip()
        if not self._redo: return
        self._undo.append(self._make_snapshot())
        if len(self._redo) > self._history_cap: self._undo.pop(0)
        snap = self._redo.pop(-1)
        self._apply_snapshot(snap)
        self._redraw()

    # ---------- View-center helpers ----------
    def _ensure_view_center(self):
        if self.img_arr is None:
            return
        H, W = self.img_arr.shape[:2]
        if self.overlay and self.overlay.get("center"):
            cx = float(self.overlay["center"].get("x", (W-1)/2.0))
            cy = float(self.overlay["center"].get("y", (H-1)/2.0))
        else:
            cx = (W-1)/2.0; cy = (H-1)/2.0
        if self.view_cx is None: self.view_cx = cx
        if self.view_cy is None: self.view_cy = cy

    # ---------- Zoom ----------
    def _apply_zoom(self):
        if self.img_arr is None:
            return
        H, W = self.img_arr.shape[:2]
        self._ensure_view_center()

        if self.zoom_val <= 0:
            self.ax.set_xlim(-0.5, W - 0.5)
            self.ax.set_ylim(H - 0.5, -0.5)
            return

        min_dim = min(H, W)
        L = int(round(min_dim - (min_dim - 50) * (self.zoom_val / 100.0)))
        L = max(50, min_dim if L < 50 else L)
        half = L / 2.0

        cx = float(self.view_cx); cy = float(self.view_cy)
        x0 = max(-0.5, cx - half); x1 = min(W - 0.5, cx + half)
        y0 = max(-0.5, cy - half); y1 = min(H - 0.5, cy + half)

        # если окно "урезано" границами — подправим
        if (x1 - x0) < L:
            if x0 <= -0.5: x1 = x0 + L
            elif x1 >= (W - 0.5): x0 = x1 - L
        if (y1 - y0) < L:
            if y0 <= -0.5: y1 = y0 + L
            elif y1 >= (H - 0.5): y0 = y1 - L

        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y1, y0)

    def _on_zoom_change(self, val):
        try: self.zoom_val = int(float(val))
        except Exception: self.zoom_val = 0
        self._clear_tooltip()
        self._redraw()

    # ---------- Mouse / Keyboard events ----------
    def _on_key(self, e):
        # закрыть tooltip по Esc
        if e.key == 'escape':
            self._clear_tooltip()

    def _on_down(self, e):
        pos = self._img_xy(e)
        # любое действие закрывает tooltip
        self._clear_tooltip()

        # Средняя кнопка: только подсказка, если попали в точку
        if e.button == 2:
            if pos is None:
                return
            y, x = pos
            idx = self._near_idx(y, x, pix_tol=8)
            if idx is not None:
                self._show_tooltip_for_idx(idx)
            return  # не считаем это добавлением/редактированием

        if e.button == 1 and not (e.key and "shift" in e.key):
            if pos is None: return
            y, x = pos
            # Перетаскивание центра — оставить
            if self._center_hit(y, x):
                self._push_undo()
                self.center_dragging = True
                self._redo.clear()
                return

            # Добавление новой точки по клику в пустое место (без последующего перетаскивания!)
            i = self._near_idx(y, x)
            if i is None:
                self._push_undo()
                self.points = np.vstack([self.points, [y, x]])
                self.values = np.append(self.values, self._sample_intensities(np.array([[y, x]]))[0])
                self._redo.clear()
            # если кликнули по существующей точке — ничего не делаем (перемещение запрещено)

        elif e.button == 3:
            if pos is None: return
            y, x = pos
            i = self._near_idx(y, x)
            if i is not None:
                self._push_undo()
                self.points = np.delete(self.points, i, axis=0)
                self.values = np.delete(self.values, i, axis=0)
                self._redo.clear()

        elif e.button == 1 and e.key and "shift" in e.key:
            self._push_undo()
            self.rect_start = pos
            self._redo.clear()

        self._redraw()

    def _on_move(self, e):
        pos = self._img_xy(e)
        # любое движение закрывает tooltip
        self._clear_tooltip()

        # Перетаскивание центра — оставить
        if self.center_dragging and pos is not None:
            y, x = pos
            if self.overlay is None: self.overlay = {}
            self.overlay["center"] = {"x": float(x), "y": float(y)}
            self._redraw()
            return

        # Прямоугольник выделения для удаления
        if self.rect_start and e.xdata and e.ydata:
            y0, x0 = self.rect_start; y1, x1 = e.ydata, e.xdata
            self._redraw()
            self.rect_artist = self.ax.add_patch(
                plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", ls="--", lw=1.5)
            )
            self.canvas.draw_idle()
            return

    def _on_up(self, e):
        # любое действие закрывает tooltip
        self._clear_tooltip()

        # Завершение перетаскивания центра
        if self.center_dragging:
            self.center_dragging = False
            self._apply_center_filters()
            if self.overlay and self.overlay.get("center"):
                self.view_cx = float(self.overlay["center"]["x"])
                self.view_cy = float(self.overlay["center"]["y"])
            self._redraw()
            return

        # Завершение прямоугольника удаления
        if self.rect_start:
            y0, x0 = self.rect_start; y1, x1 = e.ydata, e.xdata
            if y1 is not None and x1 is not None:
                ymin, ymax = sorted([y0, y1]); xmin, xmax = sorted([x0, x1])
                mask = ~((self.points[:,0] >= ymin) & (self.points[:,0] <= ymax) &
                         (self.points[:,1] >= xmin) & (self.points[:,1] <= xmax))
                self.points = self.points[mask]
                self.values = self.values[mask]
            self.rect_start = None
            if self.rect_artist is not None:
                self.rect_artist.remove(); self.rect_artist = None
            self._redraw()

    # ---------- Draw ----------
    def _redraw(self):
        self.ax.clear()
        if self.img_arr is not None:
            self.ax.imshow(self.img_arr, cmap="gray", interpolation="nearest")
        self.ax.axis("off")

        if self.overlay and self.overlay.get("center"):
            cy = float(self.overlay["center"].get("y", 0))
            cx = float(self.overlay["center"].get("x", 0))
            self.ax.scatter([cx], [cy], s=36, c="red", marker="o")
            dead = float(self.overlay.get("dead_radius") or 0)
            sr   = float(self.overlay.get("search_radius") or 0)
            for R in [dead, sr]:
                if R > 0:
                    self.ax.add_patch(Circle((cx, cy), R, fill=False, ls="--", lw=2.0, ec="red"))

        if len(self.points):
            self.ax.scatter(self.points[:,1], self.points[:,0],
                            s=22, alpha=0.9, marker="o", linewidths=0.5, edgecolors="black")

        self._apply_zoom()
        self.canvas.draw_idle()

    # ---------- Анализ ----------
    def _start_analysis(self):
        saved = self._save_points()  # сохраняет текущие точки (и интенсивности) и возвращает путь к spots.json

        # payload для fibonachi_analysis
        payload_path = None
        try:
            if self.img_arr is not None:
                geo_cy = (self.img_arr.shape[0] - 1) / 2.0
                geo_cx = (self.img_arr.shape[1] - 1) / 2.0
            else:
                geo_cy = geo_cx = None

            # актуальные точки (после правок) + интенсивности из текущего изображения
            points_list = []
            if len(self.points):
                if self.img_arr is not None:
                    H, W = self.img_arr.shape[:2]
                else:
                    H = W = None
                for (y, x) in self.points.tolist():
                    if self.img_arr is not None:
                        yi = max(0, min(H - 1, int(round(y))))
                        xi = max(0, min(W - 1, int(round(x))))
                        inten = float(self.img_arr[yi, xi])
                    else:
                        inten = None
                    points_list.append({"x": float(x), "y": float(y), "intensity": inten})

            # актуальный центр из overlay
            overlay_center = None
            if self.overlay and isinstance(self.overlay, dict) and self.overlay.get("center") is not None:
                c = self.overlay["center"]
                overlay_center = {"x": float(c.get("x")), "y": float(c.get("y"))}

            # радиусы из overlay
            dead_val = None
            search_val = None
            if self.overlay:
                if self.overlay.get("dead_radius") is not None:
                    dead_val = float(self.overlay.get("dead_radius"))
                if self.overlay.get("search_radius") is not None:
                    search_val = float(self.overlay.get("search_radius"))

            payload = {
                "image": str(self.image_path) if self.image_path else None,
                "points": points_list,
                "centers": {
                    "geometric": {"x": float(geo_cx), "y": float(geo_cy)} if geo_cx is not None else None,
                    "overlay": overlay_center
                },
                "radii": {
                    "dead": dead_val,
                    "search": search_val
                },
                "spots_json": str(saved) if saved else None
            }

            payload_path = (self.image_path.with_name("fibo_input.json") if self.image_path
                            else Path("fibo_input.json"))
            payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        except Exception as e:
            messagebox.showerror("Ошибка подготовки данных",
                                 f"Не удалось подготовить данные для fibonachi_analysis:\n{e}")

        # запуск fibonachi_analysis.exe (совместимость со старыми ключами сохранена)
        try:
            if getattr(sys, "frozen", False):
                fibexe = Path(sys.executable).with_name("fibonachi_analysis.exe")
            else:
                fibexe = Path(__file__).with_name("fibonachi_analysis.exe")

            cmd = [str(fibexe)]
            if payload_path is not None:
                cmd += ["--payload", str(payload_path)]
            if self.image_path is not None:
                cmd += ["--image", str(self.image_path)]
            if saved is not None:
                cmd += ["--points", str(saved)]
            subprocess.Popen(cmd, shell=False)
        except Exception as e:
            messagebox.showerror("Ошибка запуска",
                                 f"Не удалось запустить fibonachi_analysis.exe:\n{e}")

        # быстрый отчёт (как раньше)
        if self.img_arr is None or len(self.points) == 0:
            messagebox.showinfo("Анализ", "Нет изображения или точек для анализа.")
            return

        cy, cx = (self.img_arr.shape[0] - 1) / 2.0, (self.img_arr.shape[1] - 1) / 2.0
        radii, angles = pol_from((cy, cx), self.points)
        rc, labels, _ = cluster_rings(radii)
        ring_means = [np.mean(radii[labels == i]) for i in np.unique(labels)] if len(rc) else []
        sym = symmetry_scores(angles, radii, ring_means)

        lines = ["SAED Symmetry Analysis", "=======================", "",
                 f"Файл: {self.image_path}",
                 f"Сохранено: {saved.name if saved else '-'}",
                 f"Точек: {len(self.points)}", "", "Симметрии:"]
        for k, v in sorted(sym.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k:>7}: {v:.3f}")
        self._show_report("\n".join(lines))

    def _show_report(self, text: str):
        win = tk.Toplevel(self)
        win.title("SAED Report")
        txt = tk.Text(win, wrap="word")
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", text)
        txt.config(state=tk.DISABLED)

# -------- CLI ---------
def _parse_args(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=False, help="Путь к saed_input.json")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    app = PointEditor(args.input)
    app.mainloop()
