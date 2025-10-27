#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAED Editor (Упрощенный)
-------------------------
Главный класс, использующий архитектуру Model-View-Controller.
Панели/логика для симметрии и группировки удалены.
"""
import sys, json, subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Импорт компонентов ---
from saed_data_model import SaedDataModel
from saed_editor_state_ui import EditorUIState
from saed_editor_drawing_refactored import EditorDrawingView
from saed_editor_handlers_refactored import EditorEventHandlers
from saed_editor_io_refactored import EditorIO
# ---

try:
    from preproc import PreprocSettings, load_grayscale_with_preproc # Добавлен load_grayscale_with_preproc
    from percentile_utils import compute_percentile_map # Добавлен compute_percentile_map
except ImportError:
    messagebox.showerror("Import Error", "Failed to import 'PreprocSettings' or 'percentile_utils'.")
    if 'PreprocSettings' not in globals():
        class PreprocSettings:
            def __init__(self, mode="raw"): self.mode = mode
            @staticmethod
            def from_json(data, fallback_mode=None): return PreprocSettings(fallback_mode or "raw")
            def to_json(self): return {"mode": self.mode}
    # Добавляем заглушки для импортированных функций
    def load_grayscale_with_preproc(path, settings): raise ImportError("preproc.py not found")
    def compute_percentile_map(img): raise ImportError("percentile_utils.py not found")


class PointEditor(tk.Frame):

    def __init__(self, master: tk.Misc, controller=None,
                 input_json: str | None = None, auto_load: bool = True):
        super().__init__(master)
        self.app_controller = controller

        # --- Данные Контроллера ---
        self.image_path: Optional[Path] = None
        self.img_arr_raw: Optional[np.ndarray] = None
        self.img_arr_processed: Optional[np.ndarray] = None
        self._percent_map: Optional[np.ndarray] = None
        self._percent_lookup: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._preproc_settings = PreprocSettings(mode="raw")
        self.overlay: Dict[str, Any] = {}

        # --- Состояние UI Контроллера ---
        self.zoom_val = 0
        self.view_cx = None
        self.view_cy = None
        self.show_raw_background = tk.BooleanVar(value=False)
        self.status_message = ""
        self.default_status = "Mode: point editor"

        # --- Состояние Undo/Redo ---
        self.undo: List[Dict[str, Any]] = []
        self.redo: List[Dict[str, Any]] = []
        self._history_cap = 300

        # --- ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ ---
        self.model = SaedDataModel()
        self.ui_state = EditorUIState()
        self.view: Optional[EditorDrawingView] = None
        self.handlers: Optional[EditorEventHandlers] = None
        self.io: Optional[EditorIO] = None

        # --- Строим UI ---
        self._build_ui()

        # --- Инициализация компонентов, зависящих от UI ---
        if self.ax is None or self.canvas is None:
             raise RuntimeError("UI build failed to create ax or canvas.")
        self.view = EditorDrawingView(self.ax, self.canvas)
        self.io = EditorIO(self) # IO все еще нужен для open/save wrappers
        self.handlers = EditorEventHandlers(self)

        # --- Подключаем обработчики ---
        self.handlers.connect_mpl_events(self.canvas)
        self.zoom_scale.config(command=self.handlers.on_zoom_change_tk)
        self.chk_raw_bg.config(command=self._on_background_toggle)
        self.btn_open.config(command=self.io.open_json_wrapper)
        self.btn_save.config(command=self.io.save_points_wrapper)
        self.btn_analysis.config(command=self.io.start_analysis_wrapper)

        # --- Привязки клавиш ---
        self.bind_all('<Control-z>', self._undo_btn)
        self.bind_all('<Control-y>', self._redo_btn)

        # --- Первичная загрузка ---
        if auto_load and input_json:
            try:
                # Используем io.load_input_json для первоначальной загрузки из файла
                self.io.load_input_json(Path(input_json), push_undo=False)
            except FileNotFoundError:
                self.set_status(f"Error: Input JSON not found at {input_json}")
                self.ensure_view_center()
                self.redraw()
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to auto-load JSON:\n{e}")
        else:
            self.ensure_view_center()
            self.redraw()

        self.update_zoom_hint()
        self.set_status(self.default_status)
        self._toggle_help()

    # ---------- UI Builder (Без изменений) ----------
    def _build_ui(self):
        """Создает Tkinter виджеты и Matplotlib холст."""
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        side_panel = ttk.Frame(self, padding=(16, 16, 12, 16))
        side_panel.grid(row=0, column=0, sticky="ns")
        side_panel.columnconfigure(0, weight=1)

        scroll_host = ttk.Frame(side_panel); scroll_host.pack(fill=tk.BOTH, expand=True)
        scroll_host.rowconfigure(0, weight=1); scroll_host.columnconfigure(0, weight=1)
        canvas = tk.Canvas(scroll_host, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding=(0, 0, 10, 0))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        def _on_left_scroll(event):
             widget_under_cursor = event.widget.winfo_containing(event.x_root, event.y_root)
             if widget_under_cursor is canvas or widget_under_cursor.master is scrollable_frame:
                  canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_left_scroll); scrollable_frame.bind("<MouseWheel>", _on_left_scroll)

        controls = ttk.Frame(scrollable_frame); controls.pack(side=tk.TOP, fill=tk.X)
        header_row = ttk.Frame(controls); header_row.pack(fill=tk.X); header_row.columnconfigure(0, weight=1)
        help_button = ttk.Button(header_row, text="?", width=3, command=self._toggle_help, style="Toolbutton")
        help_button.pack(side=tk.RIGHT)
        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 10))

        zoom_group = ttk.LabelFrame(controls, text="Scale", padding=(12, 8, 12, 10)); zoom_group.pack(fill=tk.X)
        self.zoom_var = tk.DoubleVar(value=self.zoom_val)
        self.zoom_scale = ttk.Scale(zoom_group, from_=0, to=100, variable=self.zoom_var)
        self.zoom_scale.pack(fill=tk.X, padx=4, pady=(0, 6))
        self.zoom_hint = ttk.Label(zoom_group, anchor="w"); self.zoom_hint.pack(fill=tk.X, padx=4)
        self.zoom_scale.set(self.zoom_val)
        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 10))

        file_group = ttk.Frame(controls); file_group.pack(fill=tk.X)
        self.btn_open = ttk.Button(file_group, text="Open JSON…")
        self.btn_open.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_save = ttk.Button(file_group, text="Save")
        self.btn_save.pack(side=tk.LEFT, padx=(0, 6))

        analysis_group = ttk.Frame(controls); analysis_group.pack(fill=tk.X, pady=(8, 0))
        self.btn_analysis = ttk.Button(analysis_group, text="Start Fibonacci analysis")
        self.btn_analysis.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 10))

        display_group = ttk.LabelFrame(controls, text="Display Options", padding=(12, 8, 12, 10))
        display_group.pack(fill=tk.X, pady=(0, 10))
        self.chk_raw_bg = ttk.Checkbutton(display_group, text="Show Raw Image Background", variable=self.show_raw_background)
        self.chk_raw_bg.pack(anchor="w")

        self.help_panel = ttk.LabelFrame(scrollable_frame, text="Hints", padding=(16, 12, 16, 12))
        help_text = (
            "Ctrl+Z / Ctrl+Y — Undo/Redo actions\n"
            "Ctrl+S — Save current session (if in app)\n"
            "Mouse Wheel — Zoom in/out\n\n"
            "LMB on empty — add point\n"
            "LMB on center — drag center\n"
            "RMB on point — delete point\n"
            "MMB on point — show info (Radius, Angle, Area)\n\n"
            "LMB Click (Point A) -> LMB Click (Point B/Center) — measure distance\n\n"
            "Ctrl + LMB Drag — draw selection ring\n"
            "  (+/- keys change thickness)\n"
            "  (LMB Click finishes selection)\n"
            "Shift + LMB (on point) — add point to selection\n"
            "Shift + RMB (on selected point) — remove point from selection\n"
            "Enter (with points selected) — average selected points\n\n"
            "Shift + LMB Drag (no points selected) — rectangular delete"
        )
        ttk.Label(self.help_panel, text=help_text, justify="left", wraplength=280).pack(fill=tk.X)
        self._help_visible = False
        self.help_panel.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))

        status_frame = ttk.Frame(side_panel, padding=(0, 0, 0, 0)); status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))
        self.status_label = ttk.Label(status_frame, anchor="w", justify="left"); self.status_label.pack(fill=tk.X)
        def _status_wrap(event, label=self.status_label):
            if not label.winfo_exists(): return
            label.configure(wraplength=max(int(event.width) - 8, 120))
        self.status_label.bind("<Configure>", _status_wrap)

        canvas_frame = ttk.Frame(self, padding=(0, 16, 16, 16)); canvas_frame.grid(row=0, column=1, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1); canvas_frame.columnconfigure(0, weight=1)
        self.fig = plt.Figure(figsize=(9.4, 6.4)); self.ax = self.fig.add_subplot(111); self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")


    # ---------- *** НОВЫЙ МЕТОД: set_state *** ----------
    def set_state(self, state: dict):
        """Восстанавливает состояние редактора из словаря (сессии)."""
        image_path_str = state.get("image_path")
        if not image_path_str:
            self.clear_all() # Сбрасываем, если нет пути к изображению
            return

        try:
            new_image_path = Path(image_path_str).resolve()
            if not new_image_path.exists():
                raise FileNotFoundError(f"Image not found during session load: {new_image_path}")

            # --- 1. Восстанавливаем атрибуты контроллера ---
            self.image_path = new_image_path
            # Используем PreprocSettings.from_json для безопасного создания настроек
            self._preproc_settings = PreprocSettings.from_json(state.get("preproc_settings", {}))
            self.overlay = state.get("overlay", {})
            self.zoom_val = state.get("zoom_val", 0)
            self.view_cx = state.get("view_cx") # Может быть None
            self.view_cy = state.get("view_cy") # Может быть None
            # Обновляем виджеты UI
            self.zoom_var.set(self.zoom_val)
            self.show_raw_background.set(state.get("show_raw_background", False))

            # --- 2. Загружаем изображения и карты ---
            # Эта логика перенесена сюда из EditorIO.set_state
            try:
                self.img_arr_raw = load_grayscale_with_preproc(self.image_path, PreprocSettings(mode="raw"))
                self.img_arr_processed = load_grayscale_with_preproc(self.image_path, self._preproc_settings)
                # Пересчитываем карты процентилей
                p_map, uniq_vals, uniq_perc = compute_percentile_map(self.img_arr_processed)
                self._percent_map = p_map
                self._percent_lookup = (uniq_vals, uniq_perc)
            except Exception as img_load_err:
                 # Обработка ошибки загрузки/обработки изображения
                 messagebox.showerror("Image Error", f"Failed to load/process image during session load:\n{img_load_err}")
                 # Пытаемся продолжить с пустыми изображениями, если возможно
                 if self.img_arr_raw is None: self.img_arr_raw = np.zeros((100, 100), dtype=np.uint8)
                 if self.img_arr_processed is None: self.img_arr_processed = self.img_arr_raw.copy()
                 self._percent_map = None
                 self._percent_lookup = None


            # --- 3. Восстанавливаем Модель ---
            data_snapshot = state.get("data_snapshot", {})
            # Передаем снэпшот в модель для восстановления
            self.model.apply_snapshot(data_snapshot)

            # --- 4. Восстанавливаем UI State ---
            self.ui_state.measurement = state.get("measurement")
            # Восстанавливаем выделенные индексы (убедимся, что это set)
            saved_indices = state.get("ring_select_indices", [])
            self.ui_state.ring_select_indices = set(map(int, saved_indices)) # Преобразуем в int на всякий случай
            # Сбрасываем активные взаимодействия
            self.ui_state.cancel_all_interactions()
            self.ui_state.center_dragging = False
            self.ui_state.rect_start = None

            # --- 5. Финализация ---
            self.ensure_view_center() # Устанавливаем центр вида
            self.redraw() # Перерисовываем холст
            self.update_zoom_hint() # Обновляем подсказку зума
            self.set_status(f"Restored state for {self.image_path.name}")
            # Очищаем историю undo/redo при загрузке сессии
            self.undo.clear()
            self.redo.clear()

        except FileNotFoundError as e:
            messagebox.showerror("Load Session Error", str(e))
            self.clear_all() # Сбрасываем редактор при ошибке
        except Exception as e:
            messagebox.showerror("Load Session Error", f"Failed to restore editor state:\n{e}")
            import traceback
            traceback.print_exc() # Для отладки
            self.clear_all() # Сбрасываем редактор при ошибке
    # ---------- *** КОНЕЦ НОВОГО МЕТОДА *** ----------


    # ---------- Управление Контроллером (Остальное без изменений) ----------
    def redraw(self):
        if self.view: self.view.redraw(self)

    def set_status(self, text: str):
        self.status_message = text
        if hasattr(self, "status_label") and self.status_label.winfo_exists():
            self.status_label.configure(text=text)
        if self.app_controller is not None and hasattr(self.app_controller, "set_status"):
            try: self.app_controller.set_status(f"Editor: {text}")
            except Exception: pass

    def get_output_dir(self) -> Path:
        if self.app_controller and hasattr(self.app_controller, 'launcher'):
            output_dir_str = self.app_controller.launcher.ent_out.get()
            if output_dir_str:
                try:
                    output_dir = Path(output_dir_str).expanduser().resolve()
                    output_dir.mkdir(parents=True, exist_ok=True)
                    return output_dir
                except Exception as e: raise OSError(f"Invalid output directory '{output_dir_str}': {e}")
            else: raise ValueError("Output folder not specified.")
        else:
            print("Warning: Controller/Launcher not found, using default output 'saed_results'.")
            output_dir = Path("saed_results").resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir

    def clear_all(self):
        self.image_path = None
        self.img_arr_raw = None; self.img_arr_processed = None
        self._percent_map = None; self._percent_lookup = None
        self.overlay = {}; self._preproc_settings = PreprocSettings()
        self.zoom_val = 0; self.view_cx = None; self.view_cy = None
        self.zoom_var.set(0); self.show_raw_background.set(False)
        self.model = SaedDataModel(); self.ui_state = EditorUIState()
        self.undo.clear(); self.redo.clear()
        self.redraw(); self.set_status("Editor cleared.")

    # ---------- Обработчики UI Контроллера (Без изменений) ----------
    def _toggle_help(self):
        self._help_visible = not self._help_visible
        if self._help_visible: self.help_panel.pack(side=tk.TOP, fill=tk.X, pady=(0, 8)); self.set_status("Detailed hints expanded")
        else: self.help_panel.pack_forget(); self.set_status(self.default_status)

    def _on_background_toggle(self):
        self.redraw()
        mode = "Raw" if self.show_raw_background.get() else "Processed"
        self.set_status(f"Background image set to: {mode}")

    # ---------- Управление состоянием (Undo/Redo) (Без изменений) ----------
    def push_undo(self):
        if self._history_cap > 0:
            self.undo.append(self.model.get_snapshot())
            if len(self.undo) > self._history_cap: self.undo.pop(0)
            self.redo.clear()

    def push_redo(self):
        if self._history_cap > 0:
            self.redo.append(self.model.get_snapshot())
            if len(self.redo) > self._history_cap: self.redo.pop(0)

    def pop_undo(self):
        if self.undo: self.undo.pop()

    def _apply_snapshot(self, snap: Dict[str, Any]):
        self.model.apply_snapshot(snap)

    def _undo_btn(self, event=None):
        if hasattr(event, 'widget') and isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)): return
        if not self.undo: self.set_status("Nothing to undo."); return
        self.ui_state.cancel_all_interactions(); self.ui_state.clear_tooltip(self.view)
        self.push_redo(); snap_to_restore = self.undo.pop(); self._apply_snapshot(snap_to_restore)
        self.redraw(); self.set_status("Undo successful.")

    def _redo_btn(self, event=None):
        if hasattr(event, 'widget') and isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)): return
        if not self.redo: self.set_status("Nothing to redo."); return
        self.ui_state.cancel_all_interactions(); self.ui_state.clear_tooltip(self.view)
        self.push_undo(); snap_to_restore = self.redo.pop(); self._apply_snapshot(snap_to_restore)
        self.redraw(); self.set_status("Redo successful.")

    # ---------- Утилиты-аксессоры (Без изменений) ----------
    def get_image_to_display(self) -> Optional[np.ndarray]:
        img = self.img_arr_raw if self.show_raw_background.get() else self.img_arr_processed
        return img if img is not None else (self.img_arr_processed if self.img_arr_processed is not None else self.img_arr_raw)

    def ensure_view_center(self):
        if self.view: self.view._ensure_view_center(self)

    def update_zoom_hint(self):
        if hasattr(self, "zoom_hint"):
            value = int(round(self.zoom_var.get())); self.zoom_hint.configure(text=f"Current zoom: {value}% (0 = full frame)")

    def get_center(self) -> Optional[Tuple[float, float]]:
        c = self.overlay.get("center")
        return (float(c["y"]), float(c["x"])) if isinstance(c, dict) and "x" in c and "y" in c else None

    def set_center(self, x: float, y: float):
        if "center" not in self.overlay or self.overlay["center"] is None: self.overlay["center"] = {}
        self.overlay["center"]["x"] = float(x); self.overlay["center"]["y"] = float(y)

    def get_dead_radius(self) -> float: return float(self.overlay.get("dead_radius", 0.0))
    def get_search_radius(self) -> float: return float(self.overlay.get("search_radius", 0.0))

    def sample_intensities(self, pts_yx: np.ndarray) -> np.ndarray:
        if pts_yx is None or len(pts_yx) == 0: return np.zeros((0,), float)
        if self._percent_map is not None:
            src = self._percent_map; H, W = src.shape[:2]; out = []
            for y, x in pts_yx:
                yi=max(0,min(H-1,int(round(y)))); xi=max(0,min(W-1,int(round(x)))); out.append(float(src[yi,xi]))
            return np.array(out, dtype=float)
        elif self.img_arr_processed is not None and self._percent_lookup is not None:
             H, W = self.img_arr_processed.shape[:2]; raw = []
             for y,x in pts_yx: raw.append(float(self.img_arr_processed[max(0,min(H-1,int(round(y)))), max(0,min(W-1,int(round(x))))]))
             try: from percentile_utils import map_values_to_percent; return map_values_to_percent(np.array(raw,dtype=float),*self._percent_lookup)
             except ImportError: print("Warn: percentile_utils NA"); return np.zeros(len(pts_yx),dtype=float)
        return np.zeros(len(pts_yx), dtype=float)

# --- Standalone wrapper (Без изменений) ---
class PointEditorApp(tk.Tk):
    def __init__(self, input_json: str | None = None):
        super().__init__(); self.title("SAED Editor (Simplified)"); self.geometry("1100x800")
        self.resizable(True, True); self.editor = PointEditor(self, input_json=input_json); self.editor.pack(fill=tk.BOTH, expand=True)
if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    try: import preproc, percentile_utils
    except ImportError as e: print(f"ERROR: {e}"); sys.exit(1)
    root = PointEditorApp(input_file); root.mainloop()