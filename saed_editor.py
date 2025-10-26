#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAED Editor + Analysis
----------------------
Редактор точек SAED.
Логика разделена на mix-in классы:
- EditorIO: Загрузка/сохранение/сессии
- EditorState: Undo/Redo, Измерения, Тултипы, Выбор кольцом
- EditorDrawingView: Отрисовка, Зум
- EditorEventHandlers: Обработка событий мыши/клавиатуры
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

# Импортируем разделенную логику
from saed_editor_io import EditorIO
from saed_editor_state import EditorState
from saed_editor_drawing import EditorDrawingView
from saed_editor_handlers import EditorEventHandlers
# Утилиты анализа (теперь импортируются внутри EditorEventHandlers)

# Зависимости, которые должны быть в среде
import math  # Нужен для handlers


class PointEditor(tk.Frame, EditorIO, EditorState, EditorDrawingView, EditorEventHandlers):

    def __init__(self, master: tk.Misc, controller=None,
                 input_json: str | None = None, auto_load: bool = True):
        super().__init__(master)
        self.controller = controller

        # --- Данные (инициализируются здесь или в _load_input_json) ---
        self.points = np.zeros((0, 2), float)
        self.values = np.zeros((0,), float) # Интенсивности в процентилях
        self.areas = np.zeros((0,), float)  # Площади пикселей
        self.angles: np.ndarray = np.zeros((0,), dtype=float) # Углы
        self.point_types: list[Union[str, int]] = []
        self.initial_group_ids: Dict[int, Optional[int]] = {}
        self.overlay = None
        self.image_path: Optional[Path] = None
        # --- ИЗМЕНЕНИЕ: Два массива для изображений ---
        self.img_arr_raw: Optional[np.ndarray] = None      # Исходное изображение
        self.img_arr_processed: Optional[np.ndarray] = None # Обработанное (для показа по умолчанию)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
        self._percent_map: Optional[np.ndarray] = None
        self._percent_lookup: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._preproc_settings = None  # Загрузится из PreprocSettings в IO

        # --- Состояние UI ---
        self.rect_start = None
        self.rect_artist = None
        self.center_dragging = False
        self._center_hit_radius = 10.0
        self.zoom_val = 0
        self.view_cx = None
        self.view_cy = None
        self._tooltip = None
        self._tooltip_idx = None
        # --- ИЗМЕНЕНИЕ: Переменная для режима фона ---
        self.show_raw_background = tk.BooleanVar(value=False) # По умолчанию показываем обработанное
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        # --- Состояние Undo/Redo ---
        self._undo = []
        self._redo = []
        self._history_cap = 300

        # --- Состояние Измерения ---
        # Переменные инициализируются в _initialize_state()

        # --- Состояние Выбора кольцом ---
        # Переменные инициализируются в _initialize_state()

        # --- Состояние Авто-группировки ---
        self._auto_grouping_active = False # Флаг, что идет процесс авто-группировки

        # --- Инициализация состояний из Mix-ins ---
        self._initialize_state()  # Инициализируем переменные из EditorState

        # --- Строим UI ---
        self._build_ui()

        # --- Первичная загрузка ---
        if auto_load and input_json:
            try:
                # Динамический импорт для PreprocSettings
                from preproc import PreprocSettings
                self._preproc_settings = PreprocSettings(mode="raw")
                self.load_input_json(Path(input_json), push_undo=False)
            except FileNotFoundError:
                self._set_status(f"Error: Input JSON not found at {input_json}")
                self._ensure_view_center()
                self._redraw()
            except ImportError as e:
                messagebox.showerror("Import Error", f"Failed to import required module: {e}")
        else:
            try:
                from preproc import PreprocSettings
                self._preproc_settings = PreprocSettings(mode="raw")
            except ImportError:
                messagebox.showerror("Import Error",
                                     "Failed to import 'PreprocSettings'. Ensure 'preproc.py' is present.")
            self._ensure_view_center()
            self._redraw()

    # ---------- UI ----------
    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # --- Левая панель ---
        side_panel = ttk.Frame(self, padding=(16, 16, 12, 16))
        side_panel.grid(row=0, column=0, sticky="ns")
        side_panel.columnconfigure(0, weight=1) # Разрешаем панели расширяться

        # --- Контейнер для скроллинга левой панели ---
        scroll_host = ttk.Frame(side_panel)
        scroll_host.pack(fill=tk.BOTH, expand=True)
        scroll_host.rowconfigure(0, weight=1)
        scroll_host.columnconfigure(0, weight=1)

        canvas = tk.Canvas(scroll_host, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding=(0, 0, 10, 0)) # Фрейм внутри канваса

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        # Привязка колеса мыши к канвасу для скроллинга
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))


        # --- Элементы управления внутри scrollable_frame ---
        controls = ttk.Frame(scrollable_frame)
        controls.pack(side=tk.TOP, fill=tk.X)

        header_row = ttk.Frame(controls)
        header_row.pack(fill=tk.X)
        header_row.columnconfigure(0, weight=1)

        help_button = ttk.Button(header_row, text="?", width=3, command=self._toggle_help, style="Toolbutton")
        help_button.pack(side=tk.RIGHT)

        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 10))

        # --- Zoom ---
        zoom_group = ttk.LabelFrame(controls, text="Scale", padding=(12, 8, 12, 10))
        zoom_group.pack(fill=tk.X)
        self.zoom_var = tk.DoubleVar(value=self.zoom_val)
        self.zoom_scale = ttk.Scale(zoom_group, from_=0, to=100, variable=self.zoom_var, command=self._on_zoom_change)
        self.zoom_scale.pack(fill=tk.X, padx=4, pady=(0, 6))
        self.zoom_hint = ttk.Label(zoom_group, anchor="w")
        self.zoom_hint.pack(fill=tk.X, padx=4)
        self.zoom_scale.set(self.zoom_val)

        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 10))

        # --- File IO ---
        file_group = ttk.Frame(controls)
        file_group.pack(fill=tk.X)
        ttk.Button(file_group, text="Open JSON…", command=self._open_json).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(file_group, text="Save", command=self._save_points_wrapper).pack(side=tk.LEFT, padx=(0, 6))

        # --- Analysis Launch ---
        analysis_group = ttk.Frame(controls)
        analysis_group.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(analysis_group, text="Start Fibonacci analysis", command=self._start_analysis).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 10))

        # --- Auto-Grouping ---
        auto_group = ttk.LabelFrame(controls, text="Auto Ring Grouping", padding=(12, 8, 12, 10))
        auto_group.pack(fill=tk.X, pady=(0, 10))
        auto_group.columnconfigure(1, weight=1) # Разрешаем спинбоксам растягиваться

        # Функция для создания Spinbox с Label
        def _spin_param(parent, row, label, default, **kwargs):
            ttk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
            spin = ttk.Spinbox(parent, width=8, justify="right", **kwargs)
            spin.set(default)
            spin.grid(row=row, column=1, sticky="ew", pady=2)
            return spin

        self.spn_auto_radius_tol = _spin_param(auto_group, 0, "Radius Tol (px)", 3.0, from_=0.1, to=50.0, increment=0.1, format="%.1f")
        self.spn_auto_area_tol = _spin_param(auto_group, 1, "Area Tol (%)", 15.0, from_=0.0, to=100.0, increment=1.0, format="%.1f") # Теперь row=1

        self.btn_auto_group = ttk.Button(auto_group, text="Auto-Group & Save Debug", command=self._auto_group_and_save_wrapper)
        self.btn_auto_group.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0)) # Теперь row=2

        # --- ИЗМЕНЕНИЕ: Добавлен Checkbutton для фона ---
        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 10))
        display_group = ttk.LabelFrame(controls, text="Display Options", padding=(12, 8, 12, 10))
        display_group.pack(fill=tk.X, pady=(0, 10))

        self.chk_raw_bg = ttk.Checkbutton(
            display_group,
            text="Show Raw Image Background",
            variable=self.show_raw_background,
            command=self._on_background_toggle # Вызываем redraw при изменении
        )
        self.chk_raw_bg.pack(anchor="w")
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---


        # --- Help Panel ---
        self.help_panel = ttk.LabelFrame(scrollable_frame, text="Hints", padding=(16, 12, 16, 12))
        help_text = (
            # ... (текст подсказок без изменений) ...
            "Ctrl+Z / Ctrl+Y — Undo/Redo actions\n"
            "Ctrl+S — Save current session\n"
            "Mouse Wheel — Zoom in/out\n\n"
            "LMB on empty — add point\n"
            "LMB on center — drag center\n"
            "RMB on point — delete point\n"
            "MMB on point — show info (Radius, Angle, Area, Type, Group ID)\n\n" # Обновлено
            "LMB Click (Point A) -> LMB Click (Point B/Center) — measure distance\n\n"
            "Ctrl + LMB Drag — draw selection ring\n"
            "  (+/- keys change thickness)\n"
            "  (LMB Click finishes selection)\n"
            "Shift + LMB (on point) — add point to selection\n"
            "Shift + RMB (on selected point) — remove point from selection\n"
            "Enter (with points selected) — average selected points (manual only)\n\n"
            "Shift + LMB Drag (no points selected) — rectangular delete\n\n"
            "Auto Ring Grouping:\n"
            " - Groups points by Radius & Area Tolerances.\n" # Обновлено
            " - Classifies groups based on Dominant Symmetry (structural/superstructural).\n" # Добавлено
            " - Assigns Numeric IDs to other groups.\n" # Обновлено
            " - Saves results & debug data to output folder." # Обновлено
        )
        ttk.Label(self.help_panel, text=help_text, justify="left", wraplength=280).pack(fill=tk.X)
        self._help_visible = False
        # Размещаем Help Panel после Auto Group
        self.help_panel.pack(side=tk.TOP, fill=tk.X, pady=(0, 8)) # По умолчанию показана

        # --- Spacer and Status Bar (вне скроллинга) ---
        status_frame = ttk.Frame(side_panel, padding=(0, 0, 0, 0))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))
        self.status_label = ttk.Label(status_frame, anchor="w", justify="left")
        self.status_label.pack(fill=tk.X)

        def _status_wrap(event, label=self.status_label):
            if not label.winfo_exists(): return
            new_wrap = max(int(event.width) - 8, 120)
            current_wrap = int(label.cget("wraplength") or 0)
            if new_wrap != current_wrap: label.configure(wraplength=new_wrap)

        self.status_label.bind("<Configure>", _status_wrap)


        # --- Правая панель (холст Matplotlib) ---
        canvas_frame = ttk.Frame(self, padding=(0, 16, 16, 16))
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(9.4, 6.4))
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("button_press_event", self._on_down)
        self.canvas.mpl_connect("button_release_event", self._on_up)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("key_press_event", self._on_key)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

        # --- Привязки клавиш ---
        self.bind_all('<Control-z>', self._undo_btn)
        self.bind_all('<Control-y>', self._redo_btn)
        # Ctrl+S обрабатывается в pipeline_app.py

        # --- Начальный статус ---
        self._default_status = "Mode: point editor"
        self._status_message = ""
        self._update_zoom_hint()
        self._set_status(self._default_status)
        self._toggle_help() # Начинаем со скрытыми подсказками

    def _toggle_help(self):
        self._help_visible = not self._help_visible
        if self._help_visible:
            self.help_panel.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
            self._set_status("Detailed hints expanded")
        else:
            self.help_panel.pack_forget()
            self._set_status(self._default_status)

    def _set_status(self, text: str):
        self._status_message = text
        if hasattr(self, "status_label") and self.status_label.winfo_exists():
            self.status_label.configure(text=text)
        if self.controller is not None and hasattr(self.controller, "set_status"):
            try:
                self.controller.set_status(f"Editor: {text}")
            except Exception:
                pass

    # --- ИЗМЕНЕНИЕ: Callback для переключателя фона ---
    def _on_background_toggle(self):
        self._redraw()
        mode = "Raw" if self.show_raw_background.get() else "Processed"
        self._set_status(f"Background image set to: {mode}")
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    def _auto_group_and_save_wrapper(self):
        """ Обертка для кнопки Auto-Group & Save Debug """
        if self._auto_grouping_active:
             messagebox.showwarning("Busy", "Auto-grouping is already running.")
             return
        try:
             self._auto_grouping_active = True
             self.btn_auto_group.config(state=tk.DISABLED) # Блокируем кнопку
             self.update_idletasks() # Обновляем UI
             self._auto_group_rings_and_save() # Запускаем основную функцию ГРУППИРОВКИ И СОХРАНЕНИЯ
        except Exception as e:
             messagebox.showerror("Auto-Grouping Error", f"An error occurred:\n{e}")
             import traceback
             traceback.print_exc()
        finally:
             self._auto_grouping_active = False
             if hasattr(self, 'btn_auto_group') and self.btn_auto_group.winfo_exists():
                  self.btn_auto_group.config(state=tk.NORMAL) # Разблокируем кнопку


class PointEditorApp(tk.Tk):
    """Standalone wrapper embedding the editor into the root window."""

    def __init__(self, input_json: str | None = None):
        super().__init__()
        self.title("SAED Editor + Analysis")
        self.geometry("1100x800")
        self.resizable(True, True)
        self.editor = PointEditor(self, input_json=input_json)
        self.editor.pack(fill=tk.BOTH, expand=True)


# -------- CLI ---------
def _parse_args(argv):
    import argparse

    p = argparse.ArgumentParser(description="SAED Point Editor")
    p.add_argument("--input", type=str, required=False, default=None,
                   help="Optional path to saed_input.json to load on startup.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])

    # --- Необходимые импорты для standalone запуска ---
    try:
        from preproc import PreprocSettings  # Нужен для инициализации
        # Остальные импорты (percentile_utils, scipy) проверяются динамически внутри методов
    except ImportError as e:
        print(f"ERROR: Could not import required module: {e}")
        print("Make sure all .py files (preproc.py, etc.) are in the same directory or accessible.")
        sys.exit(1)

    root = PointEditorApp(args.input)
    root.mainloop()