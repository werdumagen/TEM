#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAED Editor + Analysis (Refactored)
-----------------------------------
Главный класс, использующий архитектуру Model-View-Controller.
- PointEditor: Главный класс (tk.Frame), "Контроллер"
- SaedDataModel: "Модель", хранит все данные о точках
- EditorDrawingView: "View", отвечает только за отрисовку
- EditorEventHandlers: "Контроллер", обрабатывает события мыши/клавиатуры
- EditorIO: "Контроллер", отвечает за загрузку/сохранение
- EditorUIState: Хранит состояние UI (замеры, выделения)

Изменения:
- Добавлена панель симметрии.
- Расчет начальной симметрии при загрузке данных.
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

# --- Импорт новых, разделенных компонентов ---
from saed_data_model import SaedDataModel
from saed_editor_state_ui import EditorUIState
from saed_editor_drawing_refactored import EditorDrawingView
from saed_editor_handlers_refactored import EditorEventHandlers, symmetry_scores, cluster_rings, pol_from # Импортируем утилиты симметрии
from saed_editor_io_refactored import EditorIO
# ---

# Импорт зависимостей (они должны быть доступны)
try:
    from preproc import PreprocSettings
    from percentile_utils import map_values_to_percent
except ImportError:
    messagebox.showerror("Import Error", "Failed to import 'PreprocSettings' or 'percentile_utils'.")
    # Используем заглушку, если не найден
    if 'PreprocSettings' not in globals():
        class PreprocSettings:
            def __init__(self, mode="raw"): self.mode = mode
            @staticmethod
            def from_json(data, fallback_mode=None): return PreprocSettings(fallback_mode or "raw")
            def to_json(self): return {"mode": self.mode}


class PointEditor(tk.Frame):

    def __init__(self, master: tk.Misc, controller=None,
                 input_json: str | None = None, auto_load: bool = True):
        super().__init__(master)
        self.app_controller = controller # Контроллер верхнего уровня (pipeline_app)

        # --- Данные Контроллера (не-точки) ---
        self.image_path: Optional[Path] = None
        self.img_arr_raw: Optional[np.ndarray] = None
        self.img_arr_processed: Optional[np.ndarray] = None
        self._percent_map: Optional[np.ndarray] = None
        self._percent_lookup: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._preproc_settings = PreprocSettings(mode="raw")
        self.overlay: Dict[str, Any] = {} # {"center": {...}, "dead_radius": ...}

        # --- Состояние UI Контроллера ---
        self.zoom_val = 0
        self.view_cx = None
        self.view_cy = None
        self.show_raw_background = tk.BooleanVar(value=False)
        self.status_message = ""
        self.default_status = "Mode: point editor"
        # --- НОВОЕ: Состояние симметрии ---
        self.detected_symmetry_label_var = tk.StringVar(value="Detected: N/A")
        self.selected_symmetry_var = tk.IntVar(value=0) # 0 = Auto/None
        # ---

        # --- Состояние Undo/Redo (хранится в Контроллере) ---
        self.undo: List[Dict[str, Any]] = [] # Хранит snapshots Модели
        self.redo: List[Dict[str, Any]] = []
        self._history_cap = 300

        # --- ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ ---
        self.model = SaedDataModel()
        self.ui_state = EditorUIState()

        # View и Handlers/IO инициализируются в _build_ui,
        # т.к. им нужны созданные ax и canvas
        self.view: Optional[EditorDrawingView] = None
        self.handlers: Optional[EditorEventHandlers] = None
        self.io: Optional[EditorIO] = None

        # --- Строим UI ---
        self._build_ui() # Создает self.ax, self.canvas

        # --- Инициализация компонентов, зависящих от UI ---
        if self.ax is None or self.canvas is None:
             raise RuntimeError("UI build failed to create ax or canvas.")

        self.view = EditorDrawingView(self.ax, self.canvas)
        self.io = EditorIO(self)
        self.handlers = EditorEventHandlers(self)

        # --- Подключаем обработчики ---
        self.handlers.connect_mpl_events(self.canvas)
        self.zoom_scale.config(command=self.handlers.on_zoom_change_tk)
        self.chk_raw_bg.config(command=self._on_background_toggle)
        # Подключаем кнопки
        self.btn_open.config(command=self.io.open_json_wrapper)
        self.btn_save.config(command=self.io.save_points_wrapper)
        self.btn_analysis.config(command=self.io.start_analysis_wrapper)
        self.btn_auto_group.config(command=self.handlers.auto_group_and_save_wrapper)
        # --- НОВОЕ: Подключаем обработчик изменения симметрии ---
        self.cmb_symmetry.bind("<<ComboboxSelected>>", self._on_symmetry_change)
        # ---

        # --- Привязки клавиш ---
        self.bind_all('<Control-z>', self._undo_btn)
        self.bind_all('<Control-y>', self._redo_btn)

        # --- Первичная загрузка ---
        if auto_load and input_json:
            try:
                # load_input_json вызовет _calculate_initial_symmetry внутри
                self.io.load_input_json(Path(input_json), push_undo=False)
            except FileNotFoundError:
                self.set_status(f"Error: Input JSON not found at {input_json}")
                self.ensure_view_center()
                self._calculate_initial_symmetry() # Рассчитываем симметрию для пустого состояния
                self.redraw()
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to auto-load JSON:\n{e}")
                self._calculate_initial_symmetry()
        else:
            self.ensure_view_center()
            self._calculate_initial_symmetry() # Рассчитываем симметрию для пустого состояния
            self.redraw()

        # --- Начальный статус ---
        self.update_zoom_hint()
        self.set_status(self.default_status)
        self._toggle_help() # Начинаем со скрытыми подсказками


    # ---------- UI Builder ----------
    def _build_ui(self):
        """Создает все Tkinter виджеты и Matplotlib холст."""
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        side_panel = ttk.Frame(self, padding=(16, 16, 12, 16))
        side_panel.grid(row=0, column=0, sticky="ns")
        side_panel.columnconfigure(0, weight=1)

        # ... (Код для scroll_host, canvas, vscroll, scrollable_frame без изменений) ...
        scroll_host = ttk.Frame(side_panel)
        scroll_host.pack(fill=tk.BOTH, expand=True)
        scroll_host.rowconfigure(0, weight=1)
        scroll_host.columnconfigure(0, weight=1)
        canvas = tk.Canvas(scroll_host, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding=(0, 0, 10, 0))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        # Убираем bind_all для MouseWheel здесь, т.к. он будет мешать matplotlib
        # canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        # --- Привязка колеса только к левому канвасу ---
        def _on_left_scroll(event):
             # Проверяем, находится ли курсор над левым канвасом или его дочерними элементами
             widget_under_cursor = event.widget.winfo_containing(event.x_root, event.y_root)
             if widget_under_cursor is canvas or widget_under_cursor.master is scrollable_frame:
                  canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind("<MouseWheel>", _on_left_scroll)
        scrollable_frame.bind("<MouseWheel>", _on_left_scroll) # Для дочерних элементов
        # ---

        controls = ttk.Frame(scrollable_frame)
        controls.pack(side=tk.TOP, fill=tk.X)

        # --- Help Button ---
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
        self.zoom_scale = ttk.Scale(zoom_group, from_=0, to=100, variable=self.zoom_var) # command= уст-ся в __init__
        self.zoom_scale.pack(fill=tk.X, padx=4, pady=(0, 6))
        self.zoom_hint = ttk.Label(zoom_group, anchor="w")
        self.zoom_hint.pack(fill=tk.X, padx=4)
        self.zoom_scale.set(self.zoom_val)
        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 10))

        # --- File IO ---
        file_group = ttk.Frame(controls)
        file_group.pack(fill=tk.X)
        self.btn_open = ttk.Button(file_group, text="Open JSON…") # command= уст-ся в __init__
        self.btn_open.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_save = ttk.Button(file_group, text="Save") # command= уст-ся в __init__
        self.btn_save.pack(side=tk.LEFT, padx=(0, 6))

        # --- Analysis Launch ---
        analysis_group = ttk.Frame(controls)
        analysis_group.pack(fill=tk.X, pady=(8, 0))
        self.btn_analysis = ttk.Button(analysis_group, text="Start Fibonacci analysis") # command= уст-ся в __init__
        self.btn_analysis.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 10))

        # --- Auto-Grouping ---
        auto_group = ttk.LabelFrame(controls, text="Auto Ring Grouping", padding=(12, 8, 12, 10))
        auto_group.pack(fill=tk.X, pady=(0, 10))
        auto_group.columnconfigure(1, weight=1)
        def _spin_param(parent, row, label, default, **kwargs):
            ttk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
            spin = ttk.Spinbox(parent, width=8, justify="right", **kwargs)
            spin.set(default)
            spin.grid(row=row, column=1, sticky="ew", pady=2)
            return spin
        self.spn_auto_radius_tol = _spin_param(auto_group, 0, "Radius Tol (px)", 3.0, from_=0.1, to=50.0, increment=0.1, format="%.1f")
        self.spn_auto_area_tol = _spin_param(auto_group, 1, "Area Tol (%)", 15.0, from_=0.0, to=100.0, increment=1.0, format="%.1f")
        self.btn_auto_group = ttk.Button(auto_group, text="Auto-Group & Save Debug") # command= уст-ся в __init__
        self.btn_auto_group.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        # --- НОВОЕ: Symmetry Panel ---
        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 10))
        symmetry_group = ttk.LabelFrame(controls, text="Symmetry", padding=(12, 8, 12, 10))
        symmetry_group.pack(fill=tk.X, pady=(0, 10))
        symmetry_group.columnconfigure(1, weight=1) # Allow combobox to expand if needed

        self.lbl_detected_symmetry = ttk.Label(symmetry_group, textvariable=self.detected_symmetry_label_var)
        self.lbl_detected_symmetry.grid(row=0, column=0, columnspan=2, sticky="w", padx=(0, 6), pady=2)

        ttk.Label(symmetry_group, text="Select/Override:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        # Using Combobox for predefined values + manual entry possibility
        self.cmb_symmetry = ttk.Combobox(
            symmetry_group,
            textvariable=self.selected_symmetry_var,
            values=[0, 4, 6, 8, 10, 12], # Common symmetries + 0 for None/Auto
            width=8,
            state="readonly" # Use readonly to prevent arbitrary text
        )
        self.cmb_symmetry.grid(row=1, column=1, sticky="w", pady=2)
        self.cmb_symmetry.set(0) # Default to 0 (Auto/None)
        # --- КОНЕЦ НОВОГО ---


        # --- Display Options ---
        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 10))
        display_group = ttk.LabelFrame(controls, text="Display Options", padding=(12, 8, 12, 10))
        display_group.pack(fill=tk.X, pady=(0, 10))
        self.chk_raw_bg = ttk.Checkbutton(
            display_group, text="Show Raw Image Background", variable=self.show_raw_background
        ) # command= уст-ся в __init__
        self.chk_raw_bg.pack(anchor="w")

        # --- Help Panel ---
        self.help_panel = ttk.LabelFrame(scrollable_frame, text="Hints", padding=(16, 12, 16, 12))
        # ... (текст подсказок без изменений) ...
        help_text = (
            "Ctrl+Z / Ctrl+Y — Undo/Redo actions\n"
            "Ctrl+S — Save current session (if in app)\n"
            "Mouse Wheel — Zoom in/out\n\n"
            "LMB on empty — add point\n"
            "LMB on center — drag center\n"
            "RMB on point — delete point\n"
            "MMB on point — show info (Radius, Angle, Area, Type, Group ID)\n\n"
            "LMB Click (Point A) -> LMB Click (Point B/Center) — measure distance\n\n"
            "Ctrl + LMB Drag — draw selection ring\n"
            "  (+/- keys change thickness)\n"
            "  (LMB Click finishes selection)\n"
            "Shift + LMB (on point) — add point to selection\n"
            "Shift + RMB (on selected point) — remove point from selection\n"
            "Enter (with points selected) — average selected points\n\n"
            "Shift + LMB Drag (no points selected) — rectangular delete\n\n"
            "Auto Ring Grouping:\n"
            " - Groups points by Radius & Area Tolerances.\n"
            " - Classifies groups based on Dominant Symmetry (calculated automatically).\n" # Updated hint
            " - Assigns Numeric IDs to other groups.\n"
            " - Saves results & debug data to output folder.\n\n"
            "Symmetry Panel:\n"
            " - Shows automatically detected symmetry.\n"
            " - Allows selecting a specific fold symmetry (0 = Auto/None)." # Added hint
        )
        ttk.Label(self.help_panel, text=help_text, justify="left", wraplength=280).pack(fill=tk.X)
        self._help_visible = False
        self.help_panel.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))

        # --- Status Bar (вне скроллинга) ---
        status_frame = ttk.Frame(side_panel, padding=(0, 0, 0, 0))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))
        self.status_label = ttk.Label(status_frame, anchor="w", justify="left")
        self.status_label.pack(fill=tk.X)
        def _status_wrap(event, label=self.status_label):
            if not label.winfo_exists(): return
            label.configure(wraplength=max(int(event.width) - 8, 120))
        self.status_label.bind("<Configure>", _status_wrap)

        # --- Правая панель (холст Matplotlib) ---
        canvas_frame = ttk.Frame(self, padding=(0, 16, 16, 16))
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(9.4, 6.4))
        self.ax = self.fig.add_subplot(111) # Сохраняем ax
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame) # Сохраняем canvas
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # ---------- Управление Контроллером ----------

    def redraw(self):
        """Публичный метод для запуска перерисовки из Handlers/IO."""
        if self.view:
            self.view.redraw(self)

    def set_status(self, text: str):
        """Обновляет строку статуса."""
        self.status_message = text
        if hasattr(self, "status_label") and self.status_label.winfo_exists():
            self.status_label.configure(text=text)
        if self.app_controller is not None and hasattr(self.app_controller, "set_status"):
            try:
                self.app_controller.set_status(f"Editor: {text}")
            except Exception:
                pass

    def get_output_dir(self) -> Path:
        """Получает путь к папке вывода из pipeline_app."""
        if self.app_controller and hasattr(self.app_controller, 'launcher'):
            output_dir_str = self.app_controller.launcher.ent_out.get()
            if output_dir_str:
                try:
                    output_dir = Path(output_dir_str).expanduser().resolve()
                    output_dir.mkdir(parents=True, exist_ok=True)
                    return output_dir
                except Exception as e:
                    raise OSError(f"Invalid output directory '{output_dir_str}': {e}")
            else:
                raise ValueError("Output folder not specified.")
        else:
            print("Warning: Controller/Launcher not found, using default output 'saed_results'.")
            output_dir = Path("saed_results").resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir

    def clear_all(self):
        """Сбрасывает редактор в исходное состояние."""
        self.image_path = None
        self.img_arr_raw = None; self.img_arr_processed = None
        self._percent_map = None; self._percent_lookup = None
        self.overlay = {}; self._preproc_settings = PreprocSettings()
        self.zoom_val = 0; self.view_cx = None; self.view_cy = None
        self.zoom_var.set(0)
        self.show_raw_background.set(False)

        self.model = SaedDataModel() # Новая пустая модель
        self.ui_state = EditorUIState() # Новое пустое состояние UI

        self.undo.clear(); self.redo.clear()
        self._calculate_initial_symmetry() # Обновляем отображение симметрии
        self.redraw(); self.set_status("Editor cleared.")


    # ---------- Обработчики UI Контроллера ----------

    def _toggle_help(self):
        self._help_visible = not self._help_visible
        if self._help_visible:
            self.help_panel.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
            self.set_status("Detailed hints expanded")
        else:
            self.help_panel.pack_forget()
            self.set_status(self.default_status)

    def _on_background_toggle(self):
        """Callback для переключателя фона."""
        self.redraw()
        mode = "Raw" if self.show_raw_background.get() else "Processed"
        self.set_status(f"Background image set to: {mode}")

    # --- НОВОЕ: Обработчик изменения симметрии ---
    def _on_symmetry_change(self, event=None):
        try:
            selected_sym = self.selected_symmetry_var.get()
            self.set_status(f"Symmetry override set to: {selected_sym}-fold" if selected_sym > 0 else "Symmetry override: Auto/None")
            # Можно добавить сюда логику, если нужно что-то пересчитать при изменении
            # Например, перерисовать точки другим цветом или подготовиться к авто-группировке
            # self.redraw() # Перерисовка, если нужно визуальное изменение
        except tk.TclError:
            pass # Ignore potential errors during widget initialization/destruction
    # ---

    # ---------- Управление состоянием (Undo/Redo) ----------
    # --- Undo/Redo методы без изменений ---
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
        if self.undo:
            self.undo.pop()

    def _apply_snapshot(self, snap: Dict[str, Any]):
        self.model.apply_snapshot(snap)
        # Analysis is now only run on load or via auto-group button

    def _undo_btn(self, event=None):
        if hasattr(event, 'widget') and isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)): return
        if not self.undo: self.set_status("Nothing to undo."); return
        self.ui_state.cancel_all_interactions()
        self.ui_state.clear_tooltip(self.view)
        self.push_redo()
        snap_to_restore = self.undo.pop()
        self._apply_snapshot(snap_to_restore) # Вызовет _calculate_initial_symmetry
        self.redraw(); self.set_status("Undo successful.")

    def _redo_btn(self, event=None):
        if hasattr(event, 'widget') and isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)): return
        if not self.redo: self.set_status("Nothing to redo."); return
        self.ui_state.cancel_all_interactions()
        self.ui_state.clear_tooltip(self.view)
        self.push_undo()
        snap_to_restore = self.redo.pop()
        self._apply_snapshot(snap_to_restore) # Вызовет _calculate_initial_symmetry
        self.redraw(); self.set_status("Redo successful.")
    # --- Конец Undo/Redo ---

    # ---------- Расчет симметрии ----------
    def _calculate_initial_symmetry(self):
        """Рассчитывает доминирующую симметрию и обновляет UI."""
        dominant_symmetry = 0
        symmetry_label = "Detected: N/A"

        center = self.get_center()
        # Проверяем, что модель инициализирована и не пуста
        if center and hasattr(self, 'model') and not self.model.is_empty():
            cy, cx = center
            dead_radius = self.get_dead_radius()

            # Получаем радиусы и углы ИЗ МОДЕЛИ (они должны быть актуальны)
            all_radii = np.hypot(self.model.points[:, 1] - cx, self.model.points[:, 0] - cy)
            all_angles = self.model.angles # Используем сохраненные углы

            valid_mask = ~np.isnan(all_angles) & (all_radii > dead_radius)
            if np.any(valid_mask):
                radii_valid = all_radii[valid_mask]
                angles_valid = all_angles[valid_mask]

                potential_ring_means, _, _ = cluster_rings(radii_valid)
                scores = symmetry_scores(angles_valid, radii_valid, potential_ring_means)

                if scores:
                    best_sym_str = max(scores, key=scores.get)
                    try:
                        dominant_symmetry = int(best_sym_str.split('-')[0])
                        symmetry_label = f"Detected: {dominant_symmetry}-fold"
                    except (ValueError, IndexError):
                        symmetry_label = "Detected: Error"
                else:
                    symmetry_label = "Detected: None Found"
            else:
                symmetry_label = "Detected: No valid points"
        else:
             if not center:
                  symmetry_label = "Detected: No Center"
             elif not hasattr(self, 'model') or self.model.is_empty():
                  symmetry_label = "Detected: No Points"


        # Обновляем UI
        if hasattr(self, 'detected_symmetry_label_var'):
            self.detected_symmetry_label_var.set(symmetry_label)
        if hasattr(self, 'selected_symmetry_var'):
             # Устанавливаем значение в комбобоксе/спинбоксе,
             # если оно есть в списке допустимых значений
             try:
                  current_selection = self.selected_symmetry_var.get()
                  # Если пользователь еще не выбрал вручную (значение 0),
                  # устанавливаем найденное значение
                  if current_selection == 0:
                       valid_symmetries = self.cmb_symmetry['values'] # Получаем список допустимых значений
                       # Преобразуем строки в int для сравнения
                       valid_symmetries_int = [int(v) for v in valid_symmetries]
                       if dominant_symmetry in valid_symmetries_int:
                            self.selected_symmetry_var.set(dominant_symmetry)
                       else:
                            self.selected_symmetry_var.set(0) # Сбрасываем в 0, если не найдено
             except (tk.TclError, AttributeError, ValueError):
                  # Ошибка может возникнуть, если виджет еще не создан
                  # или значение некорректно
                  pass

        return dominant_symmetry


    # ---------- Утилиты-аксессоры (для Handlers/View/IO) ----------
    # --- (Без изменений) ---
    def get_image_to_display(self) -> Optional[np.ndarray]:
        img_to_display = self.img_arr_raw if self.show_raw_background.get() else self.img_arr_processed
        if img_to_display is None:
            img_to_display = self.img_arr_processed if self.img_arr_processed is not None else self.img_arr_raw
        return img_to_display

    def ensure_view_center(self):
        if self.view:
            self.view._ensure_view_center(self)

    def update_zoom_hint(self):
        if hasattr(self, "zoom_hint"):
            value = int(round(self.zoom_var.get()))
            self.zoom_hint.configure(text=f"Current zoom: {value}% (0 = full frame)")

    def get_center(self) -> Optional[Tuple[float, float]]:
        if self.overlay and isinstance(self.overlay.get("center"), dict):
            c = self.overlay["center"]
            if "x" in c and "y" in c:
                return (float(c["y"]), float(c["x"]))
        return None

    def set_center(self, x: float, y: float):
        if "center" not in self.overlay or self.overlay["center"] is None:
            self.overlay["center"] = {}
        self.overlay["center"]["x"] = float(x)
        self.overlay["center"]["y"] = float(y)

    def get_dead_radius(self) -> float:
        return float(self.overlay.get("dead_radius", 0.0))

    def get_search_radius(self) -> float:
        return float(self.overlay.get("search_radius", 0.0))

    def sample_intensities(self, pts_yx: np.ndarray) -> np.ndarray:
        if pts_yx is None or len(pts_yx) == 0: return np.zeros((0,), float)
        if self._percent_map is not None:
            src = self._percent_map; H, W = src.shape[:2]; out = []
            for y, x in pts_yx:
                yi = max(0, min(H - 1, int(round(y))))
                xi = max(0, min(W - 1, int(round(x))))
                out.append(float(src[yi, xi]))
            return np.array(out, dtype=float)
        elif self.img_arr_processed is not None and self._percent_lookup is not None:
             H, W = self.img_arr_processed.shape[:2]; raw_values = []
             for y, x in pts_yx:
                 yi = max(0, min(H - 1, int(round(y))))
                 xi = max(0, min(W - 1, int(round(x))))
                 raw_values.append(float(self.img_arr_processed[yi, xi]))
             # Убедимся, что map_values_to_percent импортирован
             try:
                 from percentile_utils import map_values_to_percent
                 return map_values_to_percent(np.array(raw_values, dtype=float), *self._percent_lookup)
             except ImportError:
                 print("Warning: percentile_utils not available for intensity sampling fallback.")
                 return np.zeros(len(pts_yx), dtype=float)

        return np.zeros(len(pts_yx), dtype=float)


# --- Обертка для Standalone запуска ---
class PointEditorApp(tk.Tk):
    def __init__(self, input_json: str | None = None):
        super().__init__()
        self.title("SAED Editor + Analysis (Refactored w/ Symmetry)")
        self.geometry("1100x800")
        self.resizable(True, True)
        self.editor = PointEditor(self, input_json=input_json)
        self.editor.pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else None

    # Проверка импортов для standalone
    try:
        import preproc
        import percentile_utils
        # Проверим импорт scipy для симметрии (нужен для cluster_rings -> find_peaks)
        import scipy.signal
    except ImportError as e:
        print(f"ERROR: Could not import required module: {e}")
        print("Make sure all .py files (preproc.py, percentile_utils.py) and dependencies (scipy) are installed/accessible.")
        sys.exit(1)

    root = PointEditorApp(input_file)
    root.mainloop()