#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Класс IO (Упрощенный)
---------------------
Отвечает за загрузку/сохранение JSON, изображений, сессий.
Не обрабатывает типы/ID точек.
"""
import json
from pathlib import Path
from tkinter import filedialog, messagebox
import numpy as np
from typing import Tuple, Union, Optional, Dict, Any, List

try:
    from preproc import PreprocSettings, load_grayscale_with_preproc
    from percentile_utils import compute_percentile_map, map_values_to_percent
except ImportError as e:
    print(f"Ошибка импорта зависимостей: {e}")
    messagebox.showerror("Import Error", f"Failed to import dependencies: {e}")
    class PreprocSettings:
        def __init__(self, mode="raw"): self.mode = mode
        @staticmethod
        def from_json(data, fallback_mode=None): return PreprocSettings(fallback_mode or "raw")
        def to_json(self): return {"mode": self.mode}
    def load_grayscale_with_preproc(path, settings): raise ImportError("preproc.py not found")
    def compute_percentile_map(img): raise ImportError("percentile_utils.py not found")


class EditorIO:

    def __init__(self, controller):
        self.controller = controller

    # ---------- Обертки для UI ----------
    def open_json_wrapper(self):
        p = filedialog.askopenfilename(
            title="Open SAED Input",
            filetypes=[("SAED Input JSON", "*saed_input.json;*.json"), ("All", "*.*")]
        )
        if p:
            try: self.load_input_json(Path(p), push_undo=True)
            except FileNotFoundError: messagebox.showerror("Error", f"File not found: {p}")
            except Exception as e: messagebox.showerror("Error", f"Failed to load JSON:\n{e}")

    def save_points_wrapper(self):
        try: self._save_points()
        except Exception as e: messagebox.showerror("Save Error", f"Failed to save points:\n{e}")

    # ---------- Основная Логика IO ----------

    def load_input_json(self, path: Path, *, push_undo: bool = False, reset_view: bool = True):
        """Загружает JSON (без типов/ID)."""
        if not path.exists(): raise FileNotFoundError(f"Input JSON file not found: {path}")
        if push_undo: self.controller.push_undo()

        try: data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e: raise ValueError(f"Invalid JSON format in {path.name}: {e}")
        except Exception as e: raise IOError(f"Failed to read JSON file {path.name}: {e}")

        self.controller.img_arr_raw = None; self.controller.img_arr_processed = None
        self.controller._percent_map = None; self.controller._percent_lookup = None

        img_path_str = data.get("image")
        if not img_path_str: raise ValueError("JSON missing 'image' field.")
        img_p = Path(img_path_str)
        img_path = (path.parent / img_p).resolve() if not img_p.is_absolute() else img_p.resolve()
        if not img_path.exists(): raise FileNotFoundError(f"Image not found: {img_path}")
        self.controller.image_path = img_path
        fallback_mode = data.get("preproc_mode", "raw")
        self.controller._preproc_settings = PreprocSettings.from_json(data.get("preproc"), fallback_mode=fallback_mode)

        try:
            self.controller.img_arr_raw = load_grayscale_with_preproc(img_path, PreprocSettings(mode="raw"))
            self.controller.img_arr_processed = load_grayscale_with_preproc(img_path, self.controller._preproc_settings)
            p_map, uniq_vals, uniq_perc = compute_percentile_map(self.controller.img_arr_processed)
            self.controller._percent_map = p_map
            self.controller._percent_lookup = (uniq_vals, uniq_perc)
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load/process image:\n{e}")
            if self.controller.img_arr_raw is None: self.controller.img_arr_raw = np.zeros((100, 100), dtype=np.uint8)
            if self.controller.img_arr_processed is None: self.controller.img_arr_processed = self.controller.img_arr_raw.copy()

        img_h, img_w = self.controller.img_arr_processed.shape[:2]
        c = data.get("center") or {}; r = data.get("radii") or {}
        default_cx = (img_w - 1) / 2.0; default_cy = (img_h - 1) / 2.0
        center_data = {"x": float(c.get("x", default_cx)), "y": float(c.get("y", default_cy))}
        self.controller.overlay = {
            "center": center_data,
            "dead_radius": float(r.get("dead", 0.0)),
            "search_radius": float(r.get("search", 0.0))
        }
        center_yx = (center_data["y"], center_data["x"])

        try:
            pts_data = data.get("points", [])
            # Модель загрузит точки без типов/ID
            self.controller.model.from_json_list(pts_data, self.controller.sample_intensities, center_yx)
        except (ValueError, TypeError) as e:
            messagebox.showerror("Data Error", f"Invalid point data: {e}")
            self.controller.model.from_json_list([], self.controller.sample_intensities, center_yx)

        self.controller.ui_state.clear_tooltip()
        if reset_view: self.controller.view_cx = None; self.controller.view_cy = None
        self.controller.ensure_view_center()
        self.controller.undo.clear(); self.controller.redo.clear()
        # УДАЛЕНО: _calculate_initial_symmetry()
        # УДАЛЕНО: _update_group_panel()
        self.controller.redraw()
        self.controller.update_zoom_hint()
        self.controller.set_status(f"Loaded: {path.name}")

    def _save_points(self) -> Path:
        """Сохраняет точки (без типов/ID) и обновляет 'saed_input.edited.json'."""
        output_dir = self.controller.get_output_dir()
        # Модель вернет список без типов/ID
        pts_list = self.controller.model.to_json_list()

        def default_serializer(obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        spots_path = output_dir / "spots.json"
        spots_path.write_text(json.dumps({"points": pts_list}, indent=2, default=default_serializer), encoding="utf-8")

        abs_image_path = self.controller.image_path.resolve() if self.controller.image_path else None
        saed_input_edited_data = {
            "image": str(abs_image_path) if abs_image_path else None,
            "preproc_mode": self.controller._preproc_settings.mode,
            "preproc": self.controller._preproc_settings.to_json(),
            "center": self.controller.overlay.get("center"),
            "radii": {
                "dead": self.controller.overlay.get("dead_radius", 0.0),
                "search": self.controller.overlay.get("search_radius", 0.0)
            },
            "points": pts_list # Уже без типов/ID
        }
        edited_path = output_dir / "saed_input.edited.json"
        edited_path.write_text(json.dumps(saed_input_edited_data, ensure_ascii=False, indent=2, default=default_serializer), encoding="utf-8")
        self.controller.set_status(f"Points saved to {output_dir.name}")
        return spots_path

    # --- Session Save/Load (Упрощено) ---

    # ++++++++++ НАЧАЛО ИЗМЕНЕНИЯ ++++++++++
    # МЕТОД get_state(self) -> dict: БЫЛ УДАЛЕН ОТСЮДА
    # Он перемещен в saed_editor.py
    # ++++++++++ КОНЕЦ ИЗМЕНЕНИЯ ++++++++++

    # --- Сохранение отладки (Упрощено) ---

    def save_debug_data(self, filename: str = "points_debug.json"): # Изменено имя по умолчанию
        """Сохраняет отладочные данные из Модели (без типов/ID)."""
        output_dir = self.controller.get_output_dir()
        if self.controller.model.is_empty():
            messagebox.showwarning("Save Debug", "No points to save.")
            return

        # Модель вернет данные без типов/ID
        data_to_save = self.controller.model.get_all_data_for_debug()

        filepath = output_dir / filename
        try:
            def default_serializer(obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
            filepath.write_text(json.dumps(data_to_save, indent=2, default=default_serializer), encoding="utf-8")
            current_status = self.controller.status_message
            self.controller.set_status(f"{current_status} Debug data saved to {filename}")
            print(f"Debug data saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save debug data:\n{e}")
            self.controller.set_status(f"Failed to save {filename}")

    # --- Запуск Анализа (Без изменений, т.к. он просто передает данные) ---

    def start_analysis_wrapper(self):
        """Обертка для кнопки 'Start Analysis'."""
        try:
            saved_spots_path = self._save_points()
            output_dir = saved_spots_path.parent
            payload_path = output_dir / "fibo_input.json"
        except (ValueError, OSError, Exception) as e:
            messagebox.showerror("Save Error", f"Cannot proceed. Failed to save:\n{e}")
            return

        try:
            abs_image_path = self.controller.image_path.resolve() if self.controller.image_path else None
            img_to_use = self.controller.get_image_to_display()
            H, W = img_to_use.shape[:2] if img_to_use is not None else (1, 1)
            geo_center_data = {"x": (W - 1) / 2.0, "y": (H - 1) / 2.0}
            # Модель вернет точки без типов/ID
            pts_list = self.controller.model.to_json_list()

            payload = {
                "image": str(abs_image_path) if abs_image_path else None,
                "preproc_mode": self.controller._preproc_settings.mode,
                "preproc": self.controller._preproc_settings.to_json(),
                "points": pts_list,
                "centers": { "geometric": geo_center_data, "overlay": self.controller.overlay.get("center") },
                "radii": { "dead": self.controller.overlay.get("dead_radius", 0.0), "search": self.controller.overlay.get("search_radius", 0.0) },
                "spots_json": str(saved_spots_path.resolve())
            }

            def default_serializer(obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

            payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=default_serializer), encoding="utf-8")
            self.controller.set_status("Prepared data for analysis…")
        except Exception as e:
            messagebox.showerror("Data Prep Error", f"Failed to create ({payload_path.name}):\n{e}")
            return

        if self.controller.app_controller is not None:
            try:
                self.controller.app_controller.open_analysis(payload_path.resolve(), abs_image_path, saved_spots_path.resolve())
            except Exception as e:
                print(f"Error calling controller.open_analysis: {e}")
                messagebox.showerror("Launch Error", f"Failed to switch to analysis tab.\nDetails: {e}")
        else:
            messagebox.showwarning("Standalone Mode", "Cannot switch to analysis tab.")