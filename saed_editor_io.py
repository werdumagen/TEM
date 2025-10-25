#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mix-in класс для PointEditor:
Обрабатывает I/O (загрузку, сохранение, сессии, запуск анализа)
"""
import json
from pathlib import Path
from tkinter import filedialog, messagebox
import numpy as np
# Зависимости импортируются динамически

class EditorIO:

    # ---------- IO ----------
    def _open_json(self):
        # ... (без изменений) ...
        p = filedialog.askopenfilename(
            title="Open SAED Input",
            filetypes=[("SAED Input JSON", "*saed_input.json;*.json"), ("All", "*.*")]
        )
        if p:
            try:
                self.load_input_json(Path(p), push_undo=True)
            except FileNotFoundError:
                messagebox.showerror("Error", f"File not found: {p}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load JSON:\n{e}")

    def load_input_json(self, path: Path, *, push_undo: bool = False, reset_view: bool = True):
        """Public JSON loading method, also used by the tab controller."""
        # ... (без изменений) ...
        if not path.exists():
            raise FileNotFoundError(f"Input JSON file not found: {path}")

        if push_undo:
            self._push_undo()

        self._load_input_json(path)
        self._clear_tooltip()

        if reset_view:
            self.view_cx = None
            self.view_cy = None
        self._ensure_view_center()

        self._undo.clear()
        self._redo.clear()

        self._redraw()
        self._update_zoom_hint()
        self._set_status(f"Loaded: {path.name}")


    def _load_input_json(self, path: Path):
        """Internal method to load data from the JSON file."""
        from preproc import PreprocSettings, load_grayscale_with_preproc
        from percentile_utils import compute_percentile_map, map_values_to_percent
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e: raise ValueError(f"Invalid JSON format in {path.name}: {e}") from e
        except Exception as e: raise IOError(f"Failed to read JSON file {path.name}: {e}") from e

        self._percent_map = None; self._percent_lookup = None
        img_path_str = data.get("image")
        if not img_path_str: raise ValueError("JSON missing 'image' field.")
        img_p = Path(img_path_str)
        self.image_path = (path.parent / img_p).resolve() if not img_p.is_absolute() else img_p.resolve()
        if not self.image_path.exists(): raise FileNotFoundError(f"Image not found: {self.image_path}")

        fallback_mode = data.get("preproc_mode");
        if not isinstance(fallback_mode, str): fallback_mode = None
        self._preproc_settings = PreprocSettings.from_json(data.get("preproc"), fallback_mode=fallback_mode)

        try:
            self.img_arr = load_grayscale_with_preproc(self.image_path, self._preproc_settings)
            self._percent_map, uniq_vals, uniq_perc = compute_percentile_map(self.img_arr)
            self._percent_lookup = (uniq_vals, uniq_perc)
        except RuntimeError as cv_err: messagebox.showerror("Dependency Error", str(cv_err)); self.img_arr = None; self._percent_map = None; self._percent_lookup = None
        except Exception as e: messagebox.showerror("Image Error", f"Failed to load/process image:\n{e}"); self.img_arr = None; self._percent_map = None; self._percent_lookup = None

        c = data.get("center") or {}; r = data.get("radii") or {}
        img_w = self.img_arr.shape[1] if self.img_arr is not None else 0
        img_h = self.img_arr.shape[0] if self.img_arr is not None else 0
        default_cx = (img_w - 1) / 2.0 if img_w > 0 else 0.0
        default_cy = (img_h - 1) / 2.0 if img_h > 0 else 0.0
        self.overlay = {"center": {"x": float(c.get("x", default_cx)), "y": float(c.get("y", default_cy))},
                        "dead_radius": float(r.get("dead", 0.0)), "search_radius": float(r.get("search", 0.0))}

        pts_data = data.get("points", [])
        if pts_data:
            try:
                yy = [float(p.get("y", 0.0)) for p in pts_data]
                xx = [float(p.get("x", 0.0)) for p in pts_data]
                self.points = np.column_stack([yy, xx]).astype(float)
                # --- ИЗМЕНЕНИЕ: Читаем типы и ПЛОЩАДИ ---
                self.point_types = [p.get("type", "unknown") for p in pts_data]
                self.areas = np.array([float(p.get("area", 0.0)) for p in pts_data], dtype=float) # Читаем area
                # --- КОНЕЦ ИЗМЕНЕНИЯ ---

                # Загружаем или сэмплируем интенсивности
                if self._percent_map is not None:
                    self.values = self._sample_intensities(self.points)
                elif any("intensity" in p for p in pts_data):
                    # --- ИЗМЕНЕНИЕ: Загружаем intensity (это уже %?) ---
                    # Если intensity уже в %, просто берем ее
                    vv_raw = np.array([float(p.get("intensity", 0.0)) for p in pts_data], dtype=float)
                    self.values = vv_raw # Предполагаем, что это уже %
                    # --- КОНЕЦ ИЗМЕНЕНИЯ ---
                else: # Если нет intensity, сэмплируем
                    if self.img_arr is not None: self.values = self._sample_intensities(self.points)
                    else: self.values = np.zeros(len(self.points), dtype=float)

                # --- ИЗМЕНЕНИЕ: Проверяем длину types и areas ---
                if len(self.point_types) != len(self.points):
                     print(f"Warning: Mismatch in point count ({len(self.points)}) and type count ({len(self.point_types)}). Resetting types.")
                     self.point_types = ["unknown"] * len(self.points)
                if len(self.areas) != len(self.points):
                     print(f"Warning: Mismatch in point count ({len(self.points)}) and area count ({len(self.areas)}). Resetting areas.")
                     self.areas = np.zeros(len(self.points), dtype=float) # Заполняем нулями
                # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            except (ValueError, TypeError) as e:
                messagebox.showerror("Data Error", f"Invalid point data: {e}")
                self.points = np.zeros((0, 2), float); self.values = np.zeros((0,), float)
                self.point_types = []; self.areas = np.zeros((0,), float) # Инициализируем пустым списком/массивом
        else:
            self.points = np.zeros((0, 2), float); self.values = np.zeros((0,), float)
            self.point_types = []; self.areas = np.zeros((0,), float) # Инициализируем пустым списком/массивом


    def _save_points_wrapper(self):
        """Wrapper for the save button to handle potential errors."""
        # ... (без изменений) ...
        try:
            self._save_points()
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save points:\n{e}")

    def _save_points(self) -> Path:
        """Saves points and updates 'saed_input.edited.json'."""
        # ... (код определения output_dir без изменений) ...
        output_dir = Path("saed_results")
        if self.controller and hasattr(self.controller, 'launcher'):
            output_dir_str = self.controller.launcher.ent_out.get()
            if output_dir_str:
                try:
                    output_dir = Path(output_dir_str).expanduser().resolve()
                    output_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e: raise OSError(f"Invalid output directory '{output_dir_str}': {e}") from e
            else: raise ValueError("Output folder not specified.")
        else:
            print("Warning: Controller/Launcher not found, using default output 'saed_results'.")
            output_dir.mkdir(parents=True, exist_ok=True)


        pts_list = []
        # --- ИЗМЕНЕНИЕ: Используем текущие values и areas ---
        # НЕ пересчитываем интенсивности, берем те, что есть в self.values
        # current_values = self._sample_intensities(self.points)
        current_values = self.values if self.values is not None else np.zeros(len(self.points))
        current_areas = self.areas if hasattr(self, 'areas') and self.areas is not None else np.zeros(len(self.points))

        for i, (y, x) in enumerate(self.points):
            intensity = float(current_values[i]) if i < len(current_values) else 0.0
            area = int(current_areas[i]) if i < len(current_areas) else 0 # Сохраняем area как int
            pt_type = self.point_types[i] if hasattr(self, 'point_types') and i < len(self.point_types) else "unknown"
            pts_list.append({
                "y": float(y), "x": float(x),
                "intensity": intensity, # Это уже %
                "area": area,           # Добавляем area
                "type": pt_type
            })
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        spots_path = output_dir / "spots.json"
        spots_path.write_text(json.dumps({"points": pts_list}, indent=2), encoding="utf-8")

        abs_image_path = self.image_path.resolve() if self.image_path else None
        overlay_center_data = None
        if self.overlay and self.overlay.get("center"):
            center_data = self.overlay["center"]
            if isinstance(center_data, dict) and "x" in center_data and "y" in center_data:
                overlay_center_data = {"x": float(center_data["x"]), "y": float(center_data["y"])}

        saed_input_edited_data = {
            "image": str(abs_image_path) if abs_image_path else None,
            "preproc_mode": self._preproc_settings.mode if self._preproc_settings else "raw",
            "preproc": self._preproc_settings.to_json() if self._preproc_settings else {"mode": "raw"},
            "center": overlay_center_data,
            "radii": {
                "dead": float(self.overlay.get("dead_radius", 0.0)) if self.overlay else 0.0,
                "search": float(self.overlay.get("search_radius", 0.0)) if self.overlay else 0.0
            },
            "points": pts_list
        }
        edited_path = output_dir / "saed_input.edited.json"
        edited_path.write_text(json.dumps(saed_input_edited_data, ensure_ascii=False, indent=2), encoding="utf-8")

        self._set_status(f"Points saved to {output_dir.name}")
        return spots_path


    # --- Session Save/Load ---
    def get_state(self) -> dict:
        """Returns a serializable dictionary of the editor's state."""
        points_list = self.points.tolist() if self.points is not None else []
        values_list = self.values.tolist() if self.values is not None else []
        # --- ИЗМЕНЕНО: Сохраняем типы и ПЛОЩАДИ ---
        types_list = self.point_types if hasattr(self, 'point_types') else ["unknown"] * len(points_list)
        areas_list = self.areas.tolist() if hasattr(self, 'areas') and self.areas is not None else [0.0] * len(points_list)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        return {
            "image_path": str(self.image_path.resolve()) if self.image_path else None,
            "preproc_settings": self._preproc_settings.to_json() if self._preproc_settings else {"mode":"raw"},
            "points": points_list,
            "values": values_list,
            "point_types": types_list, # Добавили типы
            "areas": areas_list,       # Добавили площади
            "overlay": self.overlay,
            "zoom_val": self.zoom_val,
            "view_cx": self.view_cx,
            "view_cy": self.view_cy,
            "measurement": self._measurement,
        }

    def set_state(self, state: dict):
        """Restores the editor's state from a dictionary."""
        # ... (код загрузки image, preproc, img_arr, percent_map без изменений) ...
        from preproc import PreprocSettings, load_grayscale_with_preproc
        from percentile_utils import compute_percentile_map

        image_path_str = state.get("image_path")
        if not image_path_str:
            self.image_path = None; self.img_arr = None; self._percent_map = None; self._percent_lookup = None
            self.points = np.zeros((0, 2), float); self.values = np.zeros((0,), float)
            self.point_types = []; self.areas = np.zeros((0,), float) # Очищаем типы и площади
            self.overlay = {}; self._preproc_settings = PreprocSettings()
            self.zoom_val = 0; self.view_cx = None; self.view_cy = None
            if hasattr(self, 'zoom_var'): self.zoom_var.set(0)
            self._measurement = None; self._redraw(); self._set_status("Editor cleared."); return

        try:
            self.image_path = Path(image_path_str).resolve()
            if not self.image_path.exists(): raise FileNotFoundError(f"Image not found: {self.image_path}")
            self._preproc_settings = PreprocSettings.from_json(state.get("preproc_settings", {}))
            self.img_arr = load_grayscale_with_preproc(self.image_path, self._preproc_settings)
            self._percent_map, uniq_vals, uniq_perc = compute_percentile_map(self.img_arr)
            self._percent_lookup = (uniq_vals, uniq_perc)

            # Восстанавливаем точки, значения, типы И ПЛОЩАДИ
            self.points = np.array(state.get("points", []), dtype=float)
            saved_values = state.get("values", [])
            if len(saved_values) == len(self.points): self.values = np.array(saved_values, dtype=float)
            else: self.values = self._sample_intensities(self.points) # Пересчитываем если не совпадает

            # --- ИЗМЕНЕНО: Восстанавливаем типы и ПЛОЩАДИ ---
            saved_types = state.get("point_types", [])
            if len(saved_types) == len(self.points): self.point_types = saved_types
            else: self.point_types = ["unknown"] * len(self.points)

            saved_areas = state.get("areas", [])
            if len(saved_areas) == len(self.points): self.areas = np.array(saved_areas, dtype=float)
            else: self.areas = np.zeros(len(self.points), dtype=float) # Заполняем нулями если не совпадает
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            self.overlay = state.get("overlay", {})
            self.zoom_val = state.get("zoom_val", 0)
            self.view_cx = state.get("view_cx"); self.view_cy = state.get("view_cy")
            self._measurement = state.get("measurement")
            if hasattr(self, 'zoom_var'): self.zoom_var.set(self.zoom_val)

            self._cancel_all_interactions() # Сбрасываем все активные режимы
            self.center_dragging = False
            self.rect_start = None

            self._ensure_view_center(); self._redraw(); self._update_zoom_hint()
            self._set_status(f"Restored state for {self.image_path.name}")

        except FileNotFoundError as e: messagebox.showerror("Load Error", str(e)); self.set_state({})
        except Exception as e: messagebox.showerror("Load Error", f"Failed to restore state:\n{e}"); self.set_state({})


    # ---------- Helpers ----------
    def _sample_intensities(self, pts_yx: np.ndarray) -> np.ndarray:
        """Samples intensity values from the percentile map."""
        # --- ИЗМЕНЕНИЕ: Всегда возвращаем процентили ---
        if pts_yx is None or len(pts_yx) == 0: return np.zeros((0,), float)
        # Приоритет - карта процентилей
        if hasattr(self, '_percent_map') and self._percent_map is not None:
            src = self._percent_map; H, W = src.shape[:2]; out = []
            for y, x in pts_yx: yi = max(0, min(H - 1, int(round(y)))); xi = max(0, min(W - 1, int(round(x)))); out.append(float(src[yi, xi]))
            return np.array(out, dtype=float)
        # Если карты нет, но есть img_arr и lookup -> считаем по ним
        elif hasattr(self, 'img_arr') and self.img_arr is not None and hasattr(self, '_percent_lookup') and self._percent_lookup is not None:
             from percentile_utils import map_values_to_percent
             H, W = self.img_arr.shape[:2]; raw_values = []
             for y, x in pts_yx: yi = max(0, min(H - 1, int(round(y)))); xi = max(0, min(W - 1, int(round(x)))); raw_values.append(float(self.img_arr[yi, xi]))
             raw_values_np = np.array(raw_values, dtype=float); return map_values_to_percent(raw_values_np, *self._percent_lookup)
        # В крайнем случае - нули
        else:
            return np.zeros(len(pts_yx), dtype=float)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---


    # ---------- Анализ ----------
    def _start_analysis(self):
        # --- ИЗМЕНЕНИЕ: Передаем area в payload ---
        try: saved_spots_path = self._save_points(); output_dir = saved_spots_path.parent; payload_path = output_dir / "fibo_input.json"
        except (ValueError, OSError, Exception) as e: messagebox.showerror("Save Error", f"Cannot proceed. Failed to save:\n{e}"); return

        try:
            abs_image_path = self.image_path.resolve() if self.image_path else None
            overlay_center_data = None
            if self.overlay and isinstance(self.overlay.get("center"), dict): center_data = self.overlay["center"]; overlay_center_data = {"x": float(center_data["x"]), "y": float(center_data["y"])}
            geo_center_data = None
            if self.img_arr is not None: H, W = self.img_arr.shape[:2]; geo_center_data = {"x": (W - 1) / 2.0, "y": (H - 1) / 2.0}

            # Используем данные, которые УЖЕ сохранены в _save_points
            pts_list = []
            current_values = self.values if self.values is not None else np.zeros(len(self.points))
            current_areas = self.areas if hasattr(self, 'areas') and self.areas is not None else np.zeros(len(self.points))
            for i, (y, x) in enumerate(self.points):
                intensity = float(current_values[i]) if i < len(current_values) else 0.0
                area = int(current_areas[i]) if i < len(current_areas) else 0
                pt_type = self.point_types[i] if hasattr(self, 'point_types') and i < len(self.point_types) else "unknown"
                pts_list.append({"y": float(y), "x": float(x), "intensity": intensity, "area": area, "type": pt_type})

            payload = {"image": str(abs_image_path) if abs_image_path else None,
                       "preproc_mode": self._preproc_settings.mode if self._preproc_settings else "raw",
                       "preproc": self._preproc_settings.to_json() if self._preproc_settings else {"mode": "raw"},
                       "points": pts_list, # Уже содержит area и type
                       "centers": {"geometric": geo_center_data, "overlay": overlay_center_data},
                       "radii": {"dead": float(self.overlay.get("dead_radius", 0.0)) if self.overlay else 0.0,
                                 "search": float(self.overlay.get("search_radius", 0.0)) if self.overlay else 0.0},
                       "spots_json": str(saved_spots_path.resolve())}
            payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._set_status("Prepared data for analysis…")
        except Exception as e: messagebox.showerror("Data Prep Error", f"Failed to create ({payload_path.name}):\n{e}"); return

        if self.controller is not None:
            try: self.controller.open_analysis(payload_path.resolve(), abs_image_path, saved_spots_path.resolve())
            except Exception as e: print(f"Error calling controller.open_analysis: {e}"); messagebox.showerror("Launch Error", f"Failed to switch to analysis tab.\nDetails: {e}")
        else: messagebox.showwarning("Standalone Mode", "Cannot switch to analysis tab.")
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---