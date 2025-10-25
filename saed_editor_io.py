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

        # --- ИЗМЕНЕНИЕ: Пересчитываем углы после загрузки ---
        if self.overlay and self.overlay.get("center"):
             center_data = self.overlay["center"]
             if isinstance(center_data, dict) and "x" in center_data and "y" in center_data:
                 cy, cx = float(center_data["y"]), float(center_data["x"])
                 if len(self.points) > 0:
                     _, self.angles = self._calculate_angles((cy, cx), self.points)
                 else:
                     self.angles = np.zeros((0,), dtype=float)
             else: # Если центра нет или он некорректный
                  self.angles = np.full(len(self.points), np.nan)
        else: # Если нет overlay
             self.angles = np.full(len(self.points), np.nan)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

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
                # --- ИЗМЕНЕНИЕ: Читаем типы (str/int), ПЛОЩАДИ, УГЛЫ ---
                raw_types = [p.get("type", "unknown") for p in pts_data]
                # Преобразуем int ID обратно в int, остальное в str
                self.point_types = []
                for t in raw_types:
                     try: self.point_types.append(int(t)) # Пробуем как int
                     except (ValueError, TypeError): self.point_types.append(str(t)) # Если не int, то str

                self.areas = np.array([float(p.get("area", 0.0)) for p in pts_data], dtype=float) # Читаем area
                # Углы пока не читаем из JSON, они будут пересчитаны
                self.angles = np.full(len(self.points), np.nan)
                # Инициализируем initial_group_ids (пока None)
                self.initial_group_ids = {i: None for i in range(len(self.points))}
                # --- КОНЕЦ ИЗМЕНЕНИЯ ---

                # Загружаем или сэмплируем интенсивности
                if self._percent_map is not None:
                    self.values = self._sample_intensities(self.points)
                elif any("intensity" in p for p in pts_data):
                    vv_raw = np.array([float(p.get("intensity", 0.0)) for p in pts_data], dtype=float)
                    self.values = vv_raw
                else: # Если нет intensity, сэмплируем
                    if self.img_arr is not None: self.values = self._sample_intensities(self.points)
                    else: self.values = np.zeros(len(self.points), dtype=float)

                # --- Проверяем длину types и areas ---
                if len(self.point_types) != len(self.points):
                     print(f"Warning: Mismatch in point count ({len(self.points)}) and type count ({len(self.point_types)}). Resetting types.")
                     self.point_types = ["unknown"] * len(self.points)
                if len(self.areas) != len(self.points):
                     print(f"Warning: Mismatch in point count ({len(self.points)}) and area count ({len(self.areas)}). Resetting areas.")
                     self.areas = np.zeros(len(self.points), dtype=float)
                # --- КОНЕЦ Проверки ---

            except (ValueError, TypeError) as e:
                messagebox.showerror("Data Error", f"Invalid point data: {e}")
                self.points = np.zeros((0, 2), float); self.values = np.zeros((0,), float)
                self.point_types = []; self.areas = np.zeros((0,), float); self.angles = np.zeros((0,), float)
                self.initial_group_ids = {}
        else:
            self.points = np.zeros((0, 2), float); self.values = np.zeros((0,), float)
            self.point_types = []; self.areas = np.zeros((0,), float); self.angles = np.zeros((0,), float)
            self.initial_group_ids = {}


    def _save_points_wrapper(self):
        """Wrapper for the save button to handle potential errors."""
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
        # --- ИЗМЕНЕНИЕ: Используем текущие values, areas, types (str/int), angles ---
        current_values = self.values if self.values is not None else np.zeros(len(self.points))
        current_areas = self.areas if hasattr(self, 'areas') and self.areas is not None else np.zeros(len(self.points))
        current_angles = self.angles if hasattr(self, 'angles') and self.angles is not None else np.full(len(self.points), np.nan)

        for i, (y, x) in enumerate(self.points):
            intensity = float(current_values[i]) if i < len(current_values) else 0.0
            area = int(current_areas[i]) if i < len(current_areas) else 0 # Сохраняем area как int
            pt_type = self.point_types[i] if hasattr(self, 'point_types') and i < len(self.point_types) else "unknown" # Сохраняем как есть (str/int)
            angle = float(current_angles[i]) if i < len(current_angles) and not np.isnan(current_angles[i]) else None # Сохраняем угол, если есть

            point_data = {
                "y": float(y), "x": float(x),
                "intensity": intensity,
                "area": area,
                "type": pt_type
            }
            if angle is not None:
                point_data["angle"] = angle # Добавляем угол, если он рассчитан

            pts_list.append(point_data)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        spots_path = output_dir / "spots.json"
        # --- ИЗМЕНЕНИЕ: Сериализатор для NumPy типов ---
        def default_serializer(obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
        spots_path.write_text(json.dumps({"points": pts_list}, indent=2, default=default_serializer), encoding="utf-8")
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

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
        edited_path.write_text(json.dumps(saed_input_edited_data, ensure_ascii=False, indent=2, default=default_serializer), encoding="utf-8")

        self._set_status(f"Points saved to {output_dir.name}")
        return spots_path


    # --- Session Save/Load ---
    def get_state(self) -> dict:
        """Returns a serializable dictionary of the editor's state."""
        points_list = self.points.tolist() if self.points is not None else []
        values_list = self.values.tolist() if self.values is not None else []
        # --- ИЗМЕНЕНО: Сохраняем типы (str/int), ПЛОЩАДИ, УГЛЫ, Initial IDs ---
        types_list = list(self.point_types) if hasattr(self, 'point_types') else ["unknown"] * len(points_list) # Сохраняем как есть
        areas_list = self.areas.tolist() if hasattr(self, 'areas') and self.areas is not None else [0.0] * len(points_list)
        angles_list = self.angles.tolist() if hasattr(self, 'angles') and self.angles is not None else [None] * len(points_list) # None для NaN
        initial_ids_dict = dict(self.initial_group_ids) if hasattr(self, 'initial_group_ids') else {}
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        return {
            "image_path": str(self.image_path.resolve()) if self.image_path else None,
            "preproc_settings": self._preproc_settings.to_json() if self._preproc_settings else {"mode":"raw"},
            "points": points_list,
            "values": values_list,
            "point_types": types_list, # Сохраняем типы (str/int)
            "areas": areas_list,       # Сохраняем площади
            "angles": angles_list,     # Сохраняем углы
            "initial_group_ids": initial_ids_dict, # Сохраняем исходные ID
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
            self.point_types = []; self.areas = np.zeros((0,), float); self.angles = np.zeros((0,), float)
            self.initial_group_ids = {}
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

            # Восстанавливаем точки, значения, типы, ПЛОЩАДИ, УГЛЫ, Initial IDs
            self.points = np.array(state.get("points", []), dtype=float)
            n_points = len(self.points) # Количество точек

            saved_values = state.get("values", [])
            if len(saved_values) == n_points: self.values = np.array(saved_values, dtype=float)
            else: self.values = self._sample_intensities(self.points)

            # --- ИЗМЕНЕНО: Восстанавливаем типы (str/int), ПЛОЩАДИ, УГЛЫ, Initial IDs ---
            saved_types = state.get("point_types", [])
            if len(saved_types) == n_points: self.point_types = list(saved_types)
            else: self.point_types = ["unknown"] * n_points

            saved_areas = state.get("areas", [])
            if len(saved_areas) == n_points: self.areas = np.array(saved_areas, dtype=float)
            else: self.areas = np.zeros(n_points, dtype=float)

            saved_angles = state.get("angles", [])
            if len(saved_angles) == n_points:
                 # Заменяем None на np.nan
                self.angles = np.array([a if a is not None else np.nan for a in saved_angles], dtype=float)
            else: self.angles = np.full(n_points, np.nan)

            saved_initial_ids = state.get("initial_group_ids", {})
            # Преобразуем ключи обратно в int, если они сохранились как строки
            self.initial_group_ids = {int(k): v for k, v in saved_initial_ids.items()}
            # Добавим None для точек, которых нет в словаре
            for i in range(n_points):
                 if i not in self.initial_group_ids: self.initial_group_ids[i] = None
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            self.overlay = state.get("overlay", {})
            self.zoom_val = state.get("zoom_val", 0)
            self.view_cx = state.get("view_cx"); self.view_cy = state.get("view_cy")
            self._measurement = state.get("measurement")
            if hasattr(self, 'zoom_var'): self.zoom_var.set(self.zoom_val)

            # --- ИЗМЕНЕНИЕ: Пересчитываем углы, если они NaN ---
            if np.isnan(self.angles).any() and self.overlay and self.overlay.get("center"):
                 center_data = self.overlay["center"]
                 if isinstance(center_data, dict) and "x" in center_data and "y" in center_data:
                     cy, cx = float(center_data["y"]), float(center_data["x"])
                     if n_points > 0:
                         _, new_angles = self._calculate_angles((cy, cx), self.points)
                         self.angles = np.where(np.isnan(self.angles), new_angles, self.angles) # Заменяем только NaN
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            self._cancel_all_interactions()
            self.center_dragging = False
            self.rect_start = None

            self._ensure_view_center(); self._redraw(); self._update_zoom_hint()
            self._set_status(f"Restored state for {self.image_path.name}")

        except FileNotFoundError as e: messagebox.showerror("Load Error", str(e)); self.set_state({})
        except Exception as e: messagebox.showerror("Load Error", f"Failed to restore state:\n{e}"); self.set_state({})


    # ---------- Helpers ----------
    def _sample_intensities(self, pts_yx: np.ndarray) -> np.ndarray:
        """Samples intensity values from the percentile map."""
        # --- (Без изменений с прошлой версии) ---
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
        # --- КОНЕЦ ---

    # --- НОВЫЙ Хелпер: Расчет углов ---
    def _calculate_angles(self, center_yx: Tuple[float, float], pts_yx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates radii and angles [0, 360) for points relative to center."""
        return pol_from(center_yx, pts_yx)
    # --- КОНЕЦ Хелпера ---


    # ---------- Анализ ----------
    def _start_analysis(self):
        # --- ИЗМЕНЕНИЕ: Передаем area и type (str/int) и angle в payload ---
        try: saved_spots_path = self._save_points(); output_dir = saved_spots_path.parent; payload_path = output_dir / "fibo_input.json"
        except (ValueError, OSError, Exception) as e: messagebox.showerror("Save Error", f"Cannot proceed. Failed to save:\n{e}"); return

        try:
            abs_image_path = self.image_path.resolve() if self.image_path else None
            overlay_center_data = None
            if self.overlay and isinstance(self.overlay.get("center"), dict): center_data = self.overlay["center"]; overlay_center_data = {"x": float(center_data["x"]), "y": float(center_data["y"])}
            geo_center_data = None
            if self.img_arr is not None: H, W = self.img_arr.shape[:2]; geo_center_data = {"x": (W - 1) / 2.0, "y": (H - 1) / 2.0}

            # Используем данные, которые УЖЕ сохранены в _save_points
            # Открываем spots.json, чтобы получить pts_list
            if not saved_spots_path.exists():
                 raise FileNotFoundError("spots.json not found after saving.")
            spots_data = json.loads(saved_spots_path.read_text(encoding="utf-8"))
            pts_list = spots_data.get("points", [])

            payload = {"image": str(abs_image_path) if abs_image_path else None,
                       "preproc_mode": self._preproc_settings.mode if self._preproc_settings else "raw",
                       "preproc": self._preproc_settings.to_json() if self._preproc_settings else {"mode": "raw"},
                       "points": pts_list, # Уже содержит area, type (str/int), angle
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