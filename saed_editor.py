#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAED Editor + Analysis
----------------------
• Middle mouse button (MMB) tooltip on a point: shows x, y, intensity; closes with Esc and any actions.

Removed according to the spec:
• Moving regular points.
• Any mentions/functions of panning or moving the window with the middle mouse button.

The remaining editor functionality is preserved.
"""
import sys, json, subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from percentile_utils import compute_percentile_map, map_values_to_percent
from preproc import PreprocSettings, load_grayscale_with_preproc
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
        return np.array([]), np.zeros(0, dtype=int), ([], [])  # Return empty int array for labels
    hist, edges = np.histogram(radii, bins=60)
    centers = (edges[:-1] + edges[1:]) / 2
    # Find peaks, require a minimum height relative to max peak?
    # prominence=3 might be too small for noisy data
    if hist.max() > 0:
        prominence = max(3, hist.max() * 0.05)  # Adjust prominence based on data
    else:
        prominence = 3
    pk, _ = find_peaks(hist, prominence=prominence)

    ring_centers = centers[pk]
    if len(ring_centers) == 0:
        # Fallback: maybe use KMeans or simple thresholding if find_peaks fails?
        # For now, return empty if no clear peaks found
        return np.array([]), np.zeros_like(radii, dtype=int), (hist.tolist(), edges.tolist())  # Return numpy int array

    # Assign each point to the nearest ring center
    labels = np.argmin(np.abs(radii[:, None] - ring_centers[None, :]), axis=1)
    return ring_centers, labels, (hist.tolist(), edges.tolist())  # Convert hist/edges for JSON


def symmetry_scores(angles, radii, ring_means, top_rings=3):
    out = {}
    if not ring_means or len(ring_means) == 0:  # Check if list is empty
        return out

    # Ensure top_rings doesn't exceed available rings
    effective_top_rings = min(top_rings, len(ring_means))
    if effective_top_rings == 0: return out  # No rings to analyze

    # Determine radius cutoff based on available rings
    idx = effective_top_rings - 1
    maxR = ring_means[idx] * 1.15  # Use 1.15 multiplier as before

    # Filter points within the cutoff radius
    mask = radii <= maxR
    ang_sel = angles[mask]
    if len(ang_sel) == 0: return out  # No points selected

    # Calculate scores for different folds
    for k in [4, 6, 8, 10, 12]:
        period = 360.0 / k
        # Calculate phases relative to the period
        phases_deg = (ang_sel % period) * k
        phases_rad = np.deg2rad(phases_deg)
        # Calculate mean cosine and sine
        C = np.cos(phases_rad).mean();
        S = np.sin(phases_rad).mean()
        # Score is the magnitude of the mean vector (length of resultant vector)
        out[f"{k}-fold"] = float(np.hypot(C, S))

    return out


class PointEditor(tk.Frame):
    def __init__(self, master: tk.Misc, controller=None,
                 input_json: str | None = None, auto_load: bool = True):
        super().__init__(master)
        self.controller = controller
        # данные
        self.points = np.zeros((0, 2), float)  # [y, x]
        self.values = np.zeros((0,), float)  # интенсивности (параллельно points)
        self.rect_start = None
        self.rect_artist = None
        self.overlay = None  # {center:{x,y}, dead_radius, search_radius}
        self.image_path: Optional[Path] = None
        self.img_arr: Optional[np.ndarray] = None
        self._percent_map: Optional[np.ndarray] = None
        self._percent_lookup: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._preproc_settings: PreprocSettings = PreprocSettings(mode="raw")

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
        self._tooltip = None  # matplotlib.text.Annotation
        self._tooltip_idx = None

        # Измерение расстояний между точками
        self._measure_active = False
        self._measure_start_idx: Optional[int] = None
        self._measure_start_point: Optional[tuple[float, float]] = None
        self._measure_preview_end: Optional[tuple[float, float]] = None
        self._measure_preview_artist = None
        self._measure_line_artist = None
        self._measure_annotation = None
        self._measurement: Optional[dict[str, object]] = None

        # Объединение точек по радиусу от выбранной
        self._merge_seed_idx: Optional[int] = None
        self._merge_seed_origin: Optional[tuple[float, float]] = None
        self._last_cursor_pos: Optional[tuple[float, float]] = None

        self._build_ui()

        # первичная загрузка
        if auto_load and input_json:
            try:
                self.load_input_json(Path(input_json), push_undo=False)
            except FileNotFoundError:
                self._set_status(f"Error: Input JSON not found at {input_json}")
                # Initialize with empty state if file not found
                self._ensure_view_center()  # Still try to set a default view
                self._redraw()

        else:
            self._ensure_view_center()
            self._redraw()

    # ---------- UI ----------
    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        side_panel = ttk.Frame(self, padding=(16, 16, 12, 16))
        side_panel.grid(row=0, column=0, sticky="ns")
        side_panel.columnconfigure(0, weight=1)

        controls = ttk.Frame(side_panel)
        controls.pack(side=tk.TOP, fill=tk.X)

        header_row = ttk.Frame(controls)
        header_row.pack(fill=tk.X)
        header_row.columnconfigure(0, weight=1)  # Allow help button to align right

        help_button = ttk.Button(header_row, text="?", width=3, command=self._toggle_help, style="Toolbutton")
        help_button.pack(side=tk.RIGHT)

        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 10))

        zoom_group = ttk.LabelFrame(controls, text="Scale", padding=(12, 8, 12, 10))
        zoom_group.pack(fill=tk.X)
        self.zoom_var = tk.DoubleVar(value=self.zoom_val)
        self.zoom_scale = ttk.Scale(zoom_group, from_=0, to=100, variable=self.zoom_var, command=self._on_zoom_change)
        self.zoom_scale.pack(fill=tk.X, padx=4, pady=(0, 6))
        self.zoom_hint = ttk.Label(zoom_group, anchor="w")
        self.zoom_hint.pack(fill=tk.X, padx=4)
        self.zoom_scale.set(self.zoom_val)

        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 10))

        file_group = ttk.Frame(controls)
        file_group.pack(fill=tk.X)
        ttk.Button(file_group, text="Open JSON…", command=self._open_json).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(file_group, text="Save", command=self._save_points_wrapper).pack(side=tk.LEFT,
                                                                                    padx=(0, 6))  # Changed command

        analysis_group = ttk.Frame(controls)
        analysis_group.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(analysis_group, text="Start analysis", command=self._start_analysis).pack(side=tk.LEFT, padx=(0, 6))

        self.help_panel = ttk.LabelFrame(side_panel, text="Hints", padding=(16, 12, 16, 12))
        help_text = (
            "Ctrl+Z / Ctrl+Y — Undo/Redo actions\n"
            "Ctrl+S — Save current session\n"
            "Mouse Wheel — Zoom in/out\n\n"
            "Left mouse button on empty area — add a point\n"
            "Left mouse button on the center — drag the center\n"
            "Right mouse button on a point — delete\n"
            "Middle mouse button on a point — show info\n"
            "Left mouse button on a point — select; move and press Enter to merge\n"
            "Hold left mouse button from point to point — measure distance\n"
            "Shift + drag — rectangular range deletion"
        )
        ttk.Label(self.help_panel, text=help_text, justify="left", wraplength=280).pack(
            fill=tk.X)  # Adjusted wraplength
        self._help_visible = False

        self._side_spacer = ttk.Frame(side_panel)
        self._side_spacer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        status_frame = ttk.Frame(side_panel, padding=(0, 0, 0, 0))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))
        self.status_label = ttk.Label(status_frame, anchor="w", justify="left")
        self.status_label.pack(fill=tk.X)

        def _status_wrap(event, label=self.status_label):
            if not label.winfo_exists():
                return
            new_wrap = max(int(event.width) - 8, 120)
            current_wrap = int(label.cget("wraplength") or 0)
            if new_wrap != current_wrap:
                label.configure(wraplength=new_wrap)

        self.status_label.bind("<Configure>", _status_wrap)

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

        self.bind_all('<Control-z>', self._undo_btn)
        self.bind_all('<Control-y>', self._redo_btn)

        self._default_status = "Mode: point editor"
        self._status_message = ""
        self._update_zoom_hint()
        self._set_status(self._default_status)

    def _toggle_help(self):
        self._help_visible = not self._help_visible
        if self._help_visible:
            self.help_panel.pack(side=tk.TOP, fill=tk.X, pady=(12, 8), before=self._side_spacer)
            self._set_status("Detailed hints expanded")
        else:
            self.help_panel.pack_forget()
            self._set_status(self._default_status)  # Restore default status when hiding

    def _update_zoom_hint(self):
        if hasattr(self, "zoom_hint"):
            value = int(round(self.zoom_var.get())) if hasattr(self, "zoom_var") else int(round(self.zoom_val))
            self.zoom_hint.configure(text=f"Current zoom: {value}% (0 = full frame)")

    def _set_status(self, text: str):
        self._status_message = text
        if hasattr(self, "status_label") and self.status_label.winfo_exists():  # Check if widget exists
            self.status_label.configure(text=text)
        # Propagate status up to the main controller if it exists
        if self.controller is not None and hasattr(self.controller, "set_status"):
            try:
                self.controller.set_status(f"Editor: {text}")  # Add context
            except Exception:
                pass  # Ignore if controller is gone

    # ---------- IO ----------
    def _open_json(self):
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
        if not path.exists():
            raise FileNotFoundError(f"Input JSON file not found: {path}")

        if push_undo:
            self._push_undo()  # Save state *before* loading new data

        self._load_input_json(path)  # Load the data
        self._clear_tooltip()  # Clear any popup info

        if reset_view:
            self.view_cx = None
            self.view_cy = None
        self._ensure_view_center()  # Set view center based on loaded data

        # Clear undo/redo history after a new file load
        self._undo.clear()
        self._redo.clear()

        self._redraw()  # Redraw canvas
        self._update_zoom_hint()  # Update zoom label
        self._set_status(f"Loaded: {path.name}")

    def _load_input_json(self, path: Path):
        """Internal method to load data from the JSON file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {path.name}: {e}") from e
        except Exception as e:
            raise IOError(f"Failed to read JSON file {path.name}: {e}") from e

        self._percent_map = None
        self._percent_lookup = None

        img_path_str = data.get("image")
        if not img_path_str:
            raise ValueError("The JSON is missing the required 'image' field.")

        # --- Resolve image path ---
        img_p = Path(img_path_str)
        if not img_p.is_absolute():
            # Resolve relative to the JSON file's directory
            self.image_path = (path.parent / img_p).resolve()
        else:
            self.image_path = img_p.resolve()

        if not self.image_path.exists():
            raise FileNotFoundError(f"Image file specified in JSON not found: {self.image_path}")

        # --- Load Preprocessing Settings ---
        fallback_mode = data.get("preproc_mode")
        if not isinstance(fallback_mode, str): fallback_mode = None  # Handle missing or wrong type
        self._preproc_settings = PreprocSettings.from_json(
            data.get("preproc"), fallback_mode=fallback_mode
        )

        # --- Load and Process Image ---
        try:
            self.img_arr = load_grayscale_with_preproc(self.image_path, self._preproc_settings)
            # Compute percentile map for intensity lookups
            self._percent_map, uniq_vals, uniq_perc = compute_percentile_map(self.img_arr)
            self._percent_lookup = (uniq_vals, uniq_perc)
        except RuntimeError as cv_err:  # Catch OpenCV dependency error
            messagebox.showerror("Dependency Error", str(cv_err))
            self.img_arr = None  # Continue without image array if OpenCV fails
            self._percent_map = None
            self._percent_lookup = None
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load or process the image:\n{e}")
            self.img_arr = None  # Continue without image array on other errors
            self._percent_map = None
            self._percent_lookup = None

        # --- Load Overlay (Center and Radii) ---
        c = data.get("center") or {}
        r = data.get("radii") or {}
        # Provide defaults based on image size if available, else use 0
        img_w = self.img_arr.shape[1] if self.img_arr is not None else 0
        img_h = self.img_arr.shape[0] if self.img_arr is not None else 0
        default_cx = (img_w - 1) / 2.0 if img_w > 0 else 0.0
        default_cy = (img_h - 1) / 2.0 if img_h > 0 else 0.0

        self.overlay = {
            "center": {
                "x": float(c.get("x", default_cx)),
                "y": float(c.get("y", default_cy))
            },
            "dead_radius": float(r.get("dead", 0.0)),
            "search_radius": float(r.get("search", 0.0)),
        }

        # --- Load Points ---
        pts_data = data.get("points", [])
        if pts_data:
            try:
                yy = [float(p.get("y", 0.0)) for p in pts_data]
                xx = [float(p.get("x", 0.0)) for p in pts_data]
                self.points = np.column_stack([yy, xx]).astype(float)

                # Assign or sample intensities
                if self._percent_map is not None:
                    # Sample intensities directly from the loaded percentile map
                    self.values = self._sample_intensities(self.points)
                elif any("intensity" in p for p in pts_data):
                    # Use intensities from JSON if percentile map failed but intensities exist
                    vv_raw = np.array([float(p.get("intensity", 0.0)) for p in pts_data], dtype=float)
                    # Attempt to map to percentiles if lookup exists (from failed compute_percentile_map)
                    if self._percent_lookup is not None:
                        self.values = map_values_to_percent(vv_raw, *self._percent_lookup)
                    else:
                        # Use raw intensities as fallback if no percentile info available
                        self.values = vv_raw
                else:
                    # Fallback: Sample from img_arr if possible, else zeros
                    if self.img_arr is not None:
                        self.values = self._sample_intensities(self.points)  # Will use raw values if no lookup
                    else:
                        self.values = np.zeros(len(self.points), dtype=float)

            except (ValueError, TypeError) as e:
                messagebox.showerror("Data Error", f"Invalid point data in JSON: {e}")
                self.points = np.zeros((0, 2), float)
                self.values = np.zeros((0,), float)

        else:  # No points in JSON
            self.points = np.zeros((0, 2), float)
            self.values = np.zeros((0,), float)

    def _save_points_wrapper(self):
        """Wrapper for the save button to handle potential errors."""
        try:
            self._save_points()
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save points:\n{e}")

    def _save_points(self) -> Path:
        """
        Saves points and updates the 'saed_input.edited.json' file
        in the **output folder** defined in the Launcher tab.
        Also saves a separate 'spots.json'.
        Returns the path to 'spots.json'.
        """
        # --- Determine Output Directory ---
        output_dir = Path("saed_results")  # Default if controller/launcher missing
        if self.controller and hasattr(self.controller, 'launcher'):
            output_dir_str = self.controller.launcher.ent_out.get()
            if output_dir_str:
                try:
                    # Resolve and create the directory
                    output_dir = Path(output_dir_str).expanduser().resolve()
                    output_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise OSError(f"Invalid or inaccessible output directory '{output_dir_str}': {e}") from e
            else:
                raise ValueError("Output folder is not specified in the Launcher tab.")
        else:
            print("Warning: Controller or Launcher not found, using default output 'saed_results'.")
            output_dir.mkdir(parents=True, exist_ok=True)

        # --- Prepare Points Data ---
        pts_list = []
        # Recalculate values just before saving to ensure they match current points
        current_values = self._sample_intensities(self.points)
        for i, (y, x) in enumerate(self.points):
            # Ensure index is valid for current_values
            intensity = float(current_values[i]) if i < len(current_values) else 0.0
            pts_list.append({"y": float(y), "x": float(x), "intensity": intensity})

        # --- Save spots.json ---
        spots_path = output_dir / "spots.json"
        spots_path.write_text(json.dumps({"points": pts_list}, indent=2), encoding="utf-8")

        # --- Update and Save saed_input.edited.json ---
        abs_image_path = self.image_path.resolve() if self.image_path else None

        # Ensure overlay center is serializable
        overlay_center_data = None
        if self.overlay and self.overlay.get("center"):
            center_data = self.overlay["center"]
            if isinstance(center_data, dict) and "x" in center_data and "y" in center_data:
                overlay_center_data = {"x": float(center_data["x"]), "y": float(center_data["y"])}

        saed_input_edited_data = {
            "image": str(abs_image_path) if abs_image_path else None,
            "preproc_mode": self._preproc_settings.mode,
            "preproc": self._preproc_settings.to_json(),
            "center": overlay_center_data,  # Use the cleaned-up center data
            "radii": {
                "dead": float(self.overlay.get("dead_radius", 0.0)) if self.overlay else 0.0,
                "search": float(self.overlay.get("search_radius", 0.0)) if self.overlay else 0.0
            },
            "points": pts_list  # Use the latest points list
        }
        edited_path = output_dir / "saed_input.edited.json"
        edited_path.write_text(json.dumps(saed_input_edited_data, ensure_ascii=False, indent=2), encoding="utf-8")

        self._set_status(f"Points saved to {output_dir.name}")
        return spots_path  # Return path to spots.json

    # --- NEW: Session Save/Load ---

    def get_state(self) -> Dict[str, Any]:
        """Returns a serializable dictionary of the editor's state."""
        # Convert numpy arrays to lists for JSON
        points_list = self.points.tolist() if self.points is not None else []
        values_list = self.values.tolist() if self.values is not None else []

        return {
            "image_path": str(self.image_path.resolve()) if self.image_path else None,
            "preproc_settings": self._preproc_settings.to_json(),
            "points": points_list,
            "values": values_list,  # Save current values
            "overlay": self.overlay,
            "zoom_val": self.zoom_val,
            "view_cx": self.view_cx,
            "view_cy": self.view_cy,
            # Add measurement state if needed
            "measurement": self._measurement,
        }

    def set_state(self, state: Dict[str, Any]):
        """Restores the editor's state from a dictionary."""
        image_path_str = state.get("image_path")
        if not image_path_str:
            # Clear state if no image path
            self.image_path = None
            self.img_arr = None
            self._percent_map = None
            self._percent_lookup = None
            self.points = np.zeros((0, 2), float)
            self.values = np.zeros((0,), float)
            self.overlay = {}
            self._preproc_settings = PreprocSettings()  # Reset preproc
            self.zoom_val = 0
            self.view_cx = None
            self.view_cy = None
            if hasattr(self, 'zoom_var'): self.zoom_var.set(0)
            self._measurement = None
            self._redraw()
            self._set_status("Editor cleared (no image path in session).")
            return

        try:
            self.image_path = Path(image_path_str).resolve()  # Ensure absolute path
            if not self.image_path.exists():
                raise FileNotFoundError(f"Image from session not found: {self.image_path}")

            self._preproc_settings = PreprocSettings.from_json(state.get("preproc_settings", {}))

            # Load image and compute percentile map
            self.img_arr = load_grayscale_with_preproc(self.image_path, self._preproc_settings)
            self._percent_map, uniq_vals, uniq_perc = compute_percentile_map(self.img_arr)
            self._percent_lookup = (uniq_vals, uniq_perc)

            # Restore points and values
            self.points = np.array(state.get("points", []), dtype=float)
            # IMPORTANT: Use the saved values, don't re-sample unless necessary
            saved_values = state.get("values", [])
            if len(saved_values) == len(self.points):
                self.values = np.array(saved_values, dtype=float)
            else:
                # Resample only if saved values don't match point count (error case)
                print("Warning: Mismatch between saved points and values count. Re-sampling intensities.")
                self.values = self._sample_intensities(self.points)

            # Restore overlay, zoom, view, measurement
            self.overlay = state.get("overlay", {})
            self.zoom_val = state.get("zoom_val", 0)
            self.view_cx = state.get("view_cx")
            self.view_cy = state.get("view_cy")
            self._measurement = state.get("measurement")  # Restore measurement

            if hasattr(self, 'zoom_var'): self.zoom_var.set(self.zoom_val)

            # Clear any interactive states
            self._clear_merge_seed()
            self._cancel_measurement_preview()
            self.center_dragging = False
            self.rect_start = None

            # Final UI updates
            self._ensure_view_center()
            self._redraw()
            self._update_zoom_hint()
            self._set_status(f"Restored editor state for {self.image_path.name}")

        except FileNotFoundError as e:
            messagebox.showerror("Editor Load Error", str(e))
            self.set_state({})  # Clear state on file not found
        except Exception as e:
            messagebox.showerror("Editor Load Error", f"Failed to restore editor state:\n{e}")
            self.set_state({})  # Clear state on other errors

    # ---------- Helpers ----------
    def _sample_intensities(self, pts_yx: np.ndarray) -> np.ndarray:
        """Samples intensity values, preferring percentile map, then raw image with lookup, then raw, then zeros."""
        if pts_yx is None or len(pts_yx) == 0:
            return np.zeros((0,), float)

        # 1. Try Percentile Map directly
        if self._percent_map is not None:
            src = self._percent_map
            H, W = src.shape[:2]
            out = []
            for y, x in pts_yx:
                yi = max(0, min(H - 1, int(round(y))))
                xi = max(0, min(W - 1, int(round(x))))
                out.append(float(src[yi, xi]))
            return np.array(out, dtype=float)

        # 2. Try Raw Image with Percentile Lookup
        if self.img_arr is not None and self._percent_lookup is not None:
            H, W = self.img_arr.shape[:2]
            raw_values = []
            for y, x in pts_yx:
                yi = max(0, min(H - 1, int(round(y))))
                xi = max(0, min(W - 1, int(round(x))))
                raw_values.append(float(self.img_arr[yi, xi]))
            raw_values_np = np.array(raw_values, dtype=float)
            return map_values_to_percent(raw_values_np, *self._percent_lookup)

        # 3. Try Raw Image directly (no percentile conversion)
        if self.img_arr is not None:
            H, W = self.img_arr.shape[:2]
            raw_values = []
            for y, x in pts_yx:
                yi = max(0, min(H - 1, int(round(y))))
                xi = max(0, min(W - 1, int(round(x))))
                raw_values.append(float(self.img_arr[yi, xi]))
            return np.array(raw_values, dtype=float)

        # 4. Fallback to zeros if no image data available
        return np.zeros(len(pts_yx), dtype=float)

    def _img_xy(self, e):
        # Convert matplotlib event coordinates (x, y) to image coordinates (y, x)
        return None if (e.xdata is None or e.ydata is None) else (e.ydata, e.xdata)

    def _near_idx(self, y, x, pix_tol=8):
        # Find index of the point closest to (y, x) within tolerance
        if self.points is None or len(self.points) == 0: return None
        # Calculate squared Euclidean distances for efficiency
        dist_sq = (self.points[:, 0] - y) ** 2 + (self.points[:, 1] - x) ** 2
        i = int(np.argmin(dist_sq))  # Index of the minimum distance point
        # Check if the minimum distance is within the tolerance
        return i if dist_sq[i] <= pix_tol ** 2 else None

    def _center_hit(self, y, x):
        # Check if the click is near the defined center overlay
        if not (self.overlay and self.overlay.get("center")): return False
        center_data = self.overlay["center"]
        if not isinstance(center_data, dict): return False  # Ensure center is a dict

        cy = float(center_data.get("y", 0.0))
        cx = float(center_data.get("x", 0.0))
        dist_sq = (y - cy) ** 2 + (x - cx) ** 2
        return dist_sq <= self._center_hit_radius ** 2

    def _apply_center_filters(self):
        """Removes points that end up in the dead zone or outside the search radius after moving the center."""
        if self.points is None or len(self.points) == 0: return  # No points to filter
        if not (self.overlay and self.overlay.get("center")): return  # No center defined

        center_data = self.overlay["center"]
        if not isinstance(center_data, dict): return

        cy = float(center_data.get("y", 0.0))
        cx = float(center_data.get("x", 0.0))
        dead = float(self.overlay.get("dead_radius", 0.0))
        sr = float(self.overlay.get("search_radius", 0.0))

        if dead <= 0 and sr <= 0: return  # No filtering needed if radii are zero or less

        # Calculate distances from the new center
        r = np.hypot(self.points[:, 1] - cx, self.points[:, 0] - cy)

        # Create mask
        mask = np.ones(len(self.points), dtype=bool)
        if dead > 0: mask &= (r >= dead)
        if sr > 0: mask &= (r <= sr)

        # --- Update Merge Seed Index ---
        # If a point was selected for merging, we need to find its new index
        # *after* filtering, or clear the selection if it was removed.
        new_merge_seed_idx = None
        if self._merge_seed_idx is not None:
            original_selected_point = self._merge_seed_origin  # Use the *original* position
            if original_selected_point:
                # Check if the originally selected point is still present after filtering
                kept_indices = np.where(mask)[0]
                original_index_in_old_array = self._merge_seed_idx

                if original_index_in_old_array in kept_indices:
                    # Find the *new* index corresponding to the original one
                    try:
                        # Map old index to its position in the filtered array
                        new_merge_seed_idx = np.where(kept_indices == original_index_in_old_array)[0][0]
                    except IndexError:
                        # Should not happen if check passed, but handle defensively
                        new_merge_seed_idx = None

        # Apply the mask to points and values
        self.points = self.points[mask]
        # Ensure values array is also filtered correctly
        if self.values is not None and len(self.values) == len(mask):
            self.values = self.values[mask]
        else:
            # If values array was mismatched or None, re-sample intensities
            self.values = self._sample_intensities(self.points)

        # Update the merge seed index and origin *after* filtering
        self._merge_seed_idx = new_merge_seed_idx
        if self._merge_seed_idx is not None:
            # Update origin to the current position of the (potentially shifted) point
            current_y, current_x = self.points[self._merge_seed_idx]
            self._merge_seed_origin = (float(current_y), float(current_x))
        else:
            self._merge_seed_origin = None  # Clear origin if seed was removed or invalid

    # ---------- Объединение точек ----------
    def _clear_merge_seed(self, *, keep_status: bool = False) -> bool:
        cleared = self._merge_seed_idx is not None
        self._merge_seed_idx = None
        self._merge_seed_origin = None
        if cleared and not keep_status:
            self._set_status(self._default_status)
        return cleared

    def _select_merge_seed(self, idx: int) -> None:
        if self.points is None or idx < 0 or idx >= len(self.points):
            self._clear_merge_seed()
            return
        self._merge_seed_idx = int(idx)
        y, x = self.points[idx]
        self._merge_seed_origin = (float(y), float(x))
        self._set_status(
            "Point selected for merging. Move cursor to define radius and press Enter."
        )

    def _merge_selected_with_radius(self) -> bool:
        if self._merge_seed_idx is None: return False
        if self.points is None or len(self.points) == 0:
            self._clear_merge_seed();
            return False

        idx = int(self._merge_seed_idx)
        if idx < 0 or idx >= len(self.points):  # Check bounds again
            self._clear_merge_seed()
            self._set_status("Selected point became invalid. Please select again.");
            return False

        if self._last_cursor_pos is None:
            self._set_status("Move cursor inside image to set merge radius, then press Enter.");
            return False

        # Current position of the seed point
        base_cur_y, base_cur_x = map(float, self.points[idx])
        # Cursor position defines the radius
        cur_y, cur_x = self._last_cursor_pos
        radius_sq = (cur_x - base_cur_x) ** 2 + (cur_y - base_cur_y) ** 2
        if radius_sq < 1e-6:  # Radius too small
            self._set_status("Radius is too small. Move cursor further and press Enter.");
            return False

        # Use the stored origin for distance calculation to handle potential center drag effects
        origin = self._merge_seed_origin or (base_cur_y, base_cur_x)
        origin_y, origin_x = origin

        # Find points within the radius using squared distances for efficiency
        dist_sq = (self.points[:, 1] - origin_x) ** 2 + (self.points[:, 0] - origin_y) ** 2
        candidate_indices = np.where(dist_sq <= radius_sq + 1e-6)[0].tolist()

        # Ensure the seed index itself is included if somehow missed
        if idx not in candidate_indices:
            candidate_indices.append(idx)

        if len(candidate_indices) <= 1:
            self._set_status("No other points found within the selected radius.");
            return False

        # --- Proceed with merging ---
        self._push_undo()  # Save state before modification
        self._cancel_measurement_preview()
        self._clear_measurement_result()

        # Ensure values array is up-to-date
        if self.values is None or len(self.values) != len(self.points):
            self.values = self._sample_intensities(self.points)

        old_points = self.points.copy()
        old_values = self.values.copy() if self.values is not None else np.zeros(len(old_points))
        use_values = self.values is not None

        # --- Calculate the new merged point position and value ---
        merged_point_sum = np.zeros(2, dtype=float)
        merged_value_sum = 0.0
        weight_sum = 0.0  # Could use intensity for weighted average later? For now, simple average.

        for cand_idx in candidate_indices:
            merged_point_sum += old_points[cand_idx]
            if use_values:
                merged_value_sum += old_values[cand_idx]
            weight_sum += 1.0

        new_point_yx = merged_point_sum / weight_sum
        new_value = merged_value_sum / weight_sum if use_values else None

        # --- Create new points and values arrays ---
        mask_to_keep = np.ones(len(old_points), dtype=bool)
        mask_to_keep[candidate_indices] = False  # Mark merged points for removal

        new_points_list = old_points[mask_to_keep].tolist()
        new_values_list = old_values[mask_to_keep].tolist() if use_values else []

        # Find where to insert the new point (maintain rough order if possible)
        # Insert at the position of the original seed point in the *filtered* list
        insert_pos = np.count_nonzero(mask_to_keep[:idx])

        new_points_list.insert(insert_pos, new_point_yx.tolist())
        if use_values and new_value is not None:
            new_values_list.insert(insert_pos, float(new_value))

        # Update instance variables
        self.points = np.array(new_points_list, dtype=float) if new_points_list else np.zeros((0, 2), dtype=float)
        if use_values:
            self.values = np.array(new_values_list, dtype=float) if new_values_list else np.zeros((0,), dtype=float)
        else:
            # If values weren't used or were inconsistent, resample
            self.values = self._sample_intensities(self.points)

        # Update the merge seed to the newly created point's index
        self._merge_seed_idx = insert_pos
        self._merge_seed_origin = tuple(new_point_yx.tolist())  # Update origin

        self._last_cursor_pos = None  # Clear cursor pos after merge

        merged_count = len(candidate_indices)
        radius = math.sqrt(radius_sq)
        self._set_status(
            f"Merged {merged_count} points within radius {radius:.1f} px. New point selected."
        )
        return True

    # ---------- Tooltip и измерения ----------
    def _remove_measure_preview_artist(self) -> bool:
        # Removes the dashed preview line during measurement
        if self._measure_preview_artist is not None:
            try:
                self._measure_preview_artist.remove()
            except Exception:
                pass  # Ignore if already removed
            self._measure_preview_artist = None
            return True
        return False

    def _remove_measurement_artists(self) -> bool:
        # Removes the solid line and text annotation of a completed measurement
        removed = False
        if self._measure_line_artist is not None:
            try:
                self._measure_line_artist.remove()
            except Exception:
                pass
            self._measure_line_artist = None
            removed = True
        if self._measure_annotation is not None:
            try:
                self._measure_annotation.remove()
            except Exception:
                pass
            self._measure_annotation = None
            removed = True
        return removed

    def _cancel_measurement_preview(self) -> bool:
        # Cancels an ongoing measurement (before the second click)
        removed_artist = self._remove_measure_preview_artist()
        has_state = (self._measure_active or self._measure_start_point is not None)
        self._measure_active = False
        self._measure_start_idx = None
        self._measure_start_point = None
        self._measure_preview_end = None
        return removed_artist or has_state  # Return True if state was cleared or artist removed

    def _clear_measurement_result(self) -> bool:
        # Clears a completed measurement result
        removed_artists = self._remove_measurement_artists()
        had_measurement = self._measurement is not None
        self._measurement = None
        return removed_artists or had_measurement

    def _start_measurement(self, idx: int) -> None:
        # Initiates measurement mode when clicking on a point
        if self.points is None or idx < 0 or idx >= len(self.points): return
        # Clear previous measurement results first
        if self._clear_measurement_result():
            self._redraw()  # Redraw if old measurement was cleared

        self._measure_active = True
        self._measure_start_idx = idx
        y0, x0 = self.points[idx]
        self._measure_start_point = (float(y0), float(x0))
        self._measure_preview_end = None
        self._remove_measure_preview_artist()  # Ensure no old preview line exists

    def _update_measurement_preview(self, pos: Optional[tuple[float, float]]) -> None:
        # Updates the dashed preview line as the mouse moves
        if not self._measure_active or self._measure_start_point is None: return

        if pos is None:  # Mouse moved out of axes
            self._measure_preview_end = None
            if self._remove_measure_preview_artist():
                if hasattr(self, "canvas"): self.canvas.draw_idle()
            return

        y1, x1 = pos
        self._measure_preview_end = (float(y1), float(x1))
        y0, x0 = self._measure_start_point

        # Preserve current zoom limits
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()

        if self._measure_preview_artist is None:
            # Create the line artist if it doesn't exist
            (line,) = self.ax.plot(
                [x0, x1], [y0, y1],  # x, y order for plot
                color="#ffcc33", linewidth=1.6, linestyle="--", alpha=0.9,
                scalex=False, scaley=False, zorder=10  # Ensure visible
            )
            self._measure_preview_artist = line
        else:
            # Update existing line data
            self._measure_preview_artist.set_data([x0, x1], [y0, y1])

        # Restore zoom limits
        self.ax.set_xlim(current_xlim)
        self.ax.set_ylim(current_ylim)

        if hasattr(self, "canvas"): self.canvas.draw_idle()

    def _finalize_measurement(self, end_idx: Optional[int]) -> None:
        # Completes the measurement on the second click (if on a point)
        if not self._measure_active or self._measure_start_point is None:
            self._cancel_measurement_preview();
            return

        needs_redraw = False
        if self.points is None or end_idx is None or end_idx == self._measure_start_idx or end_idx < 0 or end_idx >= len(
                self.points):
            # If second click is not on a valid *different* point, cancel measurement
            if self._cancel_measurement_preview(): needs_redraw = True
        else:
            # Valid second point clicked
            start_y, start_x = self._measure_start_point
            end_y, end_x = map(float, self.points[end_idx])
            length = float(np.hypot(end_x - start_x, end_y - start_y))
            # Store measurement details
            self._measurement = {
                "start_yx": (start_y, start_x),
                "end_yx": (end_y, end_x),
                "length": length,
            }
            # Cancel the preview mode (removes dashed line)
            self._cancel_measurement_preview()
            needs_redraw = True  # Need to redraw to show final measurement line/text

        # Deactivate measurement mode regardless
        self._measure_active = False
        self._measure_start_idx = None
        self._measure_start_point = None

        if needs_redraw: self._redraw()

    def _draw_measurement_overlays(self) -> None:
        # Draws the solid line and text for a completed measurement
        # Called during the main _redraw cycle

        # Clear any old artists first (important)
        self._remove_measurement_artists()

        if self._measurement is not None and isinstance(self._measurement, dict):
            start_y, start_x = self._measurement.get("start_yx", (0, 0))
            end_y, end_x = self._measurement.get("end_yx", (0, 0))
            length = float(self._measurement.get("length", 0.0))

            # Draw the solid line
            (line,) = self.ax.plot(
                [start_x, end_x], [start_y, end_y],  # x, y order for plot
                color="#ffcc33", linewidth=1.8, alpha=0.95,
                scalex=False, scaley=False, zorder=5  # Draw below points but above image
            )
            self._measure_line_artist = line

            # Add the text annotation near the midpoint
            mid_x = (start_x + end_x) / 2.0
            mid_y = (start_y + end_y) / 2.0
            txt = f"L = {length:.1f} px"
            self._measure_annotation = self.ax.annotate(
                txt, xy=(mid_x, mid_y), xytext=(0, -14),  # Offset text below line midpoint
                textcoords="offset points", ha="center", va="top",  # Adjust alignment
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9),
                fontsize=9, zorder=6  # Ensure text is visible
            )

        # Re-draw the preview line if measurement is still active
        # (This handles cases where redraw happens during measurement)
        if (self._measure_active and self._measure_start_point is not None and self._measure_preview_end is not None):
            y0, x0 = self._measure_start_point
            y1, x1 = self._measure_preview_end
            # Ensure preview artist is removed if it exists before creating new one
            self._remove_measure_preview_artist()
            (pline,) = self.ax.plot(
                [x0, x1], [y0, y1], color="#ffcc33", linewidth=1.6,
                linestyle="--", alpha=0.9, scalex=False, scaley=False, zorder=10
            )
            self._measure_preview_artist = pline

    def _clear_tooltip(self, *, keep_measure: bool = False, keep_preview: bool = False):
        # Clears the MMB tooltip and optionally measurement artifacts
        removed_tooltip = False
        if self._tooltip is not None:
            try:
                self._tooltip.remove()
            except Exception:
                pass
            self._tooltip = None
            self._tooltip_idx = None
            removed_tooltip = True

        removed_preview = False
        if not keep_preview:
            removed_preview = self._cancel_measurement_preview()

        removed_measure = False
        if not keep_measure:
            removed_measure = self._clear_measurement_result()

        # Redraw only if something visual was actually removed
        if (removed_tooltip or removed_preview or removed_measure) and hasattr(self, "canvas"):
            self.canvas.draw_idle()

    def _show_tooltip_for_idx(self, idx):
        # Shows intensity tooltip on MMB click
        if self.points is None or idx is None or idx < 0 or idx >= len(self.points):
            return

        y, x = self.points[idx]

        # Determine intensity: Prefer sampled percentile, fallback to saved value
        inten = None
        if self._percent_map is not None:  # Best source if available
            H, W = self._percent_map.shape[:2]
            yi = max(0, min(H - 1, int(round(y))))
            xi = max(0, min(W - 1, int(round(x))))
            inten = float(self._percent_map[yi, xi])
        elif self.values is not None and idx < len(self.values):  # Use stored value if no map
            inten = float(self.values[idx])
            # Check if this stored value *looks* like a percentile (0-100) or raw
            # This is heuristic, might need adjustment based on typical raw values
            is_percentile_like = 0 <= inten <= 100

        # Format the text
        if inten is not None:
            unit = "%" if (self._percent_map is not None or is_percentile_like) else "raw"
            txt = f"x={x:.1f}, y={y:.1f}\nI={inten:.1f} {unit}"
        else:  # Fallback if intensity couldn't be determined
            txt = f"x={x:.1f}, y={y:.1f}\nIntensity: N/A"

        self._clear_tooltip()  # Clear previous tooltip

        # Create new annotation
        self._tooltip = self.ax.annotate(
            txt, xy=(x, y), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9),
            fontsize=9, zorder=20  # Ensure tooltip is on top
        )
        self._tooltip_idx = idx
        self.canvas.draw_idle()  # Update canvas to show tooltip

    # ---------- Undo/Redo ----------
    def _make_snapshot(self):
        # Creates a snapshot of the current state for undo/redo
        center_data = None
        if self.overlay and isinstance(self.overlay.get("center"), dict):
            c = self.overlay["center"]
            if "x" in c and "y" in c:
                center_data = {"x": float(c["x"]), "y": float(c["y"])}

        # Make copies of numpy arrays
        points_copy = self.points.copy() if self.points is not None else np.zeros((0, 2))
        values_copy = self.values.copy() if self.values is not None else np.zeros((0,))

        return {
            "points": points_copy,
            "values": values_copy,
            "center": center_data,  # Store cleaned center data
            # Store view state
            "view_cx": self.view_cx,
            "view_cy": self.view_cy,
            "zoom_val": self.zoom_val,
            # Store measurement state
            "measurement": self._measurement,
            # Store overlay radii
            "dead_radius": self.overlay.get("dead_radius", 0.0) if self.overlay else 0.0,
            "search_radius": self.overlay.get("search_radius", 0.0) if self.overlay else 0.0,
        }

    def _push_undo(self):
        # Add current state to undo stack
        if self._history_cap > 0:  # Only store if history is enabled
            self._undo.append(self._make_snapshot())
            # Limit stack size
            if len(self._undo) > self._history_cap:
                self._undo.pop(0)
            self._redo.clear()  # Clear redo stack on new action

    def _apply_snapshot(self, snap):
        # Restores state from a snapshot
        self.center_dragging = False  # Ensure dragging stops
        # Clear temporary visuals
        if self.rect_artist is not None:
            try:
                self.rect_artist.remove()
            except Exception:
                pass
            self.rect_artist = None
        self.rect_start = None
        self._clear_merge_seed(keep_status=True)  # Clear merge selection
        self._cancel_measurement_preview()  # Clear measurement preview
        self._last_cursor_pos = None

        # Restore core data (use copies from snapshot)
        self.points = snap["points"].copy()
        self.values = snap["values"].copy()

        # Restore overlay
        if self.overlay is None: self.overlay = {}  # Ensure overlay dict exists
        self.overlay["center"] = snap["center"]  # Restore center (can be None)
        self.overlay["dead_radius"] = snap.get("dead_radius", 0.0)
        self.overlay["search_radius"] = snap.get("search_radius", 0.0)

        # Restore view state
        self.view_cx = snap.get("view_cx")
        self.view_cy = snap.get("view_cy")
        self.zoom_val = snap.get("zoom_val", 0)
        if hasattr(self, 'zoom_var'): self.zoom_var.set(self.zoom_val)  # Update slider

        # Restore measurement state
        self._measurement = snap.get("measurement")

        self._ensure_view_center()  # Recalculate if needed
        self._update_zoom_hint()  # Update label

    def _undo_btn(self, event=None):
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)): return  # Ignore in text fields
        if not self._undo:
            self._set_status("Nothing to undo.")
            return

        self._clear_tooltip()  # Clear popups before state change
        current_state = self._make_snapshot()  # Save current state for redo
        self._redo.append(current_state)
        if len(self._redo) > self._history_cap: self._redo.pop(0)  # Limit redo stack

        snap_to_restore = self._undo.pop()  # Get previous state
        self._apply_snapshot(snap_to_restore)  # Restore it
        self._redraw()  # Update visuals
        self._set_status("Undo successful.")

    def _redo_btn(self, event=None):
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)): return  # Ignore in text fields
        if not self._redo:
            self._set_status("Nothing to redo.")
            return

        self._clear_tooltip()  # Clear popups
        current_state = self._make_snapshot()  # Save current state for undo
        self._undo.append(current_state)
        if len(self._undo) > self._history_cap: self._undo.pop(0)  # Limit undo stack

        snap_to_restore = self._redo.pop()  # Get state to redo
        self._apply_snapshot(snap_to_restore)  # Restore it
        self._redraw()  # Update visuals
        self._set_status("Redo successful.")

    # ---------- View-center helpers ----------
    def _ensure_view_center(self):
        # Set default view center if not already set
        if self.view_cx is not None and self.view_cy is not None:
            return  # Already have a center

        cx_def, cy_def = 0.0, 0.0
        # Try center from overlay first
        if self.overlay and isinstance(self.overlay.get("center"), dict):
            c = self.overlay["center"]
            cx_def = float(c.get("x", 0.0))
            cy_def = float(c.get("y", 0.0))
        # Fallback to image center if overlay center is missing/invalid
        elif self.img_arr is not None:
            H, W = self.img_arr.shape[:2]
            cx_def = (W - 1) / 2.0
            cy_def = (H - 1) / 2.0
        # Use defaults if no other info available
        if self.view_cx is None: self.view_cx = cx_def
        if self.view_cy is None: self.view_cy = cy_def

    # ---------- Zoom ----------
    def _apply_zoom(self):
        # Applies the current zoom level and view center to the matplotlib axes
        if self.img_arr is None:
            # Handle case with no image: Set default limits?
            self.ax.set_xlim(0, 100)
            self.ax.set_ylim(100, 0)
            return

        H, W = self.img_arr.shape[:2]
        self._ensure_view_center()  # Ensure view_cx, view_cy are set

        x0_full, x1_full = -0.5, W - 0.5
        y0_full, y1_full = H - 0.5, -0.5  # y-axis inverted for imshow

        if self.zoom_val <= 0:  # Zoom level 0 means full view
            self.ax.set_xlim(x0_full, x1_full)
            self.ax.set_ylim(y0_full, y1_full)
            return

        # Calculate zoom window size (L)
        min_dim = min(H, W)
        # Linear interpolation for zoom size: L ranges from min_dim (at 0%) down to 50px (at 100%)
        L = max(50.0, min_dim - (min_dim - 50.0) * (self.zoom_val / 100.0))

        # Maintain square aspect ratio for the zoom *window*
        half_w, half_h = L / 2.0, L / 2.0

        cx = float(self.view_cx);
        cy = float(self.view_cy)

        # Calculate initial window boundaries centered on (cx, cy)
        x0 = cx - half_w;
        x1 = cx + half_w
        y1 = cy - half_h;
        y0 = cy + half_h  # y-axis is inverted

        # Constrain window boundaries to image limits
        x0 = max(x0_full, x0);
        x1 = min(x1_full, x1)
        y1 = max(y1_full, y1);
        y0 = min(y0_full, y0)

        # Adjust boundaries if the window became smaller than L due to hitting edges
        current_w = x1 - x0
        current_h = y0 - y1  # y0 > y1
        if current_w < L - 1e-6:  # Check width, allow small tolerance
            if x0 == x0_full:
                x1 = min(x1_full, x0 + L)  # Expand right if hit left edge
            elif x1 == x1_full:
                x0 = max(x0_full, x1 - L)  # Expand left if hit right edge
            # Recalculate center if bounds shifted significantly? Optional.
            # cx = (x0 + x1) / 2.0

        if current_h < L - 1e-6:  # Check height
            if y1 == y1_full:
                y0 = min(y0_full, y1 + L)  # Expand down if hit top edge
            elif y0 == y0_full:
                y1 = max(y1_full, y0 - L)  # Expand up if hit bottom edge
            # Recalculate center if bounds shifted? Optional.
            # cy = (y0 + y1) / 2.0

        # Set the final axes limits
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)  # ymax, ymin

    def _on_zoom_change(self, val=None):
        try:
            # If called by slider, 'val' is a string representation
            # If called internally (e.g., scroll), 'val' might be float or None
            new_val = float(val) if val is not None else self.zoom_val
            new_zoom = max(0, min(100, int(round(new_val))))  # Clamp between 0 and 100
        except (ValueError, TypeError):
            new_zoom = 0  # Default to 0 on error

        if new_zoom != self.zoom_val:  # Only update if value changed
            self.zoom_val = new_zoom
            # Update slider if it exists and its value differs
            if hasattr(self, "zoom_var"):
                current_slider_val = int(round(self.zoom_var.get()))
                if current_slider_val != self.zoom_val:
                    self.zoom_var.set(self.zoom_val)

            self._update_zoom_hint()
            self._clear_tooltip()  # Clear tooltip on zoom change
            self._redraw()  # Redraw with new zoom level

    def _on_scroll(self, event):
        # Handles zooming with the mouse wheel
        if event.xdata is None or event.ydata is None:
            return  # Don't zoom if cursor is outside axes

        # --- Zoom centering ---
        # Set the view center to the cursor position *before* changing zoom level
        self.view_cx = event.xdata
        self.view_cy = event.ydata

        # Determine zoom direction and step
        zoom_step = 5
        if event.button == 'up':  # Scroll up zooms in
            new_zoom_val = self.zoom_val + zoom_step
        elif event.button == 'down':  # Scroll down zooms out
            new_zoom_val = self.zoom_val - zoom_step
        else:
            return  # Not a recognized scroll event

        # Clamp and apply the new zoom value
        new_zoom_val = max(0, min(100, new_zoom_val))
        self._on_zoom_change(new_zoom_val)  # Use the common handler

    # ---------- Mouse / Keyboard events ----------
    def _on_key(self, e):
        # Handles key presses on the canvas
        if e.key == "escape":
            # Clear tooltip, cancel measurement preview, cancel merge selection
            cleared_tooltip = self._tooltip is not None
            cleared_measure = self._cancel_measurement_preview()
            cleared_merge = self._clear_merge_seed()
            if cleared_tooltip or cleared_measure or cleared_merge:
                self._redraw()  # Redraw if any state was cleared
            if not (cleared_tooltip or cleared_measure or cleared_merge):
                self._set_status("Escape pressed, no action taken.")


        elif e.key in {"enter", "return"}:
            # Finalize merge operation if a seed point is selected
            if self._merge_seed_idx is not None:
                if self._merge_selected_with_radius():
                    self._redraw()  # Redraw after successful merge
                # Status is set within _merge_selected_with_radius
            else:
                self._set_status("Enter pressed, no merge point selected.")

    def _on_down(self, e):
        # Handles mouse button presses on the canvas
        pos_yx = self._img_xy(e)  # Get coordinates in (y, x) format
        self._clear_tooltip()  # Clear tooltip on any click

        # Store last known cursor position
        if pos_yx is not None:
            self._last_cursor_pos = (float(pos_yx[0]), float(pos_yx[1]))
        else:
            self._last_cursor_pos = None

        # --- Middle Mouse Button (Button 2) ---
        if e.button == 2:
            if pos_yx is None: return  # Click outside axes
            y, x = pos_yx
            idx = self._near_idx(y, x, pix_tol=8)  # Find nearest point
            if idx is not None:
                self._show_tooltip_for_idx(idx)  # Show info tooltip
            return

        # --- Left Mouse Button (Button 1) ---
        if e.button == 1:
            # --- Shift + Left Click: Start Rectangular Selection ---
            if e.key and "shift" in e.key.lower():
                if pos_yx is not None:
                    self._push_undo()  # Save state before starting rect select
                    self.rect_start = pos_yx  # Store start corner (y, x)
                    self._redo.clear()
                    self._set_status("Drag to select points for deletion.")
                return  # Don't do other actions when shift is held

            # --- Normal Left Click ---
            if pos_yx is None: return  # Click outside axes

            y, x = pos_yx

            # --- Click on Center? ---
            if self._center_hit(y, x):
                self._clear_merge_seed()  # Cancel merge selection
                self._cancel_measurement_preview()  # Cancel measurement
                self._push_undo()  # Save state before dragging center
                self.center_dragging = True
                self._redo.clear()
                self._set_status("Dragging center overlay. Release to finish.")
                self._redraw()  # Show visual feedback? (Optional)
                return

            # --- Click Near a Point? ---
            i = self._near_idx(y, x)
            if i is not None:
                # If measurement is active, finalize it
                if self._measure_active:
                    self._finalize_measurement(i)
                else:
                    # If not measuring, select point for merge OR start measurement
                    self._select_merge_seed(i)
                    self._start_measurement(i)  # Start measurement mode simultaneously
                    self._set_status("Point selected. Drag to measure distance or press Enter to merge.")

            # --- Click on Empty Area? ---
            else:
                self._clear_merge_seed()  # Cancel merge selection
                self._cancel_measurement_preview()  # Cancel measurement
                self._push_undo()  # Save state before adding point
                # Add new point
                self.points = np.vstack([self.points, [y, x]])
                # Sample its intensity
                sampled_value = self._sample_intensities(np.array([[y, x]]))[0]
                self.values = np.append(self.values, sampled_value)
                self._redo.clear()
                self._set_status(f"Added point at ({x:.1f}, {y:.1f}).")

        # --- Right Mouse Button (Button 3) ---
        elif e.button == 3:
            if pos_yx is None: return  # Click outside axes
            y, x = pos_yx
            i = self._near_idx(y, x)  # Find point to delete
            if i is not None:
                self._push_undo()  # Save state before deleting

                # --- Update merge seed if the deleted point was the seed ---
                deleted_point_was_seed = (i == self._merge_seed_idx)
                index_shift = 0
                if self._merge_seed_idx is not None and i < self._merge_seed_idx:
                    index_shift = -1  # Adjust index if point before seed is deleted

                # Delete point and value
                self.points = np.delete(self.points, i, axis=0)
                if self.values is not None and len(self.values) > i:
                    self.values = np.delete(self.values, i, axis=0)
                else:
                    # Resample if values array was inconsistent
                    self.values = self._sample_intensities(self.points)

                # --- Update merge seed state ---
                if deleted_point_was_seed:
                    self._clear_merge_seed()  # Clear if seed was deleted
                elif index_shift != 0 and self._merge_seed_idx is not None:
                    self._merge_seed_idx += index_shift  # Adjust index
                    # Update origin based on new index
                    if 0 <= self._merge_seed_idx < len(self.points):
                        new_y, new_x = self.points[self._merge_seed_idx]
                        self._merge_seed_origin = (float(new_y), float(new_x))
                    else:  # Should not happen, but clear if index becomes invalid
                        self._clear_merge_seed()

                self._redo.clear()
                self._set_status("Deleted point.")

        # Redraw after any action (add, delete, select)
        self._redraw()

    def _on_move(self, e):
        # Handles mouse movement over the canvas
        pos_yx = self._img_xy(e)  # Get coordinates (y, x)

        # Update last cursor position (used for merge radius)
        if pos_yx is not None:
            self._last_cursor_pos = (float(pos_yx[0]), float(pos_yx[1]))
        else:
            self._last_cursor_pos = None

        # --- Dragging Center ---
        if self.center_dragging and pos_yx is not None:
            y, x = pos_yx
            if self.overlay is None: self.overlay = {}  # Ensure exists
            self.overlay["center"] = {"x": float(x), "y": float(y)}
            # Update view center dynamically while dragging? Optional, can feel jerky.
            # self.view_cx = float(x)
            # self.view_cy = float(y)
            self._redraw()  # Redraw to show center moving
            self._set_status("Dragging center...")  # Update status
            return  # Don't do other move actions while dragging center

        # --- Updating Measurement Preview ---
        if self._measure_active:
            self._update_measurement_preview(pos_yx)  # Update dashed line
            # Don't return here, allow rect drag simultaneously if needed? No, measure takes priority.
            return

        # --- Updating Rectangular Selection ---
        if self.rect_start and e.xdata is not None and e.ydata is not None:
            y0, x0 = self.rect_start  # Start corner (y, x)
            y1, x1 = e.ydata, e.xdata  # Current corner (y, x)

            # --- Draw the rectangle ---
            # Remove previous rectangle artist if it exists
            if self.rect_artist is not None:
                try:
                    self.rect_artist.remove()
                except Exception:
                    pass
                self.rect_artist = None

            # Preserve zoom
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()

            # Create new rectangle patch (x, y, width, height for Rectangle)
            rect_x = min(x0, x1)
            rect_y = min(y0, y1)
            rect_w = abs(x1 - x0)
            rect_h = abs(y1 - y0)
            self.rect_artist = self.ax.add_patch(
                plt.Rectangle((rect_x, rect_y), rect_w, rect_h,
                              fill=False, ec="red", ls="--", lw=1.5, zorder=15)
            )

            # Restore zoom
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)

            self.canvas.draw_idle()  # Update canvas to show rectangle
            return  # Don't do other move actions

    def _on_up(self, e):
        # Handles mouse button releases on the canvas

        # --- Releasing after Dragging Center ---
        if self.center_dragging:
            self.center_dragging = False
            self._apply_center_filters()  # Remove points now outside radii
            # Update view center permanently after drag
            if self.overlay and isinstance(self.overlay.get("center"), dict):
                self.view_cx = float(self.overlay["center"]["x"])
                self.view_cy = float(self.overlay["center"]["y"])
            self._redraw()
            self._set_status("Center position updated.")
            return

        # --- Releasing after Measurement Start (potential finalize) ---
        if self._measure_active:
            # Measurement is finalized only by clicking a *second point* (handled in _on_down)
            # Releasing button in empty space or on same point cancels preview
            pos_yx = self._img_xy(e)
            end_idx = None
            if pos_yx is not None:
                end_idx = self._near_idx(pos_yx[0], pos_yx[1])

            # If released on the start point or empty space, just cancel the preview drawing
            if end_idx is None or end_idx == self._measure_start_idx:
                if self._cancel_measurement_preview():
                    self._redraw()  # Redraw if preview was cleared
                    self._set_status("Measurement cancelled.")
                # Important: Don't reset _measure_active here, it's reset in _finalize or _cancel
            # If released on a *different* point, _finalize_measurement was already called in _on_down
            return

        # --- Releasing after Rectangular Selection ---
        if self.rect_start:
            y0, x0 = self.rect_start;  # Start corner (y, x)
            # Ensure coordinates exist on release
            if e.ydata is not None and e.xdata is not None:
                y1, x1 = e.ydata, e.xdata  # End corner (y, x)

                # Determine bounds
                ymin, ymax = sorted([y0, y1]);
                xmin, xmax = sorted([x0, x1])

                # Find points within the rectangle
                points_in_rect_mask = (
                        (self.points[:, 0] >= ymin) & (self.points[:, 0] <= ymax) &
                        (self.points[:, 1] >= xmin) & (self.points[:, 1] <= xmax)
                )
                num_to_delete = np.count_nonzero(points_in_rect_mask)

                if num_to_delete > 0:
                    # --- Update merge seed index before deleting points ---
                    new_merge_seed_idx = None
                    if self._merge_seed_idx is not None:
                        # Check if seed is *not* being deleted
                        if not points_in_rect_mask[self._merge_seed_idx]:
                            # Calculate how many points *before* the seed are being deleted
                            num_deleted_before_seed = np.count_nonzero(points_in_rect_mask[:self._merge_seed_idx])
                            new_merge_seed_idx = self._merge_seed_idx - num_deleted_before_seed
                        # If seed is being deleted, new_merge_seed_idx remains None

                    # --- Delete points ---
                    mask_to_keep = ~points_in_rect_mask
                    self.points = self.points[mask_to_keep]
                    if self.values is not None and len(self.values) == len(mask_to_keep) + num_to_delete:
                        self.values = self.values[mask_to_keep]
                    else:
                        self.values = self._sample_intensities(self.points)  # Resample if inconsistent

                    # --- Update merge seed state ---
                    self._merge_seed_idx = new_merge_seed_idx
                    if self._merge_seed_idx is not None:
                        # Update origin to current position
                        current_y, current_x = self.points[self._merge_seed_idx]
                        self._merge_seed_origin = (float(current_y), float(current_x))
                    else:
                        self._merge_seed_origin = None  # Clear if seed was deleted or invalid

                    self._set_status(f"Deleted {num_to_delete} points in selection.")
                else:
                    self._set_status("Rectangular selection finished, no points deleted.")

            # --- Cleanup rectangle drawing ---
            self.rect_start = None
            if self.rect_artist is not None:
                try:
                    self.rect_artist.remove()
                except Exception:
                    pass
                self.rect_artist = None
            self._redraw()  # Redraw to remove rectangle and show updated points
            return

    # ---------- Draw ----------
    def _redraw(self):
        self.ax.clear()
        if self.img_arr is not None:
            self.ax.imshow(self.img_arr, cmap="gray", interpolation="nearest")
        self.ax.axis("off")

        # Draw Center and Radii Overlay
        if self.overlay and isinstance(self.overlay.get("center"), dict):
            center_data = self.overlay["center"]
            cy = float(center_data.get("y", 0))
            cx = float(center_data.get("x", 0))
            self.ax.scatter([cx], [cy], s=40, c="red", marker="o", zorder=5)  # Increase size slightly
            dead = float(self.overlay.get("dead_radius", 0))
            sr = float(self.overlay.get("search_radius", 0))
            # Draw radii only if > 0
            if dead > 0:
                self.ax.add_patch(Circle((cx, cy), dead, fill=False, ls="--", lw=1.5, ec="red", zorder=4))
            if sr > 0:
                self.ax.add_patch(Circle((cx, cy), sr, fill=False, ls=":", lw=1.0, ec="red", zorder=4))

        # Draw Points
        if self.points is not None and len(self.points) > 0:
            points_to_draw = self.points
            colors = 'cyan'  # Default color
            sizes = 22  # Default size
            zorder = 3

            # Highlight the merge seed point if selected
            if self._merge_seed_idx is not None and 0 <= self._merge_seed_idx < len(self.points):
                seed_y, seed_x = points_to_draw[self._merge_seed_idx]
                # Draw non-seed points first
                mask = np.ones(len(points_to_draw), dtype=bool)
                mask[self._merge_seed_idx] = False
                if np.any(mask):
                    self.ax.scatter(points_to_draw[mask, 1], points_to_draw[mask, 0],
                                    s=sizes, c=colors, alpha=0.8, marker="o",
                                    linewidths=0.5, edgecolors="black", zorder=zorder)
                # Draw seed point highlighted
                self.ax.scatter([seed_x], [seed_y], s=42, c="#ffd34d",  # Yellowish color
                                alpha=0.95, marker="o", linewidths=0.8, edgecolors="black", zorder=zorder + 1)
            else:
                # Draw all points normally if no seed selected
                self.ax.scatter(points_to_draw[:, 1], points_to_draw[:, 0],
                                s=sizes, c=colors, alpha=0.9, marker="o",
                                linewidths=0.5, edgecolors="black", zorder=zorder)

        # Draw Measurement Overlays (solid line + text, or dashed preview line)
        self._draw_measurement_overlays()

        # Apply Zoom
        self._apply_zoom()

        # Update Canvas
        self.canvas.draw_idle()

    # ---------- Анализ ----------
    def _start_analysis(self):
        # --- Ensure output directory and save points first ---
        try:
            saved_spots_path = self._save_points()  # This now saves both files and returns spots.json path
            # The output dir is determined and created within _save_points
            output_dir = saved_spots_path.parent
            # The saed_input.edited.json is also saved by _save_points
            payload_path = output_dir / "fibo_input.json"  # Define standard payload name

        except (ValueError, OSError, Exception) as e:
            messagebox.showerror("Save Error", f"Cannot proceed to analysis. Failed to save points/files:\n{e}")
            return

        # --- Prepare the payload for fibonachi_analysis ---
        try:
            # Most data is already in saed_input.edited.json, just need to reference it?
            # Or recreate the payload structure? Recreating is safer.

            # Reload the just saved saed_input.edited.json to get consistent data?
            # Or use current instance state? Using instance state is simpler here.

            abs_image_path = self.image_path.resolve() if self.image_path else None

            overlay_center_data = None
            if self.overlay and isinstance(self.overlay.get("center"), dict):
                center_data = self.overlay["center"]
                overlay_center_data = {"x": float(center_data["x"]), "y": float(center_data["y"])}

            # Get geometric center if possible
            geo_center_data = None
            if self.img_arr is not None:
                H, W = self.img_arr.shape[:2]
                geo_center_data = {"x": (W - 1) / 2.0, "y": (H - 1) / 2.0}

            # Use current points/values from instance
            pts_list = []
            current_values = self._sample_intensities(self.points)
            for i, (y, x) in enumerate(self.points):
                intensity = float(current_values[i]) if i < len(current_values) else 0.0
                pts_list.append({"y": float(y), "x": float(x), "intensity": intensity})

            payload = {
                "image": str(abs_image_path) if abs_image_path else None,
                "preproc_mode": self._preproc_settings.mode,
                "preproc": self._preproc_settings.to_json(),
                "points": pts_list,
                "centers": {
                    "geometric": geo_center_data,
                    "overlay": overlay_center_data
                },
                "radii": {
                    "dead": float(self.overlay.get("dead_radius", 0.0)) if self.overlay else 0.0,
                    "search": float(self.overlay.get("search_radius", 0.0)) if self.overlay else 0.0
                },
                # Pass path to the spots file generated by _save_points
                "spots_json": str(saved_spots_path.resolve())
            }

            payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._set_status("Prepared data for analysis…")

        except Exception as e:
            messagebox.showerror("Data Preparation Error",
                                 f"Failed to create analysis payload ({payload_path.name}):\n{e}")
            return  # Stop if payload creation fails

        # --- Launch analysis via controller ---
        if self.controller is not None:
            try:
                # Pass the absolute path of the generated payload
                self.controller.open_analysis(payload_path.resolve(), abs_image_path, saved_spots_path.resolve())
                # Status is updated by controller on successful tab switch
            except Exception as e:
                # Log the error for debugging
                print(f"Error calling controller.open_analysis: {e}")
                # Display a user-friendly message
                messagebox.showerror("Launch Error",
                                     f"Failed to switch to the analysis tab. Please check the data and try again.\nDetails: {e}")
        else:
            # Fallback for standalone mode (optional)
            messagebox.showwarning("Standalone Mode", "Cannot switch to analysis tab. Controller not available.")
            # If you still need external launch:
            # try:
            #     # ... code to launch fibonachi_analysis.exe ...
            #     self._set_status("External analysis started (standalone mode)")
            # except Exception as e:
            #     messagebox.showerror("Launch Error", f"Failed to launch external analysis:\n{e}")

    def _show_report(self, text: str):  # Kept for potential future use, but not called by _start_analysis now
        win = tk.Toplevel(self)
        win.title("SAED Report")
        txt = tk.Text(win, wrap="word", padx=10, pady=10, state=tk.DISABLED)  # Start disabled
        txt.pack(fill=tk.BOTH, expand=True)

        # Use tags for basic formatting
        txt.tag_configure("header", font=("TkDefaultFont", 12, "bold", "underline"))
        txt.tag_configure("bold", font=("TkDefaultFont", 10, "bold"))

        txt.config(state=tk.NORMAL)  # Enable for inserting
        lines = text.splitlines()
        if lines:
            txt.insert("1.0", lines[0] + "\n", "header")  # First line as header
            txt.insert(tk.END, "\n".join(lines[1:]))  # Insert rest

        txt.config(state=tk.DISABLED)  # Disable again
        self._set_status("Symmetry report generated")


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
    # Make --input optional for standalone startup
    p.add_argument("--input", type=str, required=False, default=None,
                   help="Optional path to saed_input.json to load on startup.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    root = PointEditorApp(args.input)
    root.mainloop()