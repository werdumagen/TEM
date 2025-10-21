#!/usr/bin/env python3
# -*- coding: utf-8 -*-
(
    "SAED Symmetry – Launcher\n"
    "========================\n"
    "Uses centroid-based spot detection and updated preprocessing.\n"
    "Removed merging logic.\n"
)
from __future__ import annotations
import json, subprocess, sys, cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np

from percentile_utils import compute_percentile_map
from preproc import PreprocSettings, load_grayscale_with_preproc

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# -------------------------- Algorithm --------------------------
@dataclass
class CenterResult:
    cy: float
    cx: float
    method: str

def detect_spots_by_centroid(
    arr: np.ndarray,
    perc: float = 99.0,
    min_area: int = 3,
    max_spots: int = 6000,
) -> np.ndarray:
    """
    Detects spots using centroiding. Uses percentile map for thresholding,
    but original array intensity for sorting.
    """
    H, W = arr.shape
    percent_map, _, _ = compute_percentile_map(arr)
    try:
        perc_val = float(np.clip(perc, 0.0, 100.0))
        th_value = float(np.percentile(percent_map, perc_val))
    except (ValueError, IndexError):
        th_value = 99.0

    binary_mask = np.where(percent_map >= th_value, 255, 0).astype(np.uint8)

    num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    kept = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        cx, cy = centroids[i]
        yi, xi = int(round(cy)), int(round(cx))
        if 0 <= yi < H and 0 <= xi < W:
            # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
            # Берем яркость из исходного предобработанного изображения arr
            v = float(arr[yi, xi])
            # ----------------------
            kept.append((float(cy), float(cx), float(v))) # Store as (y, x, v)

    # Сортируем по яркости из arr
    kept.sort(key=lambda t: -t[2])

    if len(kept) > max_spots:
        kept = kept[:max_spots]

    return np.array(kept, dtype=float) if kept else np.zeros((0, 3), dtype=float)


def geometric_midpoint(arr: np.ndarray) -> CenterResult:
    H, W = arr.shape
    return CenterResult(cy=(H - 1) / 2.0, cx=(W - 1) / 2.0, method="midpoint")

def refine_center_antipodal(center: Tuple[float,float], pts: np.ndarray, tol_ang_deg: float=8.0, tol_rel_r: float=0.06, iters: int=3) -> CenterResult:
    cy, cx = float(center[0]), float(center[1])
    if len(pts) < 4:
        return CenterResult(cy=cy, cx=cx, method="midpoint (fallback)")
    for _ in range(max(0, int(iters))):
        dy = pts[:,0]-cy; dx = pts[:,1]-cx
        r = np.hypot(dx, dy)
        r_safe = np.where(r > 1e-9, r, 1e-9)
        u = np.column_stack((dx, dy)) / r_safe[:,None]
        cos_thr = -np.cos(np.deg2rad(180.0 - float(tol_ang_deg)))
        mids = []
        for i in range(len(pts)):
            if r[i] < 1e-6: continue
            dots = (u @ u[i])
            max_r_pair = np.maximum(r, r[i])
            rel_diff = np.divide(np.abs(r - r[i]), max_r_pair, out=np.zeros_like(r), where=max_r_pair > 1e-9)
            rad_ok = (rel_diff < float(tol_rel_r)) & (max_r_pair > 1e-9)
            ang_ok = (dots < cos_thr)
            valid_match = rad_ok & ang_ok & (np.arange(len(pts)) != i) & (r > 1e-6)
            idx = np.where(valid_match)[0]
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
    (
        "Launcher tab suitable for both the standalone application and notebooks.\n"
    )

    def __init__(self, master: tk.Misc, controller=None):
        super().__init__(master)
        self.controller = controller
        self._scroll_canvas = None
        self._scroll_window_id = None
        self._build_ui()

    def _get_default_output_path(self) -> str:
        """Generates a default output path, avoiding existing directories."""
        if getattr(sys, "frozen", False):
            base_dir = Path(sys.executable).parent
        else:
            try:
                 base_dir = Path.cwd()
            except OSError:
                 base_dir = Path(__file__).parent

        base_name = "saed_results"
        output_path = base_dir / base_name
        if not output_path.exists():
            return str(output_path)

        counter = 1
        while True:
            new_name = f"{base_name}_{counter}"
            new_path = base_dir / new_name
            if not new_path.exists():
                return str(new_path)
            counter += 1
            if counter > 999:
                 return str(base_dir / f"{base_name}_temp_{np.random.randint(1000)}")

    def _build_ui(self):
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)

        fixed = ttk.Frame(outer, padding=(16, 16, 16, 0))
        fixed.pack(side=tk.TOP, fill=tk.X)
        fixed.grid_columnconfigure(0, weight=1)

        data_box = ttk.LabelFrame(fixed, text="Input data", padding=(12, 10, 12, 12))
        data_box.grid(row=0, column=0, sticky="nsew")
        data_box.grid_columnconfigure(1, weight=1)

        ttk.Label(data_box, text="Image:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.ent_img = ttk.Entry(data_box)
        self.ent_img.grid(row=0, column=1, columnspan=2, sticky="we", padx=6, pady=4)
        ttk.Button(data_box, text="Browse…", command=self._browse_img).grid(row=0, column=3, sticky="ew", padx=6, pady=4)

        ttk.Label(data_box, text="Output folder:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.ent_out = ttk.Entry(data_box)
        self.ent_out.insert(0, self._get_default_output_path())
        self.ent_out.grid(row=1, column=1, sticky="we", padx=6, pady=4)
        ttk.Button(data_box, text="Load Session…", command=self._load_session).grid(row=1, column=2, sticky="ew", padx=6, pady=4)
        ttk.Button(data_box, text="Choose…", command=self._browse_out).grid(row=1, column=3, sticky="ew", padx=6, pady=4)

        ttk.Label(data_box, text="Center X (optional):").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.ent_cx = ttk.Entry(data_box, width=12)
        self.ent_cx.grid(row=2, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(data_box, text="Center Y:").grid(row=2, column=2, sticky="w", padx=6, pady=4)
        self.ent_cy = ttk.Entry(data_box, width=12)
        self.ent_cy.grid(row=2, column=3, sticky="w", padx=6, pady=4)

        ttk.Label(
            data_box,
            text="Leave the coordinates empty to let the program find the center automatically. Use 'Load Session' to restore a previous state.",
            wraplength=520,
            foreground="#555555"
        ).grid(row=3, column=0, columnspan=4, sticky="we", padx=6, pady=(0, 4))

        pre_box = ttk.LabelFrame(fixed, text="Preprocessing", padding=(12, 10, 12, 12))
        pre_box.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        pre_box.grid_columnconfigure(1, weight=1)

        ttk.Label(pre_box, text="Mode:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.cmb_pre = ttk.Combobox(
            pre_box,
            values=["No smoothing", "NLM Denoising"], # <-- Updated modes
            state="readonly",
        )
        self.cmb_pre.current(0)
        self.cmb_pre.grid(row=0, column=1, sticky="w", padx=6, pady=4)
        self.cmb_pre.bind("<<ComboboxSelected>>", self._on_preproc_change)

        ttk.Label(pre_box, text="NLM 'h' parameter:").grid(row=1, column=0, sticky="w", padx=6, pady=4) # <-- New label
        self.spn_h = ttk.Spinbox(pre_box, from_=0.01, to=5.0, increment=0.1, width=8, justify="right") # <-- New spinner
        self._set_spinbox_value(self.spn_h, 1.0)
        self.spn_h.grid(row=1, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(
            pre_box,
            text="Select 'NLM Denoising' to reduce noise. 'h' controls filtering strength (higher = stronger).", # <-- Updated hint
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

        detect_box = ttk.LabelFrame(scrollable, text="Detector and refinement", padding=(12, 10, 12, 12))
        detect_box.grid(row=0, column=0, sticky="nsew")
        detect_box.grid_columnconfigure(1, weight=1)

        ttk.Label(
            detect_box,
            text="Peak threshold and search window",
            font=("TkDefaultFont", 10, "bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))
        self.spn_perc = self._spin_param(
            detect_box, 1, "Detection percentile (%)", 99.0,
            from_=80.0, to=100.0, increment=0.1, format_str="%.1f"
        )
        # --- Removed merge spinners ---
        self.spn_min_area = self._spin_param(
            detect_box, 2, "Min. peak area (px)", 3,
            from_=1, to=500, increment=1
        )
        self.spn_maxpts = self._spin_param(
            detect_box, 3, "Maximum detected points", 6000,
            from_=100, to=20000, increment=100
        )

        ttk.Separator(detect_box).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 8))

        ttk.Label(
            detect_box,
            text="Center refinement",
            font=("TkDefaultFont", 10, "bold")
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))
        self.spn_iters = self._spin_param(
            detect_box, 6, "Center refinement iterations", 4,
            from_=0, to=10, increment=1
        )
        self.spn_tolang = self._spin_param(
            detect_box, 7, "Antipode tolerance (°)", 8.0,
            from_=1.0, to=30.0, increment=0.5, format_str="%.1f"
        )
        self.spn_tolr = self._spin_param(
            detect_box, 8, "Radius tolerance (relative)", 0.06,
            from_=0.01, to=0.5, increment=0.01, format_str="%.2f"
        )

        ttk.Separator(detect_box).grid(row=9, column=0, columnspan=2, sticky="ew", pady=(6, 8))

        ttk.Label(
            detect_box,
            text="Geometric filters",
            font=("TkDefaultFont", 10, "bold")
        ).grid(row=10, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))
        self.spn_dead = self._spin_param(
            detect_box, 11, "Dead zone (px)", 0,
            from_=0, to=500, increment=1
        )
        self.spn_search = self._spin_param(
            detect_box, 12, "Search radius (px, 0 = unlimited)", 0,
            from_=0, to=10000, increment=25
        )

        ttk.Label(
            detect_box,
            text="Use 'Min. peak area' to filter noise. Use 'Dead zone' to filter the central beam by its position.", # <-- Updated hint
            wraplength=520,
            foreground="#555555"
        ).grid(row=13, column=0, columnspan=2, sticky="we", padx=6, pady=(2, 0))

        action_box = ttk.Frame(scrollable, padding=(0, 12, 0, 0))
        action_box.grid(row=1, column=0, sticky="nsew")
        action_box.grid_columnconfigure(0, weight=1)

        ttk.Label(
            action_box,
            text="Review the parameters and press the button below to switch to interactive editing of detected points.",
            wraplength=540,
            justify="left"
        ).grid(row=0, column=0, sticky="we", padx=4, pady=(0, 8))

        ttk.Button(action_box, text="Open point editor", command=self._go_editor).grid(
            row=1, column=0, sticky="ew", padx=4, pady=(0, 12)
        )

        filler_bg = ttk.Style().lookup("TFrame", "background") or self.winfo_toplevel().cget("background")
        bottom_filler = tk.Frame(scrollable, height=56, bg=filler_bg)
        bottom_filler.grid(row=2, column=0, sticky="ew")
        bottom_filler.grid_propagate(False)

        self._on_preproc_change(None)

    def _activate_scroll(self, _event):
        if self._scroll_canvas is None: return
        self._scroll_canvas.bind_all("<MouseWheel>", self._on_scroll_mousewheel)
        self._scroll_canvas.bind_all("<Button-4>", self._on_scroll_mousewheel)
        self._scroll_canvas.bind_all("<Button-5>", self._on_scroll_mousewheel)

    def _deactivate_scroll(self, _event):
        if self._scroll_canvas is None: return
        self._scroll_canvas.unbind_all("<MouseWheel>")
        self._scroll_canvas.unbind_all("<Button-4>")
        self._scroll_canvas.unbind_all("<Button-5>")

    def _on_scroll_mousewheel(self, event):
        if self._scroll_canvas is None: return
        delta = 0
        if sys.platform == "win32": delta = -int(event.delta / 120)
        elif sys.platform == "darwin": delta = event.delta
        elif event.num == 4: delta = -1
        elif event.num == 5: delta = 1
        if delta != 0: self._scroll_canvas.yview_scroll(delta, "units")

    def _on_preproc_change(self, _evt):
        mode = self.cmb_pre.get()
        nlm_enabled = (mode == "NLM Denoising") # <-- Check for new mode
        state = "normal" if nlm_enabled else "disabled"
        self.spn_h.configure(state=state) # <-- Configure new spinner

    def _spin_param(self, parent, row, label, default, *, from_, to, increment, format_str=None):
        ttk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        spin = ttk.Spinbox(parent, from_=from_, to=to, increment=increment, width=10, justify="right")
        if format_str: spin.configure(format=format_str)
        self._set_spinbox_value(spin, default)
        spin.grid(row=row, column=1, sticky="w", padx=6, pady=4)
        return spin

    def _set_spinbox_value(self, spinbox: ttk.Spinbox, value):
        try: spinbox.set(value)
        except tk.TclError:
            try:
                 current_value = spinbox.get()
                 if str(current_value) != str(value):
                      spinbox.delete(0, tk.END)
                      spinbox.insert(0, str(value))
            except (tk.TclError, ValueError):
                 print(f"Warning: Could not set spinbox value to {value}")

    def _browse_img(self):
        p = filedialog.askopenfilename(title="Select image", filetypes=[("Images","*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),("All","*.*")])
        if p: self.ent_img.delete(0, tk.END); self.ent_img.insert(0, p)

    def _browse_out(self):
        p = filedialog.askdirectory(title="Select output folder", mustexist=False)
        if p: self.ent_out.delete(0, tk.END); self.ent_out.insert(0, p)

    def _load_session(self):
        """Asks user for a session file and tells the controller to load it."""
        filepath = filedialog.askopenfilename(
            title="Load SAED Session",
            filetypes=[("SAED Session", "saed_session.json"), ("All files", "*.*")]
        )
        if not filepath or not self.controller: return
        try:
            self.controller.load_session_from_file(filepath)
            if self.controller: self.controller.set_status(f"Session loaded from {Path(filepath).name}")
        except FileNotFoundError: messagebox.showerror("Load Error", "Session file not found.")
        except Exception as e: messagebox.showerror("Load Error", f"Failed to load session:\n{e}")

    def get_state(self) -> Dict[str, Any]:
        """Returns a serializable dictionary of the launcher's settings."""
        return {
            "image_path": self.ent_img.get(),
            "output_folder": self.ent_out.get(),
            "center_x": self.ent_cx.get(),
            "center_y": self.ent_cy.get(),
            "preproc_mode": self.cmb_pre.get(),
            "h_param": self.spn_h.get(), # <-- Use h_param
            "detect_perc": self.spn_perc.get(),
            "min_area": self.spn_min_area.get(),
            "max_pts": self.spn_maxpts.get(),
            "refine_iters": self.spn_iters.get(),
            "tol_angle": self.spn_tolang.get(),
            "tol_radius": self.spn_tolr.get(),
            "dead_zone": self.spn_dead.get(),
            "search_radius": self.spn_search.get(),
        }

    def set_state(self, state: Dict[str, Any]):
        """Restores the launcher's settings from a dictionary."""
        def _set_entry(widget, value):
            if value is not None and isinstance(widget, (ttk.Entry, tk.Entry)):
                widget.delete(0, tk.END); widget.insert(0, str(value))

        _set_entry(self.ent_img, state.get("image_path"))
        _set_entry(self.ent_out, state.get("output_folder"))
        _set_entry(self.ent_cx, state.get("center_x"))
        _set_entry(self.ent_cy, state.get("center_y"))

        preproc_mode = state.get("preproc_mode")
        if preproc_mode and isinstance(self.cmb_pre, ttk.Combobox):
             if preproc_mode in self.cmb_pre['values']: self.cmb_pre.set(preproc_mode)
             else: print(f"Warning: Saved preproc_mode '{preproc_mode}' not found. Using default."); self.cmb_pre.current(0)
        elif isinstance(self.cmb_pre, ttk.Combobox): self.cmb_pre.current(0)

        self._on_preproc_change(None)

        self._set_spinbox_value(self.spn_h, state.get("h_param", 1.0)) # <-- Use h_param
        self._set_spinbox_value(self.spn_perc, state.get("detect_perc", 99.0))
        self._set_spinbox_value(self.spn_min_area, state.get("min_area", 3))
        self._set_spinbox_value(self.spn_maxpts, state.get("max_pts", 6000))
        self._set_spinbox_value(self.spn_iters, state.get("refine_iters", 4))
        self._set_spinbox_value(self.spn_tolang, state.get("tol_angle", 8.0))
        self._set_spinbox_value(self.spn_tolr, state.get("tol_radius", 0.06))
        self._set_spinbox_value(self.spn_dead, state.get("dead_zone", 0))
        self._set_spinbox_value(self.spn_search, state.get("search_radius", 0))


    def _go_editor(self):
        try:
            image_path_str = self.ent_img.get()
            output_dir_str = self.ent_out.get()
            if not image_path_str: messagebox.showerror("Error", "Please select an image file."); return
            if not output_dir_str: messagebox.showerror("Error", "Please specify an output folder."); return

            image_path = Path(image_path_str).expanduser().resolve()
            outdir = Path(output_dir_str).expanduser().resolve();
            outdir.mkdir(parents=True, exist_ok=True)
            if not image_path.exists(): messagebox.showerror("Error", f"Image not found at: {image_path}"); return

            perc = float(self.spn_perc.get())
            min_area = int(float(self.spn_min_area.get()))
            max_pts = int(float(self.spn_maxpts.get()))
            iters = int(float(self.spn_iters.get()))
            tol_ang = float(self.spn_tolang.get())
            tol_relr = float(self.spn_tolr.get())
            dead_r = float(self.spn_dead.get())
            search_r = float(self.spn_search.get())

            pre_mode = self.cmb_pre.get()
            if pre_mode == "No smoothing":
                settings = PreprocSettings(mode="raw") # <-- Use "raw"
            else: # NLM Denoising
                h = float(self.spn_h.get())
                settings = PreprocSettings(mode="nlm", h_param=h) # <-- Use "nlm"

            try:
                arr = load_grayscale_with_preproc(image_path, settings)
            except RuntimeError as cv_err: messagebox.showerror("Dependency Error", str(cv_err)); return
            except Exception as img_load_err: messagebox.showerror("Image Error", f"Failed to load/process image:\n{img_load_err}"); return

            mode = settings.mode
            preproc_payload = settings.to_json()

            cx_txt = self.ent_cx.get().strip(); cy_txt = self.ent_cy.get().strip()
            if cx_txt and cy_txt:
                try: center0 = CenterResult(cy=float(cy_txt), cx=float(cx_txt), method="user")
                except ValueError: messagebox.showwarning("Input Warning", "Invalid center coords. Using automatic."); center0 = geometric_midpoint(arr)
            else: center0 = geometric_midpoint(arr)

            pts = detect_spots_by_centroid(arr, perc=perc, min_area=min_area, max_spots=max_pts)
            if len(pts) == 0:
                 print("Warning: No spots detected initially.")
                 lower_perc = max(85.0, perc - 5.0)
                 print(f"Retrying with percentile {lower_perc:.1f}%...")
                 pts = detect_spots_by_centroid(arr, perc=lower_perc, min_area=min_area, max_spots=max_pts)
                 if len(pts) == 0: messagebox.showwarning("Detection Warning", "No spots detected even with lower threshold.")

            center = refine_center_antipodal((center0.cy, center0.cx), pts, tol_ang_deg=tol_ang, tol_rel_r=tol_relr, iters=iters)

            if (dead_r > 0 or search_r > 0) and len(pts) > 0:
                dy = pts[:, 0] - center.cy; dx = pts[:, 1] - center.cx; r = np.hypot(dx, dy)
                mask = np.ones(len(pts), dtype=bool)
                if dead_r > 0:   mask &= (r >= dead_r)
                if search_r > 0: mask &= (r <= search_r)
                pts = pts[mask]

            # --- Merging REMOVED ---

            points_list = [{"y": float(y), "x": float(x), "intensity": float(v)} for (y, x, v) in pts.tolist()]
            saed_input_data = {
                "image": str(image_path),
                "preproc_mode": mode,
                "preproc": preproc_payload,
                "center": {"x": float(center.cx), "y": float(center.cy), "method": center.method},
                "radii": {"dead": float(dead_r), "search": float(search_r)},
                "points": points_list
            }
            saed_input_path = outdir / "saed_input.json"
            saed_input_path.write_text(json.dumps(saed_input_data, ensure_ascii=False, indent=2), encoding="utf-8")

            if self.controller is not None:
                try: self.controller.open_editor(saed_input_path)
                except Exception as exc: messagebox.showerror("Error", f"Failed to open editor tab:\n{exc}")
            else: messagebox.showwarning("Standalone Mode", "Running standalone. Editor will open externally if available.")

            try:
                (outdir/"center_init.json").write_text(json.dumps({
                    "initial": {"x": center0.cx, "y": center0.cy, "method": center0.method},
                    "refined": {"x": center.cx,  "y": center.cy,  "method": center.method},
                    "dead_zone_px": dead_r, "search_radius_px": search_r,
                    "preproc_mode": mode, "preproc": preproc_payload,
                    "image_size": {"H": int(arr.shape[0]), "W": int(arr.shape[1])}
                }, indent=2), encoding="utf-8")
            except Exception as log_err: print(f"Warning: Could not save center_init.json - {log_err}")

        except Exception as e: messagebox.showerror("Processing Error", f"Unexpected error:\n{e}")

class SAEDApp(tk.Tk):
    ( "Backwards-compatible standalone application using the tab frame." )
    def __init__(self):
        super().__init__()
        self.title("SAED Symmetry – Launcher")
        self.geometry("980x680")
        self.resizable(True, False)
        frame = SAEDLauncherFrame(self)
        frame.pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    SAEDApp().mainloop()