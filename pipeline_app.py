#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified window with launcher, editor, and analyzer tabs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import importlib
import importlib.util
import json
import secrets
import sys
import textwrap
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import tkinter as tk
from tkinter import messagebox, ttk, filedialog

try:
    import winreg
except ImportError:
    winreg = None

try:
    from PIL import Image, ImageTk  # type: ignore[import-not-found]
except ImportError:
    Image = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]

MODULE_DIR = Path(__file__).resolve().parent

_LSP1 = "Q2hhbmdlTWVUb0FQcml"
_LSP2 = "2YXRlU2VjcmV0"
LICENSE_SECRET = base64.b64decode(_LSP1 + _LSP2).decode("utf-8")

TRIAL_DAYS = 3


class MaskedEntry(ttk.Entry):
    """An entry widget that enforces a mask for license key input."""

    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.mask = "XXXX-XXXX-XXXX-XXXX-XXXX-XXXX"
        self.char_positions = [i for i, char in enumerate(self.mask) if char == 'X']
        self.literal_positions = {i: char for i, char in enumerate(self.mask) if char != 'X'}
        self.var = tk.StringVar()
        self.configure(textvariable=self.var)
        self._last_value = ''
        self.var.trace_add("write", self._on_write)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<<Paste>>", self._on_paste)
        self._format_to_mask("")

    def _on_paste(self, _event=None):
        try:
            clipboard_content = self.clipboard_get()
            self._format_to_mask(clipboard_content)
        except tk.TclError:
            pass
        return "break"

    def _on_focus_in(self, _event=None):
        raw_content = self._get_raw_content()
        pos = len(raw_content)
        self._set_cursor_at_char_pos(pos)

    def _get_raw_content(self) -> str:
        return "".join(char for i, char in enumerate(self.var.get())
                       if i in self.char_positions and char != 'X')

    def _format_to_mask(self, text: str):
        sanitized = "".join(filter(lambda c: c in "0123456789ABCDEFabcdef", text.upper()))
        sanitized = sanitized[:len(self.char_positions)]
        new_value = list(self.mask)
        for i, char_pos in enumerate(self.char_positions):
            if i < len(sanitized):
                new_value[char_pos] = sanitized[i]
            else:
                new_value[char_pos] = 'X'
        self._last_value = "".join(new_value)
        self.var.set(self._last_value)
        self._set_cursor_at_char_pos(len(sanitized))

    def _set_cursor_at_char_pos(self, char_index: int):
        if 0 <= char_index < len(self.char_positions):
            cursor_pos = self.char_positions[char_index]
        else:
            cursor_pos = self.char_positions[-1] + 1
        self.icursor(cursor_pos)

    def _on_write(self, *_args):
        current_value = self.var.get()
        if current_value == self._last_value:
            return
        raw_content = self._get_raw_content()
        self._format_to_mask(raw_content)

    def get_key(self) -> str:
        return self._get_raw_content()


class LicenseDialog(tk.Toplevel):
    """A custom dialog for entering and validating a license key."""

    def __init__(self, parent, title, message):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title(title)
        self.resizable(False, False)
        self.configure(padx=24, pady=24)
        self.result = None
        ttk.Label(self, text=message, wraplength=360, justify="left").pack(anchor="w", pady=(0, 12))
        self.entry = MaskedEntry(self, width=32, font=("Courier", 10))
        self.entry.pack(fill=tk.X, pady=(4, 8))
        self.entry.focus_set()
        self.feedback_var = tk.StringVar(value="")
        feedback_label = ttk.Label(self, textvariable=self.feedback_var, foreground="#aa0000", wraplength=360)
        feedback_label.pack(anchor="w", pady=(0, 16))
        actions = ttk.Frame(self)
        actions.pack(fill=tk.X)
        ttk.Button(actions, text="Activate", command=self._on_activate, style="Accent.TButton").pack(side=tk.RIGHT)
        ttk.Button(actions, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT, padx=(0, 8))
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.wait_window(self)

    def _on_activate(self):
        key = self.entry.get_key()
        if len(key) != 24:
            self.feedback_var.set("Please fill in the entire license key.")
            return
        self.result = self.entry.var.get()
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()


def _resource_path(filename: str) -> Path:
    """Return an absolute path to *filename* that works in frozen bundles."""
    candidates: list[Path] = []
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir is not None:
        candidates.append(Path(bundle_dir, filename))
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / filename)
    candidates.append(MODULE_DIR / filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _import_module(name: str):
    """Import helper that falls back to sibling files when bundlers miss them."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        base_candidates = []
        frozen_base = getattr(sys, "_MEIPASS", None)
        if frozen_base is not None:
            base_candidates.append(Path(frozen_base))
        base_candidates.append(Path(__file__).resolve().parent)

        def _attempt_load(module_path: Path, *, package_dir: Path | None = None):
            spec_kwargs = {}
            if package_dir is not None:
                spec_kwargs["submodule_search_locations"] = [str(package_dir)]
            spec = importlib.util.spec_from_file_location(name, module_path, **spec_kwargs)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        for base in base_candidates:
            for suffix in (".py", ".pyc"):
                candidate = base / f"{name}{suffix}"
                if candidate.exists():
                    module = _attempt_load(candidate)
                    if module is not None:
                        return module
            package_dir = base / name
            if package_dir.is_dir():
                for suffix in (".py", ".pyc"):
                    init_file = package_dir / f"__init__{suffix}"
                    if init_file.exists():
                        module = _attempt_load(init_file, package_dir=package_dir)
                        if module is not None:
                            return module
        raise exc


class LicenseManager:
    """Handle trial and permanent license state."""
    REG_PATH = r"Software\SAEDSuite"
    REG_KEY_TRIAL_START = "TrialStartDate"
    REG_KEY_LICENSE = "LicenseKey"

    def __init__(self, *, trial_days: int = TRIAL_DAYS):
        self.trial_days = trial_days
        self._data = self._load()

    def _default_data(self) -> dict[str, Optional[str]]:
        return {"trial_start": self._now().isoformat(), "license_key": None}

    def _load(self) -> dict[str, Optional[str]]:
        if winreg is None:
            return self._default_data()
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.REG_PATH, 0, winreg.KEY_READ)
            trial_start_str, _ = winreg.QueryValueEx(key, self.REG_KEY_TRIAL_START)
            license_key_str, _ = winreg.QueryValueEx(key, self.REG_KEY_LICENSE)
            winreg.CloseKey(key)
            return {"trial_start": trial_start_str, "license_key": license_key_str or None}
        except FileNotFoundError:
            data = self._default_data()
            self._save(data)
            return data
        except Exception:
            # Handle potential registry read errors gracefully
            print("Warning: Could not read license data from registry. Using defaults.")
            data = self._default_data()
            # Attempt to save defaults back, might fail if permissions are wrong
            self._save(data)
            return data


    def _save(self, data: dict[str, Optional[str]]) -> None:
        if winreg is None:
            return
        try:
            key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, self.REG_PATH)
            trial_start = data.get("trial_start") or self._now().isoformat()
            winreg.SetValueEx(key, self.REG_KEY_TRIAL_START, 0, winreg.REG_SZ, trial_start)
            license_key = data.get("license_key") or ""
            winreg.SetValueEx(key, self.REG_KEY_LICENSE, 0, winreg.REG_SZ, license_key)
            winreg.CloseKey(key)
        except Exception as e:
            print(f"Warning: Could not save license data to registry: {e}")

    def _now(self) -> datetime:
        return datetime.utcnow()

    def _parse_timestamp(self, value: Optional[str]) -> datetime:
        if not value: return self._now()
        try:
            # Handle potential timezone info if present (though unlikely from registry)
            if value.endswith('Z'):
                 value = value[:-1] + '+00:00'
            return datetime.fromisoformat(value)
        except ValueError:
             # Fallback if the format is somehow corrupted
            print(f"Warning: Corrupted trial start date '{value}'. Resetting trial.")
            return self._now()


    def _normalize_key(self, key: str) -> str:
        cleaned = key.replace("-", "").replace(" ", "").upper()
        if not cleaned:
            raise ValueError("Empty license key")
        # Ensure it fits the XXXX-... format even if input is slightly off
        return "-".join(textwrap.wrap(cleaned.ljust(24, 'X')[:24], 4))


    def _validate_license_key(self, key: str) -> bool:
        cleaned = key.replace("-", "").upper()
        if len(cleaned) != 24 or any(char not in "0123456789ABCDEF" for char in cleaned):
            return False
        random_part = cleaned[:16]
        checksum = cleaned[16:]
        expected = hmac.new(
            LICENSE_SECRET.encode("utf-8"), random_part.encode("utf-8"), hashlib.sha256,
        ).hexdigest()[:8].upper()
        return secrets.compare_digest(checksum, expected)

    def has_valid_license(self) -> bool:
        key = self._data.get("license_key")
        if not key: return False
        return self._validate_license_key(key)

    def register_license_key(self, key: str) -> None:
        normalized = self._normalize_key(key)
        if not self._validate_license_key(normalized):
            raise ValueError("Invalid license key")
        self._data["license_key"] = normalized
        self._data["licensed_at"] = self._now().isoformat() # Optional: record activation time
        self._save(self._data)

    def clear_license(self) -> None:
        # Reset trial start date as well when clearing license
        self._data = self._default_data()
        self._save(self._data)


    def trial_start(self) -> datetime:
        return self._parse_timestamp(self._data.get("trial_start"))

    def trial_expiration(self) -> datetime:
        return self.trial_start() + timedelta(days=self.trial_days)

    def is_trial_expired(self) -> bool:
        if self.has_valid_license(): return False
        return self._now() >= self.trial_expiration()

    def trial_days_remaining(self) -> int:
        if self.has_valid_license(): return 0 # No trial days remaining if licensed
        expiration = self.trial_expiration()
        now = self._now()
        if now >= expiration: return 0
        remaining = expiration - now
        # Calculate remaining days, rounding up
        return remaining.days + (1 if remaining.seconds > 0 or remaining.microseconds > 0 else 0)


    def status_message(self) -> str:
        if self.has_valid_license():
            return "Permanent license activated. Thank you for supporting the project!"
        expiration = self.trial_expiration()
        remaining = self.trial_days_remaining()
        if remaining <= 0:
            return "Trial expired. Please enter a license key to continue using the application."
        plural = "day" if remaining == 1 else "days"
        # Show expiration date for clarity
        return (
            f"Trial mode: {remaining} {plural} remaining (expires on {expiration.date():%Y-%m-%d}). "
            "Enter a license key to unlock the full version permanently."
        )



if TYPE_CHECKING:
    from temn import SAEDLauncherFrame
    from saed_editor_refactored import PointEditor
    from fibonachi_analysis import FibonacciAnalysisFrame
else:
    try:
        from temn import SAEDLauncherFrame
        from saed_editor import PointEditor
        from fibonachi_analysis import FibonacciAnalysisFrame
    except ModuleNotFoundError:
        SAEDLauncherFrame = _import_module("temn").SAEDLauncherFrame
        PointEditor = _import_module("saed_editor").PointEditor
        FibonacciAnalysisFrame = _import_module("fibonachi_analysis").FibonacciAnalysisFrame


class PipelineController:
    """Connect the tabs and handle stage switching."""

    def __init__(self, parent: tk.Misc, *, status_callback=None, license_manager: LicenseManager):
        self.parent = parent
        self._status_callback = status_callback or (lambda _msg: None)
        self.license_manager = license_manager
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.launcher = SAEDLauncherFrame(self.notebook, controller=self)
        self.editor = PointEditor(self.notebook, controller=self)
        self.analysis = FibonacciAnalysisFrame(
            self.notebook, controller=self, auto_load=False, license_manager=self.license_manager
        )

        self.notebook.add(self.launcher, text="Launcher")
        self.notebook.add(self.editor, text="Editor")
        self.notebook.add(self.analysis, text="Analysis")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def set_status(self, message: str) -> None:
        self._status_callback(message)

    def _on_tab_changed(self, _event) -> None:
        current = self.notebook.select()
        if current:
            try:
                tab_text = self.notebook.tab(current, "text")
                self.set_status(f"Opened tab: {tab_text}")
            except tk.TclError: # Handle case where tab might be briefly invalid during changes
                self.set_status("Switching tabs...")

    def open_editor(self, saed_json_path: Path | str) -> None:
        path = Path(saed_json_path)
        if not path.exists():
            raise FileNotFoundError(f"Editor input file not found: {path}")
        try:
            # ИСПРАВЛЕННЫЙ ВЫЗОВ:
            self.editor.io.load_input_json(path, push_undo=False)
            self.notebook.select(self.editor)  # Switch to editor tab
            self.set_status(f"Editor: Loaded {path.name}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load data into the editor:\n{exc}")

    def open_analysis(
        self,
        payload_path: Path | str,
        image_path: Optional[Path | str], # These might be redundant if payload has all info
        spots_json: Optional[Path | str], # These might be redundant if payload has all info
    ) -> None:
        path = Path(payload_path)
        if not path.exists():
             raise FileNotFoundError(f"Analysis input file not found: {path}")
        try:
            # Pass the main payload path to load_json
            self.analysis.load_json(path)
            self.notebook.select(self.analysis) # Switch to analysis tab
            self.set_status(f"Analysis: Loaded {path.name}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load data into the analyzer:\n{exc}")


    # --- Session Save/Load Methods ---

    def save_session(self, filepath: Path | str) -> None:
        """Collect state from all tabs and save to a JSON file."""
        state = {
            'launcher': self.launcher.get_state(),
            'editor': self.editor.io.get_state(),
            'analysis': self.analysis.get_state(),
            'active_tab': self.notebook.index(self.notebook.select()) # Save current tab index
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            self.set_status(f"Session saved to {Path(filepath).name}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not write session file:\n{e}")
            raise # Re-raise for the caller to know


    def load_session_from_file(self, filepath: Path | str) -> None:
        """Load state from JSON and apply to all tabs."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)

            # --- Critical Order: Set launcher state FIRST ---
            # This ensures the output path is set before other modules might need it
            if 'launcher' in state:
                self.launcher.set_state(state['launcher'])
            else:
                 messagebox.showwarning("Load Warning", "Session file missing 'launcher' state. Some settings may be default.")


            # Now load editor and analysis state
            if 'editor' in state:
                self.editor.set_state(state['editor'])
            else:
                 messagebox.showwarning("Load Warning", "Session file missing 'editor' state. Editor may be empty.")
                 self.editor.set_state({}) # Clear editor state if missing


            if 'analysis' in state:
                self.analysis.set_state(state['analysis'])
            else:
                 messagebox.showwarning("Load Warning", "Session file missing 'analysis' state. Analysis tab may be empty.")
                 self.analysis.set_state({}) # Clear analysis state if missing


            # Restore the active tab
            active_tab_index = state.get('active_tab', 0)
            try:
                # Ensure the index is valid before selecting
                if 0 <= active_tab_index < self.notebook.index('end'):
                    self.notebook.select(active_tab_index)
                else:
                    self.notebook.select(0) # Fallback to first tab
            except tk.TclError:
                self.notebook.select(0) # Fallback on error

            self.set_status(f"Session loaded from {path.name}")

        except json.JSONDecodeError as e:
            messagebox.showerror("Load Error", f"Session file is corrupted or invalid JSON:\n{e}")
            raise
        except Exception as e:
            messagebox.showerror("Load Error", f"An unexpected error occurred while loading the session:\n{e}")
            raise # Re-raise for debugging


def _show_splash(
    root: tk.Tk,
    *,
    logo_path: Path | str | None = None,
    duration_ms: int = 3000,
    background: str = "#59c6f1", # This is the argument with the default value
) -> None:
    if duration_ms <= 0:
        root.deiconify()
        return

    splash = tk.Toplevel(root)
    splash.overrideredirect(True)
    # --- FIX: Use the 'background' argument directly ---
    splash.configure(background=background)
    frame = tk.Frame(splash, background=background) # Also use it here
    # --- End FIX ---
    frame.pack(fill=tk.BOTH, expand=True)

    logo_image = None
    if logo_path is not None:
        try:
            logo_file = Path(logo_path)
            if logo_file.exists():
                if Image is not None and ImageTk is not None:
                    with Image.open(logo_file) as pil_image:
                        # Optional: Resize if needed, e.g., pil_image.thumbnail((width, height))
                        logo_image = ImageTk.PhotoImage(pil_image)
                else:
                    # Fallback for systems without Pillow, might not handle all formats
                    logo_image = tk.PhotoImage(file=str(logo_file))
            else:
                 print(f"Warning: Splash logo not found at {logo_file}")
        except Exception as e: # Catch potential errors from Image.open or tk.PhotoImage
            print(f"Warning: Could not load splash logo: {e}")
            logo_image = None # Ensure it's None on failure


    if logo_image is not None:
        logo_label = tk.Label(frame, image=logo_image, background=background)
        logo_label.image = logo_image # Keep a reference!
        logo_label.pack(padx=32, pady=24)
    else:
        # Fallback text if logo fails or isn't provided
        tk.Label(
            frame, text="SAED Symmetry\nLaunching…", justify="center", background=background,
            foreground="#ffffff", font=("TkDefaultFont", 18, "bold"), padx=36, pady=28,
        ).pack()

    splash.update_idletasks() # Ensure dimensions are calculated
    width = splash.winfo_reqwidth()
    height = splash.winfo_reqheight()
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    splash.geometry(f"{width}x{height}+{x}+{y}") # Center the splash screen

    def _close_splash() -> None:
        try:
            if splash.winfo_exists():
                splash.destroy()
            if root.winfo_exists(): # Check if main window still exists
                root.deiconify() # Show main window
        except tk.TclError:
             pass # Ignore errors if widgets are already destroyed


    # Ensure the close function runs even if the app closes early
    splash.after(duration_ms, _close_splash)
    root.protocol("WM_DELETE_WINDOW", lambda: (_close_splash(), root.destroy())) # Handle main window close during splash


class TabbedPipelineApp(tk.Tk):
    """Main window containing every stage of the workflow."""

    def __init__(self, license_manager: LicenseManager, *, show_initially: bool = True):
        super().__init__()
        self.license_manager = license_manager
        if not show_initially:
            self.withdraw() # Hide main window initially
        self.title("SAED Symmetry — Suite")
        self.geometry("1520x980")
        self.resizable(True, True)

        # --- Style Configuration ---
        style = ttk.Style(self)
        available_themes = style.theme_names()
        # Prefer 'clam', 'alt', 'default' in that order
        preferred_themes = ['clam', 'alt', 'default']
        for theme in preferred_themes:
             if theme in available_themes:
                  try:
                       style.theme_use(theme)
                       break
                  except tk.TclError:
                       continue
        # Define custom styles
        style.configure("Header.TLabel", font=("TkDefaultFont", 18, "bold"))
        style.configure("Subheader.TLabel", font=("TkDefaultFont", 11))
        style.configure("Byline.TLabel", font=("TkDefaultFont", 10, "italic"), foreground="#555555")
        style.configure("Accent.TButton", font=("TkDefaultFont", 10, "bold"))
        style.configure("TNotebook", padding=(12, 10))
        style.configure("TNotebook.Tab", padding=(16, 8))
        style.configure("License.TLabel", font=("TkDefaultFont", 10))


        # --- Header ---
        header = ttk.Frame(self, padding=(20, 18, 20, 12))
        header.pack(side=tk.TOP, fill=tk.X)
        header.grid_columnconfigure(0, weight=1) # Allow title label to expand
        header.grid_columnconfigure(1, weight=0) # Column for byline
        header.grid_columnconfigure(2, weight=0) # Column for test button
        header.grid_columnconfigure(3, weight=0) # Column for help button

        ttk.Label(header, text="SAED Symmetry — Suite", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header, text="A single pipeline for electron diffraction processing from loading to analysis.",
            style="Subheader.TLabel", wraplength=720, justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(header, text="by RL 9-11 2025 v6.66 ", style="Byline.TLabel").grid(
            row=0, column=1, rowspan=2, sticky="ne", padx=(12, 0)
        )

        # --- Test Close Logic Button (moved here) ---
        # test_close_button = ttk.Button(header, text="Test Close Logic", command=self._on_close_window) # УДАЛЕНО ДЛЯ ЧИСТОТЫ
        # test_close_button.grid(row=0, column=2, rowspan=2, sticky="ne", padx=(12, 0))                 # УДАЛЕНО ДЛЯ ЧИСТОТЫ
        # --- End Move ---

        ttk.Button(header, text="Help", command=self._show_help).grid(
            row=0, column=3, rowspan=2, sticky="ne", padx=(12, 0) # Используем column=3
        )

        # --- License Info ---
        self.license_label = ttk.Label(header, text="", style="License.TLabel", wraplength=720, justify="left")
        self.license_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0)) # Spans first 2 columns
        self.license_button = ttk.Button(
            header, text="Enter License Key", command=self._prompt_for_license, style="Accent.TButton",
        )
        # Place license button in the last column, aligned right
        self.license_button.grid(row=2, column=3, sticky="e", padx=(12, 0), pady=(12, 0)) # Используем column=3


        # --- Main Content Area (Tabs) ---
        content = ttk.Frame(self, padding=(20, 0, 20, 12))
        content.pack(fill=tk.BOTH, expand=True)

        # --- Status Bar ---
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(20, 8), relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Initialize Controller (and its tabs) ---
        self.controller = PipelineController(content, status_callback=self._update_status, license_manager=self.license_manager)
        self.controller.set_status("Opened tab: Launcher") # Initial status

        # Refresh license banner after controller is initialized
        self._refresh_license_banner()

        # --- Bind save/close events ---
        self.bind_all("<Control-s>", self._on_save_shortcut)
        self.protocol("WM_DELETE_WINDOW", self._on_close_window)
        # print("DEBUG: WM_DELETE_WINDOW protocol handler SET") # УДАЛЕНО ДЛЯ ЧИСТОТЫ


    def _update_status(self, message: str) -> None:
        self.status_var.set(message)

    def _refresh_license_banner(self) -> None:
        message = self.license_manager.status_message()
        self.license_label.configure(text=message)
        # Change button text based on license status
        if self.license_manager.has_valid_license():
            self.license_button.configure(text="Update License")
        else:
            self.license_button.configure(text="Enter License Key")


    def _prompt_for_license(self) -> None:
        prompt_message = "Enter the permanent license key provided by the publisher:"
        dialog = LicenseDialog(self, "License Key", prompt_message)
        key = dialog.result # This will be None if cancelled
        if key is None:
            self._update_status("License entry cancelled.")
            return
        try:
            self.license_manager.register_license_key(key)
            messagebox.showinfo("License Key", "License activated successfully. Enjoy the full version!")
            self._update_status("License activated.")
        except ValueError:
            messagebox.showerror("License Key", "The provided license key is invalid. Please try again.")
            self._update_status("Invalid license key entered.")

        self._refresh_license_banner() # Update banner regardless of success


    def _show_help(self) -> None:
        help_window = tk.Toplevel(self)
        help_window.title("About the application")
        help_window.transient(self) # Make it behave like a dialog relative to the main window
        help_window.grab_set() # Prevent interaction with main window while help is open
        help_window.resizable(False, False)

        frame = ttk.Frame(help_window, padding=(20, 16))
        frame.pack(fill=tk.BOTH, expand=True)

        message = (
            "This application provides a workflow for SAED pattern analysis:\n\n"
            "1.  **Launcher:** Load an image, set preprocessing options, define center/radii (or use auto-detection), and generate initial points.\n"
            "2.  **Editor:** Manually refine the detected points (add, delete, merge, measure distances) and adjust the center overlay.\n"
            "3.  **Analysis:** Perform symmetry analysis based on the refined points, focusing on Fibonacci chains and polygon properties.\n\n"
            "Use **Ctrl+S** to save the current state (settings, points, analyses) to a `saed_session.json` file in the output folder. Use **Load Session...** in the Launcher to restore a previous state."
        )
        ttk.Label(frame, text=message, justify="left", wraplength=480).pack(anchor="w")

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(12, 8))

        ttk.Label(frame, text="Support the project:").pack(anchor="w", pady=(0, 2))
        donation_link = "https://donatello.to/Roynik"
        link_label = tk.Label(
            frame, text=donation_link, fg="#1a0dab", cursor="hand2",
            font=("TkDefaultFont", 10, "underline"), justify="left",
        )
        link_label.pack(anchor="w")
        link_label.bind("<Button-1>", lambda _event: webbrowser.open_new_tab(donation_link))

        # Close button at the bottom right
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        ttk.Button(button_frame, text="Close", command=help_window.destroy).pack(side=tk.RIGHT)


    # --- Session Save/Load Handlers ---

    # <<< НАЧАЛО _on_save_shortcut БЕЗ ОТЛАДКИ >>>
    def _on_save_shortcut(self, event=None) -> bool:
        """Saves the current session state to saed_session.json in the output folder."""
        # Check if controller and launcher exist
        if not hasattr(self, 'controller') or not hasattr(self.controller, 'launcher'):
            messagebox.showerror("Save Error", "Application components not fully initialized.", parent=self)
            return False

        output_dir_str = self.controller.launcher.ent_out.get()
        if not output_dir_str:
            messagebox.showerror("Save Error", "Please specify an 'Output folder' in the Launcher tab first.", parent=self)
            return False

        output_dir = Path(output_dir_str)
        try:
            output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            filepath = output_dir / "saed_session.json"
            self.controller.save_session(filepath) # Delegate saving to controller
            # Status update already done in controller.save_session
            return True
        except Exception as e:
            # Show error relative to main window
            messagebox.showerror("Save Error", f"Failed to save session:\n{e}", parent=self)
            return False
    # <<< КОНЕЦ _on_save_shortcut БЕЗ ОТЛАДКИ >>>

    # <<< ИСПРАВЛЕННАЯ ФУНКЦИЯ _on_close_window БЕЗ ОТЛАДКИ >>>
    def _on_close_window(self) -> None:
        """Prompts to save on close, then destroys the window."""
        result = messagebox.askyesnocancel(
            "Confirm Exit",
            "Save current session before closing?",
            parent=self # Make dialog modal to this window
        )

        if result is True: # Yes
            save_successful = self._on_save_shortcut() # Attempt save
            if save_successful:
                self.destroy() # Close if save worked
            else:
                # Inform the user that save failed and window stays open
                messagebox.showwarning(
                    "Save Failed",
                    "Could not save the session. Please check the output folder and try again.\n\nThe application will remain open.",
                    parent=self
                )
                # Keep the window open - do nothing more here
        elif result is False: # No
            self.destroy() # Close without saving
        # else: Cancel (result is None), do nothing - window stays open
    # <<< КОНЕЦ ИСПРАВЛЕНИЯ БЕЗ ОТЛАДКИ >>>


def _show_trial_expired_dialog(license_manager: LicenseManager) -> bool:
    root = tk.Tk()
    root.withdraw() # Keep root hidden
    message = (
        "The 3-day trial period has ended. "
        "Please enter a valid license key to unlock the full version permanently."
    )
    # Ensure dialog is transient to the hidden root
    dialog = LicenseDialog(root, "Trial Expired", message)
    key = dialog.result # Blocks until dialog is closed
    activated = False
    if key:
        try:
            license_manager.register_license_key(key)
            # Use root as parent for messagebox
            messagebox.showinfo("License Key", "License activated successfully. Thank you!", parent=root)
            activated = True
        except ValueError:
             # Use root as parent for messagebox
            messagebox.showerror("License Key", "The provided license key is invalid. Check the code and try again.", parent=root)
            activated = False # Explicitly set to False on error

    root.destroy() # Clean up hidden root window
    return activated


def main(
    *,
    splash_logo: Path | str | None = None,
    splash_duration_ms: int = 3000,
) -> None:
    license_manager = LicenseManager()

    # Check license status BEFORE creating the main app window
    if not license_manager.has_valid_license() and license_manager.is_trial_expired():
        activated = _show_trial_expired_dialog(license_manager)
        if not activated:
            print("Trial expired and no valid license provided. Exiting.")
            return # Exit if trial expired and activation failed/cancelled

    # If license is okay (or trial active), proceed to create main app
    app = TabbedPipelineApp(license_manager, show_initially=False) # Keep hidden for splash

    # Show splash screen, which will deiconify the app window when done
    _show_splash(app, logo_path=splash_logo, duration_ms=splash_duration_ms)

    app.mainloop() # Start the Tkinter event loop


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAED Symmetry pipeline application")
    parser.add_argument(
        "--splash-logo", metavar="PATH", type=Path,
        help="Custom splash logo to display when launching the application.",
    )
    parser.add_argument(
        "--no-splash", action="store_true",
        help="Skip the splash screen when launching the graphical interface.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_cli_parser()
    args = parser.parse_args()

    splash_duration = 0
    logo = None
    if not args.no_splash:
        # Determine logo path, checking if default exists
        default_logo_path = _resource_path("logo.png")
        if args.splash_logo:
             logo_path_to_use = args.splash_logo
        elif default_logo_path.exists():
             logo_path_to_use = default_logo_path
        else:
             logo_path_to_use = None # No logo found or specified

        if logo_path_to_use:
            logo = logo_path_to_use
            splash_duration = 3000 # Default duration if logo exists
        else:
             # If no logo, maybe a shorter splash or text-only splash?
             # For now, keep duration but logo will be None
             splash_duration = 2000 # Shorter splash if text only
             print("Note: No splash logo found or specified.")


    main(splash_logo=logo, splash_duration_ms=splash_duration)