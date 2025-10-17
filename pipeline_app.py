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

# --- Добавлены модули для проверки прав администратора ---
import ctypes
import os
# ---------------------------------------------------------

import tkinter as tk
from tkinter import messagebox, ttk

try:
    import winreg
except ImportError:
    winreg = None

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

MODULE_DIR = Path(__file__).resolve().parent

_LSP1 = "Q2hhbmdlTWVUb0FQcml"
_LSP2 = "2YXRlU2VjcmV0"
LICENSE_SECRET = base64.b64decode(_LSP1 + _LSP2).decode("utf-8")

TRIAL_DAYS = 3


# --- Функции для проверки и запроса прав администратора ---
def is_admin():
    """Проверяет, запущена ли программа с правами администратора."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False

def run_as_admin():
    """Перезапускает текущий скрипт с правами администратора."""
    if sys.platform == 'win32':
        script = os.path.abspath(sys.argv[0])
        params = ' '.join([f'"{p}"' for p in sys.argv[1:]])
        try:
            ret = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script}" {params}', None, 1)
            return ret > 32
        except Exception as e:
            print(f"Failed to elevate privileges: {e}")
            return False
    return False
# ---------------------------------------------------------


class MaskedEntry(ttk.Entry):
    """An entry widget that enforces a mask for license key input."""
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.mask = "XXXX-XXXX-XXXX-XXXX-XXXX-XXXX"
        self.char_positions = [i for i, char in enumerate(self.mask) if char == 'X']
        self.var = tk.StringVar()
        self.configure(textvariable=self.var)
        self._last_value = ''
        self.var.trace_add("write", self._on_write)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<<Paste>>", self._on_paste)
        self._format_to_mask("")

    def _on_paste(self, _event=None):
        try:
            self._format_to_mask(self.clipboard_get())
        except tk.TclError:
            pass
        return "break"

    def _on_focus_in(self, _event=None):
        self._set_cursor_at_char_pos(len(self._get_raw_content()))

    def _get_raw_content(self) -> str:
        return "".join(char for i, char in enumerate(self.var.get()) if i in self.char_positions and char != 'X')

    def _format_to_mask(self, text: str):
        sanitized = "".join(filter(lambda c: c in "0123456789ABCDEFabcdef", text.upper()))[:len(self.char_positions)]
        new_value = list(self.mask)
        for i, char_pos in enumerate(self.char_positions):
            new_value[char_pos] = sanitized[i] if i < len(sanitized) else 'X'
        self._last_value = "".join(new_value)
        self.var.set(self._last_value)
        self._set_cursor_at_char_pos(len(sanitized))

    def _set_cursor_at_char_pos(self, char_index: int):
        cursor_pos = self.char_positions[char_index] if 0 <= char_index < len(self.char_positions) else self.char_positions[-1] + 1
        self.icursor(cursor_pos)

    def _on_write(self, *_args):
        if self.var.get() != self._last_value:
            self._format_to_mask(self._get_raw_content())

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
        ttk.Label(self, textvariable=self.feedback_var, foreground="#aa0000", wraplength=360).pack(anchor="w", pady=(0, 16))
        actions = ttk.Frame(self)
        actions.pack(fill=tk.X)
        ttk.Button(actions, text="Activate", command=self._on_activate, style="Accent.TButton").pack(side=tk.RIGHT)
        ttk.Button(actions, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT, padx=(0, 8))
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.wait_window(self)

    def _on_activate(self):
        if len(self.entry.get_key()) != 24:
            self.feedback_var.set("Please fill in the entire license key.")
            return
        self.result = self.entry.var.get()
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()


def _resource_path(filename: str) -> Path:
    candidates = []
    if (bundle_dir := getattr(sys, "_MEIPASS", None)) is not None:
        candidates.append(Path(bundle_dir, filename))
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / filename)
    candidates.append(MODULE_DIR / filename)
    return next((c for c in candidates if c.exists()), candidates[-1])


def _import_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        base_candidates = [Path(__file__).resolve().parent]
        if (frozen_base := getattr(sys, "_MEIPASS", None)) is not None:
            base_candidates.insert(0, Path(frozen_base))
        for base in base_candidates:
            for suffix in (".py", ".pyc"):
                if (candidate := base / f"{name}{suffix}").exists() and (module := _attempt_load(candidate, name)):
                    return module
            if (package_dir := base / name).is_dir():
                for suffix in (".py", ".pyc"):
                    if (init_file := package_dir / f"__init__{suffix}").exists() and (module := _attempt_load(init_file, name, package_dir=package_dir)):
                        return module
        raise exc

def _attempt_load(module_path: Path, name: str, *, package_dir: Path | None = None):
    spec = importlib.util.spec_from_file_location(name, module_path, submodule_search_locations=[str(package_dir)] if package_dir else None)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    return None

class LicenseManager:
    REG_PATH = r"Software\SAEDSuite"
    REG_KEY_TRIAL_START = "TrialStartDate"
    REG_KEY_LICENSE = "LicenseKey"

    def __init__(self, *, trial_days: int = TRIAL_DAYS):
        self.trial_days = trial_days
        self._data = self._load()

    def _default_data(self) -> dict[str, Optional[str]]:
        return {"trial_start": self._now().isoformat(), "license_key": None}

    def _load(self) -> dict[str, Optional[str]]:
        if not winreg: return self._default_data()
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.REG_PATH, 0, winreg.KEY_READ) as key:
                trial_start, _ = winreg.QueryValueEx(key, self.REG_KEY_TRIAL_START)
                license_key, _ = winreg.QueryValueEx(key, self.REG_KEY_LICENSE)
                return {"trial_start": trial_start, "license_key": license_key or None}
        except FileNotFoundError:
            data = self._default_data()
            self._save(data)
            return data
        except Exception:
            data = self._default_data()
            self._save(data)
            return data

    def _save(self, data: dict[str, Optional[str]]) -> None:
        if not winreg: return
        try:
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, self.REG_PATH) as key:
                winreg.SetValueEx(key, self.REG_KEY_TRIAL_START, 0, winreg.REG_SZ, data.get("trial_start") or self._now().isoformat())
                winreg.SetValueEx(key, self.REG_KEY_LICENSE, 0, winreg.REG_SZ, data.get("license_key") or "")
        except Exception as e:
            print(f"Warning: Could not save license data to registry: {e}")

    def _now(self) -> datetime:
        return datetime.utcnow()

    def _parse_timestamp(self, value: Optional[str]) -> datetime:
        if not value: return self._now()
        try: return datetime.fromisoformat(value)
        except ValueError: return self._now()

    def _normalize_key(self, key: str) -> str:
        cleaned = key.replace("-", "").replace(" ", "").upper()
        if not cleaned: raise ValueError("Empty license key")
        return "-".join(textwrap.wrap(cleaned, 4))

    def _validate_license_key(self, key: str) -> bool:
        cleaned = key.replace("-", "").upper()
        if len(cleaned) != 24 or any(c not in "0123456789ABCDEF" for c in cleaned): return False
        random_part, checksum = cleaned[:16], cleaned[16:]
        expected = hmac.new(LICENSE_SECRET.encode(), random_part.encode(), hashlib.sha256).hexdigest()[:8].upper()
        return secrets.compare_digest(checksum, expected)

    def has_valid_license(self) -> bool:
        return bool(self._data.get("license_key") and self._validate_license_key(self._data["license_key"]))

    def register_license_key(self, key: str) -> None:
        normalized = self._normalize_key(key)
        if not self._validate_license_key(normalized):
            raise ValueError("Invalid license key")
        self._data["license_key"] = normalized
        self._data["licensed_at"] = self._now().isoformat()
        self._save(self._data)

    def clear_license(self) -> None:
        self._data = self._default_data()
        self._save(self._data)

    def trial_start(self) -> datetime:
        return self._parse_timestamp(self._data.get("trial_start"))

    def trial_expiration(self) -> datetime:
        return self.trial_start() + timedelta(days=self.trial_days)

    def is_trial_expired(self) -> bool:
        return not self.has_valid_license() and self._now() >= self.trial_expiration()

    def trial_days_remaining(self) -> int:
        if self.has_valid_license(): return 0
        if (remaining := self.trial_expiration() - self._now()).total_seconds() <= 0: return 0
        return remaining.days + (1 if remaining.seconds > 0 else 0)

    def status_message(self) -> str:
        if self.has_valid_license():
            return "Permanent license activated. Thank you for supporting the project!"
        remaining = self.trial_days_remaining()
        if remaining == 0:
            return "Trial expired. Please enter a license key to continue using the application."
        plural = "s" if remaining > 1 else ""
        return f"Trial mode: {remaining} day{plural} remaining (expires on {self.trial_expiration().date():%Y-%m-%d}). " \
               "Enter a license key to unlock the full version permanently."


if TYPE_CHECKING:
    from temn import SAEDLauncherFrame
    from saed_editor import PointEditor
    from fibonachi_analysis import FibonacciAnalysisFrame
else:
    SAEDLauncherFrame = _import_module("temn").SAEDLauncherFrame
    PointEditor = _import_module("saed_editor").PointEditor
    FibonacciAnalysisFrame = _import_module("fibonachi_analysis").FibonacciAnalysisFrame


class PipelineController:
    def __init__(self, parent: tk.Misc, *, status_callback=None, license_manager: LicenseManager):
        self.parent = parent
        self._status_callback = status_callback or (lambda _: None)
        self.license_manager = license_manager
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.launcher = SAEDLauncherFrame(self.notebook, controller=self)
        self.editor = PointEditor(self.notebook, controller=self)
        self.analysis = FibonacciAnalysisFrame(self.notebook, controller=self, auto_load=False, license_manager=self.license_manager)
        self.notebook.add(self.launcher, text="Launcher")
        self.notebook.add(self.editor, text="Editor")
        self.notebook.add(self.analysis, text="Analysis")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def set_status(self, message: str):
        self._status_callback(message)

    def _on_tab_changed(self, _):
        if (current := self.notebook.select()):
            self.set_status(f"Opened tab: {self.notebook.tab(current, 'text')}")

    def open_editor(self, saed_json_path: Path | str):
        try:
            self.editor.load_input_json(Path(saed_json_path), push_undo=False)
            self.notebook.select(self.editor)
            self.set_status(f"Editor: {Path(saed_json_path).name}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load data into the editor:\n{exc}")

    def open_analysis(self, payload_path: Path | str, _: Optional[Path | str], __: Optional[Path | str]):
        try:
            self.analysis.load_json(Path(payload_path))
            self.notebook.select(self.analysis)
            self.set_status(f"Analysis: {Path(payload_path).name}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load data into the analyzer:\n{exc}")


def _show_splash(root: tk.Tk, *, logo_path: Path | str | None = None, duration_ms: int = 3000, background: str = "#59c6f1"):
    if duration_ms <= 0: return root.deiconify()
    splash = tk.Toplevel(root)
    splash.overrideredirect(True)
    splash.configure(background=background)
    frame = tk.Frame(splash, background=background)
    frame.pack(fill=tk.BOTH, expand=True)
    try:
        if logo_path and Image and ImageTk:
            with Image.open(logo_path) as pil_image:
                logo_image = ImageTk.PhotoImage(pil_image)
        else: logo_image = None
    except Exception: logo_image = None
    if logo_image:
        logo_label = tk.Label(frame, image=logo_image, background=background)
        logo_label.image = logo_image
        logo_label.pack(padx=32, pady=24)
    else:
        tk.Label(frame, text="SAED Symmetry\nLaunching…", justify="center", background=background,
                 foreground="#ffffff", font=("TkDefaultFont", 18, "bold"), padx=36, pady=28).pack()
    splash.update_idletasks()
    x = (splash.winfo_screenwidth() - splash.winfo_reqwidth()) // 2
    y = (splash.winfo_screenheight() - splash.winfo_reqheight()) // 2
    splash.geometry(f"+{x}+{y}")
    root.after(duration_ms, lambda: (splash.destroy(), root.deiconify()))


class TabbedPipelineApp(tk.Tk):
    def __init__(self, license_manager: LicenseManager, *, show_initially: bool = True):
        super().__init__()
        self.license_manager = license_manager
        if not show_initially: self.withdraw()
        self.title("SAED Symmetry — Suite")
        self.geometry("1520x980")
        style = ttk.Style(self)
        try: style.theme_use("clam")
        except tk.TclError: pass
        for s, f in [("Header.TLabel", (18, "bold")), ("Subheader.TLabel", (11,)), ("Byline.TLabel", (10, "italic")),
                     ("Accent.TButton", (10, "bold")), ("License.TLabel", (10,))]:
            style.configure(s, font=("TkDefaultFont", *f))
        style.configure("Byline.TLabel", foreground="#555555")
        style.configure("TNotebook", padding=(12, 10))
        style.configure("TNotebook.Tab", padding=(16, 8))
        header = ttk.Frame(self, padding=(20, 18, 20, 12))
        header.pack(side=tk.TOP, fill=tk.X)
        header.grid_columnconfigure(0, weight=1)
        ttk.Label(header, text="SAED Symmetry — Suite", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="A single pipeline for electron diffraction processing from loading to analysis.",
                  style="Subheader.TLabel", wraplength=720, justify="left").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(header, text="by RL 9-11 2025 v2.61 ", style="Byline.TLabel").grid(row=0, column=1, rowspan=2, sticky="ne", padx=(12, 0))
        ttk.Button(header, text="Help", command=self._show_help).grid(row=0, column=2, rowspan=2, sticky="ne")
        self.license_label = ttk.Label(header, text="", style="License.TLabel", wraplength=720, justify="left")
        self.license_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.license_button = ttk.Button(header, text="Enter License Key", command=self._prompt_for_license, style="Accent.TButton")
        self.license_button.grid(row=2, column=2, sticky="e", padx=(12, 0), pady=(12, 0))
        content = ttk.Frame(self, padding=(20, 0, 20, 12))
        content.pack(fill=tk.BOTH, expand=True)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(20, 8)).pack(side=tk.BOTTOM, fill=tk.X)
        self.controller = PipelineController(content, status_callback=self.status_var.set, license_manager=self.license_manager)
        self._refresh_license_banner()

    def _refresh_license_banner(self):
        self.license_label.configure(text=self.license_manager.status_message())
        self.license_button.configure(text="Update License Key" if self.license_manager.has_valid_license() else "Enter License Key")

    def _prompt_for_license(self):
        if (key := LicenseDialog(self, "License Key", "Enter the permanent license key provided by the publisher:").result):
            try:
                self.license_manager.register_license_key(key)
                messagebox.showinfo("License Key", "License activated successfully. Enjoy the full version!")
                self._refresh_license_banner()
            except ValueError:
                messagebox.showerror("License Key", "The provided license key is invalid. Please try again.")

    def _show_help(self):
        win = tk.Toplevel(self)
        win.title("About the application")
        win.transient(self)
        win.grab_set()
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=(20, 16))
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="In the Launcher tab, prepare the image and detector parameters. The Editor tab lets you refine points and radii manually, and Analysis builds a symmetry report with Fibonacci chains.",
                  justify="left", wraplength=480).pack(anchor="w")
        ttk.Label(frame, text="Support the project:", padding=(0, 12, 0, 0)).pack(anchor="w")
        link = "https://donatello.to/Roynik"
        link_label = tk.Label(frame, text=link, fg="#1a0dab", cursor="hand2", font=("TkDefaultFont", 10, "underline"), justify="left")
        link_label.pack(anchor="w")
        link_label.bind("<Button-1>", lambda _: webbrowser.open_new_tab(link))
        ttk.Button(frame, text="Close", command=win.destroy).pack(anchor="e", pady=(20, 0))


def _show_trial_expired_dialog(license_manager: LicenseManager) -> bool:
    root = tk.Tk()
    root.withdraw()
    key = LicenseDialog(root, "Trial Expired", "The 3-day trial period has ended. Please enter a valid license key to unlock the full version permanently.").result
    activated = False
    if key:
        try:
            license_manager.register_license_key(key)
            messagebox.showinfo("License Key", "License activated successfully. Thank you!", parent=root)
            activated = True
        except ValueError:
            messagebox.showerror("License Key", "The provided license key is invalid. Check the code and try again.", parent=root)
    root.destroy()
    return activated


def main(
    *,
    license_manager: LicenseManager,
    splash_logo: Path | str | None = None,
    splash_duration_ms: int = 3000,
) -> None:
    """Run the pipeline app, enforcing the trial and license policy."""
    if license_manager.is_trial_expired():
        if not _show_trial_expired_dialog(license_manager):
            return

    app = TabbedPipelineApp(license_manager, show_initially=False)
    _show_splash(app, logo_path=splash_logo, duration_ms=splash_duration_ms)
    app.mainloop()


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAED Symmetry pipeline application")
    parser.add_argument("--splash-logo", metavar="PATH", type=Path, help="Custom splash logo.")
    parser.add_argument("--no-splash", action="store_true", help="Skip the splash screen.")
    return parser


if __name__ == "__main__":
    if sys.platform == 'win32' and not is_admin():
        if run_as_admin():
            sys.exit(0)
        else:
            ctypes.windll.user32.MessageBoxW(0, "Для работы программы требуются права администратора.", "Ошибка", 0x10)
            sys.exit(1)

    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.no_splash:
        splash_duration = 0
        logo = None
    else:
        default_logo = _resource_path("logo.png")
        logo = args.splash_logo or (default_logo if default_logo.exists() else None)
        splash_duration = 3000

    license_manager = LicenseManager()
    main(
        license_manager=license_manager,
        splash_logo=logo,
        splash_duration_ms=splash_duration
    )#!/usr/bin/env python3
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

# --- Добавлены модули для проверки прав администратора ---
import ctypes
import os
# ---------------------------------------------------------

import tkinter as tk
from tkinter import messagebox, ttk

try:
    import winreg
except ImportError:
    winreg = None

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

MODULE_DIR = Path(__file__).resolve().parent

_LSP1 = "Q2hhbmdlTWVUb0FQcml"
_LSP2 = "2YXRlU2VjcmV0"
LICENSE_SECRET = base64.b64decode(_LSP1 + _LSP2).decode("utf-8")

TRIAL_DAYS = 3


# --- Функции для проверки и запроса прав администратора ---
def is_admin():
    """Проверяет, запущена ли программа с правами администратора."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False

def run_as_admin():
    """Перезапускает текущий скрипт с правами администратора."""
    if sys.platform == 'win32':
        script = os.path.abspath(sys.argv[0])
        params = ' '.join([f'"{p}"' for p in sys.argv[1:]])
        try:
            ret = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script}" {params}', None, 1)
            return ret > 32
        except Exception as e:
            print(f"Failed to elevate privileges: {e}")
            return False
    return False
# ---------------------------------------------------------


class MaskedEntry(ttk.Entry):
    """An entry widget that enforces a mask for license key input."""
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.mask = "XXXX-XXXX-XXXX-XXXX-XXXX-XXXX"
        self.char_positions = [i for i, char in enumerate(self.mask) if char == 'X']
        self.var = tk.StringVar()
        self.configure(textvariable=self.var)
        self._last_value = ''
        self.var.trace_add("write", self._on_write)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<<Paste>>", self._on_paste)
        self._format_to_mask("")

    def _on_paste(self, _event=None):
        try:
            self._format_to_mask(self.clipboard_get())
        except tk.TclError:
            pass
        return "break"

    def _on_focus_in(self, _event=None):
        self._set_cursor_at_char_pos(len(self._get_raw_content()))

    def _get_raw_content(self) -> str:
        return "".join(char for i, char in enumerate(self.var.get()) if i in self.char_positions and char != 'X')

    def _format_to_mask(self, text: str):
        sanitized = "".join(filter(lambda c: c in "0123456789ABCDEFabcdef", text.upper()))[:len(self.char_positions)]
        new_value = list(self.mask)
        for i, char_pos in enumerate(self.char_positions):
            new_value[char_pos] = sanitized[i] if i < len(sanitized) else 'X'
        self._last_value = "".join(new_value)
        self.var.set(self._last_value)
        self._set_cursor_at_char_pos(len(sanitized))

    def _set_cursor_at_char_pos(self, char_index: int):
        cursor_pos = self.char_positions[char_index] if 0 <= char_index < len(self.char_positions) else self.char_positions[-1] + 1
        self.icursor(cursor_pos)

    def _on_write(self, *_args):
        if self.var.get() != self._last_value:
            self._format_to_mask(self._get_raw_content())

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
        ttk.Label(self, textvariable=self.feedback_var, foreground="#aa0000", wraplength=360).pack(anchor="w", pady=(0, 16))
        actions = ttk.Frame(self)
        actions.pack(fill=tk.X)
        ttk.Button(actions, text="Activate", command=self._on_activate, style="Accent.TButton").pack(side=tk.RIGHT)
        ttk.Button(actions, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT, padx=(0, 8))
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.wait_window(self)

    def _on_activate(self):
        if len(self.entry.get_key()) != 24:
            self.feedback_var.set("Please fill in the entire license key.")
            return
        self.result = self.entry.var.get()
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()


def _resource_path(filename: str) -> Path:
    candidates = []
    if (bundle_dir := getattr(sys, "_MEIPASS", None)) is not None:
        candidates.append(Path(bundle_dir, filename))
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / filename)
    candidates.append(MODULE_DIR / filename)
    return next((c for c in candidates if c.exists()), candidates[-1])


def _import_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        base_candidates = [Path(__file__).resolve().parent]
        if (frozen_base := getattr(sys, "_MEIPASS", None)) is not None:
            base_candidates.insert(0, Path(frozen_base))
        for base in base_candidates:
            for suffix in (".py", ".pyc"):
                if (candidate := base / f"{name}{suffix}").exists() and (module := _attempt_load(candidate, name)):
                    return module
            if (package_dir := base / name).is_dir():
                for suffix in (".py", ".pyc"):
                    if (init_file := package_dir / f"__init__{suffix}").exists() and (module := _attempt_load(init_file, name, package_dir=package_dir)):
                        return module
        raise exc

def _attempt_load(module_path: Path, name: str, *, package_dir: Path | None = None):
    spec = importlib.util.spec_from_file_location(name, module_path, submodule_search_locations=[str(package_dir)] if package_dir else None)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    return None

class LicenseManager:
    REG_PATH = r"Software\SAEDSuite"
    REG_KEY_TRIAL_START = "TrialStartDate"
    REG_KEY_LICENSE = "LicenseKey"

    def __init__(self, *, trial_days: int = TRIAL_DAYS):
        self.trial_days = trial_days
        self._data = self._load()

    def _default_data(self) -> dict[str, Optional[str]]:
        return {"trial_start": self._now().isoformat(), "license_key": None}

    def _load(self) -> dict[str, Optional[str]]:
        if not winreg: return self._default_data()
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.REG_PATH, 0, winreg.KEY_READ) as key:
                trial_start, _ = winreg.QueryValueEx(key, self.REG_KEY_TRIAL_START)
                license_key, _ = winreg.QueryValueEx(key, self.REG_KEY_LICENSE)
                return {"trial_start": trial_start, "license_key": license_key or None}
        except FileNotFoundError:
            data = self._default_data()
            self._save(data)
            return data
        except Exception:
            data = self._default_data()
            self._save(data)
            return data

    def _save(self, data: dict[str, Optional[str]]) -> None:
        if not winreg: return
        try:
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, self.REG_PATH) as key:
                winreg.SetValueEx(key, self.REG_KEY_TRIAL_START, 0, winreg.REG_SZ, data.get("trial_start") or self._now().isoformat())
                winreg.SetValueEx(key, self.REG_KEY_LICENSE, 0, winreg.REG_SZ, data.get("license_key") or "")
        except Exception as e:
            print(f"Warning: Could not save license data to registry: {e}")

    def _now(self) -> datetime:
        return datetime.utcnow()

    def _parse_timestamp(self, value: Optional[str]) -> datetime:
        if not value: return self._now()
        try: return datetime.fromisoformat(value)
        except ValueError: return self._now()

    def _normalize_key(self, key: str) -> str:
        cleaned = key.replace("-", "").replace(" ", "").upper()
        if not cleaned: raise ValueError("Empty license key")
        return "-".join(textwrap.wrap(cleaned, 4))

    def _validate_license_key(self, key: str) -> bool:
        cleaned = key.replace("-", "").upper()
        if len(cleaned) != 24 or any(c not in "0123456789ABCDEF" for c in cleaned): return False
        random_part, checksum = cleaned[:16], cleaned[16:]
        expected = hmac.new(LICENSE_SECRET.encode(), random_part.encode(), hashlib.sha256).hexdigest()[:8].upper()
        return secrets.compare_digest(checksum, expected)

    def has_valid_license(self) -> bool:
        return bool(self._data.get("license_key") and self._validate_license_key(self._data["license_key"]))

    def register_license_key(self, key: str) -> None:
        normalized = self._normalize_key(key)
        if not self._validate_license_key(normalized):
            raise ValueError("Invalid license key")
        self._data["license_key"] = normalized
        self._data["licensed_at"] = self._now().isoformat()
        self._save(self._data)

    def clear_license(self) -> None:
        self._data = self._default_data()
        self._save(self._data)

    def trial_start(self) -> datetime:
        return self._parse_timestamp(self._data.get("trial_start"))

    def trial_expiration(self) -> datetime:
        return self.trial_start() + timedelta(days=self.trial_days)

    def is_trial_expired(self) -> bool:
        return not self.has_valid_license() and self._now() >= self.trial_expiration()

    def trial_days_remaining(self) -> int:
        if self.has_valid_license(): return 0
        if (remaining := self.trial_expiration() - self._now()).total_seconds() <= 0: return 0
        return remaining.days + (1 if remaining.seconds > 0 else 0)

    def status_message(self) -> str:
        if self.has_valid_license():
            return "Permanent license activated. Thank you for supporting the project!"
        remaining = self.trial_days_remaining()
        if remaining == 0:
            return "Trial expired. Please enter a license key to continue using the application."
        plural = "s" if remaining > 1 else ""
        return f"Trial mode: {remaining} day{plural} remaining (expires on {self.trial_expiration().date():%Y-%m-%d}). " \
               "Enter a license key to unlock the full version permanently."


if TYPE_CHECKING:
    from temn import SAEDLauncherFrame
    from saed_editor import PointEditor
    from fibonachi_analysis import FibonacciAnalysisFrame
else:
    SAEDLauncherFrame = _import_module("temn").SAEDLauncherFrame
    PointEditor = _import_module("saed_editor").PointEditor
    FibonacciAnalysisFrame = _import_module("fibonachi_analysis").FibonacciAnalysisFrame


class PipelineController:
    def __init__(self, parent: tk.Misc, *, status_callback=None, license_manager: LicenseManager):
        self.parent = parent
        self._status_callback = status_callback or (lambda _: None)
        self.license_manager = license_manager
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.launcher = SAEDLauncherFrame(self.notebook, controller=self)
        self.editor = PointEditor(self.notebook, controller=self)
        self.analysis = FibonacciAnalysisFrame(self.notebook, controller=self, auto_load=False, license_manager=self.license_manager)
        self.notebook.add(self.launcher, text="Launcher")
        self.notebook.add(self.editor, text="Editor")
        self.notebook.add(self.analysis, text="Analysis")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def set_status(self, message: str):
        self._status_callback(message)

    def _on_tab_changed(self, _):
        if (current := self.notebook.select()):
            self.set_status(f"Opened tab: {self.notebook.tab(current, 'text')}")

    def open_editor(self, saed_json_path: Path | str):
        try:
            self.editor.load_input_json(Path(saed_json_path), push_undo=False)
            self.notebook.select(self.editor)
            self.set_status(f"Editor: {Path(saed_json_path).name}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load data into the editor:\n{exc}")

    def open_analysis(self, payload_path: Path | str, _: Optional[Path | str], __: Optional[Path | str]):
        try:
            self.analysis.load_json(Path(payload_path))
            self.notebook.select(self.analysis)
            self.set_status(f"Analysis: {Path(payload_path).name}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load data into the analyzer:\n{exc}")


def _show_splash(root: tk.Tk, *, logo_path: Path | str | None = None, duration_ms: int = 3000, background: str = "#59c6f1"):
    if duration_ms <= 0: return root.deiconify()
    splash = tk.Toplevel(root)
    splash.overrideredirect(True)
    splash.configure(background=background)
    frame = tk.Frame(splash, background=background)
    frame.pack(fill=tk.BOTH, expand=True)
    try:
        if logo_path and Image and ImageTk:
            with Image.open(logo_path) as pil_image:
                logo_image = ImageTk.PhotoImage(pil_image)
        else: logo_image = None
    except Exception: logo_image = None
    if logo_image:
        logo_label = tk.Label(frame, image=logo_image, background=background)
        logo_label.image = logo_image
        logo_label.pack(padx=32, pady=24)
    else:
        tk.Label(frame, text="SAED Symmetry\nLaunching…", justify="center", background=background,
                 foreground="#ffffff", font=("TkDefaultFont", 18, "bold"), padx=36, pady=28).pack()
    splash.update_idletasks()
    x = (splash.winfo_screenwidth() - splash.winfo_reqwidth()) // 2
    y = (splash.winfo_screenheight() - splash.winfo_reqheight()) // 2
    splash.geometry(f"+{x}+{y}")
    root.after(duration_ms, lambda: (splash.destroy(), root.deiconify()))


class TabbedPipelineApp(tk.Tk):
    def __init__(self, license_manager: LicenseManager, *, show_initially: bool = True):
        super().__init__()
        self.license_manager = license_manager
        if not show_initially: self.withdraw()
        self.title("SAED Symmetry — Suite")
        self.geometry("1520x980")
        style = ttk.Style(self)
        try: style.theme_use("clam")
        except tk.TclError: pass
        for s, f in [("Header.TLabel", (18, "bold")), ("Subheader.TLabel", (11,)), ("Byline.TLabel", (10, "italic")),
                     ("Accent.TButton", (10, "bold")), ("License.TLabel", (10,))]:
            style.configure(s, font=("TkDefaultFont", *f))
        style.configure("Byline.TLabel", foreground="#555555")
        style.configure("TNotebook", padding=(12, 10))
        style.configure("TNotebook.Tab", padding=(16, 8))
        header = ttk.Frame(self, padding=(20, 18, 20, 12))
        header.pack(side=tk.TOP, fill=tk.X)
        header.grid_columnconfigure(0, weight=1)
        ttk.Label(header, text="SAED Symmetry — Suite", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="A single pipeline for electron diffraction processing from loading to analysis.",
                  style="Subheader.TLabel", wraplength=720, justify="left").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(header, text="by RL 9-11 2025 v2.61 ", style="Byline.TLabel").grid(row=0, column=1, rowspan=2, sticky="ne", padx=(12, 0))
        ttk.Button(header, text="Help", command=self._show_help).grid(row=0, column=2, rowspan=2, sticky="ne")
        self.license_label = ttk.Label(header, text="", style="License.TLabel", wraplength=720, justify="left")
        self.license_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.license_button = ttk.Button(header, text="Enter License Key", command=self._prompt_for_license, style="Accent.TButton")
        self.license_button.grid(row=2, column=2, sticky="e", padx=(12, 0), pady=(12, 0))
        content = ttk.Frame(self, padding=(20, 0, 20, 12))
        content.pack(fill=tk.BOTH, expand=True)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(20, 8)).pack(side=tk.BOTTOM, fill=tk.X)
        self.controller = PipelineController(content, status_callback=self.status_var.set, license_manager=self.license_manager)
        self._refresh_license_banner()

    def _refresh_license_banner(self):
        self.license_label.configure(text=self.license_manager.status_message())
        self.license_button.configure(text="Update License Key" if self.license_manager.has_valid_license() else "Enter License Key")

    def _prompt_for_license(self):
        if (key := LicenseDialog(self, "License Key", "Enter the permanent license key provided by the publisher:").result):
            try:
                self.license_manager.register_license_key(key)
                messagebox.showinfo("License Key", "License activated successfully. Enjoy the full version!")
                self._refresh_license_banner()
            except ValueError:
                messagebox.showerror("License Key", "The provided license key is invalid. Please try again.")

    def _show_help(self):
        win = tk.Toplevel(self)
        win.title("About the application")
        win.transient(self)
        win.grab_set()
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=(20, 16))
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="In the Launcher tab, prepare the image and detector parameters. The Editor tab lets you refine points and radii manually, and Analysis builds a symmetry report with Fibonacci chains.",
                  justify="left", wraplength=480).pack(anchor="w")
        ttk.Label(frame, text="Support the project:", padding=(0, 12, 0, 0)).pack(anchor="w")
        link = "https://donatello.to/Roynik"
        link_label = tk.Label(frame, text=link, fg="#1a0dab", cursor="hand2", font=("TkDefaultFont", 10, "underline"), justify="left")
        link_label.pack(anchor="w")
        link_label.bind("<Button-1>", lambda _: webbrowser.open_new_tab(link))
        ttk.Button(frame, text="Close", command=win.destroy).pack(anchor="e", pady=(20, 0))


def _show_trial_expired_dialog(license_manager: LicenseManager) -> bool:
    root = tk.Tk()
    root.withdraw()
    key = LicenseDialog(root, "Trial Expired", "The 3-day trial period has ended. Please enter a valid license key to unlock the full version permanently.").result
    activated = False
    if key:
        try:
            license_manager.register_license_key(key)
            messagebox.showinfo("License Key", "License activated successfully. Thank you!", parent=root)
            activated = True
        except ValueError:
            messagebox.showerror("License Key", "The provided license key is invalid. Check the code and try again.", parent=root)
    root.destroy()
    return activated


def main(
    *,
    license_manager: LicenseManager,
    splash_logo: Path | str | None = None,
    splash_duration_ms: int = 3000,
) -> None:
    """Run the pipeline app, enforcing the trial and license policy."""
    if license_manager.is_trial_expired():
        if not _show_trial_expired_dialog(license_manager):
            return

    app = TabbedPipelineApp(license_manager, show_initially=False)
    _show_splash(app, logo_path=splash_logo, duration_ms=splash_duration_ms)
    app.mainloop()


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAED Symmetry pipeline application")
    parser.add_argument("--splash-logo", metavar="PATH", type=Path, help="Custom splash logo.")
    parser.add_argument("--no-splash", action="store_true", help="Skip the splash screen.")
    return parser


if __name__ == "__main__":
    if sys.platform == 'win32' and not is_admin():
        if run_as_admin():
            sys.exit(0)
        else:
            ctypes.windll.user32.MessageBoxW(0, "Для работы программы требуются права администратора.", "Ошибка", 0x10)
            sys.exit(1)

    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.no_splash:
        splash_duration = 0
        logo = None
    else:
        default_logo = _resource_path("logo.png")
        logo = args.splash_logo or (default_logo if default_logo.exists() else None)
        splash_duration = 3000

    license_manager = LicenseManager()
    main(
        license_manager=license_manager,
        splash_logo=logo,
        splash_duration_ms=splash_duration
    )