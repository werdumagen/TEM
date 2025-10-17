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
import os  # <-- Импорт оставлен на всякий случай, но не используется в новой логике
import ctypes  # <-- Используется для проверки прав и перезапуска
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import tkinter as tk
from tkinter import messagebox, ttk

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
            data = self._default_data()
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
            return datetime.fromisoformat(value)
        except ValueError:
            return self._now()

    def _normalize_key(self, key: str) -> str:
        cleaned = key.replace("-", "").replace(" ", "").upper()
        if not cleaned:
            raise ValueError("Empty license key")
        return "-".join(textwrap.wrap(cleaned, 4))

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
        if self.has_valid_license(): return False
        return self._now() >= self.trial_expiration()

    def trial_days_remaining(self) -> int:
        if self.has_valid_license(): return 0
        expiration = self.trial_expiration()
        now = self._now()
        if now >= expiration: return 0
        remaining = expiration - now
        return remaining.days + (1 if remaining.seconds > 0 else 0)

    def status_message(self) -> str:
        if self.has_valid_license():
            return "Permanent license activated. Thank you for supporting the project!"
        expiration = self.trial_expiration()
        remaining = self.trial_days_remaining()
        if remaining == 0:
            return "Trial expired. Please enter a license key to continue using the application."
        plural = "day" if remaining == 1 else "days"
        return (
            f"Trial mode: {remaining} {plural} remaining (expires on {expiration.date():%Y-%m-%d}). "
            "Enter a license key to unlock the full version permanently."
        )


if TYPE_CHECKING:
    from temn import SAEDLauncherFrame
    from saed_editor import PointEditor
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
            tab_text = self.notebook.tab(current, "text")
            self.set_status(f"Opened tab: {tab_text}")

    def open_editor(self, saed_json_path: Path | str) -> None:
        path = Path(saed_json_path)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            self.editor.load_input_json(path, push_undo=False)
            self.notebook.select(self.editor)
            self.set_status(f"Editor: {path.name}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load data into the editor:\n{exc}")

    def open_analysis(
            self,
            payload_path: Path | str,
            image_path: Optional[Path | str],
            spots_json: Optional[Path | str],
    ) -> None:
        path = Path(payload_path)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            self.analysis.load_json(path)
            self.notebook.select(self.analysis)
            self.set_status(f"Analysis: {path.name}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load data into the analyzer:\n{exc}")


def _show_splash(
        root: tk.Tk,
        *,
        logo_path: Path | str | None = None,
        duration_ms: int = 3000,
        background: str = "#59c6f1",
) -> None:
    if duration_ms <= 0:
        root.deiconify()
        return

    splash = tk.Toplevel(root)
    splash.overrideredirect(True)
    splash.configure(background=background)
    frame = tk.Frame(splash, background=background)
    frame.pack(fill=tk.BOTH, expand=True)
    logo_image = None
    if logo_path is not None:
        try:
            if Image is not None and ImageTk is not None:
                with Image.open(logo_path) as pil_image:
                    logo_image = ImageTk.PhotoImage(pil_image)
            else:
                logo_image = tk.PhotoImage(file=str(logo_path))
        except (OSError, tk.TclError):
            logo_image = None
    if logo_image is not None:
        logo_label = tk.Label(frame, image=logo_image, background=background)
        logo_label.image = logo_image
        logo_label.pack(padx=32, pady=24)
    else:
        tk.Label(
            frame, text="SAED Symmetry\nLaunching…", justify="center", background=background,
            foreground="#ffffff", font=("TkDefaultFont", 18, "bold"), padx=36, pady=28,
        ).pack()
    splash.update_idletasks()
    width = splash.winfo_reqwidth()
    height = splash.winfo_reqheight()
    x = (splash.winfo_screenwidth() // 2) - (width // 2)
    y = (splash.winfo_screenheight() // 2) - (height // 2)
    splash.geometry(f"{width}x{height}+{x}+{y}")

    def _close_splash() -> None:
        if splash.winfo_exists():
            splash.destroy()
        root.deiconify()

    root.after(duration_ms, _close_splash)


class TabbedPipelineApp(tk.Tk):
    """Main window containing every stage of the workflow."""

    def __init__(self, license_manager: LicenseManager, *, show_initially: bool = True):
        super().__init__()
        self.license_manager = license_manager
        if not show_initially:
            self.withdraw()
        self.title("SAED Symmetry — Suite")
        self.geometry("1520x980")
        self.resizable(True, True)
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Header.TLabel", font=("TkDefaultFont", 18, "bold"))
        style.configure("Subheader.TLabel", font=("TkDefaultFont", 11))
        style.configure("Byline.TLabel", font=("TkDefaultFont", 10, "italic"), foreground="#555555")
        style.configure("Accent.TButton", font=("TkDefaultFont", 10, "bold"))
        style.configure("TNotebook", padding=(12, 10))
        style.configure("TNotebook.Tab", padding=(16, 8))
        style.configure("License.TLabel", font=("TkDefaultFont", 10))
        header = ttk.Frame(self, padding=(20, 18, 20, 12))
        header.pack(side=tk.TOP, fill=tk.X)
        header.grid_columnconfigure(0, weight=1)
        ttk.Label(header, text="SAED Symmetry — Suite", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header, text="A single pipeline for electron diffraction processing from loading to analysis.",
            style="Subheader.TLabel", wraplength=720, justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(header, text="by RL 9-11 2025 v3.5.2 ", style="Byline.TLabel").grid(
            row=0, column=1, rowspan=2, sticky="ne", padx=(12, 0)
        )
        ttk.Button(header, text="Help", command=self._show_help).grid(
            row=0, column=2, rowspan=2, sticky="ne"
        )
        self.license_label = ttk.Label(header, text="", style="License.TLabel", wraplength=720, justify="left")
        self.license_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.license_button = ttk.Button(
            header, text="Enter License Key", command=self._prompt_for_license, style="Accent.TButton",
        )
        self.license_button.grid(row=2, column=2, sticky="e", padx=(12, 0), pady=(12, 0))
        content = ttk.Frame(self, padding=(20, 0, 20, 12))
        content.pack(fill=tk.BOTH, expand=True)
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(20, 8))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.controller = PipelineController(content, status_callback=self._update_status,
                                             license_manager=self.license_manager)
        self.controller.set_status("Opened tab: Launcher")
        self._refresh_license_banner()

    def _update_status(self, message: str) -> None:
        self.status_var.set(message)

    def _refresh_license_banner(self) -> None:
        message = self.license_manager.status_message()
        self.license_label.configure(text=message)
        if self.license_manager.has_valid_license():
            self.license_button.configure(text="Update License Key")
        else:
            self.license_button.configure(text="Enter License Key")

    def _prompt_for_license(self) -> None:
        prompt_message = "Enter the permanent license key provided by the publisher:"
        dialog = LicenseDialog(self, "License Key", prompt_message)
        key = dialog.result
        if key is None:
            return
        try:
            self.license_manager.register_license_key(key)
        except ValueError:
            messagebox.showerror("License Key", "The provided license key is invalid. Please try again.")
            return
        messagebox.showinfo("License Key", "License activated successfully. Enjoy the full version!")
        self._refresh_license_banner()

    def _show_help(self) -> None:
        help_window = tk.Toplevel(self)
        help_window.title("About the application")
        help_window.transient(self)
        help_window.grab_set()
        help_window.resizable(False, False)
        frame = ttk.Frame(help_window, padding=(20, 16))
        frame.pack(fill=tk.BOTH, expand=True)
        message = (
            "In the Launcher tab, prepare the image and detector parameters. "
            "The Editor tab lets you refine points and radii manually, and Analysis builds "
            "a symmetry report with Fibonacci chains."
        )
        ttk.Label(frame, text=message, justify="left", wraplength=480).pack(anchor="w")
        ttk.Label(frame, text="Support the project:", padding=(0, 12, 0, 0)).pack(anchor="w")
        donation_link = "https://donatello.to/Roynik"
        link_label = tk.Label(
            frame, text=donation_link, fg="#1a0dab", cursor="hand2",
            font=("TkDefaultFont", 10, "underline"), justify="left",
        )
        link_label.pack(anchor="w")
        link_label.bind("<Button-1>", lambda _event: webbrowser.open_new_tab(donation_link))
        ttk.Button(frame, text="Close", command=help_window.destroy).pack(
            anchor="e", pady=(20, 0)
        )


def _show_trial_expired_dialog(license_manager: LicenseManager) -> bool:
    root = tk.Tk()
    root.withdraw()
    message = (
        "The 3-day trial period has ended. "
        "Please enter a valid license key to unlock the full version permanently."
    )
    dialog = LicenseDialog(root, "Trial Expired", message)
    key = dialog.result
    activated = False
    if key:
        try:
            license_manager.register_license_key(key)
            messagebox.showinfo("License Key", "License activated successfully. Thank you!", parent=root)
            activated = True
        except ValueError:
            messagebox.showerror("License Key", "The provided license key is invalid. Check the code and try again.",
                                 parent=root)
            activated = False
    root.destroy()
    return activated


# --- НОВАЯ ФУНКЦИЯ ПРОВЕРКИ И ПОВЫШЕНИЯ ПРАВ ---
def _ensure_admin_privileges() -> bool:
    """
    Проверяет, есть ли права администратора. Если нет, пытается перезапустить
    приложение с запросом UAC.

    Возвращает True, если можно продолжать (уже админ).
    Возвращает False, если нужно выйти (перезапуск или отказ пользователя).
    """
    if sys.platform != 'win32':
        return True  # Проверка только для Windows

    try:
        # Проверяем, запущена ли программа УЖЕ с правами администратора
        is_admin = (ctypes.windll.shell32.IsUserAnAdmin() == 1)
    except Exception:
        is_admin = False  # Если проверка не удалась, считаем, что прав нет

    if is_admin:
        return True  # Права уже есть, продолжаем

    # --- Прав нет. Пытаемся перезапустить себя с правами ---
    try:
        # Формируем строку параметров, корректно экранируя пути с пробелами
        params_list = []
        for arg in sys.argv[1:]:
            # Добавляем кавычки, если в аргументе есть пробел и он еще не в кавычках
            if " " in arg and not (arg.startswith('"') and arg.endswith('"')):
                params_list.append(f'"{arg}"')
            else:
                params_list.append(arg)
        params = " ".join(params_list)

        # Выполняем ShellExecute с операцией "runas" (запрос UAC)
        ret = ctypes.windll.shell32.ShellExecuteW(
            None,  # hwnd
            "runas",  # lpOperation
            sys.executable,  # lpFile (python.exe или App.exe)
            params,  # lpParameters (аргументы скрипта/приложения)
            None,  # lpDirectory
            1  # nShowCmd (SW_SHOWNORMAL)
        )

        if ret > 32:
            # UAC был показан, и пользователь, вероятно, нажал "Да".
            # Новый (админский) процесс запущен.
            # Этот (старый) процесс должен немедленно завершиться.
            return False  # Сигнал главному процессу на выход
        else:
            # Код ошибки (<= 32).
            # Самая частая: 1223 (ERROR_CANCELLED) - пользователь нажал "Нет"
            temp_root = tk.Tk()
            temp_root.withdraw()
            messagebox.showerror(
                "Требуются права администратора",
                "Для корректной работы (особенно из защищенных папок) "
                "приложению требуются права администратора.\n\n"
                "В доступе отказано. Приложение будет закрыто."
            )
            temp_root.destroy()
            return False  # Сигнал главному процессу на выход

    except Exception as e:
        # Неожиданная ошибка при попытке перезапуска
        temp_root = tk.Tk()
        temp_root.withdraw()
        messagebox.showerror(
            "Ошибка запуска",
            f"Не удалось перезапустить приложение с правами администратора: {e}"
        )
        temp_root.destroy()
        return False  # Сигнал главному процессу на выход


# --- КОНЕЦ НОВОЙ ФУНКЦИИ ---


def main(
        *,
        splash_logo: Path | str | None = None,
        splash_duration_ms: int = 3000,
) -> None:
    # --- ИЗМЕНЕНИЕ: Проверка прав в самом начале ---
    # Эта функция будет вызвана до создания ЛЮБЫХ окон
    if not _ensure_admin_privileges():
        sys.exit(0)  # Если функция вернула False, завершаем работу
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    license_manager = LicenseManager()
    if not license_manager.has_valid_license() and license_manager.is_trial_expired():
        activated = _show_trial_expired_dialog(license_manager)
        if not activated:
            return

    # Если мы дошли сюда, у нас есть права, и лицензия в порядке
    app = TabbedPipelineApp(license_manager, show_initially=False)
    _show_splash(app, logo_path=splash_logo, duration_ms=splash_duration_ms)
    app.mainloop()


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
    if args.no_splash:
        splash_duration = 0
        logo = None
    else:
        default_logo = _resource_path("logo.png")
        logo = args.splash_logo if args.splash_logo is not None else (default_logo if default_logo.exists() else None)
        splash_duration = 3000
    main(splash_logo=logo, splash_duration_ms=splash_duration)