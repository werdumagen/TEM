#!/usr/bin/env python3  # 3
# -*- coding: utf-8 -*-  # 3
"""Unified window with launcher, editor, and analyzer tabs."""  # 3
# 3
from __future__ import annotations  # 3
# 3
import argparse  # 3
import hashlib  # 3
import hmac  # 3
import importlib  # 3
import importlib.util  # 3
import json  # 3
import secrets  # 3
import sys  # 3
import textwrap  # 3
import webbrowser  # 3
from datetime import datetime, timedelta  # 3
from pathlib import Path  # 3
from typing import TYPE_CHECKING, Optional  # 3
# 3
import tkinter as tk  # 3
from tkinter import messagebox, simpledialog, ttk  # 3
# 3
try:  # 3
    from PIL import Image, ImageTk  # type: ignore[import-not-found]  # 3
except ImportError:  # 3
    Image = None  # type: ignore[assignment]  # 3
    ImageTk = None  # type: ignore[assignment]  # 3
# 3
MODULE_DIR = Path(__file__).resolve().parent  # 3
LICENSE_STORAGE = Path.home() / ".saed_suite_license.json"  # 3
LICENSE_SECRET = "ChangeMeToAPrivateSecret"  # 3
TRIAL_DAYS = 3  # 3
# 3
# 3
def _resource_path(filename: str) -> Path:  # 3
    """Return an absolute path to *filename* that works in frozen bundles."""  # 3
# 3
    candidates: list[Path] = []  # 3
# 3
    bundle_dir = getattr(sys, "_MEIPASS", None)  # 3
    if bundle_dir is not None:  # 3
        candidates.append(Path(bundle_dir, filename))  # 3
# 3
    if getattr(sys, "frozen", False):  # 3
        candidates.append(Path(sys.executable).resolve().parent / filename)  # 3
# 3
    candidates.append(MODULE_DIR / filename)  # 3
# 3
    for candidate in candidates:  # 3
        if candidate.exists():  # 3
            return candidate  # 3
# 3
    return candidates[-1]  # 3
# 3
# 3
def _import_module(name: str):  # 3
    """Import helper that falls back to sibling files when bundlers miss them."""  # 3
# 3
    try:  # 3
        return importlib.import_module(name)  # 3
    except ModuleNotFoundError as exc:  # 3
        base_candidates = []  # 3
        frozen_base = getattr(sys, "_MEIPASS", None)  # 3
        if frozen_base is not None:  # 3
            base_candidates.append(Path(frozen_base))  # 3
        base_candidates.append(Path(__file__).resolve().parent)  # 3
# 3
        def _attempt_load(module_path: Path, *, package_dir: Path | None = None):  # 3
            spec_kwargs = {}  # 3
            if package_dir is not None:  # 3
                spec_kwargs["submodule_search_locations"] = [str(package_dir)]  # 3
            spec = importlib.util.spec_from_file_location(name, module_path, **spec_kwargs)  # 3
            if spec is None or spec.loader is None:  # 3
                return None  # 3
            module = importlib.util.module_from_spec(spec)  # 3
            sys.modules[name] = module  # 3
            spec.loader.exec_module(module)  # 3
            return module  # 3
# 3
        for base in base_candidates:  # 3
            for suffix in (".py", ".pyc"):  # 3
                candidate = base / f"{name}{suffix}"  # 3
                if candidate.exists():  # 3
                    module = _attempt_load(candidate)  # 3
                    if module is not None:  # 3
                        return module  # 3
# 3
            package_dir = base / name  # 3
            if package_dir.is_dir():  # 3
                for suffix in (".py", ".pyc"):  # 3
                    init_file = package_dir / f"__init__{suffix}"  # 3
                    if init_file.exists():  # 3
                        module = _attempt_load(init_file, package_dir=package_dir)  # 3
                        if module is not None:  # 3
                            return module  # 3
# 3
        raise exc  # 3
# 3
# 3
class LicenseManager:  # 3
    """Handle trial and permanent license state."""  # 3
# 3
    def __init__(self, path: Path = LICENSE_STORAGE, *, trial_days: int = TRIAL_DAYS):  # 3
        self.path = path  # 3
        self.trial_days = trial_days  # 3
        self._data = self._load()  # 3
# 3
    # --- persistence -------------------------------------------------  # 3
    def _default_data(self) -> dict[str, Optional[str]]:  # 3
        return {"trial_start": self._now().isoformat(), "license_key": None}  # 3
# 3
    def _load(self) -> dict[str, Optional[str]]:  # 3
        if self.path.exists():  # 3
            try:  # 3
                data = json.loads(self.path.read_text(encoding="utf-8"))  # 3
                if not isinstance(data, dict):  # 3
                    raise ValueError("Invalid license data structure")  # 3
                return data  # 3
            except (OSError, json.JSONDecodeError, ValueError):  # 3
                pass  # 3
        data = self._default_data()  # 3
        self._save(data)  # 3
        return data  # 3
# 3
    def _save(self, data: dict[str, Optional[str]]) -> None:  # 3
        try:  # 3
            self.path.parent.mkdir(parents=True, exist_ok=True)  # 3
        except OSError:  # 3
            pass  # 3
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")  # 3
# 3
    # --- helpers -----------------------------------------------------  # 3
    def _now(self) -> datetime:  # 3
        return datetime.utcnow()  # 3
# 3
    def _parse_timestamp(self, value: Optional[str]) -> datetime:  # 3
        if not value:  # 3
            return self._now()  # 3
        try:  # 3
            return datetime.fromisoformat(value)  # 3
        except ValueError:  # 3
            return self._now()  # 3
# 3
    def _normalize_key(self, key: str) -> str:  # 3
        cleaned = key.replace("-", "").replace(" ", "").upper()  # 3
        if not cleaned:  # 3
            raise ValueError("Empty license key")  # 3
        return "-".join(textwrap.wrap(cleaned, 4))  # 3
# 3
    def _validate_license_key(self, key: str) -> bool:  # 3
        cleaned = key.replace("-", "").upper()  # 3
        if len(cleaned) != 24 or any(char not in "0123456789ABCDEF" for char in cleaned):  # 3
            return False  # 3
        random_part = cleaned[:16]  # 3
        checksum = cleaned[16:]  # 3
        expected = hmac.new(  # 3
            LICENSE_SECRET.encode("utf-8"),  # 3
            random_part.encode("utf-8"),  # 3
            hashlib.sha256,  # 3
        ).hexdigest()[:8].upper()  # 3
        return secrets.compare_digest(checksum, expected)  # 3
# 3
    # --- public API --------------------------------------------------  # 3
    def has_valid_license(self) -> bool:  # 3
        key = self._data.get("license_key")  # 3
        if not key:  # 3
            return False  # 3
        return self._validate_license_key(key)  # 3
# 3
    def register_license_key(self, key: str) -> None:  # 3
        normalized = self._normalize_key(key)  # 3
        if not self._validate_license_key(normalized):  # 3
            raise ValueError("Invalid license key")  # 3
        self._data["license_key"] = normalized  # 3
        self._data["licensed_at"] = self._now().isoformat()  # 3
        self._save(self._data)  # 3
# 3
    def clear_license(self) -> None:  # 3
        self._data = self._default_data()  # 3
        self._save(self._data)  # 3
# 3
    def trial_start(self) -> datetime:  # 3
        return self._parse_timestamp(self._data.get("trial_start"))  # 3
# 3
    def trial_expiration(self) -> datetime:  # 3
        return self.trial_start() + timedelta(days=self.trial_days)  # 3
# 3
    def is_trial_expired(self) -> bool:  # 3
        if self.has_valid_license():  # 3
            return False  # 3
        return self._now() >= self.trial_expiration()  # 3
# 3
    def trial_days_remaining(self) -> int:  # 3
        if self.has_valid_license():  # 3
            return 0  # 3
        expiration = self.trial_expiration()  # 3
        now = self._now()  # 3
        if now >= expiration:  # 3
            return 0  # 3
        remaining = expiration - now  # 3
        return remaining.days + (1 if remaining.seconds > 0 else 0)  # 3
# 3
    def status_message(self) -> str:  # 3
        if self.has_valid_license():  # 3
            return "Permanent license activated. Thank you for supporting the project!"  # 3
        expiration = self.trial_expiration()  # 3
        remaining = self.trial_days_remaining()  # 3
        if remaining == 0:  # 3
            return "Trial expired. Please enter a license key to continue using the application."  # 3
        plural = "day" if remaining == 1 else "days"  # 3
        return (  # 3
            f"Trial mode: {remaining} {plural} remaining (expires on {expiration.date():%Y-%m-%d}). "  # 3
            "Enter a license key to unlock the full version permanently."  # 3
        )  # 3
# 3
# 3
def generate_license_key(label: Optional[str] = None) -> str:  # 3
    label_text = (label or "").strip()  # 3
    base_seed = secrets.token_hex(8).upper()  # 3
    if label_text:  # 3
        label_digest = hashlib.sha256(label_text.upper().encode("utf-8")).hexdigest().upper()  # 3
        random_part = (label_digest[:8] + base_seed)[:16]  # 3
    else:  # 3
        random_part = base_seed[:16]  # 3
    checksum = hmac.new(  # 3
        LICENSE_SECRET.encode("utf-8"),  # 3
        random_part.encode("utf-8"),  # 3
        hashlib.sha256,  # 3
    ).hexdigest()[:8].upper()  # 3
    raw_key = random_part + checksum  # 3
    return "-".join(textwrap.wrap(raw_key, 4))  # 3
# 3
# 3
if TYPE_CHECKING:  # pragma: no cover - typing only  # 3
    from temn import SAEDLauncherFrame  # 3
    from saed_editor import PointEditor  # 3
    from fibonachi_analysis import FibonacciAnalysisFrame  # 3
else:  # 3
    try:  # 3
        from temn import SAEDLauncherFrame  # 3
        from saed_editor import PointEditor  # 3
        from fibonachi_analysis import FibonacciAnalysisFrame  # 3
    except ModuleNotFoundError:  # 3
        SAEDLauncherFrame = _import_module("temn").SAEDLauncherFrame  # 3
        PointEditor = _import_module("saed_editor").PointEditor  # 3
        FibonacciAnalysisFrame = _import_module("fibonachi_analysis").FibonacciAnalysisFrame  # 3
# 3
# 3
class PipelineController:  # 3
    """Connect the tabs and handle stage switching."""  # 3
# 3
    def __init__(self, parent: tk.Misc, *, status_callback=None):  # 3
        self.parent = parent  # 3
        self._status_callback = status_callback or (lambda _msg: None)  # 3
        self.notebook = ttk.Notebook(parent)  # 3
        self.notebook.pack(fill=tk.BOTH, expand=True)  # 3
# 3
        self.launcher = SAEDLauncherFrame(self.notebook, controller=self)  # 3
        self.editor = PointEditor(self.notebook, controller=self)  # 3
        self.analysis = FibonacciAnalysisFrame(self.notebook, controller=self, auto_load=False)  # 3
# 3
        self.notebook.add(self.launcher, text="Launcher")  # 3
        self.notebook.add(self.editor, text="Editor")  # 3
        self.notebook.add(self.analysis, text="Analysis")  # 3
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)  # 3
# 3
    def set_status(self, message: str) -> None:  # 3
        self._status_callback(message)  # 3
# 3
    def _on_tab_changed(self, _event) -> None:  # 3
        current = self.notebook.select()  # 3
        if current:  # 3
            tab_text = self.notebook.tab(current, "text")  # 3
            self.set_status(f"Opened tab: {tab_text}")  # 3
# 3
    def open_editor(self, saed_json_path: Path | str) -> None:  # 3
        path = Path(saed_json_path)  # 3
        if not path.exists():  # 3
            raise FileNotFoundError(path)  # 3
        try:  # 3
            self.editor.load_input_json(path, push_undo=False)  # 3
            self.notebook.select(self.editor)  # 3
            self.set_status(f"Editor: {path.name}")  # 3
        except Exception as exc:  # pragma: no cover - GUI fallback  # 3
            messagebox.showerror("Error", f"Failed to load data into the editor:\n{exc}")  # 3
# 3
    def open_analysis(  # 3
        self,  # 3
        payload_path: Path | str,  # 3
        image_path: Optional[Path | str],  # 3
        spots_json: Optional[Path | str],  # 3
    ) -> None:  # 3
        path = Path(payload_path)  # 3
        if not path.exists():  # 3
            raise FileNotFoundError(path)  # 3
        try:  # 3
            self.analysis.load_json(path)  # 3
            self.notebook.select(self.analysis)  # 3
            self.set_status(f"Analysis: {path.name}")  # 3
        except Exception as exc:  # pragma: no cover - GUI fallback  # 3
            messagebox.showerror("Error", f"Failed to load data into the analyzer:\n{exc}")  # 3
# 3
# 3
def _show_splash(  # 3
    root: tk.Tk,  # 3
    *,  # 3
    logo_path: Path | str | None = None,  # 3
    duration_ms: int = 3000,  # 3
    background: str = "#59c6f1",  # 3
) -> None:  # 3
    """Show a centered splash screen before the main window becomes visible."""  # 3
# 3
    if duration_ms <= 0:  # 3
        root.deiconify()  # 3
        return  # 3
# 3
    splash = tk.Toplevel(root)  # 3
    splash.overrideredirect(True)  # 3
    splash.configure(background=background)  # 3
# 3
    frame = tk.Frame(splash, background=background)  # 3
    frame.pack(fill=tk.BOTH, expand=True)  # 3
# 3
    logo_image = None  # 3
    if logo_path is not None:  # 3
        try:  # 3
            if Image is not None and ImageTk is not None:  # 3
                with Image.open(logo_path) as pil_image:  # 3
                    logo_image = ImageTk.PhotoImage(pil_image)  # 3
            else:  # 3
                logo_image = tk.PhotoImage(file=str(logo_path))  # 3
        except (OSError, tk.TclError):  # 3
            logo_image = None  # 3
# 3
    if logo_image is not None:  # 3
        logo_label = tk.Label(frame, image=logo_image, background=background)  # 3
        logo_label.image = logo_image  # keep a reference to avoid garbage collection  # 3
        logo_label.pack(padx=32, pady=24)  # 3
    else:  # 3
        tk.Label(  # 3
            frame,  # 3
            text="SAED Symmetry\nLaunching…",  # 3
            justify="center",  # 3
            background=background,  # 3
            foreground="#ffffff",  # 3
            font=("TkDefaultFont", 18, "bold"),  # 3
            padx=36,  # 3
            pady=28,  # 3
        ).pack()  # 3
# 3
    splash.update_idletasks()  # 3
    width = splash.winfo_reqwidth()  # 3
    height = splash.winfo_reqheight()  # 3
    x = (splash.winfo_screenwidth() // 2) - (width // 2)  # 3
    y = (splash.winfo_screenheight() // 2) - (height // 2)  # 3
    splash.geometry(f"{width}x{height}+{x}+{y}")  # 3
# 3
    def _close_splash() -> None:  # 3
        if splash.winfo_exists():  # 3
            splash.destroy()  # 3
        root.deiconify()  # 3
# 3
    root.after(duration_ms, _close_splash)  # 3
# 3
# 3
class TabbedPipelineApp(tk.Tk):  # 3
    """Main window containing every stage of the workflow."""  # 3
# 3
    def __init__(self, license_manager: LicenseManager, *, show_initially: bool = True):  # 3
        super().__init__()  # 3
        self.license_manager = license_manager  # 3
        if not show_initially:  # 3
            self.withdraw()  # 3
        self.title("SAED Symmetry — Suite")  # 3
        self.geometry("1520x980")  # 3
        self.resizable(True, True)  # 3
# 3
        style = ttk.Style(self)  # 3
        try:  # 3
            style.theme_use("clam")  # 3
        except tk.TclError:  # 3
            pass  # 3
        style.configure("Header.TLabel", font=("TkDefaultFont", 18, "bold"))  # 3
        style.configure("Subheader.TLabel", font=("TkDefaultFont", 11))  # 3
        style.configure("Byline.TLabel", font=("TkDefaultFont", 10, "italic"), foreground="#555555")  # 3
        style.configure("Accent.TButton", font=("TkDefaultFont", 10, "bold"))  # 3
        style.configure("TNotebook", padding=(12, 10))  # 3
        style.configure("TNotebook.Tab", padding=(16, 8))  # 3
        style.configure("License.TLabel", font=("TkDefaultFont", 10))  # 3
# 3
        header = ttk.Frame(self, padding=(20, 18, 20, 12))  # 3
        header.pack(side=tk.TOP, fill=tk.X)  # 3
        header.grid_columnconfigure(0, weight=1)  # 3
# 3
        ttk.Label(header, text="SAED Symmetry — Suite", style="Header.TLabel").grid(row=0, column=0, sticky="w")  # 3
        ttk.Label(  # 3
            header,  # 3
            text="A single pipeline for electron diffraction processing from loading to analysis.",  # 3
            style="Subheader.TLabel",  # 3
            wraplength=720,  # 3
            justify="left",  # 3
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))  # 3
# 3
        ttk.Label(header, text="by RL 9-11 2025 v2.61 ", style="Byline.TLabel").grid(  # 3
            row=0, column=1, rowspan=2, sticky="ne", padx=(12, 0)  # 3
        )  # 3
        ttk.Button(header, text="Help", command=self._show_help).grid(  # 3
            row=0, column=2, rowspan=2, sticky="ne"  # 3
        )  # 3
# 3
        self.license_label = ttk.Label(header, text="", style="License.TLabel", wraplength=720, justify="left")  # 3
        self.license_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0))  # 3
# 3
        self.license_button = ttk.Button(  # 3
            header,  # 3
            text="Enter License Key",  # 3
            command=self._prompt_for_license,  # 3
            style="Accent.TButton",  # 3
        )  # 3
        self.license_button.grid(row=2, column=2, sticky="e", padx=(12, 0), pady=(12, 0))  # 3
# 3
        content = ttk.Frame(self, padding=(20, 0, 20, 12))  # 3
        content.pack(fill=tk.BOTH, expand=True)  # 3
# 3
        self.status_var = tk.StringVar(value="Ready")  # 3
        status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(20, 8))  # 3
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)  # 3
# 3
        self.controller = PipelineController(content, status_callback=self._update_status)  # 3
        self.controller.set_status("Opened tab: Launcher")  # 3
        self._refresh_license_banner()  # 3
# 3
    def _update_status(self, message: str) -> None:  # 3
        self.status_var.set(message)  # 3
# 3
    def _refresh_license_banner(self) -> None:  # 3
        message = self.license_manager.status_message()  # 3
        self.license_label.configure(text=message)  # 3
        if self.license_manager.has_valid_license():  # 3
            self.license_button.configure(text="Update License Key")  # 3
        else:  # 3
            self.license_button.configure(text="Enter License Key")  # 3
# 3
    def _prompt_for_license(self) -> None:  # 3
        prompt_message = "Enter the permanent license key provided by the publisher:"  # 3
        key = simpledialog.askstring("License Key", prompt_message, parent=self)  # 3
        if key is None:  # 3
            return  # 3
        try:  # 3
            self.license_manager.register_license_key(key)  # 3
        except ValueError:  # 3
            messagebox.showerror("License Key", "The provided license key is invalid. Please try again.")  # 3
            return  # 3
        messagebox.showinfo("License Key", "License activated successfully. Enjoy the full version!")  # 3
        self._refresh_license_banner()  # 3
# 3
    def _show_help(self) -> None:  # 3
        help_window = tk.Toplevel(self)  # 3
        help_window.title("About the application")  # 3
        help_window.transient(self)  # 3
        help_window.grab_set()  # 3
        help_window.resizable(False, False)  # 3
# 3
        frame = ttk.Frame(help_window, padding=(20, 16))  # 3
        frame.pack(fill=tk.BOTH, expand=True)  # 3
# 3
        message = (  # 3
            "In the Launcher tab, prepare the image and detector parameters. "  # 3
            "The Editor tab lets you refine points and radii manually, and Analysis builds "  # 3
            "a symmetry report with Fibonacci chains."  # 3
        )  # 3
        ttk.Label(frame, text=message, justify="left", wraplength=480).pack(anchor="w")  # 3
# 3
        ttk.Label(frame, text="Support the project:", padding=(0, 12, 0, 0)).pack(anchor="w")  # 3
# 3
        donation_link = "https://donatello.to/Roynik"  # 3
        link_label = tk.Label(  # 3
            frame,  # 3
            text=donation_link,  # 3
            fg="#1a0dab",  # 3
            cursor="hand2",  # 3
            font=("TkDefaultFont", 10, "underline"),  # 3
            justify="left",  # 3
        )  # 3
        link_label.pack(anchor="w")  # 3
        link_label.bind("<Button-1>", lambda _event: webbrowser.open_new_tab(donation_link))  # 3
# 3
        ttk.Button(frame, text="Close", command=help_window.destroy).pack(  # 3
            anchor="e", pady=(20, 0)  # 3
        )  # 3
# 3
# 3
def _show_trial_expired_dialog(license_manager: LicenseManager) -> bool:  # 3
    root = tk.Tk()  # 3
    root.title("Trial Expired")  # 3
    root.geometry("420x240")  # 3
    root.resizable(False, False)  # 3
    root.configure(padx=24, pady=24)  # 3
# 3
    message = (  # 3
        "The 3-day trial period has ended. "  # 3
        "Please enter a valid license key to unlock the full version permanently."  # 3
    )  # 3
    ttk.Label(root, text=message, wraplength=360, justify="left").pack(anchor="w")  # 3
# 3
    entry = ttk.Entry(root)  # 3
    entry.pack(fill=tk.X, pady=(16, 8))  # 3
    entry.focus_set()  # 3
# 3
    feedback_var = tk.StringVar(value="")  # 3
    feedback_label = ttk.Label(root, textvariable=feedback_var, foreground="#aa0000", wraplength=360)  # 3
    feedback_label.pack(anchor="w")  # 3
# 3
    actions = ttk.Frame(root)  # 3
    actions.pack(fill=tk.X, pady=(18, 0))  # 3
# 3
    result = {"activated": False}  # 3
# 3
    def _activate() -> None:  # 3
        key = entry.get().strip()  # 3
        if not key:  # 3
            feedback_var.set("Please enter a license key before continuing.")  # 3
            return  # 3
        try:  # 3
            license_manager.register_license_key(key)  # 3
        except ValueError:  # 3
            feedback_var.set("The provided license key is invalid. Check the code and try again.")  # 3
            return  # 3
        messagebox.showinfo("License Key", "License activated successfully. Thank you!")  # 3
        result["activated"] = True  # 3
        root.destroy()  # 3
# 3
    def _quit() -> None:  # 3
        root.destroy()  # 3
# 3
    ttk.Button(actions, text="Activate", command=_activate, style="Accent.TButton").pack(side=tk.RIGHT)  # 3
    ttk.Button(actions, text="Quit", command=_quit).pack(side=tk.RIGHT, padx=(0, 8))  # 3
# 3
    root.protocol("WM_DELETE_WINDOW", _quit)  # 3
    root.mainloop()  # 3
    return result["activated"]  # 3
# 3
# 3
def main(  # 3
    *,  # 3
    splash_logo: Path | str | None = None,  # 3
    splash_duration_ms: int = 3000,  # 3
) -> None:  # 3
    """Run the pipeline app, enforcing the trial and license policy."""  # 3
# 3
    license_manager = LicenseManager()  # 3
    if not license_manager.has_valid_license() and license_manager.is_trial_expired():  # 3
        activated = _show_trial_expired_dialog(license_manager)  # 3
        if not activated:  # 3
            return  # 3
# 3
    app = TabbedPipelineApp(license_manager, show_initially=False)  # 3
    _show_splash(app, logo_path=splash_logo, duration_ms=splash_duration_ms)  # 3
    app.mainloop()  # 3
# 3
# 3
def _build_cli_parser() -> argparse.ArgumentParser:  # 3
    parser = argparse.ArgumentParser(description="SAED Symmetry pipeline application")  # 3
    parser.add_argument(  # 3
        "--generate-license",  # 3
        metavar="LABEL",  # 3
        nargs="?",  # 3
        const="",  # 3
        help=(  # 3
            "Generate a new license key for distribution. Optionally provide a label "  # 3
            "to derive part of the key from customer information."  # 3
        ),  # 3
    )  # 3
    parser.add_argument(  # 3
        "--splash-logo",  # 3
        metavar="PATH",  # 3
        type=Path,  # 3
        help="Custom splash logo to display when launching the application.",  # 3
    )  # 3
    parser.add_argument(  # 3
        "--no-splash",  # 3
        action="store_true",  # 3
        help="Skip the splash screen when launching the graphical interface.",  # 3
    )  # 3
    return parser  # 3
# 3
# 3
if __name__ == "__main__":  # 3
    parser = _build_cli_parser()  # 3
    args = parser.parse_args()  # 3
# 3
    if args.generate_license is not None:  # 3
        key = generate_license_key(args.generate_license or None)  # 3
        print(key)  # 3
        raise SystemExit(0)  # 3
# 3
    if args.no_splash:  # 3
        splash_duration = 0  # 3
        logo = None  # 3
    else:  # 3
        default_logo = _resource_path("logo.png")  # 3
        logo = args.splash_logo if args.splash_logo is not None else (default_logo if default_logo.exists() else None)  # 3
        splash_duration = 3000  # 3
# 3
    main(splash_logo=logo, splash_duration_ms=splash_duration)  # 3