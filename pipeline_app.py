#!/usr/bin/env python3  # 1
# -*- coding: utf-8 -*-  # 2
(  # 3
    "Unified window with launcher, editor, and analyzer tabs.\n"  # 4
)  # 5
# 6
from __future__ import annotations  # 7
# 8
import importlib  # 9
import importlib.util  # 10
import sys  # 11
import webbrowser  # 12
from pathlib import Path  # 13
from typing import Optional, TYPE_CHECKING  # 14
# 15
import tkinter as tk  # 16
from tkinter import ttk, messagebox  # 17

try:  # 18
    from PIL import Image, ImageTk  # type: ignore[import-not-found]
except ImportError:  # 19
    Image = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]
# 18
# 19
MODULE_DIR = Path(__file__).resolve().parent  # 20
if str(MODULE_DIR) not in sys.path:  # 21
    sys.path.insert(0, str(MODULE_DIR))  # 22
# 23
# 24
def _resource_path(filename: str) -> Path:
    """Return an absolute path to *filename* that works in frozen bundles."""

    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir is not None:
        return Path(bundle_dir, filename)
    return MODULE_DIR / filename


def _import_module(name: str):  # 25
    (  # 26
        "Import helper that falls back to sibling files when bundlers miss them.\n"  # 27
    )  # 28
# 29
    try:  # 30
        return importlib.import_module(name)  # 31
    except ModuleNotFoundError as exc:  # 32
        base_candidates = []  # 33
        frozen_base = getattr(sys, "_MEIPASS", None)  # 34
        if frozen_base is not None:  # 35
            base_candidates.append(Path(frozen_base))  # 36
        base_candidates.append(Path(__file__).resolve().parent)  # 37
# 38
        def _attempt_load(module_path: Path, *, package_dir: Path | None = None):  # 39
            spec_kwargs = {}  # 40
            if package_dir is not None:  # 41
                spec_kwargs["submodule_search_locations"] = [str(package_dir)]  # 42
            spec = importlib.util.spec_from_file_location(name, module_path, **spec_kwargs)  # 43
            if spec is None or spec.loader is None:  # pragma: no cover - importlib guard  # 44
                return None  # 45
            module = importlib.util.module_from_spec(spec)  # 46
            sys.modules[name] = module  # 47
            spec.loader.exec_module(module)  # 48
            return module  # 49
# 50
        for base in base_candidates:  # 51
            for suffix in (".py", ".pyc"):  # 52
                candidate = base / f"{name}{suffix}"  # 53
                if candidate.exists():  # 54
                    module = _attempt_load(candidate)  # 55
                    if module is not None:  # 56
                        return module  # 57
# 58
            package_dir = base / name  # 59
            if package_dir.is_dir():  # 60
                for suffix in (".py", ".pyc"):  # 61
                    init_file = package_dir / f"__init__{suffix}"  # 62
                    if init_file.exists():  # 63
                        module = _attempt_load(init_file, package_dir=package_dir)  # 64
                        if module is not None:  # 65
                            return module  # 66
# 67
        raise exc  # 68
# 69
# 70
if TYPE_CHECKING:  # pragma: no cover - typing only  # 71
    from temn import SAEDLauncherFrame  # 72
    from saed_editor import PointEditor  # 73
    from fibonachi_analysis import FibonacciAnalysisFrame  # 74
else:  # 75
    try:  # 76
        from temn import SAEDLauncherFrame  # 77
        from saed_editor import PointEditor  # 78
        from fibonachi_analysis import FibonacciAnalysisFrame  # 79
    except ModuleNotFoundError:  # 80
        SAEDLauncherFrame = _import_module("temn").SAEDLauncherFrame  # 81
        PointEditor = _import_module("saed_editor").PointEditor  # 82
        FibonacciAnalysisFrame = _import_module("fibonachi_analysis").FibonacciAnalysisFrame  # 83
# 84
# 85
class PipelineController:  # 86
    (  # 87
        "Connects the tabs and handles stage switching.\n"  # 88
    )  # 89
# 90
    def __init__(self, parent: tk.Misc, *, status_callback=None):  # 91
        self.parent = parent  # 92
        self._status_callback = status_callback or (lambda _msg: None)  # 93
        self.notebook = ttk.Notebook(parent)  # 94
        self.notebook.pack(fill=tk.BOTH, expand=True)  # 95
# 96
        self.launcher = SAEDLauncherFrame(self.notebook, controller=self)  # 97
        self.editor = PointEditor(self.notebook, controller=self)  # 98
        self.analysis = FibonacciAnalysisFrame(self.notebook, controller=self, auto_load=False)  # 99
# 100
        self.notebook.add(self.launcher, text="Launcher")  # 101
        self.notebook.add(self.editor, text="Editor")  # 102
        self.notebook.add(self.analysis, text="Analysis")  # 103
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)  # 104
# 105
    # --- callbacks from tabs ---  # 106
    def set_status(self, message: str) -> None:  # 107
        self._status_callback(message)  # 108
# 109
    def _on_tab_changed(self, _event) -> None:  # 110
        current = self.notebook.select()  # 111
        if current:  # 112
            tab_text = self.notebook.tab(current, "text")  # 113
            self.set_status(f"Opened tab: {tab_text}")  # 114
# 115
    def open_editor(self, saed_json_path: Path | str) -> None:  # 116
        path = Path(saed_json_path)  # 117
        if not path.exists():  # 118
            raise FileNotFoundError(path)  # 119
        try:  # 120
            self.editor.load_input_json(path, push_undo=False)  # 121
            self.notebook.select(self.editor)  # 122
            self.set_status(f"Editor: {path.name}")  # 123
        except Exception as exc:  # pragma: no cover - GUI fallback  # 124
            messagebox.showerror("Error", f"Failed to load data into the editor:\n{exc}")  # 125
# 126
    def open_analysis(  # 127
        self,  # 128
        payload_path: Path | str,  # 129
        image_path: Optional[Path | str],  # 130
        spots_json: Optional[Path | str],  # 131
    ) -> None:  # 132
        path = Path(payload_path)  # 133
        if not path.exists():  # 134
            raise FileNotFoundError(path)  # 135
        try:  # 136
            self.analysis.load_json(path)  # 137
            self.notebook.select(self.analysis)  # 138
            self.set_status(f"Analysis: {path.name}")  # 139
        except Exception as exc:  # pragma: no cover - GUI fallback  # 140
            messagebox.showerror("Error", f"Failed to load data into the analyzer:\n{exc}")  # 141
# 142
# 143
def _show_splash(
    root: tk.Tk,
    *,
    logo_path: Path | str | None = None,
    duration_ms: int = 3000,
    background: str = "#59c6f1",
) -> None:
    """Show a centered splash screen before the main window becomes visible."""

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
        logo_label.image = logo_image  # keep a reference to avoid garbage collection
        logo_label.pack(padx=32, pady=24)
    else:
        # When bundling without Pillow support (e.g. frozen EXE), loading JPEG logos
        # falls back to the Tk image loader which cannot open them.  In that case the
        # splash window used to shrink to a 1x1 pixel rectangle and effectively remain
        # invisible.  Display a textual fallback so that the splash is always visible.
        tk.Label(
            frame,
            text="SAED Symmetry\nLaunching…",
            justify="center",
            background=background,
            foreground="#ffffff",
            font=("TkDefaultFont", 18, "bold"),
            padx=36,
            pady=28,
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


class TabbedPipelineApp(tk.Tk):  # 144
    (  # 145
        "Main window containing every stage of the workflow.\n"  # 146
    )  # 147
# 148
    def __init__(self, *, show_initially: bool = True):  # 149
        super().__init__()  # 150
        if not show_initially:  # 151
            self.withdraw()  # 152
        self.title("SAED Symmetry — Suite")  # 153
        self.geometry("1520x980")  # 154
        self.resizable(True, True)  # 155
# 154
        style = ttk.Style(self)  # 155
        try:  # 156
            style.theme_use("clam")  # 157
        except tk.TclError:  # 158
            pass  # 159
        style.configure("Header.TLabel", font=("TkDefaultFont", 18, "bold"))  # 160
        style.configure("Subheader.TLabel", font=("TkDefaultFont", 11))  # 161
        style.configure("Byline.TLabel", font=("TkDefaultFont", 10, "italic"), foreground="#555555")  # 162
        style.configure("Accent.TButton", font=("TkDefaultFont", 10, "bold"))  # 163
        style.configure("TNotebook", padding=(12, 10))  # 164
        style.configure("TNotebook.Tab", padding=(16, 8))  # 165
# 166
        header = ttk.Frame(self, padding=(20, 18, 20, 12))  # 167
        header.pack(side=tk.TOP, fill=tk.X)  # 168
        header.grid_columnconfigure(0, weight=1)  # 169
# 170
        ttk.Label(header, text="SAED Symmetry — Suite", style="Header.TLabel").grid(row=0, column=0, sticky="w")  # 171
        ttk.Label(  # 172
            header,  # 173
            text="A single pipeline for electron diffraction processing from loading to analysis.",  # 174
            style="Subheader.TLabel",  # 175
            wraplength=720,  # 176
            justify="left",  # 177
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))  # 178
# 179
        ttk.Label(header, text="by Roynik 2025 v1.5", style="Byline.TLabel").grid(  # 180
            row=0, column=1, rowspan=2, sticky="ne", padx=(12, 0)  # 181
        )  # 182
        ttk.Button(header, text="Help", command=self._show_help).grid(  # 183
            row=0, column=2, rowspan=2, sticky="ne"  # 184
        )  # 185
# 186
        content = ttk.Frame(self, padding=(20, 0, 20, 12))  # 187
        content.pack(fill=tk.BOTH, expand=True)  # 188
# 189
        self.status_var = tk.StringVar(value="Ready")  # 190
        status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(20, 8))  # 191
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)  # 192
# 193
        self.controller = PipelineController(content, status_callback=self._update_status)  # 194
        self.controller.set_status("Opened tab: Launcher")  # 195
# 196
    def _update_status(self, message: str) -> None:  # 197
        self.status_var.set(message)  # 198
# 199
    def _show_help(self) -> None:  # 200
        help_window = tk.Toplevel(self)  # 201
        help_window.title("About the application")  # 202
        help_window.transient(self)  # 203
        help_window.grab_set()  # 204
        help_window.resizable(False, False)  # 205
# 206
        frame = ttk.Frame(help_window, padding=(20, 16))  # 207
        frame.pack(fill=tk.BOTH, expand=True)  # 208
# 209
        message = (  # 210
            "In the Launcher tab, prepare the image and detector parameters. "  # 211
            "The Editor tab lets you refine points and radii manually, and Analysis builds "  # 212
            "a symmetry report with Fibonacci chains."  # 213
        )  # 214
        ttk.Label(frame, text=message, justify="left", wraplength=480).pack(anchor="w")  # 215
# 216
        ttk.Label(frame, text="Support the project:", padding=(0, 12, 0, 0)).pack(anchor="w")  # 217
# 218
        donation_link = "https://donatello.to/Roynik"  # 219
        link_label = tk.Label(  # 220
            frame,  # 221
            text=donation_link,  # 222
            fg="#1a0dab",  # 223
            cursor="hand2",  # 224
            font=("TkDefaultFont", 10, "underline"),  # 225
            justify="left",  # 226
        )  # 227
        link_label.pack(anchor="w")  # 228
        link_label.bind("<Button-1>", lambda _event: webbrowser.open_new_tab(donation_link))  # 229
# 230
        ttk.Button(frame, text="Close", command=help_window.destroy).pack(  # 231
            anchor="e", pady=(20, 0)  # 232
        )  # 233
# 234
def main(
    *,
    splash_logo: Path | str | None = None,
    splash_duration_ms: int = 3000,
) -> None:
    """Run the pipeline app, optionally showing a splash screen first."""

    app = TabbedPipelineApp(show_initially=False)
    _show_splash(app, logo_path=splash_logo, duration_ms=splash_duration_ms)
    app.mainloop()


if __name__ == "__main__":
    default_logo = _resource_path("logo.png")
    logo = default_logo if default_logo.exists() else None
    main(splash_logo=logo) #1