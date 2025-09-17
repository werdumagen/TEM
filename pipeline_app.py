#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Объединённое окно с вкладками лаунчера, редактора и анализатора."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import tkinter as tk
from tkinter import ttk, messagebox


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


def _import_module(name: str):
    """Import helper that falls back to sibling files when bundlers miss them."""

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        base_candidates = []
        frozen_base = getattr(sys, "_MEIPASS", None)
        if frozen_base is not None:
            base_candidates.append(Path(frozen_base))
        base_candidates.append(Path(__file__).resolve().parent)

        for base in base_candidates:
            candidate = base / f"{name}.py"
            if candidate.exists():
                spec = importlib.util.spec_from_file_location(name, candidate)
                if spec is None or spec.loader is None:  # pragma: no cover - importlib guard
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                return module
        raise


if TYPE_CHECKING:  # pragma: no cover - typing only
    from temn import SAEDLauncherFrame
    from saed_editor import PointEditor
    from fibonachi_analysis import FibonacciAnalysisFrame
else:
    SAEDLauncherFrame = _import_module("temn").SAEDLauncherFrame
    PointEditor = _import_module("saed_editor").PointEditor
    FibonacciAnalysisFrame = _import_module("fibonachi_analysis").FibonacciAnalysisFrame


class PipelineController:
    """Связывает вкладки и занимается переключением между этапами."""

    def __init__(self, parent: tk.Misc, *, status_callback=None):
        self.parent = parent
        self._status_callback = status_callback or (lambda _msg: None)
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.launcher = SAEDLauncherFrame(self.notebook, controller=self)
        self.editor = PointEditor(self.notebook, controller=self)
        self.analysis = FibonacciAnalysisFrame(self.notebook, controller=self, auto_load=False)

        self.notebook.add(self.launcher, text="Лаунчер")
        self.notebook.add(self.editor, text="Редактор")
        self.notebook.add(self.analysis, text="Анализ")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    # --- вызовы из вкладок ---
    def set_status(self, message: str) -> None:
        self._status_callback(message)

    def _on_tab_changed(self, _event) -> None:
        current = self.notebook.select()
        if current:
            tab_text = self.notebook.tab(current, "text")
            self.set_status(f"Открыта вкладка: {tab_text}")

    def open_editor(self, saed_json_path: Path | str) -> None:
        path = Path(saed_json_path)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            self.editor.load_input_json(path, push_undo=False)
            self.notebook.select(self.editor)
            self.set_status(f"Редактор: {path.name}")
        except Exception as exc:  # pragma: no cover - GUI fallback
            messagebox.showerror("Ошибка", f"Не удалось загрузить данные в редактор:\n{exc}")

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
            self.set_status(f"Анализ: {path.name}")
        except Exception as exc:  # pragma: no cover - GUI fallback
            messagebox.showerror("Ошибка", f"Не удалось загрузить данные в анализатор:\n{exc}")


class TabbedPipelineApp(tk.Tk):
    """Главное окно, содержащее все этапы работы."""

    def __init__(self):
        super().__init__()
        self.title("SAED Symmetry — Комплекс")
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

        header = ttk.Frame(self, padding=(20, 18, 20, 12))
        header.pack(side=tk.TOP, fill=tk.X)
        header.grid_columnconfigure(0, weight=1)

        ttk.Label(header, text="SAED Symmetry — Комплекс", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Единый конвейер обработки электронограммы от загрузки до анализа.",
            style="Subheader.TLabel",
            wraplength=720,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        ttk.Label(header, text="by Roynik 2025", style="Byline.TLabel").grid(
            row=0, column=1, rowspan=2, sticky="ne", padx=(12, 0)
        )
        ttk.Button(header, text="Справка", command=self._show_help).grid(
            row=0, column=2, rowspan=2, sticky="ne"
        )

        content = ttk.Frame(self, padding=(20, 0, 20, 12))
        content.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Готово")
        status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(20, 8))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.controller = PipelineController(content, status_callback=self._update_status)
        self.controller.set_status("Открыта вкладка: Лаунчер")

    def _update_status(self, message: str) -> None:
        self.status_var.set(message)

    def _show_help(self) -> None:
        message = (
            "Во вкладке «Лаунчер» подготовьте изображение и параметры детектора. "
            "«Редактор» позволит вручную уточнить точки и радиусы, а «Анализ» — построить отчёт "
            "по симметрии и цепочкам Фибоначчи."
        )
        messagebox.showinfo("О приложении", message)

def main() -> None:
    app = TabbedPipelineApp()
    app.mainloop()


if __name__ == "__main__":
    main()