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
                    break
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

    def __init__(self, root: tk.Misc):
        self.root = root
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.launcher = SAEDLauncherFrame(self.notebook, controller=self)
        self.editor = PointEditor(self.notebook, controller=self)
        self.analysis = FibonacciAnalysisFrame(self.notebook, controller=self, auto_load=False)

        self.notebook.add(self.launcher, text="Лаунчер")
        self.notebook.add(self.editor, text="Редактор")
        self.notebook.add(self.analysis, text="Анализ")

    # --- вызовы из вкладок ---
    def open_editor(self, saed_json_path: Path | str) -> None:
        path = Path(saed_json_path)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            self.editor.load_input_json(path, push_undo=False)
            self.notebook.select(self.editor)
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
        except Exception as exc:  # pragma: no cover - GUI fallback
            messagebox.showerror("Ошибка", f"Не удалось загрузить данные в анализатор:\n{exc}")


class TabbedPipelineApp(tk.Tk):
    """Главное окно, содержащее все этапы работы."""

    def __init__(self):
        super().__init__()
        self.title("SAED Symmetry — Комплекс")
        self.geometry("1520x980")
        self.resizable(True, True)
        self.controller = PipelineController(self)


def main() -> None:
    app = TabbedPipelineApp()
    app.mainloop()


if __name__ == "__main__":
    main()