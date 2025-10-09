from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

try:  # optional dependency, used only for CLAHE mode
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    cv2 = None  # type: ignore


@dataclass(frozen=True)
class PreprocSettings:
    """Настройки предобработки изображения перед детекцией."""

    mode: str = "standard"
    clahe_clip: float = 1.5
    clahe_tiles: int = 8

    def normalized(self) -> "PreprocSettings":
        mode = (self.mode or "").lower()
        if mode not in {"standard", "raw", "clahe"}:
            mode = "standard"
        clip = max(0.1, float(self.clahe_clip))
        tiles = max(2, int(float(self.clahe_tiles)))
        return PreprocSettings(mode=mode, clahe_clip=clip, clahe_tiles=tiles)

    def to_json(self) -> dict[str, Any]:
        """Возвращает словарь с параметрами для сериализации."""

        cfg = self.normalized()
        return {
            "mode": cfg.mode,
            "clahe_clip": float(cfg.clahe_clip),
            "clahe_tiles": int(cfg.clahe_tiles),
        }

    @classmethod
    def from_json(cls, data: Any, *, fallback_mode: str | None = None) -> "PreprocSettings":
        """Создаёт настройки из словаря/строки, с откатом к fallback_mode."""

        mode = fallback_mode or "raw"
        clip = 1.5
        tiles = 8

        if isinstance(data, dict):
            raw_mode = data.get("mode")
            if isinstance(raw_mode, str):
                mode = raw_mode
            elif fallback_mode is not None:
                mode = fallback_mode
            clip_val = data.get("clahe_clip")
            if clip_val is not None:
                clip = float(clip_val)
            tiles_val = data.get("clahe_tiles")
            if tiles_val is not None:
                tiles = int(float(tiles_val))
        elif isinstance(data, str):
            mode = data
        elif fallback_mode is not None:
            mode = fallback_mode

        return cls(mode=mode, clahe_clip=clip, clahe_tiles=tiles).normalized()


def load_grayscale_with_preproc(path: Path | str, settings: PreprocSettings) -> np.ndarray:
    """Загружает изображение в оттенках серого и применяет заданную предобработку."""

    path = Path(path)
    cfg = settings.normalized()
    mode = cfg.mode

    if mode == "clahe":
        if cv2 is None:
            raise RuntimeError(
                "Для режима CLAHE требуется пакет opencv-python (или opencv-python-headless)."
            )
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("Не удалось прочитать изображение через OpenCV.")
        clahe = cv2.createCLAHE(
            clipLimit=float(cfg.clahe_clip),
            tileGridSize=(int(cfg.clahe_tiles), int(cfg.clahe_tiles)),
        )
        img = clahe.apply(img)
        return img.astype(np.float32)

    pil = Image.open(path).convert("L")
    if mode == "standard":
        pil = ImageOps.equalize(pil)
        pil = pil.filter(ImageFilter.GaussianBlur(radius=0.8))
    elif mode == "raw":
        pass  # только grayscale
    else:
        raise ValueError(f"Unknown preproc mode: {mode}")
    return np.array(pil, dtype=np.float32)