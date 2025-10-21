from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:  # NLM требует scikit-image
    import skimage.restoration as skres
    from skimage.util import img_as_float
except ImportError:
    skres = None
    img_as_float = None
    print("ПРЕДУПРЕЖДЕНИЕ: scikit-image не найден. Режим 'NLM Denoising' не будет работать.")


@dataclass(frozen=True)
class PreprocSettings:
    """Настройки предобработки изображения перед детекцией."""

    mode: str = "raw"
    h_param: float = 1.0  # Параметр H для NLM

    def normalized(self) -> "PreprocSettings":
        mode = (self.mode or "").lower()
        if mode not in {"raw", "nlm"}:
            mode = "raw"
        h = max(0.01, float(self.h_param))
        return PreprocSettings(mode=mode, h_param=h)

    def to_json(self) -> dict[str, Any]:
        """Возвращает словарь с параметрами для сериализации."""
        cfg = self.normalized()
        return {
            "mode": cfg.mode,
            "h_param": float(cfg.h_param),
        }

    @classmethod
    def from_json(cls, data: Any, *, fallback_mode: str | None = None) -> "PreprocSettings":
        """Создаёт настройки из словаря/строки, с откатом к fallback_mode."""
        mode = fallback_mode or "raw"
        h = 1.0

        if isinstance(data, dict):
            raw_mode = data.get("mode")
            if isinstance(raw_mode, str):
                mode = raw_mode
            h_val = data.get("h_param")
            if h_val is not None:
                h = float(h_val)
        elif isinstance(data, str):
            mode = data
        elif fallback_mode is not None:
            mode = fallback_mode

        return cls(mode=mode, h_param=h).normalized()


def load_grayscale_with_preproc(path: Path | str, settings: PreprocSettings) -> np.ndarray:
    """Загружает изображение в оттенках серого и применяет заданную предобработку."""

    path = Path(path)
    cfg = settings.normalized()
    mode = cfg.mode

    # Загружаем изображение с помощью PIL
    try:
        pil = Image.open(path).convert("L")
    except Exception as e:
        raise RuntimeError(f"Не удалось прочитать изображение через PIL: {e}")

    if mode == "nlm":
        if skres is None or img_as_float is None:
            raise RuntimeError(
                "Для режима NLM требуется пакет scikit-image. "
                "Установите его: pip install scikit-image"
            )

        img_arr = np.array(pil)
        img_float = img_as_float(img_arr)

        # Применяем NLM Denoising
        denoised_float = skres.denoise_nl_means(
            img_float,
            h=float(cfg.h_param),
            fast_mode=True,
            channel_axis=None  # для оттенков серого
        )

        # Возвращаем в том же формате, что и другие методы (float32, 0-255)
        return (denoised_float * 255.0).astype(np.float32)

    elif mode == "raw":
        pass  # только grayscale
    else:
        raise ValueError(f"Unknown preproc mode: {mode}")

    # Возвращаем результат для 'raw'
    return np.array(pil, dtype=np.float32)