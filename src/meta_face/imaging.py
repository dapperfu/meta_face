"""Image loading utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from meta_face.config import IMAGE_EXTENSIONS

_HEIF_REGISTERED = False


def _register_heif() -> None:
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
        _HEIF_REGISTERED = True
    except ImportError:
        _HEIF_REGISTERED = True


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def load_image(path: Path) -> np.ndarray:
    """Load an image as BGR uint8 ndarray (OpenCV convention)."""
    suffix = path.suffix.lower()
    if suffix in {".heic", ".heif"}:
        _register_heif()
        with Image.open(path) as img:
            rgb = np.array(img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        _register_heif()
        with Image.open(path) as img:
            rgb = np.array(img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return image
