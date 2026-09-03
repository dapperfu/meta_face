"""Bounding-box expansion and face cropping utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def expand_bbox(
    bbox: Sequence[float],
    buffer_pct: float,
    *,
    image_size: tuple[int, int] | None = None,
) -> list[float]:
    """
    Expand [x1, y1, x2, y2] per-side by buffer_pct% of width/height.

    When image_size (width, height) is given, coordinates are clamped to image bounds.
    """
    if len(bbox) < 4:
        raise ValueError("bbox must have at least four values")

    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        raise ValueError("bbox width and height must be positive")

    pad_x = w * buffer_pct / 100.0
    pad_y = h * buffer_pct / 100.0
    ex1 = x1 - pad_x
    ey1 = y1 - pad_y
    ex2 = x2 + pad_x
    ey2 = y2 + pad_y

    if image_size is not None:
        img_w, img_h = image_size
        ex1 = max(0.0, ex1)
        ey1 = max(0.0, ey1)
        ex2 = min(float(img_w), ex2)
        ey2 = min(float(img_h), ey2)

    ix1 = int(round(ex1))
    iy1 = int(round(ey1))
    ix2 = int(round(ex2))
    iy2 = int(round(ey2))

    if image_size is not None:
        img_w, img_h = image_size
        ix1 = max(0, min(ix1, img_w - 1))
        iy1 = max(0, min(iy1, img_h - 1))
        ix2 = max(ix1 + 1, min(ix2, img_w))
        iy2 = max(iy1 + 1, min(iy2, img_h))

    return [float(ix1), float(iy1), float(ix2), float(iy2)]


def crop_face(
    image: np.ndarray,
    bbox: Sequence[float],
    *,
    buffer_pct: float = 0.0,
) -> np.ndarray:
    """Return a BGR crop from image using bbox, optionally expanded by buffer_pct."""
    if image.ndim < 2:
        raise ValueError("image must be at least 2-dimensional")

    h_img, w_img = image.shape[:2]
    ex1, ey1, ex2, ey2 = expand_bbox(
        bbox,
        buffer_pct,
        image_size=(w_img, h_img),
    )
    x1, y1, x2, y2 = (int(v) for v in (ex1, ey1, ex2, ey2))
    return image[y1:y2, x1:x2].copy()
