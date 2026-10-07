"""Download pretrained YOLOv8-Face weights (lindevs release); stdlib only."""

from __future__ import annotations

import logging
import os
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

YOLO_FACE_RELEASE = "1.0.1"
_RELEASE_URL = f"https://github.com/lindevs/yolov8-face/releases/download/{YOLO_FACE_RELEASE}"
YOLO_FACE_VARIANTS: tuple[str, ...] = ("n", "s", "m", "l", "x")
DEFAULT_VARIANT = "n"


def yolo_weight_filename(variant: str) -> str:
    if variant not in YOLO_FACE_VARIANTS:
        raise ValueError(f"Unknown YOLO face variant '{variant}'. Known: {YOLO_FACE_VARIANTS}")
    return f"yolov8{variant}-face-lindevs.pt"


YOLO_FACE_WEIGHTS: dict[str, str] = {
    v: f"{_RELEASE_URL}/{yolo_weight_filename(v)}" for v in YOLO_FACE_VARIANTS
}


def default_model_dir() -> Path:
    return Path(os.environ.get("META_FACE_YOLO_MODEL_DIR", "models"))


def yolo_model_path(variant: str = DEFAULT_VARIANT, model_dir: Path | None = None) -> Path:
    return (model_dir or default_model_dir()) / yolo_weight_filename(variant)


def is_yolo_available(variant: str = DEFAULT_VARIANT, model_dir: Path | None = None) -> bool:
    path = yolo_model_path(variant, model_dir)
    return path.is_file() and path.stat().st_size > 0


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh, urllib.request.urlopen(url, timeout=60) as resp:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        tmp.chmod(0o644)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def download_yolo_weights(
    variant: str = DEFAULT_VARIANT,
    *,
    force: bool = False,
    model_dir: Path | None = None,
) -> Path:
    """Download one variant's ``.pt`` weights; skip if present unless ``force``."""
    dest = yolo_model_path(variant, model_dir)
    if is_yolo_available(variant, model_dir) and not force:
        return dest
    url = YOLO_FACE_WEIGHTS[variant]
    logger.info("Downloading %s -> %s", url, dest)
    _fetch(url, dest)
    return dest
