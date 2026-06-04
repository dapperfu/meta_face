"""Face model weight management (insightface model packs).

Downloading weights is a standalone step so the model pack can be fetched
explicitly (for example before starting GPU workers) instead of lazily on the
first inference call. This only needs network access; no GPU or onnxruntime.
"""

from __future__ import annotations

from pathlib import Path

from meta_face.config import (
    INSIGHTFACE_MODEL,
    INSIGHTFACE_ROOT,
    insightface_model_dir,
)


def model_dir(name: str | None = None) -> Path:
    """Return the directory where the given model pack lives (may not exist)."""
    return insightface_model_dir(name)


def is_available(name: str | None = None) -> bool:
    """True when the model pack directory exists and contains ONNX weights."""
    path = model_dir(name)
    return path.is_dir() and any(path.glob("*.onnx"))


def download(name: str | None = None, *, force: bool = False) -> Path:
    """Download and unzip an insightface model pack; return its directory."""
    from insightface.utils import storage

    pack = name or INSIGHTFACE_MODEL
    dir_path = storage.download("models", pack, force=force, root=INSIGHTFACE_ROOT)
    return Path(dir_path)
