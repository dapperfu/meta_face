"""Detectron2 model zoo resolution and weight caching."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from meta_face.config import (
    DETECTRON2_CONFIG_PATH,
    DETECTRON2_MODEL_ZOO,
    DETECTRON2_WEIGHTS_PATH,
    detectron2_dir,
)


def detectron2_package_installed() -> bool:
    try:
        import detectron2  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def uses_custom_detectron2_paths() -> bool:
    """True when explicit config/weights paths are set via environment."""
    return (
        "META_FACE_DETECTRON2_CONFIG" in os.environ
        or "META_FACE_DETECTRON2_WEIGHTS" in os.environ
    )


@dataclass(frozen=True)
class Detectron2ModelPaths:
    config: str
    weights: str
    model_zoo: str | None = None


def cached_model_zoo_weights_path() -> Path:
    """Local cache path for the active model-zoo checkpoint filename."""
    from detectron2 import model_zoo

    url = model_zoo.get_checkpoint_url(DETECTRON2_MODEL_ZOO)
    return detectron2_dir() / Path(url).name


def resolve_detectron2_model() -> Detectron2ModelPaths:
    """Resolve config and weights for inference (custom paths or model zoo)."""
    if uses_custom_detectron2_paths():
        if not DETECTRON2_CONFIG_PATH.is_file():
            raise FileNotFoundError(
                f"Detectron2 config not found: {DETECTRON2_CONFIG_PATH}"
            )
        if not DETECTRON2_WEIGHTS_PATH.is_file():
            raise FileNotFoundError(
                f"Detectron2 weights not found: {DETECTRON2_WEIGHTS_PATH}"
            )
        return Detectron2ModelPaths(
            config=str(DETECTRON2_CONFIG_PATH),
            weights=str(DETECTRON2_WEIGHTS_PATH),
        )

    from detectron2 import model_zoo

    config = model_zoo.get_config_file(DETECTRON2_MODEL_ZOO)
    cached = cached_model_zoo_weights_path()
    if cached.is_file() and cached.stat().st_size > 1024:
        weights = str(cached)
    else:
        weights = model_zoo.get_checkpoint_url(DETECTRON2_MODEL_ZOO)
    return Detectron2ModelPaths(
        config=config,
        weights=weights,
        model_zoo=DETECTRON2_MODEL_ZOO,
    )


def is_detectron2_available() -> bool:
    """True when detectron2 is importable and model paths can be resolved."""
    if not detectron2_package_installed():
        return False
    if uses_custom_detectron2_paths():
        return DETECTRON2_CONFIG_PATH.is_file() and DETECTRON2_WEIGHTS_PATH.is_file()
    return True


def download_detectron2_weights(*, force: bool = False) -> Path:
    """Pre-download model-zoo weights, or verify custom weight files."""
    detectron2_dir()
    if uses_custom_detectron2_paths():
        if not DETECTRON2_CONFIG_PATH.is_file():
            raise RuntimeError(f"Detectron2 config missing: {DETECTRON2_CONFIG_PATH}")
        if not DETECTRON2_WEIGHTS_PATH.is_file() or DETECTRON2_WEIGHTS_PATH.stat().st_size < 1024:
            raise RuntimeError(f"Detectron2 weights missing: {DETECTRON2_WEIGHTS_PATH}")
        return DETECTRON2_WEIGHTS_PATH

    from detectron2 import model_zoo

    url = model_zoo.get_checkpoint_url(DETECTRON2_MODEL_ZOO)
    dest = cached_model_zoo_weights_path()
    if force or not dest.is_file() or dest.stat().st_size < 1024:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, dest)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to download Detectron2 weights from {url}. "
                f"Original error: {exc}"
            ) from exc
    if not dest.is_file() or dest.stat().st_size < 1024:
        raise RuntimeError(f"Detectron2 weights at {dest} are missing or too small.")
    return dest
