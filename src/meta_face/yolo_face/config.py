"""Layered configuration: built-in defaults < TOML file < CLI overrides."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from meta_face.yolo_face.detector import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU,
    DEFAULT_MODEL_PATH,
)

DEFAULT_CONFIG_PATH = Path("config/yolo_face.toml")


@dataclass(frozen=True)
class ModelConfig:
    path: Path = DEFAULT_MODEL_PATH
    confidence: float = DEFAULT_CONFIDENCE
    iou: float = DEFAULT_IOU
    image_size: int = DEFAULT_IMAGE_SIZE
    device: str = "auto"
    half: bool = False


@dataclass(frozen=True)
class DisplayConfig:
    draw_boxes: bool = True
    show_confidence: bool = True
    show_center: bool = False
    show_fps: bool = True


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int | None = 1280
    height: int | None = 720
    fps: float | None = None


@dataclass(frozen=True)
class PerformanceConfig:
    frame_skip: int = 0


@dataclass(frozen=True)
class FilterConfig:
    min_face_width: int = 0
    min_face_height: int = 0


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)


T = TypeVar("T")


def _merge(obj: T, values: Mapping[str, Any], where: str) -> T:
    assert is_dataclass(obj)
    known = {f.name: f for f in fields(obj)}
    updates: dict[str, Any] = {}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"Unknown config key '{where}{key}'")
        if value is None:
            continue
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, Mapping):
                raise ValueError(f"Config section '{where}{key}' must be a table")
            updates[key] = _merge(current, value, f"{where}{key}.")
        elif isinstance(current, Path) or key == "path":
            updates[key] = Path(value)
        else:
            updates[key] = value
    return replace(obj, **updates)  # type: ignore[type-var]


def load_config(
    path: Path | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> AppConfig:
    """
    Build an :class:`AppConfig`.

    ``path`` defaults to ``config/yolo_face.toml`` when it exists. ``overrides`` is a
    nested mapping like ``{"model": {"confidence": 0.5}}``; ``None`` values are ignored
    so unset CLI options do not clobber file settings.
    """
    config = AppConfig()
    if path is None and DEFAULT_CONFIG_PATH.is_file():
        path = DEFAULT_CONFIG_PATH
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"Config file does not exist: {path}")
        with path.open("rb") as fh:
            config = _merge(config, tomllib.load(fh), "")
    if overrides:
        config = _merge(config, overrides, "")
    return config
