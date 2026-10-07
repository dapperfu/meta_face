"""Inference device selection."""

from __future__ import annotations

import logging
import re
from typing import Any

from meta_face.yolo_face.errors import DeviceUnavailableError

logger = logging.getLogger(__name__)

_CUDA_RE = re.compile(r"^cuda(?::(\d+))?$")


def _torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _cuda_count(torch: Any | None) -> int:
    if torch is None:
        return 0
    try:
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001 - broken CUDA installs raise assorted errors
        return 0


def _mps_available(torch: Any | None) -> bool:
    if torch is None:
        return False
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(requested: str = "auto") -> str:
    """
    Map a device request to a concrete device string.

    ``auto`` prefers CUDA, then Apple MPS, then CPU and never fails. An explicit
    ``cuda``/``cuda:N``/``mps`` request that cannot be satisfied raises
    :class:`DeviceUnavailableError`.
    """
    key = requested.strip().lower()
    if key == "cpu":
        device = "cpu"
    elif key == "auto":
        torch = _torch()
        if _cuda_count(torch) > 0:
            device = "cuda:0"
        elif _mps_available(torch):
            device = "mps"
        else:
            device = "cpu"
    elif key == "mps":
        if not _mps_available(_torch()):
            raise DeviceUnavailableError("Device 'mps' was requested but MPS is not available.")
        device = "mps"
    elif match := _CUDA_RE.match(key):
        index = int(match.group(1) or 0)
        count = _cuda_count(_torch())
        if index >= count:
            raise DeviceUnavailableError(
                f"Device '{requested}' was requested but only {count} CUDA device(s) are "
                "available. Use --device auto or --device cpu."
            )
        device = f"cuda:{index}"
    else:
        raise ValueError(f"Unsupported device '{requested}'. Use auto, cpu, cuda, cuda:N, or mps.")
    logger.info("Device: %s", device)
    return device
