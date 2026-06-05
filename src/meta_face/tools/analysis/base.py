"""Shared helpers for crop-based analysis tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from meta_face.tools.sidecar_encode import json_safe


@dataclass(frozen=True)
class FaceContext:
    """One detected face with its crop and sidecar metadata."""

    face_index: int
    bbox: list[float]
    crop_bgr: np.ndarray
    metadata: dict[str, Any]


class AnalysisTool(Protocol):
    """Interface implemented by each analysis tool module."""

    TOOL_NAME: str
    TOOL_VERSION: str

    def availability(self) -> str | None:
        """Return an error message when the tool cannot run, else None."""

    def analyze_faces(
        self,
        image_bgr: np.ndarray,
        faces: list[FaceContext],
    ) -> dict[str, Any]:
        """Run inference and return sidecar payload (without version/processed_at)."""


def face_results_payload(
    per_face: list[dict[str, Any]],
    *,
    model: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Standard sidecar payload shape for per-face analysis tools."""
    payload: dict[str, Any] = {
        "faces": per_face,
        "face_count": len(per_face),
        "model": model,
    }
    if extra:
        payload.update(extra)
    return json_safe(payload)


def scores_dict(labels: list[str], values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {label: float(arr[idx]) for idx, label in enumerate(labels[: len(arr)])}
