"""Face detection backend registry."""

from __future__ import annotations

from meta_face.backends.base import FaceDetectionBackend
from meta_face.backends.detectron2_backend import Detectron2Backend
from meta_face.backends.insightface_backend import InsightFaceBackend

_BACKENDS: dict[str, FaceDetectionBackend] = {
    "scrfd": InsightFaceBackend(),
    "detectron2": Detectron2Backend(),
}

DETECTION_BACKEND_NAMES: frozenset[str] = frozenset(_BACKENDS.keys())


def detection_backend_names() -> frozenset[str]:
    return DETECTION_BACKEND_NAMES


def get_detection_backend(name: str) -> FaceDetectionBackend:
    key = name.strip().lower()
    if key not in _BACKENDS:
        known = ", ".join(sorted(_BACKENDS))
        raise KeyError(f"Unknown detection backend '{name}'. Known: {known}")
    return _BACKENDS[key]


def list_detection_backends() -> list[FaceDetectionBackend]:
    return [get_detection_backend(n) for n in sorted(_BACKENDS)]
