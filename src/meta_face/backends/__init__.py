"""Face detection backends."""

from meta_face.backends.registry import (
    detection_backend_names,
    get_detection_backend,
    list_detection_backends,
)

__all__ = [
    "detection_backend_names",
    "get_detection_backend",
    "list_detection_backends",
]
