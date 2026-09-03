"""InsightFace SCRFD detection backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from meta_face.backends.base import FaceDetectionBackend
from meta_face.tools.scrfd import detect_faces, faces_to_records


class InsightFaceBackend(FaceDetectionBackend):
    @property
    def name(self) -> str:
        return "scrfd"

    def available(self) -> bool:
        try:
            import insightface  # noqa: F401
            import onnxruntime  # noqa: F401

            return hasattr(onnxruntime, "InferenceSession")
        except (ImportError, AttributeError):
            return False

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        self.ensure_available()
        faces = detect_faces(image)
        return faces_to_records(faces)

