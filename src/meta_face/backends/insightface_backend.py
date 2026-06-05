"""InsightFace SCRFD detection backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from meta_face.backends.base import FaceDetectionBackend
from meta_face.tools.scrfd import detect_faces


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
        return [
            {
                "bbox": [float(x) for x in face.bbox.tolist()],
                "landmarks": [[float(x), float(y)] for x, y in face.kps.tolist()],
                "det_score": float(face.det_score),
            }
            for face in faces
        ]

