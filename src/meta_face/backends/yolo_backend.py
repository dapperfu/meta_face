"""YOLOv8-Face detection backend (wraps :mod:`meta_face.yolo_face`)."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from meta_face.backends.base import FaceDetectionBackend
from meta_face.config import YOLO_FACE_MODEL_DIR


class YoloFaceBackend(FaceDetectionBackend):
    """Boxes and confidences only; the model is loaded on first use."""

    def __init__(self) -> None:
        self._detector: Any = None

    @property
    def name(self) -> str:
        return "yolo"

    def available(self) -> bool:
        from meta_face.yolo_face.weights import is_yolo_available

        return importlib.util.find_spec("ultralytics") is not None and is_yolo_available(
            model_dir=YOLO_FACE_MODEL_DIR
        )

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        self.ensure_available()
        if self._detector is None:
            from meta_face.yolo_face import FaceDetector
            from meta_face.yolo_face.weights import yolo_model_path

            self._detector = FaceDetector(model_path=yolo_model_path(model_dir=YOLO_FACE_MODEL_DIR))
        return [
            {"bbox": [float(v) for v in det.xyxy], "det_score": float(det.confidence)}
            for det in self._detector.detect(image)
        ]
