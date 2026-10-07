"""PyTorch inference through the Ultralytics YOLO API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from meta_face.yolo_face.detection import Detection, make_detection
from meta_face.yolo_face.errors import MissingDependencyError, ModelLoadError

logger = logging.getLogger(__name__)

_WARMUP_SIZE = 64


class UltralyticsBackend:
    """Loads a YOLO ``.pt`` (or any Ultralytics-loadable) model once and runs ``predict``."""

    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float,
        iou_threshold: float,
        image_size: int,
        device: str,
        half: bool = False,
        warmup: bool = True,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise MissingDependencyError(
                "ultralytics is not installed. Install it with: uv pip install -e '.[yolo]'"
            ) from None

        self._confidence = confidence_threshold
        self._iou = iou_threshold
        self._image_size = image_size
        self._device = device
        self._half = half and device.startswith("cuda")
        try:
            self._model: Any = YOLO(str(model_path), task="detect")
        except Exception as exc:
            raise ModelLoadError(f"Could not load YOLO model '{model_path}': {exc}") from exc
        if warmup:
            self.detect(np.zeros((_WARMUP_SIZE, _WARMUP_SIZE, 3), dtype=np.uint8))

    @property
    def name(self) -> str:
        return "ultralytics"

    @property
    def device(self) -> str:
        return self._device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        options: dict[str, Any] = {
            "conf": self._confidence,
            "iou": self._iou,
            "imgsz": self._image_size,
            "device": self._device,
            "verbose": False,
        }
        if self._half:
            options["half"] = True
        results = self._model.predict(frame, **options)
        height, width = frame.shape[:2]
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            for (x1, y1, x2, y2), score, class_id in zip(xyxy, conf, cls):
                det = make_detection(
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(score),
                    width,
                    height,
                    int(class_id),
                )
                if det is not None:
                    detections.append(det)
        return detections
