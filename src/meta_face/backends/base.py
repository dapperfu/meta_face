"""Abstract face detection backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class FaceDetectionBackend(ABC):
    """Detect faces in a single image array (BGR, OpenCV layout)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry / sidecar tool name (e.g. scrfd, detectron2)."""

    @abstractmethod
    def available(self) -> bool:
        """True when optional dependencies are importable and models are configured."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """
        Run detection.

        Each dict has at least: bbox [x1,y1,x2,y2], det_score float.
        landmarks is optional: list of [x, y] pairs.
        """

    def ensure_available(self) -> None:
        if not self.available():
            raise RuntimeError(
                f"Detection backend '{self.name}' is not available. "
                f"See: mf backends"
            )

    def to_records(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize detections to sidecar face record shape."""
        records: list[dict[str, Any]] = []
        for det in detections:
            bbox = [float(x) for x in det["bbox"][:4]]
            record: dict[str, Any] = {
                "bbox": bbox,
                "det_score": float(det["det_score"]),
            }
            landmarks = det.get("landmarks")
            if landmarks is not None:
                record["landmarks"] = [[float(x), float(y)] for x, y in landmarks]
            records.append(record)
        return records
