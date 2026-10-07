"""Inference backend interface for the YOLO face detector."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from meta_face.yolo_face.detection import Detection


@runtime_checkable
class DetectorBackend(Protocol):
    """
    Runs a face model on one BGR frame.

    Implementations return detections in original-frame pixel coordinates, already
    confidence-thresholded and NMS-filtered. Runtime-specific objects never escape.
    """

    @property
    def name(self) -> str: ...

    @property
    def device(self) -> str: ...

    def detect(self, frame: np.ndarray) -> list[Detection]: ...
