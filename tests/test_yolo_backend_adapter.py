"""The `yolo` entry in meta_face's detection backend registry."""

from __future__ import annotations

import numpy as np
import pytest

from meta_face.backends import get_detection_backend
from meta_face.backends.yolo_backend import YoloFaceBackend
from meta_face.yolo_face.detection import Detection


class _Stub:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [Detection(1, 2, 30, 40, 0.75)]


def test_registered() -> None:
    assert get_detection_backend("yolo").name == "yolo"


def test_records(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = YoloFaceBackend()
    monkeypatch.setattr(backend, "available", lambda: True)
    backend._detector = _Stub()
    assert backend.detect(np.zeros((50, 50, 3), np.uint8)) == [
        {"bbox": [1.0, 2.0, 30.0, 40.0], "det_score": 0.75}
    ]


def test_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = YoloFaceBackend()
    monkeypatch.setattr(backend, "available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        backend.detect(np.zeros((5, 5, 3), np.uint8))
