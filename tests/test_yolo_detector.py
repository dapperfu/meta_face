"""FaceDetector tests using a fake backend (no model or GPU required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from meta_face.yolo_face import FaceDetector
from meta_face.yolo_face.backends import DetectorBackend
from meta_face.yolo_face.detection import Detection
from meta_face.yolo_face.device import resolve_device
from meta_face.yolo_face.errors import (
    DeviceUnavailableError,
    ImageDecodeError,
    ModelNotFoundError,
)


class FakeBackend:
    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    @property
    def name(self) -> str:
        return "fake"

    @property
    def device(self) -> str:
        return "cpu"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return list(self._detections)


def _frame(w: int = 100, h: int = 80) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_fake_backend_matches_protocol() -> None:
    assert isinstance(FakeBackend([]), DetectorBackend)


def test_detect_clips_and_drops() -> None:
    backend = FakeBackend(
        [
            Detection(-5, -5, 40, 40, 0.9),
            Detection(90, 70, 130, 120, 0.8),
            Detection(200, 200, 220, 220, 0.9),
        ]
    )
    dets = FaceDetector(backend=backend).detect(_frame())
    assert [d.xyxy for d in dets] == [[0, 0, 40, 40], [90, 70, 100, 80]]
    for d in dets:
        assert 0 <= d.x1 < d.x2 <= 100 and 0 <= d.y1 < d.y2 <= 80
        assert 0.0 <= d.confidence <= 1.0


def test_confidence_threshold() -> None:
    backend = FakeBackend([Detection(0, 0, 10, 10, 0.3), Detection(0, 0, 10, 10, 0.5)])
    dets = FaceDetector(confidence_threshold=0.4, backend=backend).detect(_frame())
    assert [d.confidence for d in dets] == [0.5]


def test_min_size_filter_disabled_by_default() -> None:
    backend = FakeBackend([Detection(0, 0, 3, 3, 0.9), Detection(0, 0, 30, 30, 0.9)])
    assert len(FaceDetector(backend=backend).detect(_frame())) == 2
    filtered = FaceDetector(min_face_width=10, min_face_height=10, backend=backend)
    assert len(filtered.detect(_frame())) == 1


def test_missing_model(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match="does not exist"):
        FaceDetector(model_path=tmp_path / "nope.pt")


def test_bad_frame() -> None:
    detector = FaceDetector(backend=FakeBackend([]))
    with pytest.raises(ImageDecodeError):
        detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ImageDecodeError):
        detector.detect(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("kwargs", [{"confidence_threshold": 1.5}, {"image_size": 500}])
def test_invalid_params(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        FaceDetector(backend=FakeBackend([]), **kwargs)  # type: ignore[arg-type]


def test_device_cpu_and_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    from meta_face.yolo_face import device

    monkeypatch.setattr(device, "_torch", lambda: None)
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") == "cpu"


def test_explicit_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from meta_face.yolo_face import device

    monkeypatch.setattr(device, "_torch", lambda: None)
    with pytest.raises(DeviceUnavailableError):
        resolve_device("cuda:0")
    with pytest.raises(DeviceUnavailableError):
        resolve_device("mps")
    with pytest.raises(ValueError):
        resolve_device("tpu")
