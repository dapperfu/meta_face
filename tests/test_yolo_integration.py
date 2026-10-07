"""Real-model checks; skipped unless ultralytics and the nano weights are installed."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from meta_face.yolo_face.weights import yolo_model_path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / yolo_model_path()
ONNX_MODEL = MODEL.with_suffix(".onnx")
IMAGES = sorted((ROOT / "test_images").glob("*.jpg"))[:3]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(importlib.util.find_spec("ultralytics") is None, reason="needs ultralytics"),
    pytest.mark.skipif(not MODEL.is_file(), reason="run `mf download --backend yolo` first"),
    pytest.mark.skipif(not IMAGES, reason="no test images"),
]


@pytest.fixture(scope="module")
def detector():  # type: ignore[no-untyped-def]
    from meta_face.yolo_face import FaceDetector

    return FaceDetector(model_path=MODEL, device="cpu")


def _check(dets: list, image: np.ndarray) -> None:  # type: ignore[type-arg]
    height, width = image.shape[:2]
    assert len(dets) >= 1
    for d in dets:
        assert 0.0 <= d.confidence <= 1.0
        assert 0 <= d.x1 < d.x2 <= width
        assert 0 <= d.y1 < d.y2 <= height
        assert d.width > 0 and d.height > 0


def test_detects_faces_in_sample(detector) -> None:  # type: ignore[no-untyped-def]
    from meta_face.yolo_face.sources import load_image

    image = load_image(IMAGES[0])
    _check(detector.detect(image), image)


def test_blank_frame_has_no_faces(detector) -> None:  # type: ignore[no-untyped-def]
    assert detector.detect(np.zeros((480, 640, 3), np.uint8)) == []


@pytest.mark.skipif(not ONNX_MODEL.is_file(), reason="run `mf yolo export` first")
def test_onnx_matches_pytorch(detector) -> None:  # type: ignore[no-untyped-def]
    from meta_face.yolo_face.export import compare_detectors

    from meta_face.yolo_face import FaceDetector

    onnx = FaceDetector(model_path=ONNX_MODEL, device="cpu")
    report = compare_detectors(detector, onnx, IMAGES)
    assert report.ok, report.summary()
