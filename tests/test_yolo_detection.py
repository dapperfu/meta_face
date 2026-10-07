"""Tests for the YOLO face Detection type and clipping."""

from __future__ import annotations

import math

import pytest

from meta_face.yolo_face.detection import Detection, clip_detection, make_detection


def test_geometry() -> None:
    det = Detection(100, 50, 300, 250, 0.9)
    assert det.width == 200
    assert det.height == 200
    assert (det.center_x, det.center_y) == (200, 150)
    assert det.area == 40000
    assert det.xyxy == [100, 50, 300, 250]
    assert det.xywh == [100, 50, 200, 200]


def test_normalized() -> None:
    det = Detection(100, 50, 300, 250, 0.9)
    norm = det.normalized(1000, 500)
    assert norm == pytest.approx({"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.5})


def test_normalized_rejects_bad_size() -> None:
    with pytest.raises(ValueError):
        Detection(0, 0, 1, 1, 0.5).normalized(0, 10)


def test_to_dict() -> None:
    out = Detection(100, 50, 300, 250, 0.93612).to_dict(1000, 500, normalized=True)
    assert out["width"] == 200
    assert out["confidence"] == 0.9361
    assert out["normalized"]["x2"] == 0.3
    assert "normalized" not in Detection(0, 0, 1, 1, 0.5).to_dict()


def test_clip_negative() -> None:
    det = clip_detection(Detection(-5, -3, 50, 60, 0.8), 100, 100)
    assert det is not None
    assert (det.x1, det.y1) == (0, 0)


def test_clip_past_edge() -> None:
    det = clip_detection(Detection(90, 90, 150, 140, 0.8), 100, 100)
    assert det is not None
    assert (det.x2, det.y2) == (100, 100)


@pytest.mark.parametrize(
    "box",
    [
        (200, 200, 300, 300),  # fully outside
        (10, 10, 10, 40),  # zero width
        (50, 40, 20, 80),  # inverted
    ],
)
def test_clip_drops_degenerate(box: tuple[int, int, int, int]) -> None:
    assert clip_detection(Detection(*box, 0.8), 100, 100) is None


def test_make_detection_floats_and_confidence() -> None:
    det = make_detection(10.4, 20.6, 30.2, 40.9, 1.3, 100, 100)
    assert det is not None
    assert det.xyxy == [10, 20, 31, 41]
    assert det.confidence == 1.0
    assert make_detection(math.nan, 0, 10, 10, 0.5, 100, 100) is None
