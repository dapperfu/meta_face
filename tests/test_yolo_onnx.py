"""ONNX backend helpers and export comparison logic (no model required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from meta_face.yolo_face.backends.onnx_backend import _providers, letterbox
from meta_face.yolo_face.detection import Detection
from meta_face.yolo_face.errors import DeviceUnavailableError
from meta_face.yolo_face.export import _compare, iou


def test_letterbox_landscape_is_stride_aligned() -> None:
    frame = np.full((2112, 2816, 3), 200, np.uint8)
    image, scale, (pad_x, pad_y) = letterbox(frame, 640)
    assert image.shape == (640, 640, 3)
    assert scale == pytest.approx(640 / 2816)
    assert (pad_x, pad_y) == (0, 0)
    assert image[479, 0, 0] == 200 and image[480, 0, 0] == 114


def test_letterbox_odd_size_pads_to_stride() -> None:
    _, _, (pad_x, pad_y) = letterbox(np.zeros((300, 640, 3), np.uint8), 640)
    assert pad_x == 0
    assert pad_y == 10  # 300 -> 320 pads 10 px on top


def test_providers() -> None:
    assert _providers("cpu", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    with pytest.raises(DeviceUnavailableError):
        _providers("cuda:0", ["CPUExecutionProvider"])
    cuda = _providers("cuda:1", ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert cuda[0] == ("CUDAExecutionProvider", {"device_id": 1})


def test_iou() -> None:
    a = Detection(0, 0, 10, 10, 0.9)
    assert iou(a, a) == 1.0
    assert iou(a, Detection(5, 0, 15, 10, 0.9)) == pytest.approx(50 / 150)
    assert iou(a, Detection(20, 20, 30, 30, 0.9)) == 0.0


def test_compare_tolerates_borderline_only() -> None:
    ref = [Detection(0, 0, 10, 10, 0.90), Detection(50, 50, 60, 60, 0.42)]
    cand = [Detection(0, 0, 10, 10, 0.89)]
    assert not _compare(Path("x"), ref, cand, 0.4, 0.9, 0.05).problems

    missing = _compare(Path("x"), ref, [], 0.4, 0.9, 0.05)
    assert any("missing" in p for p in missing.problems)

    shifted = _compare(Path("x"), ref[:1], [Detection(1, 1, 11, 11, 0.9)], 0.4, 0.9, 0.05)
    assert any("IoU" in p for p in shifted.problems)
