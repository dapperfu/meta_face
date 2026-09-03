"""Contracts for sports-review ONNX decoding (no model weights required)."""

from __future__ import annotations

import numpy as np

from meta_face.tools.analysis.decode import (
    OPENCV_FER_LABELS,
    class_pixel_percent,
    expand_crop,
    gaze_angles_degrees,
    parsing_argmax,
    softmax,
)


def test_softmax_and_opencv_label_order() -> None:
    probs = softmax([0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert OPENCV_FER_LABELS[int(probs.argmax())] == "disgust"
    assert OPENCV_FER_LABELS[0] == "anger"
    assert "contempt" not in OPENCV_FER_LABELS


def test_gaze_two_heads() -> None:
    yaw_logits = np.full(90, -20.0)
    pitch_logits = np.full(90, -20.0)
    yaw_logits[45] = 20.0
    pitch_logits[45] = 20.0
    yaw, pitch = gaze_angles_degrees([yaw_logits, pitch_logits])
    assert yaw == 0.0
    assert pitch == 0.0


def test_parsing_argmax_not_unique_logits() -> None:
    logits = np.zeros((1, 19, 2, 2), dtype=np.float32)
    logits[0, 3, 0, 0] = 5.0
    logits[0, 7, 1, 1] = 5.0
    mask = parsing_argmax(logits)
    assert mask.shape == (2, 2)
    assert int(mask[0, 0]) == 3
    assert int(mask[1, 1]) == 7
    percents = class_pixel_percent(mask)
    assert percents["3"] == 25.0


def test_expand_crop_scale() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    crop, bounds = expand_crop(image, [40.0, 40.0, 60.0, 60.0], scale=2.7)
    assert crop.size > 0
    assert bounds[0] < 40
    assert bounds[2] > 60
