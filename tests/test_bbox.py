"""Tests for bbox expansion and face cropping."""

from __future__ import annotations

import numpy as np
import pytest

from meta_face.bbox import crop_face, expand_bbox


def test_expand_bbox_per_side() -> None:
    # 100x100 box at (50, 40, 150, 140); 10% buffer -> pad 10 on each side
    result = expand_bbox([50.0, 40.0, 150.0, 140.0], 10.0)
    assert result == [40.0, 30.0, 160.0, 150.0]


def test_expand_bbox_zero_buffer() -> None:
    bbox = [10.0, 20.0, 110.0, 120.0]
    assert expand_bbox(bbox, 0.0) == [10.0, 20.0, 110.0, 120.0]


def test_expand_bbox_clamps_to_image() -> None:
    # Box near top-left corner; expansion should not go negative
    result = expand_bbox([0.0, 0.0, 50.0, 50.0], 50.0, image_size=(100, 100))
    assert result[0] >= 0.0
    assert result[1] >= 0.0
    assert result[2] <= 100.0
    assert result[3] <= 100.0


def test_expand_bbox_clamps_bottom_right() -> None:
    result = expand_bbox([80.0, 80.0, 100.0, 100.0], 50.0, image_size=(100, 100))
    assert result[2] <= 100.0
    assert result[3] <= 100.0


def test_expand_bbox_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        expand_bbox([10.0, 20.0, 10.0, 120.0], 10.0)


def test_expand_bbox_too_short() -> None:
    with pytest.raises(ValueError, match="four values"):
        expand_bbox([1.0, 2.0, 3.0], 10.0)


def test_crop_face_shape() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    crop = crop_face(image, [50.0, 40.0, 150.0, 140.0], buffer_pct=0.0)
    assert crop.shape == (100, 100, 3)


def test_crop_face_buffered_larger() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    tight = crop_face(image, [50.0, 40.0, 150.0, 140.0], buffer_pct=0.0)
    buffered = crop_face(image, [50.0, 40.0, 150.0, 140.0], buffer_pct=10.0)
    assert buffered.shape[0] >= tight.shape[0]
    assert buffered.shape[1] >= tight.shape[1]
