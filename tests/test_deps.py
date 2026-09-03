"""Tests for runtime dependency helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from meta_face.deps import (
    PipelineDependencyError,
    adjust_per_image_tools_for_runtime,
)


def test_adjust_per_image_tools_keeps_detectron2_when_available() -> None:
    with patch("meta_face.deps.detectron2_runtime_issue", return_value=None):
        tools, warnings = adjust_per_image_tools_for_runtime(
            ["scrfd", "detectron2"],
            detectron2_explicit=False,
        )
    assert tools == ["scrfd", "detectron2"]
    assert warnings == []


def test_adjust_per_image_tools_skips_detectron2_from_default() -> None:
    with patch("meta_face.deps.detectron2_runtime_issue", return_value="missing weights"):
        tools, warnings = adjust_per_image_tools_for_runtime(
            ["scrfd", "detectron2"],
            detectron2_explicit=False,
        )
    assert tools == ["scrfd"]
    assert len(warnings) == 1
    assert "Skipping detectron2" in warnings[0]


def test_adjust_per_image_tools_fails_when_detectron2_explicit() -> None:
    with patch("meta_face.deps.detectron2_runtime_issue", return_value="missing package"):
        with pytest.raises(PipelineDependencyError, match="missing package"):
            adjust_per_image_tools_for_runtime(
                ["detectron2"],
                detectron2_explicit=True,
            )
