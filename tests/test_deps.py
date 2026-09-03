"""Tests for runtime dependency helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from meta_face.deps import (
    PipelineDependencyError,
    adjust_per_image_tools_for_runtime,
)


def test_adjust_per_image_tools_keeps_available_analysis() -> None:
    with patch("meta_face.tools.analysis.registry.tool_availability", return_value=None):
        tools, warnings = adjust_per_image_tools_for_runtime(
            ["scrfd", "opencv_fer"],
            analysis_explicit=set(),
        )
    assert tools == ["scrfd", "opencv_fer"]
    assert warnings == []


def test_adjust_per_image_tools_skips_unavailable_analysis() -> None:
    with patch("meta_face.tools.analysis.registry.tool_availability", return_value="missing weights"):
        tools, warnings = adjust_per_image_tools_for_runtime(
            ["scrfd", "opencv_fer"],
            analysis_explicit=set(),
        )
    assert tools == ["scrfd"]
    assert len(warnings) == 1
    assert "Skipping opencv_fer" in warnings[0]


def test_adjust_per_image_tools_fails_when_analysis_explicit() -> None:
    with patch("meta_face.tools.analysis.registry.tool_availability", return_value="missing package"):
        with pytest.raises(PipelineDependencyError, match="missing package"):
            adjust_per_image_tools_for_runtime(
                ["opencv_fer"],
                analysis_explicit={"opencv_fer"},
            )
