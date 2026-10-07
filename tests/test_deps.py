"""Tests for runtime dependency helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from meta_face.deps import (
    PipelineDependencyError,
    adjust_per_image_tools_for_runtime,
    ensure_pkg_resources,
    require_dlib_runtime,
)


def test_face_recognition_models_imports_without_pkg_resources() -> None:
    import sys

    sys.modules.pop("face_recognition_models", None)
    sys.modules.pop("pkg_resources", None)
    ensure_pkg_resources()
    import face_recognition_models

    landmark_model = face_recognition_models.pose_predictor_model_location()
    assert landmark_model.endswith("shape_predictor_68_face_landmarks.dat")
    require_dlib_runtime()


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
