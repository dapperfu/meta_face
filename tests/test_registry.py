"""Tests for tool registry and dependency resolution."""

from __future__ import annotations

import pytest

from meta_face.config import (
    ALL_TOOLS,
    DEFAULT_SCAN_META_TOOLS,
    DEFAULT_TOOLS,
    PER_IMAGE_TOOL_ORDER,
    PER_IMAGE_TOOLS,
    TOOL_SCANS_FOR,
)
from meta_face.scanner import (
    resolve_backend_job_groups,
    resolve_per_image_tools,
    run_cluster_requested,
)
from meta_face.tools.registry import expand_dependencies, expand_group, validate_tools


def test_face_recognition_expands_to_dlib_tools() -> None:
    assert expand_group("face_recognition") == ["dlib_detect", "dlib_embed"]


def test_expand_dependencies_dlib_embed_includes_detect() -> None:
    assert expand_dependencies(["dlib_embed"]) == ["dlib_detect", "dlib_embed"]


def test_validate_tools_accepts_new_names() -> None:
    tools = validate_tools(["face_recognition", "hdbscan_dlib"])
    assert "dlib_detect" in tools
    assert "dlib_embed" in tools
    assert "cluster_dlib" in tools


def test_validate_tools_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown tools"):
        validate_tools(["not_a_tool"])


def test_detect_and_mediapipe_aliases() -> None:
    assert validate_tools(["detect"]) == ["scrfd"]
    assert validate_tools(["mediapipe"]) == ["mediapipe_blendshapes"]


def test_all_expands_to_every_per_image_tool() -> None:
    expanded = validate_tools(["all"])
    assert expanded == list(PER_IMAGE_TOOL_ORDER)
    assert set(expanded) == PER_IMAGE_TOOLS
    assert "cluster" not in expanded
    groups = resolve_backend_job_groups(resolve_per_image_tools(expanded))
    assert [key for key, tools in groups] == list(PER_IMAGE_TOOL_ORDER)
    assert all(tools == [key] for key, tools in groups)
    assert len(groups) == len(PER_IMAGE_TOOL_ORDER)


def test_tool_scans_for_covers_every_tool() -> None:
    assert set(TOOL_SCANS_FOR) == ALL_TOOLS
    assert len(PER_IMAGE_TOOL_ORDER) == len(set(PER_IMAGE_TOOL_ORDER))


def test_default_scan_runs_detection_without_clustering() -> None:
    assert DEFAULT_SCAN_META_TOOLS == (
        "insightface",
        "face_recognition",
    )
    expanded = validate_tools(list(DEFAULT_SCAN_META_TOOLS))
    assert resolve_per_image_tools(expanded) == list(DEFAULT_TOOLS)
    assert run_cluster_requested(expanded) is False
    assert set(expanded) == set(DEFAULT_TOOLS)


def test_scan_with_hdbscan_requests_clustering() -> None:
    expanded = validate_tools([*DEFAULT_SCAN_META_TOOLS, "hdbscan"])
    assert run_cluster_requested(expanded) is True
    assert set(expanded) == set(DEFAULT_TOOLS) | {"cluster"}


def test_resolve_backend_job_groups_one_job_per_tool() -> None:
    per_image = resolve_per_image_tools(validate_tools(list(DEFAULT_SCAN_META_TOOLS)))
    groups = resolve_backend_job_groups(per_image)
    assert groups == [
        ("scrfd", ["scrfd"]),
        ("arcface", ["arcface"]),
        ("dlib_detect", ["dlib_detect"]),
        ("dlib_embed", ["dlib_embed"]),
    ]


def test_resolve_backend_job_groups_includes_analysis() -> None:
    per_image = resolve_per_image_tools(validate_tools(["opencv_fer", "mediapipe_blendshapes"]))
    groups = resolve_backend_job_groups(per_image)
    keys = [key for key, _ in groups]
    assert keys[0] == "scrfd"
    assert "opencv_fer" in keys
    assert "mediapipe_blendshapes" in keys
    assert groups[keys.index("opencv_fer")][1] == ["opencv_fer"]
    assert groups[keys.index("mediapipe_blendshapes")][1] == ["mediapipe_blendshapes"]


def test_resolve_backend_job_groups_splits_dlib_tools() -> None:
    groups = resolve_backend_job_groups(["dlib_detect", "dlib_embed"])
    assert groups == [
        ("dlib_detect", ["dlib_detect"]),
        ("dlib_embed", ["dlib_embed"]),
    ]
