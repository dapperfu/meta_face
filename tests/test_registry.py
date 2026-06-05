"""Tests for tool registry and dependency resolution."""

from __future__ import annotations

import pytest

from meta_face.config import DEFAULT_SCAN_META_TOOLS, DEFAULT_TOOLS
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


def test_default_scan_runs_detection_without_clustering() -> None:
    assert DEFAULT_SCAN_META_TOOLS == (
        "insightface",
        "face_recognition",
        "detectron2",
    )
    expanded = validate_tools(list(DEFAULT_SCAN_META_TOOLS))
    assert resolve_per_image_tools(expanded) == list(DEFAULT_TOOLS)
    assert run_cluster_requested(expanded) is False
    assert set(expanded) == set(DEFAULT_TOOLS)


def test_scan_with_hdbscan_requests_clustering() -> None:
    expanded = validate_tools([*DEFAULT_SCAN_META_TOOLS, "hdbscan"])
    assert run_cluster_requested(expanded) is True
    assert set(expanded) == set(DEFAULT_TOOLS) | {"cluster"}


def test_resolve_backend_job_groups_splits_default_tools() -> None:
    per_image = resolve_per_image_tools(validate_tools(list(DEFAULT_SCAN_META_TOOLS)))
    groups = resolve_backend_job_groups(per_image)
    assert [key for key, _ in groups] == [
        "insightface",
        "face_recognition",
        "detectron2",
    ]
    assert groups[0][1] == ["scrfd", "arcface"]
    assert groups[1][1] == ["dlib_detect", "dlib_embed"]
    assert groups[2][1] == ["detectron2"]


def test_resolve_backend_job_groups_includes_analysis() -> None:
    per_image = resolve_per_image_tools(validate_tools(["expression"]))
    groups = resolve_backend_job_groups(per_image)
    keys = [key for key, _ in groups]
    assert "insightface" in keys
    assert "analysis" in keys


def test_resolve_backend_job_groups_single_backend() -> None:
    groups = resolve_backend_job_groups(["detectron2"])
    assert groups == [("detectron2", ["detectron2"])]
