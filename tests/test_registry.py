"""Tests for tool registry and dependency resolution."""

from __future__ import annotations

import pytest

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
