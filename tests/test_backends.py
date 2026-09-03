"""Tests for face detection backends and tool wiring."""

from __future__ import annotations

from meta_face.config import DEFAULT_SCAN_META_TOOLS, DEFAULT_TOOLS
from meta_face.backends.registry import get_detection_backend
from meta_face.scanner import resolve_per_image_tools, run_cluster_requested
from meta_face.tools.registry import expand_dependencies, validate_tools


def test_registry_resolves_scrfd() -> None:
    assert get_detection_backend("scrfd").name == "scrfd"


def test_validate_tools_default_scan_meta_tools() -> None:
    expanded = validate_tools(list(DEFAULT_SCAN_META_TOOLS))
    assert set(expanded) == set(DEFAULT_TOOLS)
    assert resolve_per_image_tools(expanded) == list(DEFAULT_TOOLS)
    assert run_cluster_requested(expanded) is False


def test_resolve_per_image_tools_empty_uses_all_default_pipelines() -> None:
    assert resolve_per_image_tools([]) == list(DEFAULT_TOOLS)


def test_expand_dependencies_arcface_still_includes_scrfd() -> None:
    assert expand_dependencies(["arcface"]) == ["scrfd", "arcface"]
