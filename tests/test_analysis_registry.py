"""Smoke tests for analysis tool registry and dependencies."""

from __future__ import annotations

import importlib

import pytest

from meta_face.config import ALL_TOOLS, ANALYSIS_TOOLS, TOOL_VERSIONS
from meta_face.tools.analysis.registry import ANALYSIS_TOOL_NAMES, list_analysis_tools, tool_availability
from meta_face.tools.registry import TOOL_GROUPS, expand_dependencies, validate_tools


def test_all_analysis_tools_registered() -> None:
    assert ANALYSIS_TOOL_NAMES == ANALYSIS_TOOLS
    assert len(list_analysis_tools()) == len(ANALYSIS_TOOLS)


def test_analysis_tools_have_versions() -> None:
    missing = ANALYSIS_TOOLS - set(TOOL_VERSIONS)
    assert not missing, f"Missing TOOL_VERSIONS for: {missing}"


def test_analysis_tools_in_all_tools() -> None:
    assert ANALYSIS_TOOLS <= ALL_TOOLS


@pytest.mark.parametrize("tool_name", sorted(ANALYSIS_TOOLS))
def test_analysis_module_imports(tool_name: str) -> None:
    mod = importlib.import_module(f"meta_face.tools.analysis.{tool_name}")
    assert mod.TOOL_NAME == tool_name
    assert hasattr(mod, "availability")
    assert hasattr(mod, "analyze_faces")


def test_expression_meta_tool_expands() -> None:
    tools = validate_tools(["expression"])
    assert "emotiefflib" in tools
    assert "opencv_fer" in tools
    assert "mediapipe_blendshapes" in tools


def test_expand_dependencies_adds_scrfd_for_analysis() -> None:
    deps = expand_dependencies(["emotiefflib"])
    assert deps[0] == "scrfd"
    assert "emotiefflib" in deps


def test_tool_availability_returns_message_or_none() -> None:
    for name in list_analysis_tools():
        issue = tool_availability(name)
        assert issue is None or isinstance(issue, str)


def test_face_analysis_group_is_phase1_subset() -> None:
    members = set(TOOL_GROUPS["face_analysis"])
    assert members <= ANALYSIS_TOOLS
    assert "emotiefflib" in members
