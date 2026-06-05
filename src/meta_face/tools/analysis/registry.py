"""Registry of crop-based analysis tools."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from meta_face.tools.analysis import (
    bisenet,
    deepface,
    emotiefflib,
    emonet,
    face_anti_spoofing,
    face_antispoof_onnx,
    fairface,
    fer_plus,
    inspireface,
    l2cs_net,
    libreface,
    mediapipe_blendshapes,
    openface3,
    opencv_fer,
    py_feat,
    uniface,
    yakhyo_gaze,
)
from meta_face.tools.analysis.base import AnalysisTool, FaceContext

_ANALYSIS_MODULES: tuple[Any, ...] = (
    emotiefflib,
    opencv_fer,
    mediapipe_blendshapes,
    fer_plus,
    libreface,
    openface3,
    py_feat,
    emonet,
    deepface,
    yakhyo_gaze,
    l2cs_net,
    fairface,
    bisenet,
    face_antispoof_onnx,
    face_anti_spoofing,
    uniface,
    inspireface,
)

ANALYSIS_TOOL_NAMES: frozenset[str] = frozenset(mod.TOOL_NAME for mod in _ANALYSIS_MODULES)

_REGISTRY: dict[str, Any] = {mod.TOOL_NAME: mod for mod in _ANALYSIS_MODULES}


def list_analysis_tools() -> list[str]:
    return sorted(_REGISTRY)


def get_analysis_tool(name: str) -> Any:
    key = name.strip().lower()
    if key not in _REGISTRY:
        known = ", ".join(list_analysis_tools())
        raise KeyError(f"Unknown analysis tool '{name}'. Known: {known}")
    return _REGISTRY[key]


def tool_availability(name: str) -> str | None:
    mod = get_analysis_tool(name)
    return mod.availability()


def run_analysis_tool(
    name: str,
    image_bgr: Any,
    faces: list[FaceContext],
) -> dict[str, Any]:
    mod = get_analysis_tool(name)
    issue = mod.availability()
    if issue is not None:
        raise RuntimeError(issue)
    return mod.analyze_faces(image_bgr, faces)


def import_analysis_module(name: str) -> AnalysisTool:
    """Import an analysis tool module by name (for smoke tests)."""
    return import_module(f"meta_face.tools.analysis.{name}")
