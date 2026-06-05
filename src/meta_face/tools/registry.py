"""Tool registry and dependency resolution."""

from __future__ import annotations

from meta_face.config import AGGREGATE_TOOLS, ALL_TOOLS, ANALYSIS_TOOLS, PER_IMAGE_TOOLS

TOOL_ALIASES: dict[str, str] = {
    "hdbscan": "cluster",
    "hdbscan_dlib": "cluster_dlib",
}

# Meta-tool names that expand to several real tools. "insightface" is the
# single SCRFD detection + ArcFace recognition pass.
TOOL_GROUPS: dict[str, tuple[str, ...]] = {
    "insightface": ("scrfd", "arcface"),
    "face_recognition": ("dlib_detect", "dlib_embed"),
    "detectron2": ("detectron2",),
    # Analysis meta-tools (crop-based; require scrfd).
    "expression": (
        "emotiefflib",
        "opencv_fer",
        "mediapipe_blendshapes",
        "fer_plus",
    ),
    "emotion": (
        "emotiefflib",
        "opencv_fer",
        "fer_plus",
        "deepface",
        "emonet",
    ),
    "au": ("libreface", "openface3", "py_feat"),
    "gaze": ("yakhyo_gaze", "l2cs_net", "libreface", "openface3", "uniface"),
    "blendshapes": ("mediapipe_blendshapes",),
    "attributes": ("fairface", "deepface"),
    "parsing": ("bisenet", "uniface"),
    "liveness": ("face_antispoof_onnx", "face_anti_spoofing", "uniface", "inspireface"),
    "face_analysis": (
        "emotiefflib",
        "opencv_fer",
        "mediapipe_blendshapes",
    ),
    "all_analysis": tuple(sorted(ANALYSIS_TOOLS)),
}


def normalize_tool_name(name: str) -> str:
    key = name.strip().lower()
    return TOOL_ALIASES.get(key, key)


def expand_group(name: str) -> list[str]:
    """Expand a meta-tool (e.g. 'insightface') to its real tools."""
    key = name.strip().lower()
    if key in TOOL_GROUPS:
        return list(TOOL_GROUPS[key])
    return [normalize_tool_name(name)]


def validate_tools(tools: list[str]) -> list[str]:
    normalized: list[str] = []
    for tool in tools:
        if not tool.strip():
            continue
        for expanded in expand_group(tool):
            if expanded not in normalized:
                normalized.append(expanded)
    unknown = [t for t in normalized if t not in ALL_TOOLS]
    if unknown:
        valid = ", ".join(sorted(ALL_TOOLS | set(TOOL_ALIASES) | set(TOOL_GROUPS)))
        raise ValueError(f"Unknown tools: {', '.join(unknown)}. Valid: {valid}")
    return normalized


def is_per_image(tool: str) -> bool:
    return normalize_tool_name(tool) in PER_IMAGE_TOOLS


def is_aggregate(tool: str) -> bool:
    return normalize_tool_name(tool) in AGGREGATE_TOOLS


def is_analysis_tool(tool: str) -> bool:
    return normalize_tool_name(tool) in ANALYSIS_TOOLS


def analysis_tools_requested(tools: list[str]) -> bool:
    names = set(validate_tools(tools))
    return bool(names & ANALYSIS_TOOLS)


def expand_dependencies(tools: list[str]) -> list[str]:
    names = validate_tools(tools)
    result: list[str] = []
    if "scrfd" in names or "arcface" in names or analysis_tools_requested(names):
        if "scrfd" not in result:
            result.append("scrfd")
        if "arcface" in names and "arcface" not in result:
            result.append("arcface")
    if "dlib_detect" in names or "dlib_embed" in names:
        if "dlib_detect" not in result:
            result.append("dlib_detect")
        if "dlib_embed" in names and "dlib_embed" not in result:
            result.append("dlib_embed")
    if "detectron2" in names and "detectron2" not in result:
        result.append("detectron2")
    for tool in names:
        if tool in ANALYSIS_TOOLS and tool not in result:
            result.append(tool)
    return result


def insightface_tools_requested(tools: list[str]) -> bool:
    names = set(validate_tools(tools))
    return bool(names & {"scrfd", "arcface"})


def dlib_tools_requested(tools: list[str]) -> bool:
    names = set(validate_tools(tools))
    return bool(names & {"dlib_detect", "dlib_embed"})


def detectron2_tools_requested(tools: list[str]) -> bool:
    names = set(validate_tools(tools))
    return "detectron2" in names
