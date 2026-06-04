"""Tool registry and dependency resolution."""

from __future__ import annotations

from meta_face.config import AGGREGATE_TOOLS, ALL_TOOLS, PER_IMAGE_TOOLS

TOOL_ALIASES: dict[str, str] = {
    "hdbscan": "cluster",
}

# Meta-tool names that expand to several real tools. "insightface" is the
# single SCRFD detection + ArcFace recognition pass.
TOOL_GROUPS: dict[str, tuple[str, ...]] = {
    "insightface": ("scrfd", "arcface"),
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


def expand_dependencies(tools: list[str]) -> list[str]:
    names = validate_tools(tools)
    result: list[str] = []
    if "scrfd" in names or "arcface" in names:
        if "scrfd" not in result:
            result.append("scrfd")
        if "arcface" in names and "arcface" not in result:
            result.append("arcface")
    return result
