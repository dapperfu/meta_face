"""Tool registry and dependency resolution."""

from __future__ import annotations

from meta_face.config import AGGREGATE_TOOLS, ALL_TOOLS, PER_IMAGE_TOOLS

TOOL_ALIASES: dict[str, str] = {
    "hdbscan": "cluster",
}


def normalize_tool_name(name: str) -> str:
    key = name.strip().lower()
    return TOOL_ALIASES.get(key, key)


def validate_tools(tools: list[str]) -> list[str]:
    normalized = [normalize_tool_name(t) for t in tools if t.strip()]
    unknown = [t for t in normalized if t not in ALL_TOOLS]
    if unknown:
        valid = ", ".join(sorted(ALL_TOOLS | {"hdbscan"}))
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
