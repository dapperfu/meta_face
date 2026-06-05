"""Directory scanning and skip logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from meta_face.config import AGGREGATE_TOOLS, ANALYSIS_TOOLS, DEFAULT_TOOLS, PER_IMAGE_TOOLS
from meta_face.imaging import is_image_path
from meta_face.sidecar import has_tool, load_or_create

logger = logging.getLogger(__name__)


@dataclass
class ScanStats:
    discovered: int = 0
    enqueued: int = 0
    skipped: int = 0


def normalize_tools(tools: list[str]) -> list[str]:
    """Expand aliases and preserve order."""
    normalized: list[str] = []
    for tool in tools:
        name = tool.strip().lower()
        if name == "hdbscan":
            name = "cluster"
        elif name == "hdbscan_dlib":
            name = "cluster_dlib"
        if name and name not in normalized:
            normalized.append(name)
    return normalized


# Backend pipelines enqueued as separate RQ jobs (one job per group).
BACKEND_JOB_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    ("insightface", frozenset({"scrfd", "arcface"})),
    ("face_recognition", frozenset({"dlib_detect", "dlib_embed"})),
    ("detectron2", frozenset({"detectron2"})),
    ("analysis", ANALYSIS_TOOLS),
)


def resolve_backend_job_groups(per_image_tools: list[str]) -> list[tuple[str, list[str]]]:
    """Split per-image tools into one (backend_key, tools) job per backend pipeline."""
    groups: list[tuple[str, list[str]]] = []
    for backend_key, members in BACKEND_JOB_GROUPS:
        group_tools = [tool for tool in per_image_tools if tool in members]
        if group_tools:
            groups.append((backend_key, group_tools))
    return groups


def resolve_per_image_tools(tools: list[str]) -> list[str]:
    """Return ordered per-image tools including dependencies."""
    requested = set(normalize_tools(tools))
    result: list[str] = []
    if requested & ({"scrfd", "arcface"} | ANALYSIS_TOOLS):
        if "scrfd" not in result:
            result.append("scrfd")
        if "arcface" in requested:
            result.append("arcface")
    if "dlib_detect" in requested or "dlib_embed" in requested:
        if "dlib_detect" not in result:
            result.append("dlib_detect")
        if "dlib_embed" in requested:
            result.append("dlib_embed")
    if "detectron2" in requested:
        result.append("detectron2")
    for tool in normalize_tools(tools):
        if tool in ANALYSIS_TOOLS and tool not in result:
            result.append(tool)
    if not result and not requested & AGGREGATE_TOOLS:
        result.extend(DEFAULT_TOOLS)
    return result


def run_cluster_requested(tools: list[str]) -> bool:
    requested = normalize_tools(tools)
    return "cluster" in requested or "cluster_dlib" in requested


def resolve_cluster_embedding_tool(tools: list[str], embedding_tool: str) -> str:
    """Pick embedding source for clustering from tools list and --embeddings."""
    from meta_face.config import normalize_embedding_tool

    requested = set(normalize_tools(tools))
    if "cluster_dlib" in requested:
        return "dlib_embed"
    if embedding_tool:
        return normalize_embedding_tool(embedding_tool)
    return "arcface"


def needs_processing(media_path: Path, tools: list[str], force: bool) -> bool:
    if force:
        return True
    doc, _ = load_or_create(media_path)
    for tool in tools:
        if tool in PER_IMAGE_TOOLS and not has_tool(doc, tool):
            return True
    return False


def scan_directory_level(
    root: Path,
    tools: list[str],
    *,
    force: bool = False,
) -> tuple[ScanStats, list[Path], list[Path]]:
    """
    Scan a single directory level (or a single file path).

    Returns (stats, paths_to_enqueue, subdirs).
    """
    per_image_tools = resolve_per_image_tools(tools)
    stats = ScanStats()
    to_enqueue: list[Path] = []
    subdirs: list[Path] = []

    if root.is_file():
        if is_image_path(root):
            stats.discovered += 1
            if per_image_tools and needs_processing(root, per_image_tools, force):
                to_enqueue.append(root)
                stats.enqueued += 1
            elif per_image_tools:
                stats.skipped += 1
        return stats, to_enqueue, subdirs

    if not root.is_dir():
        return stats, to_enqueue, subdirs

    try:
        entries = list(root.iterdir())
    except PermissionError:
        logger.warning("Permission denied scanning %s", root)
        return stats, to_enqueue, subdirs

    for entry in sorted(entries):
        if entry.is_file() and is_image_path(entry):
            stats.discovered += 1
            if not per_image_tools:
                continue
            if needs_processing(entry, per_image_tools, force):
                to_enqueue.append(entry)
                stats.enqueued += 1
            else:
                stats.skipped += 1
        elif entry.is_dir():
            subdirs.append(entry)

    return stats, to_enqueue, subdirs
