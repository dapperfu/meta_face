"""Directory scanning and skip logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from meta_face.config import AGGREGATE_TOOLS, DEFAULT_TOOLS, PER_IMAGE_TOOLS
from meta_face.imaging import is_image_path
from meta_face.sidecar import has_tool, load_or_create


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
        if name and name not in normalized:
            normalized.append(name)
    return normalized


def resolve_per_image_tools(tools: list[str]) -> list[str]:
    """Return ordered per-image tools including dependencies."""
    requested = set(normalize_tools(tools))
    result: list[str] = []
    if "scrfd" in requested or "arcface" in requested:
        if "scrfd" in requested:
            result.append("scrfd")
        if "arcface" in requested:
            if "scrfd" not in result:
                result.insert(0, "scrfd")
            result.append("arcface")
    elif not requested & AGGREGATE_TOOLS:
        result.extend(DEFAULT_TOOLS)
    return result


def needs_processing(media_path: Path, tools: list[str], force: bool) -> bool:
    if force:
        return True
    doc, _ = load_or_create(media_path)
    for tool in tools:
        if tool in PER_IMAGE_TOOLS and not has_tool(doc, tool):
            return True
    return False


def iter_images(root: Path, recursive: bool = True) -> list[Path]:
    if root.is_file():
        return [root] if is_image_path(root) else []
    if not root.is_dir():
        return []
    if recursive:
        paths = sorted(p for p in root.rglob("*") if p.is_file() and is_image_path(p))
    else:
        paths = sorted(p for p in root.iterdir() if p.is_file() and is_image_path(p))
    return paths


def scan_directory(
    root: Path,
    tools: list[str],
    *,
    force: bool = False,
    recursive: bool = True,
) -> tuple[list[Path], ScanStats, bool]:
    """
    Scan a directory and return paths to enqueue plus stats.

    Returns (paths_to_enqueue, stats, run_cluster).
    """
    normalized = normalize_tools(tools)
    per_image_tools = resolve_per_image_tools(normalized)
    run_cluster = "cluster" in normalized

    stats = ScanStats()
    to_enqueue: list[Path] = []

    for image_path in iter_images(root, recursive=recursive):
        stats.discovered += 1
        if not per_image_tools:
            continue
        if needs_processing(image_path, per_image_tools, force):
            to_enqueue.append(image_path)
            stats.enqueued += 1
        else:
            stats.skipped += 1

    return to_enqueue, stats, run_cluster
