"""Lightweight RQ scan, image-query, and add-image jobs.

This module must not import inference (cv2 / CUDA / dlib). Those workers only
list directories, list images, and enqueue the next stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def scan_path(
    directory: str,
    tools: list[str],
    force: bool = False,
    recursive: bool = True,
) -> dict[str, Any]:
    """List one directory, enqueue child scans, then one image-query job."""
    from meta_face.queue import enqueue_add_image, enqueue_query_images, enqueue_scan_path
    from meta_face.scanner import resolve_per_image_tools, scan_directory_level

    dir_path = Path(directory).resolve()
    per_image_tools = resolve_per_image_tools(tools)
    stats, images, subdirs = scan_directory_level(dir_path, tools, force=force)

    if dir_path.is_file():
        add_jobs = [
            enqueue_add_image(image_path, per_image_tools, force=force) for image_path in images
        ]
        return {
            "status": "ok",
            "path": str(dir_path),
            "discovered": stats.discovered,
            "enqueued": stats.enqueued,
            "query_job": None,
            "add_jobs": add_jobs,
            "subdir_jobs": 0,
            "skipped": stats.skipped,
            "subdirs": 0,
        }

    subdir_jobs = 0
    if recursive and subdirs:
        for subdir in subdirs:
            enqueue_scan_path(subdir, tools, force=force, recursive=recursive)
            subdir_jobs += 1

    query_job = None
    if per_image_tools:
        query_job = enqueue_query_images(dir_path, per_image_tools, force=force)

    return {
        "status": "ok",
        "path": str(dir_path),
        "discovered": stats.discovered,
        "enqueued": stats.enqueued,
        "query_job": query_job,
        "subdir_jobs": subdir_jobs,
        "skipped": stats.skipped,
        "subdirs": len(subdirs),
    }


def query_images(
    directory: str,
    tools: list[str],
    force: bool = False,
) -> dict[str, Any]:
    """List images in one directory and enqueue one add-image job per file."""
    from meta_face.queue import enqueue_add_image
    from meta_face.scanner import resolve_per_image_tools, scan_directory_level

    dir_path = Path(directory).resolve()
    per_image_tools = resolve_per_image_tools(tools)
    stats, images, _subdirs = scan_directory_level(dir_path, tools, force=force)
    job_ids = [
        enqueue_add_image(image_path, per_image_tools, force=force) for image_path in images
    ]
    return {
        "status": "ok",
        "path": str(dir_path),
        "discovered": stats.discovered,
        "enqueued": len(job_ids),
        "job_ids": job_ids,
        "skipped": stats.skipped,
    }


def add_image(
    image_path: str,
    tools: list[str],
    force: bool = False,
) -> dict[str, Any]:
    """Enqueue one face job per tool that still needs to run on this image."""
    from meta_face.config import CROP_ANALYSIS_TOOLS
    from meta_face.queue import enqueue_face_tool
    from meta_face.scanner import resolve_image_tasks, resolve_per_image_tools
    from meta_face.sidecar import load_or_create, tool_is_current

    media_path = Path(image_path).resolve()
    tasks = resolve_image_tasks(resolve_per_image_tools(tools))
    if force:
        pending = tasks
    else:
        doc, _scar_path = load_or_create(media_path)
        pending = [tool for tool in tasks if not tool_is_current(doc, tool)]

    if not pending:
        return {"status": "skipped", "path": str(media_path), "tools": [], "job_ids": []}

    ordered = [tool for tool in pending if tool == "scrfd"] + [
        tool for tool in pending if tool != "scrfd"
    ]
    scrfd_job_id: str | None = None
    job_ids: list[str] = []
    for tool in ordered:
        depends_on = (
            scrfd_job_id if tool in CROP_ANALYSIS_TOOLS and scrfd_job_id is not None else None
        )
        job_id = enqueue_face_tool(media_path, tool, force=force, depends_on=depends_on)
        if tool == "scrfd":
            scrfd_job_id = job_id
        job_ids.append(job_id)

    return {
        "status": "ok",
        "path": str(media_path),
        "tools": ordered,
        "job_ids": job_ids,
    }
