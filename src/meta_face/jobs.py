"""RQ job entrypoints."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import cv2

from meta_face.backends.registry import get_detection_backend
from meta_face.deps import (
    require_cluster_runtime,
    require_dlib_runtime,
    require_detectron2_runtime,
    require_inference_runtime,
    require_insightface_runtime,
)
from meta_face.imaging import load_image
from meta_face.sidecar import has_tool, load_or_create, update_sidecar, write_tool_result
from meta_face.tools.registry import expand_dependencies

logger = logging.getLogger(__name__)


def _tools_to_run(doc: object, tools: list[str], force: bool) -> list[str]:
    if force:
        return tools
    return [t for t in tools if not has_tool(doc, t)]  # type: ignore[arg-type]


def process_image(image_path: str, tools: list[str], force: bool = False) -> dict[str, Any]:
    """RQ job: run selected per-image face tools and write sidecar data."""
    per_image_tools = expand_dependencies(tools)
    require_inference_runtime(per_image_tools)

    media_path = Path(image_path).resolve()
    doc, _ = load_or_create(media_path)
    pending = _tools_to_run(doc, per_image_tools, force)

    if not pending:
        return {"status": "skipped", "path": str(media_path), "reason": "all_tools_present"}

    image = load_image(media_path)
    face_count = 0
    pending_set = set(pending)

    insightface_faces = None
    dlib_rgb_faces: tuple[Any, list[Any]] | None = None
    detectron2_records: list[dict[str, Any]] | None = None

    if pending_set & {"scrfd", "arcface"}:
        require_insightface_runtime()
        from meta_face.tools.scrfd import detect_faces

        insightface_faces = detect_faces(image)
        face_count = max(face_count, len(insightface_faces))

    if pending_set & {"dlib_detect", "dlib_embed"}:
        require_dlib_runtime()
        from meta_face.tools.dlib_detect import detect_faces as dlib_detect_faces

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dlib_faces = dlib_detect_faces(rgb)
        dlib_rgb_faces = (rgb, dlib_faces)
        face_count = max(face_count, len(dlib_faces))

    if "detectron2" in pending_set:
        require_detectron2_runtime()
        backend = get_detection_backend("detectron2")
        detections = backend.detect(image)
        detectron2_records = backend.to_records(detections)
        face_count = max(face_count, len(detections))

    def _patch(doc: object) -> None:
        if insightface_faces is not None:
            from meta_face.tools.arcface import embeddings_from_faces
            from meta_face.tools.scrfd import faces_to_records

            if "scrfd" in pending_set:
                write_tool_result(doc, "scrfd", {"faces": faces_to_records(insightface_faces)})  # type: ignore[arg-type]
            if "arcface" in pending_set:
                write_tool_result(
                    doc,
                    "arcface",
                    {"embeddings": embeddings_from_faces(insightface_faces)},
                )  # type: ignore[arg-type]

        if dlib_rgb_faces is not None:
            from meta_face.tools.dlib_detect import faces_to_records as dlib_faces_to_records
            from meta_face.tools.dlib_embed import embeddings_from_faces as dlib_embeddings

            rgb, dlib_faces = dlib_rgb_faces
            if "dlib_detect" in pending_set:
                write_tool_result(
                    doc,
                    "dlib_detect",
                    {"faces": dlib_faces_to_records(dlib_faces)},
                )  # type: ignore[arg-type]
            if "dlib_embed" in pending_set:
                write_tool_result(
                    doc,
                    "dlib_embed",
                    {"embeddings": dlib_embeddings(rgb, dlib_faces)},
                )  # type: ignore[arg-type]

        if detectron2_records is not None:
            write_tool_result(doc, "detectron2", {"faces": detectron2_records})  # type: ignore[arg-type]

    scar_path = update_sidecar(media_path, _patch)
    return {
        "status": "ok",
        "path": str(media_path),
        "tools": pending,
        "face_count": face_count,
        "sidecar": str(scar_path),
    }


def run_cluster(
    root_path: str,
    force: bool = False,
    embedding_tool: str = "arcface",
) -> dict[str, Any]:
    """RQ job: aggregate clustering with FAISS + HDBSCAN."""
    require_cluster_runtime()
    from meta_face.config import normalize_embedding_tool
    from meta_face.tools.cluster import run_cluster_pipeline

    root = Path(root_path).resolve()
    emb_tool = normalize_embedding_tool(embedding_tool)
    result = run_cluster_pipeline(root, force=force, embedding_tool=emb_tool)
    result["root"] = str(root)
    result["embedding_tool"] = emb_tool
    return result


def scan_path(
    directory: str,
    tools: list[str],
    force: bool = False,
    recursive: bool = True,
) -> dict[str, Any]:
    """RQ job: scan one directory, enqueue child scans then image jobs."""
    from meta_face.queue import enqueue_process_image, enqueue_scan_path
    from meta_face.scanner import resolve_per_image_tools, scan_directory_level

    dir_path = Path(directory).resolve()
    per_image_tools = resolve_per_image_tools(tools)
    stats, to_enqueue, subdirs = scan_directory_level(dir_path, tools, force=force)

    # Fan out to child directories first (high-priority scan queue).
    if recursive and subdirs:
        for subdir in subdirs:
            enqueue_scan_path(subdir, tools, force=force, recursive=recursive)

    for image_path in to_enqueue:
        enqueue_process_image(image_path, per_image_tools, force=force)

    return {
        "status": "ok",
        "path": str(dir_path),
        "discovered": stats.discovered,
        "enqueued": stats.enqueued,
        "skipped": stats.skipped,
        "subdirs": len(subdirs),
    }


def job_id_for_path(prefix: str, path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"
