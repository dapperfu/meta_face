"""RQ job entrypoints."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from meta_face.imaging import load_image
from meta_face.sidecar import has_tool, load_or_create, save, write_tool_result
from meta_face.tools.arcface import embeddings_from_faces
from meta_face.tools.cluster import run_cluster_pipeline
from meta_face.tools.registry import expand_dependencies, normalize_tool_name
from meta_face.tools.scrfd import detect_faces, faces_to_records

logger = logging.getLogger(__name__)


def _tools_to_run(doc: object, tools: list[str], force: bool) -> list[str]:
    if force:
        return tools
    return [t for t in tools if not has_tool(doc, t)]  # type: ignore[arg-type]


def process_image(image_path: str, tools: list[str], force: bool = False) -> dict[str, Any]:
    """RQ job: run selected per-image face tools and write sidecar data."""
    media_path = Path(image_path).resolve()
    per_image_tools = expand_dependencies(tools)
    doc, scar_path = load_or_create(media_path)
    pending = _tools_to_run(doc, per_image_tools, force)

    if not pending:
        return {"status": "skipped", "path": str(media_path), "reason": "all_tools_present"}

    image = load_image(media_path)
    faces = detect_faces(image)

    if "scrfd" in pending:
        write_tool_result(doc, "scrfd", {"faces": faces_to_records(faces)})

    if "arcface" in pending:
        write_tool_result(doc, "arcface", {"embeddings": embeddings_from_faces(faces)})

    save(doc, scar_path)
    return {
        "status": "ok",
        "path": str(media_path),
        "tools": pending,
        "face_count": len(faces),
        "sidecar": str(scar_path),
    }


def run_cluster(root_path: str, force: bool = False) -> dict[str, Any]:
    """RQ job: aggregate clustering with FAISS + HDBSCAN."""
    root = Path(root_path).resolve()
    result = run_cluster_pipeline(root, force=force)
    result["root"] = str(root)
    return result


def job_id_for_path(prefix: str, path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"
