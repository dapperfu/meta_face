"""Run crop-based analysis tools against SCRFD face detections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from meta_face.tools.analysis.crops import (
    face_contexts_from_insightface_faces,
    load_scrfd_face_contexts,
    scrfd_faces_from_doc,
)
from meta_face.tools.analysis.registry import run_analysis_tool, tool_availability


def resolve_face_contexts(
    media_path: Path,
    image_bgr: np.ndarray,
    *,
    doc: object | None = None,
    insightface_faces: list[Any] | None = None,
) -> list[Any]:
    """Prefer live SCRFD faces; fall back to sidecar records."""
    if insightface_faces is not None:
        return face_contexts_from_insightface_faces(image_bgr, insightface_faces)
    if doc is not None:
        faces = scrfd_faces_from_doc(doc)
        if faces is not None:
            from meta_face.tools.analysis.crops import face_contexts_from_records

            return face_contexts_from_records(image_bgr, faces)
    return load_scrfd_face_contexts(media_path, image_bgr)


def run_pending_analysis_tools(
    media_path: Path,
    image_bgr: np.ndarray,
    tool_names: list[str],
    *,
    doc: object | None = None,
    insightface_faces: list[Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run analysis tools and return {tool_name: sidecar_payload}."""
    if not tool_names:
        return {}

    contexts = resolve_face_contexts(
        media_path,
        image_bgr,
        doc=doc,
        insightface_faces=insightface_faces,
    )
    results: dict[str, dict[str, Any]] = {}
    for tool in tool_names:
        issue = tool_availability(tool)
        if issue is not None:
            raise RuntimeError(issue)
        results[tool] = run_analysis_tool(tool, image_bgr, contexts)
    return results
