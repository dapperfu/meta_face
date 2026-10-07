"""RQ job entrypoints."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
from rq import Retry, get_current_job
from sidecar_rs import LockTimeout

from meta_face.config import ANALYSIS_TOOLS, CROP_ANALYSIS_TOOLS, SIDECAR_LOCK_MAX_REQUEUES
from meta_face.deps import (
    require_cluster_runtime,
    require_dlib_runtime,
    require_inference_runtime,
    require_insightface_runtime,
)
from meta_face.imaging import load_image
from meta_face.sidecar import (
    has_tool,
    load_or_create,
    sidecar_path_for_media,
    tool_is_current,
    update_sidecar,
    write_tool_result,
)
from meta_face.tools.registry import expand_dependencies

logger = logging.getLogger(__name__)


def _requeue_on_lockfile_timeout(
    job_func: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any] | Retry]:
    """Requeue an RQ job on lockfile timeout. Outside RQ the same error is a failure."""

    @functools.wraps(job_func)
    def run(*args: Any, **kwargs: Any) -> dict[str, Any] | Retry:
        try:
            return job_func(*args, **kwargs)
        except LockTimeout as exc:
            job = get_current_job()
            if job is None or (job.number_of_retries or 0) >= SIDECAR_LOCK_MAX_REQUEUES:
                raise
            logger.warning("lockfile timeout (%s); putting job %s back on its queue", exc, job.id)
            return Retry(max=SIDECAR_LOCK_MAX_REQUEUES)

    return run


def _tools_to_run(doc: object, tools: list[str], force: bool) -> list[str]:
    if force:
        return tools
    return [t for t in tools if not tool_is_current(doc, t)]  # type: ignore[arg-type]


def _write_tool_payload(
    media_path: Path,
    tool: str,
    payload: dict[str, Any],
    image_size: tuple[int, int],
) -> Path:
    """Hold the sidecar lock only long enough to merge one tool's result."""

    def _patch(doc: object) -> None:
        write_tool_result(doc, tool, payload, image_size=image_size)  # type: ignore[arg-type]

    return update_sidecar(media_path, _patch)


@_requeue_on_lockfile_timeout
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
    h_img, w_img = image.shape[:2]
    image_size = (w_img, h_img)
    face_count = 0
    pending_set = set(pending)
    pending_writes: list[tuple[str, dict[str, Any]]] = []

    insightface_faces = None
    dlib_rgb_faces: tuple[Any, list[Any]] | None = None

    pending_analysis = [t for t in pending if t in ANALYSIS_TOOLS]
    scrfd_in_sidecar = has_tool(doc, "scrfd")  # type: ignore[arg-type]
    needs_scrfd_detect = (
        bool(pending_set & {"scrfd", "arcface"})
        or (bool(pending_set & CROP_ANALYSIS_TOOLS) and not scrfd_in_sidecar)
    )

    if needs_scrfd_detect:
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

    if insightface_faces is not None:
        if "scrfd" in pending_set:
            from meta_face.tools.face_record import scrfd_to_sidecar_payload

            pending_writes.append(
                (
                    "scrfd",
                    scrfd_to_sidecar_payload(insightface_faces, image_size=image_size),
                )
            )
        if "arcface" in pending_set:
            from meta_face.tools.arcface import arcface_to_sidecar_payload

            pending_writes.append(("arcface", arcface_to_sidecar_payload(insightface_faces)))

    if dlib_rgb_faces is not None:
        from meta_face.tools.dlib_detect import dlib_detect_to_sidecar_payload
        from meta_face.tools.dlib_embed import dlib_embed_to_sidecar_payload

        rgb, dlib_faces = dlib_rgb_faces
        if "dlib_detect" in pending_set:
            pending_writes.append(
                (
                    "dlib_detect",
                    dlib_detect_to_sidecar_payload(dlib_faces, image_size=image_size),
                )
            )
        if "dlib_embed" in pending_set:
            pending_writes.append(("dlib_embed", dlib_embed_to_sidecar_payload(rgb, dlib_faces)))

    analysis_errors: list[str] = []
    if pending_analysis:
        from meta_face.tools.analysis.runner import run_pending_analysis_tools

        for tool_name in pending_analysis:
            try:
                analysis_results = run_pending_analysis_tools(
                    media_path,
                    image,
                    [tool_name],
                    doc=doc,
                    insightface_faces=insightface_faces,
                )
            except Exception as exc:
                logger.exception("analysis tool %s failed for %s", tool_name, media_path)
                analysis_errors.append(f"{tool_name}: {exc}")
                continue
            payload = analysis_results[tool_name]
            face_count = max(face_count, int(payload.get("face_count", 0)))
            pending_writes.append((tool_name, payload))

    scar_path = sidecar_path_for_media(media_path)
    for tool_name, payload in pending_writes:
        scar_path = _write_tool_payload(media_path, tool_name, payload, image_size)

    if analysis_errors:
        raise RuntimeError(
            "Analysis tool(s) failed after independent sidecar writes: "
            + "; ".join(analysis_errors)
        )

    return {
        "status": "ok",
        "path": str(media_path),
        "tools": pending,
        "face_count": face_count,
        "sidecar": str(scar_path),
    }


@_requeue_on_lockfile_timeout
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
