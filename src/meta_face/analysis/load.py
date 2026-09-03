"""Load and summarize a single image sidecar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sidecar_rs import SidecarDocument

from meta_face.sidecar import list_face_tools, sidecar_path_for_media

from .records import ImageSummary

_DETECTION_TOOLS: dict[str, str] = {
    "scrfd": "scrfd_face_count",
    "dlib_detect": "dlib_face_count",
}


def _face_count(section: dict[str, Any], field: str = "faces") -> int | None:
    faces = section.get(field)
    if not isinstance(faces, list):
        return None
    return len(faces)


def _avg_det_score(section: dict[str, Any]) -> float | None:
    faces = section.get("faces")
    if not isinstance(faces, list) or not faces:
        return None
    scores: list[float] = []
    for face in faces:
        if not isinstance(face, dict):
            continue
        det = face.get("det_score")
        if det is not None:
            scores.append(float(det))
    if not scores:
        return None
    return sum(scores) / len(scores)


def _cluster_stats(labels: Any) -> tuple[list[int] | None, int | None, int | None]:
    if not isinstance(labels, list):
        return None, None, None
    parsed = [int(x) for x in labels]
    unique = len({x for x in parsed if x >= 0})
    noise = sum(1 for x in parsed if x < 0)
    return parsed, unique, noise


def _section(doc: SidecarDocument, tool: str) -> dict[str, Any]:
    prefix = f"face.{tool}."
    section: dict[str, Any] = {}
    for key, value in doc.entries().items():
        if key.startswith(prefix):
            section[key[len(prefix) :]] = value
    return section


def summarize_sidecar(media_path: Path) -> ImageSummary:
    """Extract an ImageSummary from an image path (or .scar path)."""
    media_path = Path(media_path).resolve()
    scar_path = sidecar_path_for_media(media_path)
    parent_name = media_path.parent.name
    year = ImageSummary.year_from_parent(parent_name)

    if not scar_path.exists():
        return ImageSummary(
            media_path=media_path,
            sidecar_path=None,
            parent_name=parent_name,
            year=year,
            has_sidecar=False,
        )

    doc = SidecarDocument.from_path(str(scar_path))
    tools = list_face_tools(doc)

    scrfd_section = _section(doc, "scrfd")
    dlib_section = _section(doc, "dlib_detect")
    cluster_section = _section(doc, "cluster")
    cluster_dlib_section = _section(doc, "cluster_dlib")

    cluster_labels, cluster_unique, cluster_noise = _cluster_stats(cluster_section.get("labels"))
    (
        cluster_dlib_labels,
        cluster_dlib_unique,
        cluster_dlib_noise,
    ) = _cluster_stats(cluster_dlib_section.get("labels"))

    return ImageSummary(
        media_path=media_path,
        sidecar_path=scar_path,
        parent_name=parent_name,
        year=year,
        has_sidecar=True,
        tools_present=tools,
        scrfd_face_count=_face_count(scrfd_section),
        dlib_face_count=_face_count(dlib_section),
        scrfd_avg_det_score=_avg_det_score(scrfd_section),
        cluster_labels=cluster_labels,
        cluster_dlib_labels=cluster_dlib_labels,
        cluster_unique=cluster_unique,
        cluster_noise=cluster_noise,
        cluster_dlib_unique=cluster_dlib_unique,
        cluster_dlib_noise=cluster_dlib_noise,
    )


def summarize_sidecar_str(media_path_str: str) -> ImageSummary:
    """Picklable wrapper for parallel workers."""
    return summarize_sidecar(Path(media_path_str))
