"""Extract face crops from SCRFD sidecar records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from meta_face.bbox import crop_face
from meta_face.coordinates import record_to_pixels
from meta_face.sidecar import get_face_section, load_or_create
from meta_face.tools.analysis.base import FaceContext


DEFAULT_CROP_BUFFER_PCT = 10.0


def _face_index(face: dict[str, Any], fallback: int) -> int:
    value = face.get("face_index")
    if isinstance(value, int):
        return value
    return fallback


def face_contexts_from_records(
    image_bgr: np.ndarray,
    face_records: list[dict[str, Any]],
    *,
    buffer_pct: float = DEFAULT_CROP_BUFFER_PCT,
) -> list[FaceContext]:
    """Build FaceContext list from sidecar scrfd face dicts."""
    contexts: list[FaceContext] = []
    for idx, face in enumerate(face_records):
        h_img, w_img = image_bgr.shape[:2]
        face = record_to_pixels(face, (w_img, h_img))
        bbox = face.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in bbox[:4]):
            continue
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue  # Clipping an entirely out-of-frame detection leaves no crop.
        crop = crop_face(image_bgr, bbox, buffer_pct=buffer_pct)
        if crop.size == 0:
            continue
        contexts.append(
            FaceContext(
                face_index=_face_index(face, idx),
                bbox=[float(v) for v in bbox[:4]],
                crop_bgr=crop,
                metadata=face,
            )
        )
    return contexts


def scrfd_faces_from_doc(doc: object) -> list[dict[str, Any]] | None:
    """Return scrfd face records from a sidecar document, or None when absent."""
    section = get_face_section(doc, "scrfd")  # type: ignore[arg-type]
    faces = section.get("faces")
    if not isinstance(faces, list):
        return None
    records = []
    for face in faces:
        record = dict(face)
        if "coordinates" in section:
            record.setdefault("coordinates", section["coordinates"])
        if "image_size" in section:
            record.setdefault("source_image_size", section["image_size"])
        records.append(record)
    return records


def load_scrfd_face_contexts(
    media_path: Path,
    image_bgr: np.ndarray,
    *,
    buffer_pct: float = DEFAULT_CROP_BUFFER_PCT,
) -> list[FaceContext]:
    """Load SCRFD faces from sidecar and extract crops."""
    doc, _ = load_or_create(media_path)
    faces = scrfd_faces_from_doc(doc)
    if faces is None:
        raise FileNotFoundError(
            f"No scrfd face data for {media_path}. Run: mf scan {media_path} --tools scrfd"
        )
    return face_contexts_from_records(image_bgr, faces, buffer_pct=buffer_pct)


def face_contexts_from_insightface_faces(
    image_bgr: np.ndarray,
    insightface_faces: list[Any],
    *,
    buffer_pct: float = DEFAULT_CROP_BUFFER_PCT,
) -> list[FaceContext]:
    """Build FaceContext list from live insightface Face objects."""
    from meta_face.tools.face_record import faces_to_sidecar_records

    records = faces_to_sidecar_records(insightface_faces)
    return face_contexts_from_records(image_bgr, records, buffer_pct=buffer_pct)
