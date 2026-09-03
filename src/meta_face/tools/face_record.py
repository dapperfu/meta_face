"""Serialize insightface Face objects for sidecar storage and annotation rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from meta_face.config import INSIGHTFACE_MODEL
from meta_face.coordinates import section_records_in_pixels
from meta_face.sidecar import get_face_section, load_or_create, sidecar_path_for_media
from meta_face.tools.sidecar_encode import json_safe

# ArcFace owns embedding vectors; scrfd stores every other insightface Face field.
_ARCFACE_OWNED_KEYS = frozenset({"embedding", "normed_embedding"})


def _to_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return [float(x) for x in value.tolist()]
    return [float(x) for x in value]


def _extract_landmarks(face: Any) -> dict[str, list[list[float]]]:
    """Collect landmark_* arrays present on the face object."""
    out: dict[str, list[list[float]]] = {}
    for key, value in face.items():
        if not isinstance(key, str) or not key.startswith("landmark_"):
            continue
        if value is None or not hasattr(value, "tolist"):
            continue
        rows = value.tolist()
        if not rows or isinstance(rows[0], (int, float)):
            continue
        out[key] = [[float(c) for c in row] for row in rows]
    return out


def _extract_remaining_face_fields(face: Any, record: dict[str, Any]) -> None:
    """Copy any insightface Face keys not already stored (except embeddings)."""
    if not hasattr(face, "keys"):
        return
    for key in face.keys():
        if key in _ARCFACE_OWNED_KEYS or key in record:
            continue
        try:
            record[key] = json_safe(face[key])
        except (TypeError, ValueError):
            continue


def face_to_sidecar_record(face: Any, *, face_index: int | None = None) -> dict[str, Any]:
    """
    Build a JSON-serializable dict of all insightface Face attributes for scrfd.

    Stores every field on the Face object except embedding vectors (face.arcface).
    """
    bbox = _to_float_list(face.bbox)
    if bbox is None or len(bbox) < 4:
        raise ValueError("face bbox is missing or invalid")

    kps_raw = getattr(face, "kps", None)
    landmarks: list[list[float]] | None = None
    if kps_raw is not None:
        landmarks = [[float(x), float(y)] for x, y in kps_raw.tolist()]

    record: dict[str, Any] = {
        "bbox": bbox[:4],
        "det_score": float(face.det_score),
    }
    if face_index is not None:
        record["face_index"] = face_index
    if landmarks is not None:
        record["landmarks"] = landmarks
        record["kps"] = landmarks

    pose = getattr(face, "pose", None)
    if pose is not None:
        pose_list = _to_float_list(pose)
        if pose_list is not None and len(pose_list) >= 3:
            record["pose"] = pose_list[:3]

    gender = getattr(face, "gender", None)
    if gender is not None:
        record["gender"] = int(gender)

    age = getattr(face, "age", None)
    if age is not None:
        record["age"] = int(age)

    sex = getattr(face, "sex", None)
    if sex is not None:
        record["sex"] = str(sex)

    record.update(_extract_landmarks(face))
    _extract_remaining_face_fields(face, record)
    return record


def faces_to_sidecar_records(faces: list[Any]) -> list[dict[str, Any]]:
    return [face_to_sidecar_record(face, face_index=idx) for idx, face in enumerate(faces)]


def scrfd_to_sidecar_payload(
    faces: list[Any],
    *,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """All scrfd tool outputs for face.scrfd.* sidecar keys."""
    payload: dict[str, Any] = {
        "faces": faces_to_sidecar_records(faces),
        "face_count": len(faces),
        "model": INSIGHTFACE_MODEL,
        "det_size": [640, 640],
    }
    if image_size is not None:
        w, h = image_size
        payload["image_size"] = [int(w), int(h)]
    return json_safe(payload)


def face_to_annotation_record(face: Any) -> dict[str, Any]:
    """Build a dict for draw_annotations from a live insightface Face."""
    return sidecar_face_to_annotation_record(face_to_sidecar_record(face))


def faces_to_annotation_records(faces: list[Any]) -> list[dict[str, Any]]:
    return [face_to_annotation_record(face) for face in faces]


def sidecar_face_to_annotation_record(face: dict[str, Any]) -> dict[str, Any]:
    """Normalize a sidecar face dict to annotation record shape."""
    bbox = face.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4:
        raise ValueError("sidecar face bbox is missing or invalid")

    record: dict[str, Any] = {
        "bbox": [float(v) for v in bbox[:4]],
        "det_score": float(face["det_score"]) if face.get("det_score") is not None else 0.0,
    }

    landmarks = face.get("landmarks")
    if isinstance(landmarks, list) and landmarks:
        record["kps"] = [[float(x), float(y)] for x, y in landmarks]
    elif isinstance(face.get("kps"), list):
        record["kps"] = [[float(x), float(y)] for x, y in face["kps"]]

    for key in ("pose", "gender", "age", "sex"):
        if key in face:
            record[key] = face[key]

    for key in ("coordinates", "source_image_size"):
        if key in face:
            record[key] = face[key]

    for key, value in face.items():
        if isinstance(key, str) and key.startswith("landmark_"):
            record[key] = value

    return record


def records_from_sidecar(
    media_path: Path,
    *,
    tool: str = "scrfd",
    image: Any | None = None,
) -> list[dict[str, Any]] | None:
    """Load face records from a .scar sidecar, or None when absent."""
    media_path = Path(media_path).resolve()
    scar_path = sidecar_path_for_media(media_path)
    if not scar_path.exists():
        return None

    doc, _ = load_or_create(media_path)
    section = get_face_section(doc, tool)
    faces = section.get("faces")
    if not isinstance(faces, list):
        return None

    if faces and ("image_size" in section or "coordinates" in section
                  or any("coordinates" in face for face in faces)):
        if image is None:
            from meta_face.imaging import load_image

            image = load_image(media_path)
        h_img, w_img = image.shape[:2]
        faces = section_records_in_pixels(section, (w_img, h_img))

    return [sidecar_face_to_annotation_record(face) for face in faces]


def resolve_face_records(
    media_path: Path,
    *,
    tool: str = "scrfd",
    force: bool = False,
    image: Any | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Load face records from sidecar by default; run SCRFD only when force=True.

    Returns (records, source) where source is "sidecar" or "detect".
    Pass a pre-loaded image when force=True to avoid a second load_image call.
    """
    media_path = Path(media_path).resolve()

    if not force:
        records = records_from_sidecar(media_path, tool=tool, image=image)
        if records is not None:
            return records, "sidecar"
        scar_path = sidecar_path_for_media(media_path)
        raise FileNotFoundError(
            f"No {tool} face data in {scar_path}. "
            f"Run: mf scan {media_path} --tools {tool} "
            f"or set FORCE_DETECT=True."
        )

    from meta_face.deps import require_insightface_runtime
    from meta_face.imaging import load_image
    from meta_face.tools.scrfd import detect_faces

    require_insightface_runtime()
    if image is None:
        image = load_image(media_path)
    faces = detect_faces(image)
    return faces_to_annotation_records(faces), "detect"
