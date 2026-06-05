"""Serialize insightface Face objects for annotation rendering."""

from __future__ import annotations

from typing import Any


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


def face_to_annotation_record(face: Any) -> dict[str, Any]:
    """Build a JSON-serializable dict of drawable face attributes."""
    bbox = _to_float_list(face.bbox)
    if bbox is None or len(bbox) < 4:
        raise ValueError("face bbox is missing or invalid")

    kps_raw = getattr(face, "kps", None)
    kps: list[list[float]] | None = None
    if kps_raw is not None:
        kps = [[float(x), float(y)] for x, y in kps_raw.tolist()]

    record: dict[str, Any] = {
        "bbox": bbox[:4],
        "det_score": float(face.det_score),
        "kps": kps,
    }

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
    return record


def faces_to_annotation_records(faces: list[Any]) -> list[dict[str, Any]]:
    return [face_to_annotation_record(face) for face in faces]
