"""Face detection via face_recognition (dlib HOG/CNN)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from meta_face.config import DLIB_MODEL
from meta_face.tools.sidecar_encode import json_safe

# Stable order for flattening 68-point landmark dicts from face_recognition.
LANDMARK_KEY_ORDER: tuple[str, ...] = (
    "chin",
    "left_eyebrow",
    "right_eyebrow",
    "nose_bridge",
    "nose_tip",
    "left_eye",
    "right_eye",
    "top_lip",
    "bottom_lip",
)


@dataclass
class DlibFace:
    """Detected face with optional 68-point landmarks."""

    location: tuple[int, int, int, int]  # (top, right, bottom, left)
    landmarks: dict[str, list[tuple[int, int]]] | None = None


def location_to_bbox(location: tuple[int, int, int, int]) -> list[float]:
    """Convert face_recognition (top, right, bottom, left) to [x1, y1, x2, y2]."""
    top, right, bottom, left = location
    return [float(left), float(top), float(right), float(bottom)]


def flatten_landmarks(landmarks: dict[str, list[tuple[int, int]]]) -> list[list[float]]:
    """Flatten 68-point landmark dict to a list of [x, y] pairs."""
    points: list[list[float]] = []
    for key in LANDMARK_KEY_ORDER:
        for x, y in landmarks.get(key, []):
            points.append([float(x), float(y)])
    return points


def landmarks_named_to_json(
    landmarks: dict[str, list[tuple[int, int]]],
) -> dict[str, list[list[float]]]:
    """Structured 68-point landmarks keyed by facial region."""
    return {
        region: [[float(x), float(y)] for x, y in points]
        for region, points in landmarks.items()
    }


def locations_to_dlib_faces(
    locations: list[tuple[int, int, int, int]],
    landmarks_list: list[dict[str, list[tuple[int, int]]]] | None = None,
) -> list[DlibFace]:
    faces: list[DlibFace] = []
    for idx, location in enumerate(locations):
        lm = landmarks_list[idx] if landmarks_list and idx < len(landmarks_list) else None
        faces.append(DlibFace(location=location, landmarks=lm))
    return faces


# 68-point predictor regions used by face_recognition / dlib.
_LANDMARK_SLICES: tuple[tuple[str, int, int], ...] = (
    ("chin", 0, 17),
    ("left_eyebrow", 17, 22),
    ("right_eyebrow", 22, 27),
    ("nose_bridge", 27, 31),
    ("nose_tip", 31, 36),
    ("left_eye", 36, 42),
    ("right_eye", 42, 48),
    ("top_lip", 48, 60),
    ("bottom_lip", 60, 68),
)


def named_landmarks_from_parts(parts: list[tuple[int, int]]) -> dict[str, list[tuple[int, int]]]:
    """Group a 68-point shape into the face_recognition region names."""
    named: dict[str, list[tuple[int, int]]] = {}
    for key, start, end in _LANDMARK_SLICES:
        named[key] = [(int(x), int(y)) for x, y in parts[start:end]]
    return named


def _hog_detect(image_rgb: np.ndarray, *, upsample: int = 1) -> list[DlibFace]:
    """CPU HOG + 68-point predictor without importing face_recognition's CUDA CNN."""
    import dlib
    import face_recognition_models

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_recognition_models.pose_predictor_model_location())
    rectangles = detector(image_rgb, upsample)
    faces: list[DlibFace] = []
    for rect in sorted(rectangles, key=lambda box: (box.top(), box.left())):
        shape = predictor(image_rgb, rect)
        parts = [(int(p.x), int(p.y)) for p in shape.parts()]
        location = (int(rect.top()), int(rect.right()), int(rect.bottom()), int(rect.left()))
        faces.append(DlibFace(location=location, landmarks=named_landmarks_from_parts(parts)))
    return faces


def detect_faces(image_rgb: np.ndarray, *, model: str | None = None) -> list[DlibFace]:
    """Detect faces. Default HOG uses dlib directly; CNN still uses face_recognition."""
    det_model = model or DLIB_MODEL
    if det_model == "hog":
        return _hog_detect(image_rgb)
    import face_recognition

    locations = face_recognition.face_locations(image_rgb, model=det_model)
    if not locations:
        return []
    landmarks_list = face_recognition.face_landmarks(image_rgb, face_locations=locations)
    return locations_to_dlib_faces(locations, landmarks_list)


def dlib_face_to_sidecar_record(
    face: DlibFace,
    *,
    det_model: str | None = None,
    face_index: int | None = None,
) -> dict[str, Any]:
    """All dlib_detect fields for one face in face.dlib_detect.faces."""
    model = det_model or DLIB_MODEL
    top, right, bottom, left = face.location
    record: dict[str, Any] = {
        "bbox": location_to_bbox(face.location),
        "location": [int(top), int(right), int(bottom), int(left)],
        "det_score": 1.0,
        "det_model": model,
    }
    if face_index is not None:
        record["face_index"] = face_index
    if face.landmarks:
        record["landmarks"] = flatten_landmarks(face.landmarks)
        record["landmarks_named"] = landmarks_named_to_json(face.landmarks)
        record["landmark_count"] = len(record["landmarks"])
    return json_safe(record)


def faces_to_records(
    faces: list[DlibFace],
    *,
    det_model: str | None = None,
) -> list[dict[str, Any]]:
    return [
        dlib_face_to_sidecar_record(face, det_model=det_model, face_index=idx)
        for idx, face in enumerate(faces)
    ]


def dlib_detect_to_sidecar_payload(
    faces: list[DlibFace],
    *,
    det_model: str | None = None,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """All dlib_detect tool outputs for face.dlib_detect.* sidecar keys."""
    model = det_model or DLIB_MODEL
    payload: dict[str, Any] = {
        "faces": faces_to_records(faces, det_model=model),
        "face_count": len(faces),
        "det_model": model,
    }
    if image_size is not None:
        w, h = image_size
        payload["image_size"] = [int(w), int(h)]
    return json_safe(payload)
