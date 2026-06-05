"""Face detection via face_recognition (dlib HOG/CNN)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from meta_face.config import DLIB_MODEL

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


def locations_to_dlib_faces(
    locations: list[tuple[int, int, int, int]],
    landmarks_list: list[dict[str, list[tuple[int, int]]]] | None = None,
) -> list[DlibFace]:
    faces: list[DlibFace] = []
    for idx, location in enumerate(locations):
        lm = landmarks_list[idx] if landmarks_list and idx < len(landmarks_list) else None
        faces.append(DlibFace(location=location, landmarks=lm))
    return faces


def detect_faces(image_rgb: np.ndarray, *, model: str | None = None) -> list[DlibFace]:
    """Run face_recognition detection on an RGB image."""
    import face_recognition

    det_model = model or DLIB_MODEL
    locations = face_recognition.face_locations(image_rgb, model=det_model)
    if not locations:
        return []

    landmarks_list = face_recognition.face_landmarks(image_rgb, face_locations=locations)
    return locations_to_dlib_faces(locations, landmarks_list)


def faces_to_records(faces: list[DlibFace]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for face in faces:
        record: dict[str, Any] = {
            "bbox": location_to_bbox(face.location),
            "det_score": 1.0,
        }
        if face.landmarks:
            record["landmarks"] = flatten_landmarks(face.landmarks)
        records.append(record)
    return records
