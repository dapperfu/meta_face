"""MediaPipe Face Landmarker blendshape (52 ARKit coefficients)."""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from meta_face.config import mediapipe_model_path
from meta_face.tools.analysis.base import FaceContext, face_results_payload
from meta_face.tools.analysis.decode import (
    MEDIAPIPE_CROP_SCALE,
    MEDIAPIPE_CROP_SIZE,
    expand_crop,
)

TOOL_NAME = "mediapipe_blendshapes"
TOOL_VERSION = "1.1.0"
MODEL_NAME = "face_landmarker_blendshapes"


def _disable_optional_audio() -> None:
    """MediaPipe imports sounddevice; PortAudio init can hang on photo hosts."""
    sys.modules.setdefault("sounddevice", None)


@lru_cache(maxsize=1)
def _get_landmarker():
    _disable_optional_audio()
    from mediapipe.tasks.python.core import base_options as base_options_module
    from mediapipe.tasks.python.vision import face_landmarker

    model_path = mediapipe_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"MediaPipe face landmarker model missing at {model_path}. "
            "Run: mf download --backend mediapipe"
        )
    options = face_landmarker.FaceLandmarkerOptions(
        base_options=base_options_module.BaseOptions(model_asset_path=str(model_path)),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    return face_landmarker.FaceLandmarker.create_from_options(options)


def availability() -> str | None:
    try:
        _disable_optional_audio()
        import mediapipe  # noqa: F401
    except ImportError:
        return (
            "mediapipe is not installed. Install optional extras: "
            "pip install -e '.[dev]'"
        )
    try:
        _get_landmarker()
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        return f"mediapipe_blendshapes failed to initialize: {exc}"
    return None


def _center_in_bbox(points: np.ndarray, bbox: list[float]) -> bool:
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    lo = points[:, :2].min(axis=0)
    hi = points[:, :2].max(axis=0)
    center = (lo + hi) / 2.0
    return bool(x1 <= center[0] <= x2 and y1 <= center[1] <= y2)


def analyze_faces(
    image_bgr: np.ndarray,
    faces: list[FaceContext],
) -> dict[str, Any]:
    _disable_optional_audio()
    import mediapipe as mp

    landmarker = _get_landmarker()
    per_face: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for ctx in faces:
        crop, bounds = expand_crop(image_bgr, ctx.bbox, MEDIAPIPE_CROP_SCALE)
        if crop.size == 0:
            missing.append({"face_index": ctx.face_index, "reason": "empty crop"})
            continue
        rgb = cv2.cvtColor(cv2.resize(crop, (MEDIAPIPE_CROP_SIZE, MEDIAPIPE_CROP_SIZE)), cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            missing.append({"face_index": ctx.face_index, "reason": "no landmarks detected in source crop"})
            continue
        left, top, right, bottom = bounds
        width, height = max(right - left, 1), max(bottom - top, 1)
        landmarks = [
            [left + p.x * width, top + p.y * height, p.z * width]
            for p in result.face_landmarks[0]
        ]
        points = np.asarray(landmarks, dtype=np.float64)
        if not _center_in_bbox(points, ctx.bbox):
            missing.append({"face_index": ctx.face_index, "reason": "mesh center outside expected face box"})
            continue
        coefficients = {bs.category_name: float(bs.score) for bs in result.face_blendshapes[0]}
        matrix = None
        if result.facial_transformation_matrixes:
            matrix = np.asarray(result.facial_transformation_matrixes[0]).tolist()
        record: dict[str, Any] = {
            "face_index": ctx.face_index,
            "blendshape_coefficients": coefficients,
            "landmark_3d_478": landmarks,
        }
        if matrix is not None:
            record["facial_transformation_matrix"] = matrix
        per_face.append(record)
    return face_results_payload(
        per_face,
        model=MODEL_NAME,
        extra={
            "blendshape_count": 52,
            "attempted_face_count": len(faces),
            "missing_faces": missing,
            "face_index_source": "scrfd",
        },
    )
