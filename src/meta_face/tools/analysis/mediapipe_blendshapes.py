"""MediaPipe Face Landmarker blendshape (52 ARKit coefficients)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from meta_face.config import mediapipe_model_path
from meta_face.tools.analysis.base import FaceContext, face_results_payload

TOOL_NAME = "mediapipe_blendshapes"
TOOL_VERSION = "1.0.0"
MODEL_NAME = "face_landmarker_blendshapes"


@lru_cache(maxsize=1)
def _get_landmarker():
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
        output_facial_transformation_matrixes=False,
        num_faces=10,
    )
    return face_landmarker.FaceLandmarker.create_from_options(options)


def availability() -> str | None:
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        return (
            "mediapipe is not installed. Install optional extras: "
            "pip install -e '.[expression]'"
        )
    try:
        _get_landmarker()
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        return f"mediapipe_blendshapes failed to initialize: {exc}"
    return None


def analyze_faces(
    image_bgr: np.ndarray,
    faces: list[FaceContext],
) -> dict[str, Any]:
    import mediapipe as mp

    landmarker = _get_landmarker()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    per_face: list[dict[str, Any]] = []
    if not result.face_blendshapes:
        return face_results_payload(per_face, model=MODEL_NAME)

    for idx, blendshapes in enumerate(result.face_blendshapes):
        coefficients = {bs.category_name: float(bs.score) for bs in blendshapes}
        face_index = faces[idx].face_index if idx < len(faces) else idx
        per_face.append(
            {
                "face_index": face_index,
                "blendshape_coefficients": coefficients,
            }
        )
    return face_results_payload(per_face, model=MODEL_NAME, extra={"blendshape_count": 52})
