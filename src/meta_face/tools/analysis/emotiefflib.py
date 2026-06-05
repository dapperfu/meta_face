"""EmotiEffLib ONNX emotion recognition on face crops."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from meta_face.tools.analysis.base import FaceContext, face_results_payload, scores_dict

TOOL_NAME = "emotiefflib"
TOOL_VERSION = "1.1.0"
MODEL_NAME = "mobilenet_7.h5_onnx"


@lru_cache(maxsize=1)
def _get_recognizer():
    from emotiefflib.facial_emotion_recognition import EmotiEffLibRecognizer

    return EmotiEffLibRecognizer(engine="onnx")


def availability() -> str | None:
    try:
        import emotiefflib  # noqa: F401
    except ImportError:
        return (
            "emotiefflib is not installed. Install optional extras: "
            "pip install -e '.[emotion]'"
        )
    try:
        _get_recognizer()
    except Exception as exc:
        return f"emotiefflib failed to initialize: {exc}"
    return None


def analyze_faces(
    image_bgr: np.ndarray,
    faces: list[FaceContext],
) -> dict[str, Any]:
    del image_bgr
    recognizer = _get_recognizer()
    per_face: list[dict[str, Any]] = []
    for ctx in faces:
        rgb = cv2.cvtColor(ctx.crop_bgr, cv2.COLOR_BGR2RGB)
        emotion, scores = recognizer.predict_emotions(rgb, logits=False)
        label_scores = scores_dict(recognizer.emotions, np.asarray(scores))
        per_face.append(
            {
                "face_index": ctx.face_index,
                "emotion_label": str(emotion),
                "emotion_scores": label_scores,
            }
        )
    return face_results_payload(per_face, model=MODEL_NAME)
