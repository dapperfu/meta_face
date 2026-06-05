"""OpenCV Progressive Teacher ONNX facial expression recognition."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from meta_face.config import opencv_fer_model_path
from meta_face.tools.analysis.base import FaceContext, face_results_payload, scores_dict

TOOL_NAME = "opencv_fer"
TOOL_VERSION = "1.0.0"
MODEL_NAME = "facial_expression_recognition"

# OpenCV zoo FER+ style labels (7 classes).
EMOTION_LABELS = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
)


@lru_cache(maxsize=1)
def _get_net():
    model_path = opencv_fer_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"OpenCV FER ONNX model missing at {model_path}. "
            "Run: mf download --backend opencv_fer"
        )
    return cv2.dnn.readNetFromONNX(str(model_path))


def availability() -> str | None:
    try:
        _get_net()
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        return f"opencv_fer failed to load ONNX model: {exc}"
    return None


def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
    blob = resized.astype(np.float32) / 255.0
    blob = (blob - 0.5) / 0.5
    return blob.transpose(2, 0, 1)[np.newaxis, ...]


def analyze_faces(
    image_bgr: np.ndarray,
    faces: list[FaceContext],
) -> dict[str, Any]:
    del image_bgr
    net = _get_net()
    per_face: list[dict[str, Any]] = []
    for ctx in faces:
        blob = _preprocess(ctx.crop_bgr)
        net.setInput(blob)
        output = net.forward()
        scores = output.reshape(-1)
        if scores.size == 0:
            continue
        best_idx = int(np.argmax(scores))
        labels = list(EMOTION_LABELS[: scores.size])
        per_face.append(
            {
                "face_index": ctx.face_index,
                "emotion_label": labels[best_idx] if best_idx < len(labels) else str(best_idx),
                "emotion_scores": scores_dict(labels, scores),
            }
        )
    return face_results_payload(per_face, model=MODEL_NAME)
