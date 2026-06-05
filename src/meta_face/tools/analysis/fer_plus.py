"""Legacy FER+ ONNX emotion recognition on face crops."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from meta_face.config import fer_plus_model_path
from meta_face.tools.analysis.base import FaceContext, face_results_payload, scores_dict

TOOL_NAME = "fer_plus"
TOOL_VERSION = "1.0.0"
MODEL_NAME = "emotion-ferplus-8"

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
def _get_session():
    import onnxruntime as ort

    model_path = fer_plus_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"FER+ ONNX model missing at {model_path}. Run: mf download --backend fer_plus"
        )
    return ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def availability() -> str | None:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return "onnxruntime is required for fer_plus (included in base install)."
    try:
        _get_session()
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        return f"fer_plus failed to load ONNX model: {exc}"
    return None


def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR)
    blob = resized.astype(np.float32).reshape(1, 1, 64, 64)
    return blob


def analyze_faces(
    image_bgr: np.ndarray,
    faces: list[FaceContext],
) -> dict[str, Any]:
    del image_bgr
    session = _get_session()
    input_name = session.get_inputs()[0].name
    per_face: list[dict[str, Any]] = []
    for ctx in faces:
        blob = _preprocess(ctx.crop_bgr)
        output = session.run(None, {input_name: blob})[0].reshape(-1)
        if output.size == 0:
            continue
        best_idx = int(np.argmax(output))
        labels = list(EMOTION_LABELS[: output.size])
        per_face.append(
            {
                "face_index": ctx.face_index,
                "emotion_label": labels[best_idx] if best_idx < len(labels) else str(best_idx),
                "emotion_scores": scores_dict(labels, output),
            }
        )
    return face_results_payload(per_face, model=MODEL_NAME)
