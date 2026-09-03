"""Legacy FER+ ONNX emotion recognition on face crops."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from meta_face.config import fer_plus_model_path
from meta_face.tools.analysis.base import FaceContext, face_results_payload, scores_dict
from meta_face.tools.analysis.decode import FER_PLUS_LABELS, softmax

TOOL_NAME = "fer_plus"
TOOL_VERSION = "1.1.0"
MODEL_NAME = "emotion-ferplus-8"
EMOTION_LABELS = FER_PLUS_LABELS


@lru_cache(maxsize=1)
def _get_session():
    from meta_face.onnx_runtime import inference_session

    model_path = fer_plus_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"FER+ ONNX model missing at {model_path}. Run: mf download --backend fer_plus"
        )
    return inference_session(model_path)


def availability() -> str | None:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return "onnxruntime-gpu is required for fer_plus (included in the base install)."
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
    return resized.astype(np.float32).reshape(1, 1, 64, 64)


def analyze_faces(
    image_bgr: np.ndarray,
    faces: list[FaceContext],
) -> dict[str, Any]:
    del image_bgr
    session = _get_session()
    input_name = session.get_inputs()[0].name
    labels = list(EMOTION_LABELS)
    per_face: list[dict[str, Any]] = []
    for ctx in faces:
        logits = session.run(None, {input_name: _preprocess(ctx.crop_bgr)})[0].reshape(-1)
        if logits.size == 0:
            continue
        used = labels[: logits.size]
        probs = softmax(logits)
        best_idx = int(np.argmax(probs))
        per_face.append(
            {
                "face_index": ctx.face_index,
                "emotion_label": used[best_idx] if best_idx < len(used) else str(best_idx),
                "emotion_logits": scores_dict(used, logits),
                "emotion_scores": scores_dict(used, probs),
            }
        )
    return face_results_payload(per_face, model=MODEL_NAME)
