"""OpenCV Progressive Teacher ONNX facial expression recognition."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from meta_face.config import opencv_fer_model_path
from meta_face.tools.analysis.base import FaceContext, face_results_payload, scores_dict
from meta_face.tools.analysis.decode import (
    OPENCV_FER_INPUT,
    OPENCV_FER_LABELS,
    rgb_normalized_tensor,
    softmax,
)

TOOL_NAME = "opencv_fer"
TOOL_VERSION = "1.1.0"
MODEL_NAME = "facial_expression_recognition"
EMOTION_LABELS = OPENCV_FER_LABELS


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


def _aligned_crop(image_bgr: np.ndarray, ctx: FaceContext) -> np.ndarray:
    kps = ctx.metadata.get("kps") or ctx.metadata.get("landmarks")
    if isinstance(kps, list) and len(kps) >= 5:
        try:
            from insightface.utils.face_align import norm_crop

            points = np.asarray(kps[:5], dtype=np.float32)
            aligned = norm_crop(image_bgr, points, image_size=OPENCV_FER_INPUT)
            if aligned is not None and aligned.size:
                return aligned
        except Exception:
            pass
    if ctx.crop_bgr.size:
        return ctx.crop_bgr
    return image_bgr


def analyze_faces(
    image_bgr: np.ndarray,
    faces: list[FaceContext],
) -> dict[str, Any]:
    net = _get_net()
    labels = list(EMOTION_LABELS)
    per_face: list[dict[str, Any]] = []
    for ctx in faces:
        blob = rgb_normalized_tensor(_aligned_crop(image_bgr, ctx), OPENCV_FER_INPUT)
        net.setInput(blob)
        logits = np.asarray(net.forward()).reshape(-1)
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
