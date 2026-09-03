"""face-antispoof-onnx liveness detection (optional dependency)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from meta_face.config import face_antispoof_onnx_model_path
from meta_face.tools.analysis.base import FaceContext, face_results_payload
from meta_face.tools.analysis.decode import (
    ANTISPOOF_CROP_SCALE,
    ANTISPOOF_INPUT,
    bgr_float_tensor,
    expand_crop,
    softmax,
)

TOOL_NAME = "face_antispoof_onnx"
TOOL_VERSION = "1.1.0"
MODEL_NAME = "face_antispoof_onnx"


@lru_cache(maxsize=1)
def _get_session():
    from meta_face.onnx_runtime import inference_session

    model_path = face_antispoof_onnx_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"face-antispoof ONNX model missing at {model_path}. "
            "Run: mf download --backend face_antispoof_onnx"
        )
    return inference_session(model_path)


def availability() -> str | None:
    try:
        _get_session()
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        return f"face_antispoof_onnx failed to load: {exc}"
    return None


def analyze_faces(
    image_bgr: Any,
    faces: list[FaceContext],
) -> dict[str, Any]:
    session = _get_session()
    input_name = session.get_inputs()[0].name
    per_face: list[dict[str, Any]] = []
    for ctx in faces:
        context, _bounds = expand_crop(image_bgr, ctx.bbox, ANTISPOOF_CROP_SCALE)
        if context.size == 0:
            context = ctx.crop_bgr
        logits = session.run(None, {input_name: bgr_float_tensor(context, ANTISPOOF_INPUT)})[0]
        probs = softmax(logits)
        live = float(probs[1]) if probs.size > 1 else float(probs[0]) if probs.size else 0.0
        per_face.append(
            {
                "face_index": ctx.face_index,
                "class_probabilities": [float(v) for v in probs.tolist()],
                "liveness_score": live,
                "is_live": live >= 0.5,
            }
        )
    return face_results_payload(per_face, model=MODEL_NAME)
