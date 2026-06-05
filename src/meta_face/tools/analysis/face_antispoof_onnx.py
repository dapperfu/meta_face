"""face-antispoof-onnx liveness detection (optional dependency)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from meta_face.config import face_antispoof_onnx_model_path
from meta_face.tools.analysis.base import FaceContext, face_results_payload

TOOL_NAME = "face_antispoof_onnx"
TOOL_VERSION = "1.0.0"
MODEL_NAME = "face_antispoof_onnx"


@lru_cache(maxsize=1)
def _get_session():
    import onnxruntime as ort

    model_path = face_antispoof_onnx_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"face-antispoof ONNX model missing at {model_path}. "
            "Run: mf download --backend face_antispoof_onnx"
        )
    return ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def availability() -> str | None:
    try:
        _get_session()
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        return f"face_antispoof_onnx failed to load: {exc}"
    return None


def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (80, 80), interpolation=cv2.INTER_LINEAR)
    blob = resized.astype(np.float32) / 255.0
    return blob.transpose(2, 0, 1)[np.newaxis, ...]


def analyze_faces(
    image_bgr: Any,
    faces: list[FaceContext],
) -> dict[str, Any]:
    del image_bgr
    session = _get_session()
    input_name = session.get_inputs()[0].name
    per_face: list[dict[str, Any]] = []
    for ctx in faces:
        logits = session.run(None, {input_name: _preprocess(ctx.crop_bgr)})[0].reshape(-1)
        score = float(logits[1]) if logits.size >= 2 else float(logits[0])
        per_face.append(
            {
                "face_index": ctx.face_index,
                "liveness_score": score,
                "is_live": score >= 0.5,
            }
        )
    return face_results_payload(per_face, model=MODEL_NAME)
