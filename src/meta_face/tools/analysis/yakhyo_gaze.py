"""yakhyo/gaze-estimation gaze direction (optional dependency)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from meta_face.config import yakhyo_gaze_model_path
from meta_face.tools.analysis.base import FaceContext, face_results_payload

TOOL_NAME = "yakhyo_gaze"
TOOL_VERSION = "1.0.0"
MODEL_NAME = "yakhyo_gaze"


@lru_cache(maxsize=1)
def _get_session():
    import onnxruntime as ort

    model_path = yakhyo_gaze_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"yakhyo gaze ONNX model missing at {model_path}. "
            "Run: mf download --backend yakhyo_gaze"
        )
    return ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def availability() -> str | None:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return "onnxruntime is required for yakhyo_gaze."
    try:
        _get_session()
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        return f"yakhyo_gaze failed to load: {exc}"
    return None


def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (448, 448), interpolation=cv2.INTER_LINEAR)
    blob = resized.astype(np.float32) / 255.0
    blob = (blob - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
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
        output = session.run(None, {input_name: _preprocess(ctx.crop_bgr)})
        yaw_pitch = output[0].reshape(-1)
        record: dict[str, Any] = {"face_index": ctx.face_index, "gaze": {}}
        if yaw_pitch.size >= 2:
            record["gaze"] = {"yaw": float(yaw_pitch[0]), "pitch": float(yaw_pitch[1])}
        per_face.append(record)
    return face_results_payload(per_face, model=MODEL_NAME)
