"""FairFace race/ethnicity classification (optional dependency)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from meta_face.config import fairface_model_path
from meta_face.tools.analysis.base import FaceContext, face_results_payload, scores_dict

TOOL_NAME = "fairface"
TOOL_VERSION = "1.0.0"
MODEL_NAME = "fairface"

RACE_LABELS = (
    "White",
    "Black",
    "Latino_Hispanic",
    "East_Asian",
    "Southeast_Asian",
    "Indian",
    "Middle_Eastern",
)


@lru_cache(maxsize=1)
def _get_session():
    import onnxruntime as ort

    model_path = fairface_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"FairFace ONNX model missing at {model_path}. "
            "Run: mf download --backend fairface"
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
        return f"fairface failed to load: {exc}"
    return None


def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
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
        logits = session.run(None, {input_name: _preprocess(ctx.crop_bgr)})[0].reshape(-1)
        best_idx = int(np.argmax(logits))
        labels = list(RACE_LABELS[: logits.size])
        per_face.append(
            {
                "face_index": ctx.face_index,
                "race_label": labels[best_idx] if best_idx < len(labels) else str(best_idx),
                "race_scores": scores_dict(labels, logits),
            }
        )
    return face_results_payload(per_face, model=MODEL_NAME)
