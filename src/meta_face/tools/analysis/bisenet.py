"""BiSeNet face parsing (optional dependency)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from meta_face.config import bisenet_model_path
from meta_face.tools.analysis.base import FaceContext, face_results_payload
from meta_face.tools.analysis.decode import (
    BISENET_INPUT,
    class_pixel_percent,
    parsing_argmax,
    rgb_normalized_tensor,
)

TOOL_NAME = "bisenet"
TOOL_VERSION = "1.1.0"
MODEL_NAME = "bisenet_face_parsing"


@lru_cache(maxsize=1)
def _get_session():
    import onnxruntime as ort

    model_path = bisenet_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"BiSeNet ONNX model missing at {model_path}. "
            "Run: mf download --backend bisenet"
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
        return f"bisenet failed to load: {exc}"
    return None


def analyze_faces(
    image_bgr: Any,
    faces: list[FaceContext],
) -> dict[str, Any]:
    del image_bgr
    session = _get_session()
    input_name = session.get_inputs()[0].name
    per_face: list[dict[str, Any]] = []
    for ctx in faces:
        logits = session.run(None, {input_name: rgb_normalized_tensor(ctx.crop_bgr, BISENET_INPUT, imagenet=True)})[0]
        mask = parsing_argmax(logits)
        labels = sorted({int(v) for v in np.unique(mask).tolist()})
        per_face.append(
            {
                "face_index": ctx.face_index,
                "parsing_labels_present": labels,
                "parsing_shape": list(mask.shape),
                "class_pixel_percent": class_pixel_percent(mask),
                "parsing_mask": mask,
                "mask_coordinate_space": "model crop; pixel values are class IDs 0-18",
            }
        )
    return face_results_payload(per_face, model=MODEL_NAME)
