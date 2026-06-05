"""L2CS-Net gaze estimation (optional dependency)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from meta_face.tools.analysis.base import FaceContext, face_results_payload

TOOL_NAME = "l2cs_net"
TOOL_VERSION = "1.0.0"
MODEL_NAME = "L2CS-Net"


@lru_cache(maxsize=1)
def _get_gaze():
    try:
        from l2cs import Pipeline, select_device

        device = select_device("cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
        return Pipeline(weights=None, arch="ResNet50", device=device)
    except ImportError:
        raise ImportError(
            "l2cs is not installed. Install optional extras: pip install -e '.[gaze]'"
        ) from None


def availability() -> str | None:
    try:
        _get_gaze()
    except ImportError as exc:
        return str(exc)
    except Exception as exc:
        return f"L2CS-Net failed to initialize: {exc}"
    return None


def analyze_faces(
    image_bgr: Any,
    faces: list[FaceContext],
) -> dict[str, Any]:
    gaze = _get_gaze()
    per_face: list[dict[str, Any]] = []
    for ctx in faces:
        results = gaze.predict(ctx.crop_bgr)
        record: dict[str, Any] = {"face_index": ctx.face_index, "gaze": {}}
        if results is not None:
            pitch_yaw = np.asarray(results).reshape(-1)
            if pitch_yaw.size >= 2:
                record["gaze"] = {"pitch": float(pitch_yaw[0]), "yaw": float(pitch_yaw[1])}
        per_face.append(record)
    return face_results_payload(per_face, model=MODEL_NAME)
