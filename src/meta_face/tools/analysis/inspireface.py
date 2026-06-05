"""InspireFace SDK wrapper (optional Python bindings)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2

from meta_face.tools.analysis.base import FaceContext, face_results_payload

TOOL_NAME = "inspireface"
TOOL_VERSION = "1.0.0"
MODEL_NAME = "inspireface"


@lru_cache(maxsize=1)
def _get_session():
    try:
        import inspireface as isf  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "InspireFace Python bindings are not installed. "
            "Install optional extras: pip install -e '.[all-tools]'"
        ) from exc
    isf.launch(use_cuda=True)
    return isf


def availability() -> str | None:
    try:
        _get_session()
    except ImportError as exc:
        return str(exc)
    except Exception as exc:
        return f"InspireFace failed to initialize: {exc}"
    return None


def analyze_faces(
    image_bgr: Any,
    faces: list[FaceContext],
) -> dict[str, Any]:
    isf = _get_session()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    per_face: list[dict[str, Any]] = []
    try:
        session = isf.InspireFaceSession(detect_mode=isf.HF_DETECT_MODE_LIGHT_TRACK)
        results = session.face_detection(rgb)
    except Exception as exc:
        raise RuntimeError(f"InspireFace inference failed: {exc}") from exc

    if results:
        for idx, face in enumerate(results):
            face_index = faces[idx].face_index if idx < len(faces) else idx
            record: dict[str, Any] = {"face_index": face_index}
            if hasattr(face, "emotion"):
                record["emotion_label"] = str(face.emotion)
            if hasattr(face, "liveness"):
                record["liveness_score"] = float(face.liveness)
            per_face.append(record)
    return face_results_payload(per_face, model=MODEL_NAME)
