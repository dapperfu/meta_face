"""py-feat action units and emotion (optional dependency)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2

from meta_face.tools.analysis.base import FaceContext, face_results_payload

TOOL_NAME = "py_feat"
TOOL_VERSION = "1.0.0"
MODEL_NAME = "py-feat"


@lru_cache(maxsize=1)
def _get_detector():
    from feat import Detector

    return Detector()


def availability() -> str | None:
    try:
        import feat  # noqa: F401
    except ImportError:
        return (
            "py-feat is not installed. Install optional extras: "
            "pip install -e '.[expression]'"
        )
    try:
        _get_detector()
    except Exception as exc:
        return f"py-feat failed to initialize: {exc}"
    return None


def analyze_faces(
    image_bgr: Any,
    faces: list[FaceContext],
) -> dict[str, Any]:
    detector = _get_detector()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fex = detector.detect_image(rgb)
    per_face: list[dict[str, Any]] = []
    if fex is None or len(fex) == 0:
        return face_results_payload(per_face, model=MODEL_NAME)

    for idx, row in fex.iterrows():
        face_index = faces[idx].face_index if idx < len(faces) else int(idx)
        record: dict[str, Any] = {"face_index": face_index}
        au_cols = [c for c in fex.columns if str(c).startswith("AU")]
        if au_cols:
            record["action_units"] = {col: float(row[col]) for col in au_cols}
        for col in ("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"):
            if col in fex.columns:
                record.setdefault("emotion_scores", {})[col] = float(row[col])
        if "gaze_0_x" in fex.columns:
            record["gaze"] = {
                "x": float(row.get("gaze_0_x", 0.0)),
                "y": float(row.get("gaze_0_y", 0.0)),
            }
        per_face.append(record)
    return face_results_payload(per_face, model=MODEL_NAME)
