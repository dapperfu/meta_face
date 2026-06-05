"""DeepFace emotion, age, gender, and race wrapper (optional dependency)."""

from __future__ import annotations

from typing import Any

import cv2

from meta_face.tools.analysis.base import FaceContext, face_results_payload

TOOL_NAME = "deepface"
TOOL_VERSION = "1.0.0"
MODEL_NAME = "deepface"


def availability() -> str | None:
    try:
        import deepface  # noqa: F401
    except ImportError:
        return (
            "DeepFace is not installed. Install optional extras: "
            "pip install -e '.[attributes]'"
        )
    return None


def analyze_faces(
    image_bgr: Any,
    faces: list[FaceContext],
) -> dict[str, Any]:
    from deepface import DeepFace

    per_face: list[dict[str, Any]] = []
    for ctx in faces:
        rgb = cv2.cvtColor(ctx.crop_bgr, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(
            rgb,
            actions=("emotion", "age", "gender", "race"),
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0] if result else {}
        record: dict[str, Any] = {"face_index": ctx.face_index}
        if isinstance(result, dict):
            if "dominant_emotion" in result:
                record["emotion_label"] = result["dominant_emotion"]
            if "emotion" in result and isinstance(result["emotion"], dict):
                record["emotion_scores"] = {
                    k: float(v) for k, v in result["emotion"].items()
                }
            if "age" in result:
                record["age"] = int(result["age"])
            if "dominant_gender" in result:
                record["gender"] = result["dominant_gender"]
            if "dominant_race" in result:
                record["race"] = result["dominant_race"]
            if "race" in result and isinstance(result["race"], dict):
                record["race_scores"] = {k: float(v) for k, v in result["race"].items()}
        per_face.append(record)
    return face_results_payload(per_face, model=MODEL_NAME)
