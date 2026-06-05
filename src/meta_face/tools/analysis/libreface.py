"""LibreFace AU, emotion, and gaze analysis (optional dependency)."""

from __future__ import annotations

from typing import Any

from meta_face.tools.analysis.base import FaceContext, face_results_payload

TOOL_NAME = "libreface"
TOOL_VERSION = "1.0.0"
MODEL_NAME = "libreface"


def availability() -> str | None:
    try:
        import libreface  # noqa: F401
    except ImportError:
        return (
            "libreface is not installed. Install optional extras: "
            "pip install -e '.[expression]'  # requires LibreFace package"
        )
    return None


def analyze_faces(
    image_bgr: Any,
    faces: list[FaceContext],
) -> dict[str, Any]:
    import cv2
    import libreface

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    per_face: list[dict[str, Any]] = []
    for ctx in faces:
        result = libreface.get_facial_attributes(rgb, crop_face=False)
        record: dict[str, Any] = {"face_index": ctx.face_index}
        if isinstance(result, dict):
            au = {k: float(v) for k, v in result.items() if str(k).startswith("AU")}
            if au:
                record["action_units"] = au
            for key in ("emotion", "valence", "arousal"):
                if key in result:
                    record[key] = result[key]
            gaze = result.get("gaze") or result.get("gaze_direction")
            if gaze is not None:
                record["gaze"] = gaze
        per_face.append(record)
    return face_results_payload(per_face, model=MODEL_NAME)
