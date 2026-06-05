"""face-anti-spoofing liveness detection (optional dependency)."""

from __future__ import annotations

from typing import Any

import cv2

from meta_face.tools.analysis.base import FaceContext, face_results_payload

TOOL_NAME = "face_anti_spoofing"
TOOL_VERSION = "1.0.0"
MODEL_NAME = "face_anti_spoofing"


def availability() -> str | None:
    try:
        from src.anti_spoof_predict import AntiSpoofPredict  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return (
            "face-anti-spoofing is not installed. Clone "
            "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing and install, "
            "or use pip install -e '.[liveness]' when packaged."
        )
    return None


def analyze_faces(
    image_bgr: Any,
    faces: list[FaceContext],
) -> dict[str, Any]:
    from src.anti_spoof_predict import AntiSpoofPredict  # type: ignore[import-not-found]

    predictor = AntiSpoofPredict(device_id=0)
    per_face: list[dict[str, Any]] = []
    for ctx in faces:
        rgb = cv2.cvtColor(ctx.crop_bgr, cv2.COLOR_BGR2RGB)
        label, score = predictor.predict(rgb)
        per_face.append(
            {
                "face_index": ctx.face_index,
                "liveness_label": str(label),
                "liveness_score": float(score),
                "is_live": str(label).lower() in {"real", "live", "0"},
            }
        )
    return face_results_payload(per_face, model=MODEL_NAME)
