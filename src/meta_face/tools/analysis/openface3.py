"""OpenFace 3.0 AU, emotion, and gaze analysis (optional dependency)."""

from __future__ import annotations

from typing import Any

from meta_face.tools.analysis.base import FaceContext, face_results_payload

TOOL_NAME = "openface3"
TOOL_VERSION = "1.0.0"
MODEL_NAME = "openface3"


def availability() -> str | None:
    try:
        import openface  # noqa: F401
    except ImportError:
        return (
            "OpenFace 3.0 Python bindings are not installed. Install optional extras: "
            "pip install -e '.[expression]'  # requires openface package from OpenFace 3.0"
        )
    return None


def analyze_faces(
    image_bgr: Any,
    faces: list[FaceContext],
) -> dict[str, Any]:
    import cv2

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    per_face: list[dict[str, Any]] = []
    try:
        from openface import OpenFace  # type: ignore[import-not-found]

        model = OpenFace()
        output = model.predict(rgb)
    except Exception as exc:
        raise RuntimeError(f"openface3 inference failed: {exc}") from exc

    if isinstance(output, dict):
        per_face.append({"face_index": 0, **output})
    elif isinstance(output, list):
        for idx, item in enumerate(output):
            face_index = faces[idx].face_index if idx < len(faces) else idx
            if isinstance(item, dict):
                per_face.append({"face_index": face_index, **item})
    return face_results_payload(per_face, model=MODEL_NAME)
