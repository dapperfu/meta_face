"""UniFace SDK wrapper (emotion, gaze, FairFace, parsing, liveness)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2

from meta_face.tools.analysis.base import FaceContext, face_results_payload

TOOL_NAME = "uniface"
TOOL_VERSION = "1.0.0"
MODEL_NAME = "uniface"


@lru_cache(maxsize=1)
def _get_analyzer():
    try:
        from uniface import UniFace  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "UniFace is not installed. Install optional extras: pip install -e '.[all-tools]'"
        ) from exc
    return UniFace()


def availability() -> str | None:
    try:
        _get_analyzer()
    except ImportError as exc:
        return str(exc)
    except Exception as exc:
        return f"UniFace failed to initialize: {exc}"
    return None


def analyze_faces(
    image_bgr: Any,
    faces: list[FaceContext],
) -> dict[str, Any]:
    analyzer = _get_analyzer()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    per_face: list[dict[str, Any]] = []
    try:
        output = analyzer.analyze(rgb)
    except Exception as exc:
        raise RuntimeError(f"UniFace inference failed: {exc}") from exc

    if isinstance(output, dict):
        per_face.append({"face_index": 0, **output})
    elif isinstance(output, list):
        for idx, item in enumerate(output):
            face_index = faces[idx].face_index if idx < len(faces) else idx
            if isinstance(item, dict):
                per_face.append({"face_index": face_index, **item})
    return face_results_payload(per_face, model=MODEL_NAME)
