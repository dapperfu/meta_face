"""SCRFD face detection via insightface."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from meta_face.config import INSIGHTFACE_CTX_ID, INSIGHTFACE_MODEL, INSIGHTFACE_ROOT, ONNX_PROVIDERS


@lru_cache(maxsize=1)
def get_face_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name=INSIGHTFACE_MODEL,
        root=INSIGHTFACE_ROOT,
        providers=list(ONNX_PROVIDERS),
    )
    app.prepare(ctx_id=INSIGHTFACE_CTX_ID, det_size=(640, 640))
    return app


def detect_faces(image: np.ndarray) -> list[Any]:
    """Run SCRFD detection and return insightface Face objects."""
    app = get_face_app()
    return app.get(image)


def faces_to_records(faces: list[Any]) -> list[dict[str, Any]]:
    """Serialize insightface faces for face.scrfd.faces (see faces_to_sidecar_records)."""
    from meta_face.tools.face_record import faces_to_sidecar_records

    return faces_to_sidecar_records(faces)


def scrfd_tool_payload(
    faces: list[Any],
    *,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    from meta_face.tools.face_record import scrfd_to_sidecar_payload

    return scrfd_to_sidecar_payload(faces, image_size=image_size)
