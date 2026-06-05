"""SCRFD face detection via insightface."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from meta_face.config import INSIGHTFACE_CTX_ID, INSIGHTFACE_MODEL, INSIGHTFACE_ROOT


@lru_cache(maxsize=1)
def get_face_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name=INSIGHTFACE_MODEL,
        root=INSIGHTFACE_ROOT,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=INSIGHTFACE_CTX_ID, det_size=(640, 640))
    return app


def detect_faces(image: np.ndarray) -> list[Any]:
    """Run SCRFD detection and return insightface Face objects."""
    app = get_face_app()
    return app.get(image)


def faces_to_records(faces: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for face in faces:
        bbox = [float(x) for x in face.bbox.tolist()]
        landmarks = [[float(x), float(y)] for x, y in face.kps.tolist()]
        det_score = float(face.det_score)
        records.append(
            {
                "bbox": bbox,
                "landmarks": landmarks,
                "det_score": det_score,
            }
        )
    return records
