"""Face embeddings via face_recognition (dlib ResNet, 128-d)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from meta_face.config import DLIB_MODEL
from meta_face.tools.sidecar_encode import json_safe

if TYPE_CHECKING:
    from meta_face.tools.dlib_detect import DlibFace

DEFAULT_NUM_JITTERS = 1


def embeddings_from_faces(
    image_rgb: np.ndarray,
    faces: list[DlibFace],
    *,
    num_jitters: int = DEFAULT_NUM_JITTERS,
) -> list[list[float]]:
    """Extract 128-d descriptors via dlib without importing face_recognition."""
    if not faces:
        return []

    import dlib
    import face_recognition_models

    predictor = dlib.shape_predictor(face_recognition_models.pose_predictor_model_location())
    recognizer = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )
    vectors: list[list[float]] = []
    for face in faces:
        top, right, bottom, left = face.location
        rect = dlib.rectangle(int(left), int(top), int(right), int(bottom))
        shape = predictor(image_rgb, rect)
        encoding = recognizer.compute_face_descriptor(image_rgb, shape, num_jitters)
        vec = np.asarray(encoding, dtype=np.float64)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append([float(x) for x in vec.tolist()])
    return vectors


def dlib_embed_to_sidecar_payload(
    image_rgb: np.ndarray,
    faces: list[DlibFace],
    *,
    num_jitters: int = DEFAULT_NUM_JITTERS,
) -> dict[str, Any]:
    """All dlib_embed tool outputs for face.dlib_embed.* sidecar keys."""
    embeddings = embeddings_from_faces(image_rgb, faces, num_jitters=num_jitters)
    payload: dict[str, Any] = {
        "embeddings": embeddings,
        "face_count": len(embeddings),
        "det_model": DLIB_MODEL,
        "num_jitters": num_jitters,
        "model": "dlib_resnet_face_recognition",
    }
    if embeddings:
        payload["embedding_dim"] = len(embeddings[0])
    else:
        payload["embedding_dim"] = 0
    return json_safe(payload)
