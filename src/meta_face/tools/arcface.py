"""ArcFace embeddings via insightface."""

from __future__ import annotations

from typing import Any

import numpy as np

from meta_face.config import INSIGHTFACE_MODEL
from meta_face.tools.sidecar_encode import json_safe


def embeddings_from_faces(faces: list[Any]) -> list[list[float]]:
    """Extract normalized ArcFace embeddings from insightface Face objects."""
    vectors: list[list[float]] = []
    for face in faces:
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = face.embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        vectors.append([float(x) for x in embedding.tolist()])
    return vectors


def arcface_to_sidecar_payload(faces: list[Any]) -> dict[str, Any]:
    """All ArcFace tool outputs for face.arcface.* sidecar keys."""
    embeddings = embeddings_from_faces(faces)
    payload: dict[str, Any] = {
        "embeddings": embeddings,
        "face_count": len(embeddings),
        "model": INSIGHTFACE_MODEL,
    }
    if embeddings:
        payload["embedding_dim"] = len(embeddings[0])
    else:
        payload["embedding_dim"] = 0
    return json_safe(payload)
