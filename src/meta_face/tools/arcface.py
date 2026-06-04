"""ArcFace embeddings via insightface."""

from __future__ import annotations

from typing import Any

import numpy as np


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
