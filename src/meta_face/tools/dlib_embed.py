"""Face embeddings via face_recognition (dlib ResNet, 128-d)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from meta_face.tools.dlib_detect import DlibFace


def embeddings_from_faces(image_rgb: np.ndarray, faces: list[DlibFace]) -> list[list[float]]:
    """Extract 128-d face encodings for detected faces."""
    if not faces:
        return []

    import face_recognition

    locations = [face.location for face in faces]
    encodings = face_recognition.face_encodings(
        image_rgb,
        known_face_locations=locations,
        num_jitters=1,
    )
    vectors: list[list[float]] = []
    for encoding in encodings:
        vec = np.asarray(encoding, dtype=np.float64)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append([float(x) for x in vec.tolist()])
    return vectors
