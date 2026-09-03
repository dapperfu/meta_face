"""Tests for sidecar JSON encoding and full backend payloads."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from meta_face.tools.arcface import arcface_to_sidecar_payload
from meta_face.tools.dlib_detect import dlib_detect_to_sidecar_payload, locations_to_dlib_faces
from meta_face.tools.dlib_embed import dlib_embed_to_sidecar_payload
from meta_face.tools.face_record import scrfd_to_sidecar_payload
from meta_face.tools.sidecar_encode import json_safe


def test_json_safe_numpy() -> None:
    assert json_safe(np.array([1.0, 2.0])) == [1.0, 2.0]


def test_scrfd_sidecar_payload_keys() -> None:
    fake_face = MagicMock()
    fake_face.bbox.tolist.return_value = [0.0, 0.0, 10.0, 10.0]
    fake_face.kps.tolist.return_value = [[1.0, 2.0]]
    fake_face.det_score = 0.9
    fake_face.pose = None
    fake_face.gender = 1
    fake_face.age = 30
    fake_face.sex = None
    fake_face.items.return_value = []
    fake_face.keys.return_value = ["bbox", "kps", "det_score", "gender", "age"]

    payload = scrfd_to_sidecar_payload([fake_face], image_size=(100, 200))
    assert payload["face_count"] == 1
    assert payload["image_size"] == [100, 200]
    assert "model" in payload
    assert payload["faces"][0]["age"] == 30
    assert payload["faces"][0]["kps"] == [[1.0, 2.0]]


def test_arcface_sidecar_payload_metadata() -> None:
    fake_face = MagicMock()
    fake_face.normed_embedding = np.array([1.0, 0.0], dtype=np.float32)
    payload = arcface_to_sidecar_payload([fake_face])
    assert payload["embedding_dim"] == 2
    assert payload["face_count"] == 1
    assert "model" in payload


def test_dlib_detect_sidecar_payload_full() -> None:
    faces = locations_to_dlib_faces([(10, 100, 90, 20)])
    payload = dlib_detect_to_sidecar_payload(faces, image_size=(640, 480))
    assert payload["face_count"] == 1
    assert payload["image_size"] == [640, 480]
    face = payload["faces"][0]
    assert face["location"] == [10, 100, 90, 20]
    assert "det_model" in face


def test_dlib_embed_sidecar_payload_metadata() -> None:
    faces = locations_to_dlib_faces([(10, 100, 90, 20)])
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch(
        "meta_face.tools.dlib_embed.embeddings_from_faces",
        return_value=[[0.1, 0.2]],
    ):
        payload = dlib_embed_to_sidecar_payload(rgb, faces)

    assert payload["embedding_dim"] == 2
    assert payload["num_jitters"] == 1
    assert payload["model"] == "dlib_resnet_face_recognition"
