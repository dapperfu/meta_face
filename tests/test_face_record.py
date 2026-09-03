"""Tests for face record serialization and sidecar resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meta_face.sidecar import update_sidecar
from meta_face.tools.face_record import (
    face_to_sidecar_record,
    records_from_sidecar,
    resolve_face_records,
    sidecar_face_to_annotation_record,
)


def _write_scrfd_sidecar(image: Path, faces: list[dict] | None = None) -> None:
    if faces is None:
        faces = [
            {"bbox": [0.0, 0.0, 10.0, 10.0], "landmarks": [[1.0, 2.0]], "det_score": 0.9},
            {"bbox": [20.0, 20.0, 30.0, 30.0], "landmarks": [], "det_score": 0.7},
        ]

    def apply(doc: object) -> None:
        # These fixtures deliberately exercise legacy pixel sidecars without dimensions.
        doc.set("face.scrfd.faces", faces)
        doc.set("face.scrfd.version", "1.1.0")

    update_sidecar(image, apply)


def test_sidecar_face_to_annotation_record_maps_landmarks_to_kps() -> None:
    record = sidecar_face_to_annotation_record(
        {
            "bbox": [1, 2, 3, 4],
            "landmarks": [[5.0, 6.0], [7.0, 8.0]],
            "det_score": 0.95,
            "age": 34,
            "gender": 1,
            "pose": [1.0, -2.0, 0.5],
            "landmark_2d_106": [[0.0, 0.0], [1.0, 1.0]],
        }
    )
    assert record["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert record["kps"] == [[5.0, 6.0], [7.0, 8.0]]
    assert record["det_score"] == 0.95
    assert record["age"] == 34
    assert record["gender"] == 1
    assert record["pose"] == [1.0, -2.0, 0.5]
    assert record["landmark_2d_106"] == [[0.0, 0.0], [1.0, 1.0]]


def test_face_to_sidecar_record_full_fields() -> None:
    fake_face = MagicMock()
    fake_face.bbox.tolist.return_value = [10.0, 20.0, 110.0, 120.0]
    fake_face.kps.tolist.return_value = [[30.0, 40.0], [50.0, 60.0]]
    fake_face.det_score = 0.88
    fake_face.pose = None
    fake_face.gender = 0
    fake_face.age = 29
    fake_face.sex = None
    fake_face.items.return_value = [
        ("landmark_2d_106", MagicMock(tolist=lambda: [[1.0, 2.0]])),
    ]

    record = face_to_sidecar_record(fake_face)
    assert record["bbox"] == [10.0, 20.0, 110.0, 120.0]
    assert record["landmarks"] == [[30.0, 40.0], [50.0, 60.0]]
    assert record["det_score"] == 0.88
    assert record["gender"] == 0
    assert record["age"] == 29
    assert record["landmark_2d_106"] == [[1.0, 2.0]]
    assert "kps" in record
    assert "landmarks" in record


def test_records_from_sidecar_roundtrip_full_attributes(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"fake")
    faces = [
        {
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "landmarks": [[1.0, 2.0]],
            "det_score": 0.9,
            "age": 22,
            "gender": 1,
            "pose": [0.1, 0.2, 0.3],
            "landmark_3d_68": [[1.0, 2.0, 3.0]],
        }
    ]
    _write_scrfd_sidecar(image, faces)

    records = records_from_sidecar(image)
    assert records is not None
    assert records[0]["age"] == 22
    assert records[0]["gender"] == 1
    assert records[0]["pose"] == [0.1, 0.2, 0.3]
    assert records[0]["landmark_3d_68"] == [[1.0, 2.0, 3.0]]


def test_records_from_sidecar_returns_normalized_records(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"fake")
    _write_scrfd_sidecar(image)

    records = records_from_sidecar(image)
    assert records is not None
    assert len(records) == 2
    assert records[0]["kps"] == [[1.0, 2.0]]


def test_records_from_sidecar_missing_sidecar(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"fake")
    assert records_from_sidecar(image) is None


def test_resolve_face_records_loads_from_sidecar_without_detect(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"fake")
    _write_scrfd_sidecar(image)

    with patch("meta_face.tools.scrfd.detect_faces") as mock_detect:
        records, source = resolve_face_records(image, force=False)
        mock_detect.assert_not_called()

    assert source == "sidecar"
    assert len(records) == 2


def test_resolve_face_records_raises_when_sidecar_missing(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"fake")

    with pytest.raises(FileNotFoundError, match="mf scan"):
        resolve_face_records(image, force=False)


def test_resolve_face_records_force_runs_detect(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"fake")
    _write_scrfd_sidecar(image)

    fake_face = MagicMock()
    fake_face.bbox.tolist.return_value = [0.0, 0.0, 10.0, 10.0]
    fake_face.kps.tolist.return_value = [[1.0, 2.0]]
    fake_face.det_score = 0.99
    fake_face.pose = None
    fake_face.gender = None
    fake_face.age = None
    fake_face.sex = None
    fake_face.items.return_value = []

    fake_image = object()
    with (
        patch("meta_face.deps.require_insightface_runtime"),
        patch("meta_face.tools.scrfd.detect_faces", return_value=[fake_face]) as mock_detect,
    ):
        records, source = resolve_face_records(image, force=True, image=fake_image)

    mock_detect.assert_called_once_with(fake_image)
    assert source == "detect"
    assert len(records) == 1
    assert records[0]["bbox"] == [0.0, 0.0, 10.0, 10.0]
