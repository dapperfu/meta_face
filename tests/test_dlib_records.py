"""Tests for dlib detection record conversion (no dlib import required)."""

from __future__ import annotations

from meta_face.tools.dlib_detect import (
    DlibFace,
    flatten_landmarks,
    location_to_bbox,
    locations_to_dlib_faces,
    faces_to_records,
    named_landmarks_from_parts,
)


def test_location_to_bbox() -> None:
    # face_recognition: (top, right, bottom, left)
    assert location_to_bbox((10, 100, 90, 20)) == [20.0, 10.0, 100.0, 90.0]


def test_flatten_landmarks_order() -> None:
    landmarks = {
        "chin": [(1, 2)],
        "left_eyebrow": [(3, 4)],
        "right_eyebrow": [],
        "nose_bridge": [],
        "nose_tip": [],
        "left_eye": [],
        "right_eye": [],
        "top_lip": [],
        "bottom_lip": [],
    }
    assert flatten_landmarks(landmarks) == [[1.0, 2.0], [3.0, 4.0]]


def test_faces_to_records_shape() -> None:
    faces = locations_to_dlib_faces([(10, 100, 90, 20)])
    records = faces_to_records(faces)
    assert len(records) == 1
    assert records[0]["bbox"] == [20.0, 10.0, 100.0, 90.0]
    assert records[0]["det_score"] == 1.0


def test_dlib_face_dataclass() -> None:
    face = DlibFace(location=(0, 1, 2, 3), landmarks=None)
    assert face.location == (0, 1, 2, 3)


def test_named_landmarks_from_68_parts() -> None:
    parts = [(i, i) for i in range(68)]
    named = named_landmarks_from_parts(parts)
    assert len(named["chin"]) == 17
    assert named["chin"][0] == (0, 0)
    assert named["bottom_lip"][-1] == (67, 67)
