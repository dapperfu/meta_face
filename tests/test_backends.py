"""Tests for face detection backends and detectron2 tool wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meta_face.backends.registry import get_detection_backend
from meta_face.jobs import process_image
from meta_face.scanner import resolve_per_image_tools
from meta_face.tools.registry import expand_dependencies, validate_tools


def test_registry_resolves_scrfd_and_detectron2() -> None:
    assert get_detection_backend("scrfd").name == "scrfd"
    assert get_detection_backend("detectron2").name == "detectron2"


def test_validate_tools_accepts_detectron2() -> None:
    assert validate_tools(["detectron2"]) == ["detectron2"]


def test_resolve_per_image_tools_detectron2_without_scrfd() -> None:
    assert resolve_per_image_tools(["detectron2"]) == ["detectron2"]


def test_expand_dependencies_arcface_still_includes_scrfd() -> None:
    assert expand_dependencies(["arcface"]) == ["scrfd", "arcface"]


def test_expand_dependencies_detectron2_alone() -> None:
    assert expand_dependencies(["detectron2"]) == ["detectron2"]


def test_process_image_writes_detectron2_faces(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xd9"
    )

    mock_detections = [
        {
            "bbox": [10.0, 20.0, 110.0, 120.0],
            "det_score": 0.95,
        }
    ]
    mock_backend = MagicMock()
    mock_backend.name = "detectron2"
    mock_backend.detect.return_value = mock_detections
    mock_backend.to_records.return_value = [
        {"bbox": [10.0, 20.0, 110.0, 120.0], "det_score": 0.95}
    ]

    fake_image = np.zeros((64, 64, 3), dtype=np.uint8)

    with (
        patch("meta_face.jobs.load_image", return_value=fake_image),
        patch("meta_face.jobs.require_inference_runtime"),
        patch("meta_face.jobs.require_detectron2_runtime"),
        patch("meta_face.jobs.get_detection_backend", return_value=mock_backend),
    ):
        result = process_image(str(image_path), ["detectron2"], force=True)

    assert result["status"] == "ok"
    assert "detectron2" in result["tools"]

    from meta_face.sidecar import get_face_section, load_or_create

    doc, _ = load_or_create(image_path)
    section = get_face_section(doc, "detectron2")
    assert "faces" in section
    assert len(section["faces"]) == 1
    assert section["faces"][0]["bbox"] == [10.0, 20.0, 110.0, 120.0]


def test_detectron2_backend_available_without_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = get_detection_backend("detectron2")
    monkeypatch.setattr(
        "meta_face.backends.detectron2_backend.is_detectron2_available",
        lambda: False,
    )
    with patch.dict("sys.modules", {"detectron2": MagicMock(), "torch": MagicMock()}):
        assert backend.available() is False
