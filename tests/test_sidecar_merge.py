"""Tests for locked sidecar merge across face and pose namespaces."""

from __future__ import annotations

import threading
from pathlib import Path

from sidecar_rs import SidecarDocument

from meta_face.sidecar import update_sidecar, write_tool_result


def _pose_result() -> dict:
    return {
        "native": {},
        "keypoints": [[{"name": "nose", "x": 1.0, "y": 2.0, "score": 0.5}]],
        "bboxes": [[0.0, 0.0, 100.0, 200.0]],
        "model": "yolo",
        "version": "0.1",
        "created": "2026-06-04T12:00:00+00:00",
    }


def _write_pose(image: Path, tool: str = "yolo") -> None:
    from meta_pose.sidecar_io import write_pose_result

    write_pose_result(image, tool, _pose_result())


def test_write_preserves_pose_namespace(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"x")
    _write_pose(image)

    def apply(doc: SidecarDocument) -> None:
        write_tool_result(doc, "scrfd", {"faces": []})

    update_sidecar(image, apply)

    doc = SidecarDocument.from_path(image.with_suffix(".scar"))
    assert doc["pose.yolo.model"] == "yolo"
    assert doc["face.scrfd.version"] == "1.0.0"


def test_sequential_face_then_pose(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"x")

    def apply_face(doc: SidecarDocument) -> None:
        write_tool_result(doc, "arcface", {"embeddings": [[0.1, 0.2]]})

    update_sidecar(image, apply_face)
    _write_pose(image)

    doc = SidecarDocument.from_path(image.with_suffix(".scar"))
    assert doc["face.arcface.version"] == "1.0.0"
    assert doc["pose.yolo.keypoints"]


def test_concurrent_face_and_pose_namespaces(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"x")
    scar = image.with_suffix(".scar")
    errors: list[BaseException] = []

    def write_face() -> None:
        try:
            for _ in range(30):

                def apply(doc: SidecarDocument) -> None:
                    write_tool_result(doc, "scrfd", {"faces": []})

                SidecarDocument.update_path(scar, apply)
        except BaseException as exc:
            errors.append(exc)

    def write_pose() -> None:
        try:
            for _ in range(30):
                _write_pose(image)
        except BaseException as exc:
            errors.append(exc)

    threads: list[threading.Thread] = []
    for _ in range(4):
        threads.append(threading.Thread(target=write_face))
        threads.append(threading.Thread(target=write_pose))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    doc = SidecarDocument.from_path(scar)
    assert doc["face.scrfd.version"] == "1.0.0"
    assert doc["pose.yolo.model"] == "yolo"
