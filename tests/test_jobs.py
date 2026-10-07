"""Tests for per-tool sidecar writes from process_image."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rq import Retry
from sidecar_rs import LockTimeout

from meta_face.config import SIDECAR_LOCK_MAX_REQUEUES
from meta_face.jobs import process_image
from meta_face.sidecar import get_face_section, load_or_create, update_sidecar, write_tool_result


def test_process_image_writes_each_analysis_tool_separately(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"x")
    lock_tools: list[str] = []

    def tracking_write(media_path, tool, payload, image_size):
        lock_tools.append(tool)

        def _patch(doc):
            write_tool_result(doc, tool, payload, image_size=image_size)

        return update_sidecar(media_path, _patch)

    monkeypatch.setattr("meta_face.jobs._write_tool_payload", tracking_write)
    monkeypatch.setattr("meta_face.jobs.require_inference_runtime", lambda tools: None)
    monkeypatch.setattr("meta_face.jobs.load_image", lambda path: np.zeros((8, 8, 3), dtype=np.uint8))

    def fake_run(media_path, image_bgr, tool_names, **kwargs):
        tool = tool_names[0]
        return {tool: {"faces": [{"face_index": 0}], "face_count": 1}}

    monkeypatch.setattr(
        "meta_face.tools.analysis.runner.run_pending_analysis_tools",
        fake_run,
    )

    def seed_scrfd(doc):
        write_tool_result(doc, "scrfd", {"faces": [{"bbox": [0, 0, 4, 4]}]}, image_size=(8, 8))

    update_sidecar(image, seed_scrfd)
    result = process_image(str(image), ["opencv_fer", "fer_plus"], force=False)
    assert result["status"] == "ok"
    assert lock_tools == ["opencv_fer", "fer_plus"]
    doc, _ = load_or_create(image)
    assert get_face_section(doc, "opencv_fer")["face_count"] == 1
    assert get_face_section(doc, "fer_plus")["face_count"] == 1
    assert get_face_section(doc, "scrfd")["faces"]


def test_process_image_requeues_on_lockfile_timeout(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"x")
    job = SimpleNamespace(id="image-opencv_fer-1", number_of_retries=None)

    def locked_write(media_path, tool, payload, image_size):
        raise LockTimeout("lockfile timeout after 10.000s waiting for exclusive lock")

    monkeypatch.setattr("meta_face.jobs.get_current_job", lambda: job)
    monkeypatch.setattr("meta_face.jobs._write_tool_payload", locked_write)
    monkeypatch.setattr("meta_face.jobs.require_inference_runtime", lambda tools: None)
    monkeypatch.setattr("meta_face.jobs.load_image", lambda path: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(
        "meta_face.tools.analysis.runner.run_pending_analysis_tools",
        lambda media_path, image_bgr, tool_names, **kwargs: {
            tool_names[0]: {"faces": [], "face_count": 0}
        },
    )
    update_sidecar(
        image,
        lambda doc: write_tool_result(doc, "scrfd", {"faces": []}, image_size=(8, 8)),
    )

    result = process_image(str(image), ["opencv_fer"], force=False)
    assert isinstance(result, Retry)
    assert result.max == SIDECAR_LOCK_MAX_REQUEUES
    assert result.intervals == [0]

    job.number_of_retries = SIDECAR_LOCK_MAX_REQUEUES
    with pytest.raises(LockTimeout, match="lockfile timeout"):
        process_image(str(image), ["opencv_fer"], force=False)

    monkeypatch.setattr("meta_face.jobs.get_current_job", lambda: None)
    with pytest.raises(LockTimeout, match="lockfile timeout"):
        process_image(str(image), ["opencv_fer"], force=False)
