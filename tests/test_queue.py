"""Tests for RQ enqueue helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from meta_face.config import DEFAULT_TOOLS
from meta_face.queue import enqueue_process_image


def test_enqueue_process_image_one_job_per_backend(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")

    mock_queue = MagicMock()
    mock_jobs = [MagicMock(id=f"job-{i}") for i in range(3)]
    mock_queue.enqueue.side_effect = mock_jobs

    with (
        patch("meta_face.queue.get_queue", return_value=mock_queue),
        patch("meta_face.scanner.needs_processing", return_value=True),
    ):
        job_ids = enqueue_process_image(image_path, list(DEFAULT_TOOLS), force=False)

    assert job_ids == ["job-0", "job-1", "job-2"]
    assert mock_queue.enqueue.call_count == 3
    job_id_prefixes = [call.kwargs["job_id"] for call in mock_queue.enqueue.call_args_list]
    assert job_id_prefixes[0].startswith("image-insightface-")
    assert job_id_prefixes[1].startswith("image-face_recognition-")
    assert job_id_prefixes[2].startswith("image-detectron2-")


def test_enqueue_process_image_skips_satisfied_backends(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")

    mock_queue = MagicMock()
    mock_queue.enqueue.return_value = MagicMock(id="job-d2")

    def needs_processing(_path: Path, tools: list[str], force: bool) -> bool:
        return "detectron2" in tools

    with (
        patch("meta_face.queue.get_queue", return_value=mock_queue),
        patch("meta_face.scanner.needs_processing", side_effect=needs_processing),
    ):
        job_ids = enqueue_process_image(image_path, list(DEFAULT_TOOLS), force=False)

    assert job_ids == ["job-d2"]
    assert mock_queue.enqueue.call_count == 1
    assert mock_queue.enqueue.call_args.args[2] == ["detectron2"]
