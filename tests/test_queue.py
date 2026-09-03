"""Tests for RQ enqueue helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from meta_face.config import (
    CROP_ANALYSIS_TOOLS,
    DEFAULT_TOOLS,
    PER_IMAGE_TOOL_ORDER,
    RQ_ANALYSIS_JOB_TIMEOUT,
    RQ_DETECT_JOB_TIMEOUT,
    RQ_MEDIAPIPE_JOB_TIMEOUT,
)
from meta_face.queue import enqueue_annotate, enqueue_process_image, enqueue_sdk_run
from meta_face.scanner import resolve_per_image_tools
from meta_face.tools.registry import validate_tools


def test_enqueue_process_image_one_job_per_tool(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")

    mock_queue = MagicMock()
    mock_jobs = [MagicMock(id=f"job-{i}") for i in range(4)]
    mock_queue.enqueue.side_effect = mock_jobs

    with (
        patch("meta_face.queue.get_queue", return_value=mock_queue),
        patch("meta_face.scanner.needs_processing", return_value=True),
    ):
        job_ids = enqueue_process_image(image_path, list(DEFAULT_TOOLS), force=False)

    assert job_ids == ["job-0", "job-1", "job-2", "job-3"]
    assert mock_queue.enqueue.call_count == 4
    prefixes = [call.kwargs["job_id"] for call in mock_queue.enqueue.call_args_list]
    assert prefixes[0].startswith("image-scrfd-")
    assert prefixes[1].startswith("image-arcface-")
    assert prefixes[2].startswith("image-dlib_detect-")
    assert prefixes[3].startswith("image-dlib_embed-")
    timeouts = [call.kwargs["job_timeout"] for call in mock_queue.enqueue.call_args_list]
    assert timeouts == [RQ_DETECT_JOB_TIMEOUT] * 4


def test_enqueue_process_image_skips_satisfied_tools(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")

    mock_queue = MagicMock()
    mock_queue.enqueue.return_value = MagicMock(id="job-dlib")

    def needs_processing(_path: Path, tools: list[str], force: bool) -> bool:
        return "dlib_detect" in tools

    with (
        patch("meta_face.queue.get_queue", return_value=mock_queue),
        patch("meta_face.scanner.needs_processing", side_effect=needs_processing),
    ):
        job_ids = enqueue_process_image(image_path, list(DEFAULT_TOOLS), force=False)

    assert job_ids == ["job-dlib"]
    assert mock_queue.enqueue.call_count == 1
    assert mock_queue.enqueue.call_args.args[2] == ["dlib_detect"]


def test_enqueue_crop_analysis_waits_for_scrfd(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")

    tools = resolve_per_image_tools(
        validate_tools(
            [
                "detect",
                "opencv_fer",
                "fer_plus",
                "yakhyo_gaze",
                "bisenet",
                "face_antispoof_onnx",
                "mediapipe",
            ]
        )
    )
    mock_queue = MagicMock()
    mock_jobs = [MagicMock(id=f"job-{i}") for i in range(7)]
    mock_queue.enqueue.side_effect = mock_jobs

    with (
        patch("meta_face.queue.get_queue", return_value=mock_queue),
        patch("meta_face.scanner.needs_processing", return_value=True),
    ):
        job_ids = enqueue_process_image(image_path, tools, force=False)

    assert job_ids == [f"job-{i}" for i in range(7)]
    prefixes = [call.kwargs["job_id"].split("-")[1] for call in mock_queue.enqueue.call_args_list]
    assert prefixes == [
        "scrfd",
        "opencv_fer",
        "fer_plus",
        "yakhyo_gaze",
        "bisenet",
        "face_antispoof_onnx",
        "mediapipe_blendshapes",
    ]
    detect_job = mock_jobs[0]
    for call in mock_queue.enqueue.call_args_list[1:]:
        assert call.kwargs["depends_on"] is detect_job
    timeouts = [call.kwargs["job_timeout"] for call in mock_queue.enqueue.call_args_list]
    assert timeouts[0] == RQ_DETECT_JOB_TIMEOUT
    assert timeouts[1] == RQ_ANALYSIS_JOB_TIMEOUT
    assert timeouts[-1] == RQ_MEDIAPIPE_JOB_TIMEOUT


def test_enqueue_all_is_one_job_per_per_image_tool(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")

    tools = resolve_per_image_tools(validate_tools(["all"]))
    mock_queue = MagicMock()
    mock_jobs = [MagicMock(id=f"job-{i}") for i in range(len(PER_IMAGE_TOOL_ORDER))]
    mock_queue.enqueue.side_effect = mock_jobs

    with (
        patch("meta_face.queue.get_queue", return_value=mock_queue),
        patch("meta_face.scanner.needs_processing", return_value=True),
    ):
        job_ids = enqueue_process_image(image_path, tools, force=False)

    assert len(job_ids) == len(PER_IMAGE_TOOL_ORDER)
    prefixes = [call.kwargs["job_id"].split("-")[1] for call in mock_queue.enqueue.call_args_list]
    assert prefixes == list(PER_IMAGE_TOOL_ORDER)
    detect_job = mock_jobs[0]
    for call, tool in zip(
        mock_queue.enqueue.call_args_list,
        PER_IMAGE_TOOL_ORDER,
        strict=True,
    ):
        if tool in CROP_ANALYSIS_TOOLS:
            assert call.kwargs["depends_on"] is detect_job
        else:
            assert "depends_on" not in call.kwargs


def test_enqueue_annotate_uses_image_queue(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")
    mock_queue = MagicMock()
    mock_queue.enqueue.return_value = MagicMock(id="annotate-job")

    with patch("meta_face.queue.get_queue", return_value=mock_queue):
        job_id = enqueue_annotate(image_path, force=True, dense_landmarks=False)

    assert job_id == "annotate-job"
    assert mock_queue.enqueue.call_args.kwargs["job_id"].startswith("annotate-")
    assert mock_queue.enqueue.call_args.kwargs["job_timeout"] == RQ_DETECT_JOB_TIMEOUT
    assert mock_queue.enqueue.call_args.args[1] == str(image_path.resolve())
    assert mock_queue.enqueue.call_args.args[2] is True
    assert mock_queue.enqueue.call_args.args[3] is False


def test_enqueue_sdk_run_uses_image_queue() -> None:
    mock_queue = MagicMock()
    mock_queue.enqueue.return_value = MagicMock(id="sdk-job")
    recipe = {"steps": [{"id": "pair", "call": "verify", "kwargs": {"img1_path": "a.jpg"}}]}

    with patch("meta_face.queue.get_queue", return_value=mock_queue):
        job_id = enqueue_sdk_run("deepface", recipe, output=None, output_format="json")

    assert job_id == "sdk-job"
    assert mock_queue.enqueue.call_args.kwargs["job_id"].startswith("sdk-deepface-")
    assert mock_queue.enqueue.call_args.kwargs["job_timeout"] == RQ_ANALYSIS_JOB_TIMEOUT
    assert mock_queue.enqueue.call_args.args[1] == "deepface"
    assert mock_queue.enqueue.call_args.args[2] == recipe
