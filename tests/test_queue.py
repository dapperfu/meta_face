"""Tests for RQ stage enqueue helpers and the scan fan-out."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from meta_face.config import (
    CROP_ANALYSIS_TOOLS,
    DEFAULT_TOOLS,
    RQ_ANALYSIS_JOB_TIMEOUT,
    RQ_DETECT_JOB_TIMEOUT,
    RQ_IMAGE_ADD_QUEUE_NAME,
    RQ_IMAGE_QUEUE_NAME,
    RQ_MEDIAPIPE_JOB_TIMEOUT,
    RQ_QUEUE_NAME,
    RQ_SCAN_QUEUE_NAME,
)
from meta_face.queue import (
    enqueue_add_image,
    enqueue_face_tool,
    enqueue_query_images,
    enqueue_scan_path,
)
from meta_face.scan_jobs import add_image, query_images, scan_path
from meta_face.scanner import resolve_per_image_tools
from meta_face.tools.registry import validate_tools


def _face_job(path: Path, tool: str, force: bool = False, depends_on: str | None = None) -> str:
    del path, force, depends_on
    return f"job-{tool}"


def test_stage_enqueues_use_separate_queues(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")
    queues: dict[str, MagicMock] = {}

    def make_queue(name: str | None = None) -> MagicMock:
        key = str(name)
        if key not in queues:
            queue = MagicMock()
            queue.enqueue.return_value = MagicMock(id=f"id-{name}")
            queues[key] = queue
        return queues[key]

    with patch("meta_face.queue.get_queue", side_effect=make_queue):
        enqueue_scan_path(tmp_path, ["scrfd"])
        enqueue_query_images(tmp_path, ["scrfd"])
        enqueue_add_image(image_path, ["scrfd"])
        enqueue_face_tool(image_path, "scrfd", force=False)
        enqueue_face_tool(image_path, "opencv_fer", force=False, depends_on="id-scrfd")
        enqueue_face_tool(image_path, "mediapipe_blendshapes", force=False)

    assert queues[RQ_SCAN_QUEUE_NAME].enqueue.call_args.args[0] == "meta_face.scan_jobs.scan_path"
    image_query = queues[RQ_IMAGE_QUEUE_NAME].enqueue.call_args.args[0]
    assert image_query == "meta_face.scan_jobs.query_images"
    assert (
        queues[RQ_IMAGE_ADD_QUEUE_NAME].enqueue.call_args.args[0] == "meta_face.scan_jobs.add_image"
    )
    face_calls = queues[RQ_QUEUE_NAME].enqueue.call_args_list
    assert [call.args[0] for call in face_calls] == ["meta_face.jobs.process_image"] * 3
    assert [call.args[2] for call in face_calls] == [
        ["scrfd"],
        ["opencv_fer"],
        ["mediapipe_blendshapes"],
    ]
    assert [call.kwargs["job_id"].split("-")[1] for call in face_calls] == [
        "scrfd",
        "opencv_fer",
        "mediapipe_blendshapes",
    ]
    assert face_calls[0].kwargs["job_timeout"] == RQ_DETECT_JOB_TIMEOUT
    assert face_calls[1].kwargs["job_timeout"] == RQ_ANALYSIS_JOB_TIMEOUT
    assert face_calls[1].kwargs["depends_on"] == "id-scrfd"
    assert "depends_on" not in face_calls[0].kwargs
    assert face_calls[2].kwargs["job_timeout"] == RQ_MEDIAPIPE_JOB_TIMEOUT


def test_scan_path_queues_child_directories_and_one_image_query(tmp_path: Path) -> None:
    photos = tmp_path / "album"
    nested = photos / "nested"
    nested.mkdir(parents=True)
    (photos / "keep.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (nested / "child.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    with (
        patch("meta_face.queue.enqueue_scan_path", return_value="scan-child") as scan,
        patch("meta_face.queue.enqueue_query_images", return_value="images-root") as query,
        patch("meta_face.queue.enqueue_add_image") as add_image_job,
    ):
        result = scan_path(str(photos), ["scrfd", "arcface"], force=False, recursive=True)

    scan.assert_called_once()
    assert scan.call_args.args[0] == nested
    query.assert_called_once()
    assert query.call_args.args[0] == photos.resolve()
    assert query.call_args.args[1] == ["scrfd", "arcface"]
    add_image_job.assert_not_called()
    assert result["query_job"] == "images-root"
    assert result["subdir_jobs"] == 1
    assert result["discovered"] == 1


def test_query_images_enqueues_one_add_image_job_per_file(tmp_path: Path) -> None:
    photos = tmp_path / "album"
    nested = photos / "nested"
    nested.mkdir(parents=True)
    keep = photos / "keep.jpg"
    keep.write_bytes(b"\xff\xd8\xff\xd9")
    (photos / "notes.txt").write_text("skip")
    (nested / "child.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    with patch("meta_face.queue.enqueue_add_image", return_value="add-1") as add:
        result = query_images(str(photos), ["scrfd"], force=False)

    add.assert_called_once()
    assert add.call_args.args[0] == keep
    assert add.call_args.args[1] == ["scrfd"]
    assert result["enqueued"] == 1
    assert result["job_ids"] == ["add-1"]


def test_add_image_enqueues_one_face_job_per_tool(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")

    with patch("meta_face.queue.enqueue_face_tool", side_effect=_face_job) as enqueue:
        result = add_image(str(image_path), list(DEFAULT_TOOLS), force=True)

    assert result["tools"] == ["scrfd", "arcface", "dlib_detect", "dlib_embed"]
    assert result["job_ids"] == ["job-scrfd", "job-arcface", "job-dlib_detect", "job-dlib_embed"]
    assert enqueue.call_count == 4
    for call in enqueue.call_args_list:
        assert call.kwargs["depends_on"] is None


def test_add_image_skips_current_tools(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")

    def current(_doc: object, tool: str) -> bool:
        return tool not in {"dlib_detect", "dlib_embed"}

    with (
        patch("meta_face.sidecar.tool_is_current", side_effect=current),
        patch("meta_face.queue.enqueue_face_tool", side_effect=_face_job) as enqueue,
    ):
        result = add_image(str(image_path), list(DEFAULT_TOOLS), force=False)

    assert result["tools"] == ["dlib_detect", "dlib_embed"]
    assert enqueue.call_count == 2
    assert [call.args[1] for call in enqueue.call_args_list] == ["dlib_detect", "dlib_embed"]


def test_add_image_crop_tools_depend_on_scrfd(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")
    tools = resolve_per_image_tools(validate_tools(["detect", "analysis", "mediapipe"]))

    with patch("meta_face.queue.enqueue_face_tool", side_effect=_face_job) as enqueue:
        result = add_image(str(image_path), tools, force=True)

    assert result["tools"][0] == "scrfd"
    assert enqueue.call_args_list[0].kwargs["depends_on"] is None
    for call in enqueue.call_args_list[1:]:
        tool = call.args[1]
        if tool in CROP_ANALYSIS_TOOLS:
            assert call.kwargs["depends_on"] == "job-scrfd"
        else:
            assert call.kwargs["depends_on"] is None
