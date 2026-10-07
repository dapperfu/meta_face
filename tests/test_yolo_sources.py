"""Frame source, URL sanitization, and output writer tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from meta_face.yolo_face.detection import Detection
from meta_face.yolo_face.drawing import draw_detections
from meta_face.yolo_face.errors import ImageDecodeError, SourceOpenError
from meta_face.yolo_face.output import JsonLinesWriter, VideoSink, image_result
from meta_face.yolo_face.sources import FrameSource, expand_url, load_image, sanitize_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("rtsp://admin:secret123@192.168.1.25/stream", "rtsp://192.168.1.25/stream"),
        ("rtsp://admin@cam:554/s?token=x", "rtsp://cam:554/s"),
        ("rtsp://cam/stream", "rtsp://cam/stream"),
        ("video.mp4", "video.mp4"),
    ],
)
def test_sanitize_url(url: str, expected: str) -> None:
    assert sanitize_url(url) == expected


def test_expand_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAM_USER", "u")
    monkeypatch.setenv("CAM_PASS", "p")
    assert expand_url("rtsp://${CAM_USER}:${CAM_PASS}@h/s") == "rtsp://u:p@h/s"


def test_stream_display_name_hides_credentials() -> None:
    src = FrameSource("rtsp://admin:secret@cam/stream")
    assert src.is_stream and src.is_live
    assert "secret" not in src.display_name


def test_load_image_errors(tmp_path: Path) -> None:
    with pytest.raises(ImageDecodeError):
        load_image(tmp_path / "missing.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ImageDecodeError, match="Could not decode image"):
        load_image(bad)


def test_missing_video(tmp_path: Path) -> None:
    with pytest.raises(SourceOpenError, match="Unable to open input video"):
        FrameSource(str(tmp_path / "nope.mp4")).open()


def test_bad_camera() -> None:
    with pytest.raises(SourceOpenError, match="Unable to open camera index 99"):
        FrameSource(99).open()


def test_video_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "v.mp4"
    with VideoSink(path, 12.0, (64, 48)) as sink:
        for i in range(5):
            sink.write(np.full((48, 64, 3), i * 40, np.uint8))
    with FrameSource(str(path)) as src:
        assert (src.width, src.height) == (64, 48)
        assert src.fps == pytest.approx(12.0)
        frames = list(src)
    assert [f.index for f in frames] == [0, 1, 2, 3, 4]
    assert frames[2].timestamp_ms == pytest.approx(2000 / 12)


def test_json_outputs(tmp_path: Path) -> None:
    dets = [Detection(463, 212, 621, 401, 0.936)]
    result = image_result(dets, 1920, 1080)
    assert result["faces"][0] == {
        "x1": 463,
        "y1": 212,
        "x2": 621,
        "y2": 401,
        "width": 158,
        "height": 189,
        "confidence": 0.936,
    }
    path = tmp_path / "o.jsonl"
    with JsonLinesWriter(path) as writer:
        writer.write({"frame": 0})
        writer.write({"frame": 1})
    assert [json.loads(line)["frame"] for line in path.read_text().splitlines()] == [0, 1]


def test_drawing_does_not_mutate() -> None:
    frame = np.zeros((100, 100, 3), np.uint8)
    dets = [Detection(10, 10, 50, 50, 0.9)]
    canvas = draw_detections(frame, dets, show_center=True, track_ids=[3])
    assert not frame.any()
    assert canvas.any()
    assert dets == [Detection(10, 10, 50, 50, 0.9)]
    assert cv2.countNonZero(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)) > 0
