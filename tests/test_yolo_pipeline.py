"""Shared frame-loop tests with a synthetic source and fake detector."""

from __future__ import annotations

import numpy as np
import pytest

from meta_face.yolo_face.detection import Detection
from meta_face.yolo_face.metrics import RollingFps
from meta_face.yolo_face.pipeline import run_stream
from meta_face.yolo_face.sources import Frame


class CountingDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        return [Detection(0, 0, 10, 10, 0.9, class_id=self.calls)]


def _frames(n: int) -> list[Frame]:
    return [Frame(i, i * 33.3, np.zeros((20, 20, 3), np.uint8)) for i in range(n)]


def test_every_frame_without_skip() -> None:
    detector = CountingDetector()
    stats = run_stream(_frames(4), detector)
    assert detector.calls == 4
    assert stats.frames == 4 and stats.inferred_frames == 4


def test_frame_skip_reuses_previous_detections() -> None:
    detector = CountingDetector()
    seen: list[tuple[int, int]] = []
    stats = run_stream(
        _frames(5),
        detector,
        frame_skip=1,
        on_detections=lambda idx, dets: seen.append((idx, dets[0].class_id)),
    )
    assert detector.calls == 3
    assert stats.inferred_frames == 3
    assert seen == [(0, 1), (1, 1), (2, 2), (3, 2), (4, 3)]


def test_negative_skip_rejected() -> None:
    with pytest.raises(ValueError):
        run_stream([], CountingDetector(), frame_skip=-1)


def test_rolling_fps() -> None:
    fps = RollingFps(window=3)
    for d in (1.0, 0.1, 0.1, 0.1):
        fps.add(d)
    assert fps.fps == pytest.approx(10.0)
