"""Benchmark statistics with a fake backend."""

from __future__ import annotations

import numpy as np
import pytest

from meta_face.yolo_face import FaceDetector
from meta_face.yolo_face.benchmark import BenchmarkResult, run_benchmark
from meta_face.yolo_face.detection import Detection


class _Backend:
    name = "fake"
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        return [Detection(0, 0, 4, 4, 0.9)]


def test_run_benchmark_counts() -> None:
    backend = _Backend()
    result = run_benchmark(
        lambda: FaceDetector(backend=backend),
        np.zeros((10, 10, 3), np.uint8),
        warmup=3,
        iterations=7,
    )
    assert backend.calls == 10
    assert len(result.timings_ms) == 7
    assert result.faces == 1
    assert "Approximate maximum inference FPS" in result.report()


def test_statistics() -> None:
    result = BenchmarkResult(
        model="m",
        device="cpu",
        backend="fake",
        image_size=640,
        frame_shape=(1, 1),
        load_ms=1.0,
        timings_ms=tuple(float(i) for i in range(1, 21)),
        faces=0,
    )
    assert result.mean_ms == pytest.approx(10.5)
    assert result.median_ms == pytest.approx(10.5)
    assert result.p95_ms == 19.0
    assert result.max_fps == pytest.approx(1000 / 10.5)
