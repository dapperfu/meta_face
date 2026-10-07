"""
Inference latency benchmark.

Run with ``mf yolo benchmark`` or ``python -m meta_face.yolo_face.benchmark``.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from meta_face.yolo_face.detector import FaceDetector

DEFAULT_WARMUP = 10
DEFAULT_ITERATIONS = 100


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    device: str
    backend: str
    image_size: int
    frame_shape: tuple[int, int]
    load_ms: float
    timings_ms: tuple[float, ...]
    faces: int

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.timings_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.timings_ms)

    @property
    def p95_ms(self) -> float:
        ordered = sorted(self.timings_ms)
        return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]

    @property
    def max_fps(self) -> float:
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0

    def report(self) -> str:
        h, w = self.frame_shape
        return "\n".join(
            [
                f"Model: {self.model}",
                f"Backend: {self.backend}",
                f"Device: {self.device}",
                f"Input: {self.image_size}x{self.image_size} (frame {w}x{h})",
                f"Faces per frame: {self.faces}",
                "",
                f"Model load: {self.load_ms:.1f} ms",
                f"Average inference: {self.mean_ms:.1f} ms",
                f"Median inference: {self.median_ms:.1f} ms",
                f"P95 inference: {self.p95_ms:.1f} ms",
                f"Approximate maximum inference FPS: {self.max_fps:.1f}",
            ]
        )


def run_benchmark(
    load: Callable[[], FaceDetector],
    frame: np.ndarray,
    *,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
) -> BenchmarkResult:
    """Time model loading, then ``iterations`` detections after ``warmup`` untimed ones."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    start = time.perf_counter()
    detector = load()
    load_ms = (time.perf_counter() - start) * 1000.0
    for _ in range(warmup):
        detector.detect(frame)
    timings: list[float] = []
    faces = 0
    for _ in range(iterations):
        t0 = time.perf_counter()
        faces = len(detector.detect(frame))
        timings.append((time.perf_counter() - t0) * 1000.0)
    return BenchmarkResult(
        model=detector.model_path.stem,
        device=detector.device,
        backend=detector.backend.name,
        image_size=detector.image_size,
        frame_shape=(int(frame.shape[0]), int(frame.shape[1])),
        load_ms=load_ms,
        timings_ms=tuple(timings),
        faces=faces,
    )


if __name__ == "__main__":
    from meta_face.yolo_face.cli import benchmark_cmd

    benchmark_cmd()
