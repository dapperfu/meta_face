"""The single frame loop shared by camera, video, and stream processing."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from meta_face.yolo_face.detection import Detection
from meta_face.yolo_face.drawing import draw_detections, draw_stats
from meta_face.yolo_face.metrics import FrameTimer, RollingFps
from meta_face.yolo_face.output import JsonLinesWriter, VideoSink, frame_result
from meta_face.yolo_face.sources import Frame

logger = logging.getLogger(__name__)

DetectionCallback = Callable[[int, list[Detection]], None]
QUIT_KEYS = frozenset({ord("q"), ord("Q"), 27})
LOG_INTERVAL_S = 5.0


class FaceDetectorLike(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class Tracker(Protocol):
    """Future hook (e.g. ByteTrack): assigns per-object track IDs, never identities."""

    def update(self, detections: Sequence[Detection]) -> list[int]: ...


@dataclass(frozen=True)
class DrawOptions:
    draw_boxes: bool = True
    show_confidence: bool = True
    show_center: bool = False
    show_fps: bool = True


@dataclass
class StreamStats:
    frames: int = 0
    inferred_frames: int = 0
    faces: int = 0
    total_inference_ms: float = 0.0
    elapsed_s: float = 0.0
    stopped_by_user: bool = False

    @property
    def mean_fps(self) -> float:
        return self.frames / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def mean_inference_ms(self) -> float:
        return self.total_inference_ms / self.inferred_frames if self.inferred_frames else 0.0


def run_stream(
    frames: Iterable[Frame],
    detector: FaceDetectorLike,
    *,
    frame_skip: int = 0,
    on_detections: DetectionCallback | None = None,
    sink: VideoSink | None = None,
    window: str | None = None,
    json_writer: JsonLinesWriter | None = None,
    draw: DrawOptions | None = None,
    normalized: bool = False,
    tracker: Tracker | None = None,
) -> StreamStats:
    """
    Detect faces on every ``frame_skip + 1``-th frame and fan results out.

    Skipped frames reuse the previous detections so overlays do not flicker. When
    ``window`` is set, frames are shown with ``cv2.imshow`` and ``q``/Esc stops the loop.
    """
    if frame_skip < 0:
        raise ValueError("frame_skip must be >= 0")
    draw = draw or DrawOptions()
    want_canvas = sink is not None or window is not None
    stats = StreamStats()
    fps = RollingFps()
    detections: list[Detection] = []
    track_ids: list[int] | None = None
    inference_ms = 0.0
    start = last_log = time.perf_counter()
    previous = start
    capture_start = start

    for frame in frames:
        timer = FrameTimer()
        timer.stages["capture"] = (time.perf_counter() - capture_start) * 1000.0
        if frame.index % (frame_skip + 1) == 0:
            with timer.stage("inference"):
                detections = detector.detect(frame.image)
            inference_ms = timer.get("inference")
            stats.inferred_frames += 1
            stats.total_inference_ms += inference_ms
            if tracker is not None:
                track_ids = tracker.update(detections)
        stats.frames += 1
        stats.faces += len(detections)

        height, width = frame.image.shape[:2]
        if on_detections is not None:
            on_detections(frame.index, list(detections))
        if json_writer is not None:
            json_writer.write(
                frame_result(
                    frame.index,
                    frame.timestamp_ms,
                    detections,
                    width,
                    height,
                    normalized=normalized,
                )
            )

        now = time.perf_counter()
        fps.add(now - previous)
        previous = now

        if want_canvas:
            with timer.stage("draw"):
                canvas = (
                    draw_detections(
                        frame.image, detections, draw.show_confidence, draw.show_center, track_ids
                    )
                    if draw.draw_boxes
                    else frame.image.copy()
                )
                if draw.show_fps:
                    draw_stats(
                        canvas,
                        [
                            f"FPS: {fps.fps:.1f}",
                            f"Inference: {inference_ms:.1f} ms",
                            f"Faces: {len(detections)}",
                        ],
                    )
            if sink is not None:
                sink.write(canvas)
            if window is not None:
                cv2.imshow(window, canvas)
                if (cv2.waitKey(1) & 0xFF) in QUIT_KEYS:
                    stats.stopped_by_user = True
                    break

        timer.stages["total"] = (time.perf_counter() - capture_start) * 1000.0
        logger.debug(
            "frame=%d faces=%d capture=%.1fms inference=%.1fms draw=%.1fms total=%.1fms",
            frame.index,
            len(detections),
            timer.get("capture"),
            timer.get("inference"),
            timer.get("draw"),
            timer.get("total"),
        )
        if now - last_log >= LOG_INTERVAL_S:
            logger.info(
                "FPS: %.1f  Inference: %.1f ms  Faces: %d", fps.fps, inference_ms, len(detections)
            )
            last_log = now
        capture_start = time.perf_counter()

    stats.elapsed_s = time.perf_counter() - start
    if window is not None:
        cv2.destroyWindow(window)
    return stats
