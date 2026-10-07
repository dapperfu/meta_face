"""Serialization of detections and annotated video output."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import IO, Any, final

import cv2
import numpy as np

from meta_face.yolo_face.detection import Detection
from meta_face.yolo_face.errors import FaceDetectorError

logger = logging.getLogger(__name__)

DEFAULT_FOURCC = "mp4v"


def faces_payload(
    detections: Sequence[Detection], width: int, height: int, *, normalized: bool = False
) -> list[dict[str, Any]]:
    return [d.to_dict(width, height, normalized=normalized) for d in detections]


def image_result(
    detections: Sequence[Detection], width: int, height: int, *, normalized: bool = False
) -> dict[str, Any]:
    """``{"width", "height", "faces": [...]}`` for a single image."""
    return {
        "width": width,
        "height": height,
        "faces": faces_payload(detections, width, height, normalized=normalized),
    }


def frame_result(
    frame_index: int,
    timestamp_ms: float,
    detections: Sequence[Detection],
    width: int,
    height: int,
    *,
    normalized: bool = False,
) -> dict[str, Any]:
    """Per-frame record for video/stream output."""
    return {
        "frame": frame_index,
        "timestamp_ms": round(timestamp_ms),
        "faces": faces_payload(detections, width, height, normalized=normalized),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@final
class JsonLinesWriter:
    """Writes one JSON object per line; usable as a context manager."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: IO[str] | None = None

    def __enter__(self) -> JsonLinesWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        return self

    def write(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("JsonLinesWriter is not open")
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


@final
class VideoSink:
    """``cv2.VideoWriter`` wrapper that keeps source resolution and frame rate."""

    def __init__(
        self, path: Path, fps: float, size: tuple[int, int], fourcc: str = DEFAULT_FOURCC
    ) -> None:
        self.path = path
        self.fps = fps if fps > 0 else 30.0
        self.size = size
        self.fourcc = fourcc
        self._writer: cv2.VideoWriter | None = None

    def __enter__(self) -> VideoSink:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*self.fourcc), self.fps, self.size
        )
        if not writer.isOpened():
            raise FaceDetectorError(f"Unable to open output video: {self.path}")
        self._writer = writer
        logger.info("Writing %s (%dx%d @ %.2f FPS)", self.path, *self.size, self.fps)
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("VideoSink is not open")
        self._writer.write(frame)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
