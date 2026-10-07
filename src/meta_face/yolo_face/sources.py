"""Frame acquisition from images, cameras, video files, and network streams."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import final
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from meta_face.yolo_face.errors import ImageDecodeError, SourceOpenError

logger = logging.getLogger(__name__)

STREAM_SCHEMES = frozenset({"rtsp", "rtsps", "rtmp", "http", "https", "udp", "tcp"})


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_ms: float
    image: np.ndarray


def load_image(path: Path) -> np.ndarray:
    """Read an image as BGR, raising :class:`ImageDecodeError` on failure."""
    if not path.is_file():
        raise ImageDecodeError(f"Could not decode image: {path} does not exist.")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ImageDecodeError(f"Could not decode image: {path}")
    return image


def is_stream_url(spec: str) -> bool:
    return urlsplit(spec).scheme.lower() in STREAM_SCHEMES


def expand_url(url: str) -> str:
    """Substitute ``${VAR}`` / ``$VAR`` placeholders so credentials can live in the env."""
    return os.path.expandvars(url)


def sanitize_url(url: str) -> str:
    """Drop ``user:password@`` from a URL so it is safe to log."""
    parts = urlsplit(url)
    if not parts.scheme or "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


@final
class FrameSource:
    """
    Context-managed ``cv2.VideoCapture`` over a camera index, video file, or stream URL.

    Iterating yields :class:`Frame` objects until the source ends or fails.
    """

    def __init__(
        self,
        spec: int | str,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
    ) -> None:
        if isinstance(spec, str) and spec.isdigit():
            spec = int(spec)
        self.spec = spec
        self.is_camera = isinstance(spec, int)
        self.is_stream = isinstance(spec, str) and is_stream_url(spec)
        self.is_live = self.is_camera or self.is_stream
        self._requested = (width, height, fps)
        self._cap: cv2.VideoCapture | None = None

    @property
    def display_name(self) -> str:
        if self.is_camera:
            return f"camera {self.spec}"
        return sanitize_url(str(self.spec))

    def open(self) -> FrameSource:
        if self.is_camera:
            cap = cv2.VideoCapture(int(self.spec))
            if not cap.isOpened():
                cap.release()
                raise SourceOpenError(f"Unable to open camera index {self.spec}.")
            self._configure_camera(cap)
        else:
            target = expand_url(str(self.spec)) if self.is_stream else str(self.spec)
            if not self.is_stream and not Path(target).is_file():
                raise SourceOpenError(f"Unable to open input video: {target} does not exist.")
            cap = cv2.VideoCapture(target)
            if not cap.isOpened():
                cap.release()
                raise SourceOpenError(f"Unable to open input video: {self.display_name}")
        self._cap = cap
        label = "Camera opened" if self.is_camera else "Opened"
        logger.info(
            "%s: %s %dx%d @ %.0f FPS",
            label,
            self.display_name,
            self.width,
            self.height,
            self.fps,
        )
        return self

    def _configure_camera(self, cap: cv2.VideoCapture) -> None:
        width, height, fps = self._requested
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)

    def _capture(self) -> cv2.VideoCapture:
        if self._cap is None:
            raise RuntimeError("FrameSource is not open")
        return self._cap

    @property
    def width(self) -> int:
        return int(self._capture().get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._capture().get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        value = float(self._capture().get(cv2.CAP_PROP_FPS))
        return value if value > 0 else 30.0

    @property
    def frame_count(self) -> int | None:
        if self.is_live:
            return None
        count = int(self._capture().get(cv2.CAP_PROP_FRAME_COUNT))
        return count if count > 0 else None

    def __iter__(self) -> Iterator[Frame]:
        cap = self._capture()
        start = time.monotonic()
        index = 0
        while True:
            ok, image = cap.read()
            if not ok or image is None:
                return
            if self.is_live:
                timestamp = (time.monotonic() - start) * 1000.0
            else:
                timestamp = index * 1000.0 / self.fps
            yield Frame(index, timestamp, image)
            index += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> FrameSource:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
