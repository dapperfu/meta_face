"""Error types raised by the YOLO face pipeline."""

from __future__ import annotations


class FaceDetectorError(RuntimeError):
    """Base class for YOLO face pipeline errors."""


class ModelNotFoundError(FaceDetectorError):
    """The configured model weights file does not exist."""


class ModelLoadError(FaceDetectorError):
    """The weights exist but could not be loaded (corrupt, unsupported format, ...)."""


class ImageDecodeError(FaceDetectorError):
    """An input image could not be read or decoded."""


class SourceOpenError(FaceDetectorError):
    """A camera, video file, or stream could not be opened."""


class DeviceUnavailableError(FaceDetectorError):
    """An explicitly requested inference device is not available."""


class MissingDependencyError(FaceDetectorError):
    """An optional runtime dependency (ultralytics, onnxruntime, ...) is not installed."""
