"""Public face detector API."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from meta_face.yolo_face.backends import DetectorBackend, create_backend
from meta_face.yolo_face.detection import Detection, clip_detection
from meta_face.yolo_face.device import resolve_device
from meta_face.yolo_face.errors import (
    ImageDecodeError,
    MissingDependencyError,
    ModelLoadError,
    ModelNotFoundError,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/yolov8n-face-lindevs.pt")
DEFAULT_CONFIDENCE = 0.40
DEFAULT_IOU = 0.45
DEFAULT_IMAGE_SIZE = 640


class FaceDetector:
    """
    Detect faces in BGR frames and return clean :class:`Detection` objects.

    The model is loaded once in the constructor. Every returned box satisfies
    ``0 <= x1 < x2 <= width``, ``0 <= y1 < y2 <= height``, and ``0 <= confidence <= 1``.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU,
        device: str = "auto",
        image_size: int = DEFAULT_IMAGE_SIZE,
        *,
        min_face_width: int = 0,
        min_face_height: int = 0,
        half: bool = False,
        backend: DetectorBackend | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if image_size <= 0 or image_size % 32 != 0:
            raise ValueError("image_size must be a positive multiple of 32")

        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.min_face_width = min_face_width
        self.min_face_height = min_face_height

        if backend is None:
            if not self.model_path.is_file():
                raise ModelNotFoundError(
                    f"Model file does not exist: {self.model_path}. "
                    "Download it with: mf download --backend yolo"
                )
            logger.info("Loading model %s", self.model_path)
            resolved = resolve_device(device)
            logger.info("Input size: %d", image_size)
            logger.info("Confidence threshold: %.2f", confidence_threshold)
            backend = self._load_backend(resolved, half, allow_cpu_fallback=device == "auto")
        self.backend = backend

    def _load_backend(
        self, device: str, half: bool, *, allow_cpu_fallback: bool
    ) -> DetectorBackend:
        try:
            return create_backend(
                self.model_path,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold,
                image_size=self.image_size,
                device=device,
                half=half,
            )
        except (ModelLoadError, RuntimeError) as exc:
            if not allow_cpu_fallback or device == "cpu" or isinstance(exc, MissingDependencyError):
                raise
            logger.warning("Failed to initialise on %s (%s); falling back to CPU", device, exc)
            logger.info("Device: cpu")
            return self._load_backend("cpu", False, allow_cpu_fallback=False)

    @property
    def device(self) -> str:
        return self.backend.device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect faces in an OpenCV/NumPy BGR image of shape ``(H, W, 3)``."""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.size == 0:
            raise ImageDecodeError("Expected a non-empty HxWx3 BGR image array.")
        height, width = frame.shape[:2]
        out: list[Detection] = []
        for raw in self.backend.detect(frame):
            det = clip_detection(raw, width, height)
            if det is None or det.confidence < self.confidence_threshold:
                continue
            if det.width < self.min_face_width or det.height < self.min_face_height:
                continue
            out.append(det)
        return out
