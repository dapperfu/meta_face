"""ONNX Runtime inference for YOLOv8 detection exports (no Ultralytics/PyTorch needed)."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from meta_face.yolo_face.detection import Detection, make_detection
from meta_face.yolo_face.errors import (
    DeviceUnavailableError,
    MissingDependencyError,
    ModelLoadError,
)

logger = logging.getLogger(__name__)

LETTERBOX_FILL = 114
STRIDE = 32


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """
    Resize keeping aspect ratio into a ``size x size`` canvas; returns (image, scale, pad).

    The image is offset exactly as Ultralytics' minimal (stride-aligned) letterbox would
    place it, so the detection grid lines up with PyTorch inference and boxes match.
    """
    height, width = frame.shape[:2]
    scale = min(size / height, size / width)
    new_w, new_h = round(width * scale), round(height * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (-new_w % STRIDE) / 2
    pad_y = (-new_h % STRIDE) / 2
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    canvas = np.full((size, size, 3), LETTERBOX_FILL, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, scale, (left, top)


def _providers(device: str, available: list[str]) -> list[Any]:
    if device.startswith("cuda"):
        if "CUDAExecutionProvider" not in available:
            raise DeviceUnavailableError(
                f"Device '{device}' requested but onnxruntime has no CUDAExecutionProvider "
                f"(available: {', '.join(available)}). Install onnxruntime-gpu only."
            )
        index = int(device.split(":")[1]) if ":" in device else 0
        return [("CUDAExecutionProvider", {"device_id": index}), "CPUExecutionProvider"]
    if device == "mps":
        if "CoreMLExecutionProvider" in available:
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        logger.warning("CoreML provider unavailable; running ONNX model on CPU")
    return ["CPUExecutionProvider"]


class OnnxBackend:
    """
    Runs a YOLOv8 ``.onnx`` export with output shaped ``(1, 4 + num_classes, anchors)``.

    Preprocessing (letterbox) and NMS are done here because the raw export includes neither.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float,
        iou_threshold: float,
        image_size: int,
        device: str,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise MissingDependencyError(
                "onnxruntime is not installed. Install onnxruntime or onnxruntime-gpu."
            ) from None

        self._confidence = confidence_threshold
        self._iou = iou_threshold
        providers = _providers(device, list(ort.get_available_providers()))
        try:
            self._session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as exc:
            raise ModelLoadError(f"Could not load ONNX model '{model_path}': {exc}") from exc
        self._device = device if device.startswith("cuda") else "cpu"

        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        fixed = model_input.shape[2] if isinstance(model_input.shape[2], int) else None
        if fixed is not None and fixed != image_size:
            logger.warning(
                "ONNX model has fixed input %d; ignoring requested size %d", fixed, image_size
            )
        self._image_size = fixed or image_size
        self._num_classes = self._read_num_classes()

    def _read_num_classes(self) -> int:
        meta = self._session.get_modelmeta().custom_metadata_map
        names = meta.get("names")
        if names:
            try:
                return max(1, len(ast.literal_eval(names)))
            except (ValueError, SyntaxError):
                pass
        return 1

    @property
    def name(self) -> str:
        return "onnxruntime"

    @property
    def device(self) -> str:
        return self._device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        image, scale, (pad_x, pad_y) = letterbox(frame, self._image_size)
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, swapRB=True)
        output = self._session.run(None, {self._input_name: blob})[0]
        preds = np.squeeze(output, axis=0)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T
        scores_all = preds[:, 4 : 4 + self._num_classes]
        class_ids = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(len(scores_all)), class_ids]
        keep = scores >= self._confidence
        if not np.any(keep):
            return []
        boxes, scores, class_ids = preds[keep, :4], scores[keep], class_ids[keep]

        cx, cy, w, h = boxes.T
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        bw, bh = w / scale, h / scale
        rects = np.stack([x1, y1, bw, bh], axis=1)
        indices = cv2.dnn.NMSBoxesBatched(
            rects.tolist(), scores.tolist(), class_ids.tolist(), self._confidence, self._iou
        )

        height, width = frame.shape[:2]
        out: list[Detection] = []
        for i in np.asarray(indices).reshape(-1):
            bx, by, bw_i, bh_i = rects[i]
            det = make_detection(
                float(bx),
                float(by),
                float(bx + bw_i),
                float(by + bh_i),
                float(scores[i]),
                width,
                height,
                int(class_ids[i]),
            )
            if det is not None:
                out.append(det)
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out
