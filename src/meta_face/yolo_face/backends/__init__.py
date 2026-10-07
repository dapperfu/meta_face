"""Inference backends; the runtime is chosen from the model file suffix."""

from __future__ import annotations

from pathlib import Path

from meta_face.yolo_face.backends.base import DetectorBackend

ONNX_SUFFIXES = frozenset({".onnx"})


def create_backend(
    model_path: Path,
    *,
    confidence_threshold: float,
    iou_threshold: float,
    image_size: int,
    device: str,
    half: bool = False,
) -> DetectorBackend:
    """``.onnx`` runs on ONNX Runtime; everything else goes through Ultralytics."""
    if model_path.suffix.lower() in ONNX_SUFFIXES:
        from meta_face.yolo_face.backends.onnx_backend import OnnxBackend

        return OnnxBackend(
            model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
            device=device,
        )
    from meta_face.yolo_face.backends.ultralytics_backend import UltralyticsBackend

    return UltralyticsBackend(
        model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        image_size=image_size,
        device=device,
        half=half,
    )


__all__ = ["DetectorBackend", "create_backend"]
