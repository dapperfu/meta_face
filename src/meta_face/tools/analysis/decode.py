"""Shared ONNX decoding used by crop analysis adapters.

These match the sports-review contracts: softmax probabilities, 90-bin gaze
heads, expanded crops, and BiSeNet class argmax. Geometry conversion stays in
``meta_face.coordinates``.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


OPENCV_FER_LABELS: tuple[str, ...] = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
)

FER_PLUS_LABELS: tuple[str, ...] = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
)

GAZE_BINS = 90
GAZE_BIN_DEGREES = 4.0
ANTISPOOF_CROP_SCALE = 2.7
MEDIAPIPE_CROP_SCALE = 1.8
MEDIAPIPE_CROP_SIZE = 256
BISENET_INPUT = 512
GAZE_INPUT = 448
ANTISPOOF_INPUT = 80
OPENCV_FER_INPUT = 112


def softmax(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    weights = np.exp(arr - np.max(arr))
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full_like(arr, 1.0 / arr.size)
    return weights / total


def gaze_angles_degrees(outputs: list[Any]) -> tuple[float, float]:
    """Decode MiniFASNet-style 90-bin yaw and pitch logits to degrees."""
    if len(outputs) < 2:
        raise ValueError("gaze model must return separate yaw and pitch heads")
    yaw_bins = np.asarray(outputs[0]).reshape(-1)
    pitch_bins = np.asarray(outputs[1]).reshape(-1)
    if yaw_bins.size != GAZE_BINS or pitch_bins.size != GAZE_BINS:
        raise ValueError("gaze heads must each have 90 bins")
    bins = np.arange(GAZE_BINS, dtype=np.float64)
    yaw = float(softmax(yaw_bins) @ bins * GAZE_BIN_DEGREES - 180.0)
    pitch = float(softmax(pitch_bins) @ bins * GAZE_BIN_DEGREES - 180.0)
    return yaw, pitch


def expand_crop(
    image_bgr: np.ndarray,
    bbox: list[float],
    scale: float = 1.0,
) -> tuple[np.ndarray, list[int]]:
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = (x2 - x1) * scale / 2.0, (y2 - y1) * scale / 2.0
    height, width = image_bgr.shape[:2]
    left = max(0, int(cx - hw))
    top = max(0, int(cy - hh))
    right = min(width, int(np.ceil(cx + hw)))
    bottom = min(height, int(np.ceil(cy + hh)))
    if right <= left or bottom <= top:
        return image_bgr[0:0, 0:0], [left, top, right, bottom]
    return image_bgr[top:bottom, left:right], [left, top, right, bottom]


def rgb_normalized_tensor(
    crop_bgr: np.ndarray,
    size: int,
    *,
    imagenet: bool = False,
) -> np.ndarray:
    resized = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if imagenet:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
    else:
        rgb = (rgb - 0.5) / 0.5
    return rgb.transpose(2, 0, 1)[np.newaxis, ...]


def bgr_float_tensor(crop_bgr: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]


def parsing_argmax(logits: Any) -> np.ndarray:
    arr = np.asarray(logits)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError("BiSeNet output must be (C, H, W) or (1, C, H, W)")
    return arr.argmax(axis=0).astype(np.uint8)


def class_pixel_percent(mask: np.ndarray) -> dict[str, float]:
    values, counts = np.unique(mask, return_counts=True)
    total = float(mask.size) if mask.size else 1.0
    return {str(int(v)): float(c / total * 100.0) for v, c in zip(values, counts)}
