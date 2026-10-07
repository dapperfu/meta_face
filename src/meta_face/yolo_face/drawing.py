"""Overlay rendering; never mutates detections or the input frame."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from meta_face.yolo_face.detection import Detection

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)
STATS_COLOR = (0, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _thickness(frame: np.ndarray) -> int:
    return max(1, round(max(frame.shape[:2]) / 640))


def draw_detections(
    frame: np.ndarray,
    detections: Sequence[Detection],
    show_confidence: bool = True,
    show_center: bool = False,
    track_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """Return a copy of ``frame`` with a box and ``Face 0.94``-style label per detection."""
    canvas = frame.copy()
    thick = _thickness(canvas)
    scale = 0.3 + 0.15 * thick
    text_thick = max(1, thick // 2)
    for i, det in enumerate(detections):
        cv2.rectangle(canvas, (det.x1, det.y1), (det.x2 - 1, det.y2 - 1), BOX_COLOR, thick)
        label = "Face" if track_ids is None else f"Face {track_ids[i]}"
        if show_confidence:
            label = f"{label} {det.confidence:.2f}"
        (tw, th), base = cv2.getTextSize(label, FONT, scale, text_thick)
        top = det.y1 - th - base - 2
        if top < 0:
            top = det.y1
        cv2.rectangle(canvas, (det.x1, top), (det.x1 + tw + 2, top + th + base + 2), BOX_COLOR, -1)
        cv2.putText(canvas, label, (det.x1 + 1, top + th + 1), FONT, scale, TEXT_COLOR, text_thick)
        if show_center:
            center = (round(det.center_x), round(det.center_y))
            cv2.circle(canvas, center, thick * 2, BOX_COLOR, -1)
    return canvas


def draw_stats(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """Draw status lines (FPS, inference time, ...) in the top-left corner, in place."""
    thick = _thickness(frame)
    scale = 0.6 * thick
    y = 0
    for line in lines:
        (_, th), base = cv2.getTextSize(line, FONT, scale, thick)
        y += th + base + 4
        cv2.putText(frame, line, (8, y), FONT, scale, (0, 0, 0), thick + 2)
        cv2.putText(frame, line, (8, y), FONT, scale, STATS_COLOR, thick)
    return frame
