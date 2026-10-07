"""
Minimal use of the YOLO face detector from another application.

    python examples/yolo_face.py path/to/video.mp4    # or a camera index, e.g. 0

Each frame's detections go to ``on_detections``; swap its body for MQTT, a web hook,
Home Assistant, etc. Only coordinates leave this function, never pixels.
"""

from __future__ import annotations

import sys

from meta_face.yolo_face import Detection, FaceDetector
from meta_face.yolo_face.pipeline import run_stream
from meta_face.yolo_face.sources import FrameSource


def on_detections(frame_id: int, detections: list[Detection]) -> None:
    if detections:
        boxes = ", ".join(f"{d.xywh} @ {d.confidence:.2f}" for d in detections)
        print(f"frame {frame_id}: {len(detections)} face(s): {boxes}")


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    detector = FaceDetector(model_path="models/yolov8n-face-lindevs.pt")
    with FrameSource(source) as frames:
        run_stream(frames, detector, on_detections=on_detections)


if __name__ == "__main__":
    main()
