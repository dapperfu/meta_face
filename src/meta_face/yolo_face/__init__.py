"""
Detection-only YOLOv8-Face pipeline.

Answers "is there a face, where, and how confident" -- no recognition or embeddings.
Imports nothing from the rest of meta_face so it can be used standalone.
"""

from __future__ import annotations

from meta_face.yolo_face.detection import Detection
from meta_face.yolo_face.detector import FaceDetector

__all__ = ["Detection", "FaceDetector"]
