"""Detectron2 face detection backend."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from meta_face.backends.base import FaceDetectionBackend
from meta_face.config import (
    DETECTRON2_CLASS_FILTER,
    DETECTRON2_SCORE_THRESH,
    resolve_detectron2_device,
)
from meta_face.detectron2_model import is_detectron2_available, resolve_detectron2_model
from meta_face.tools.sidecar_encode import json_safe


class Detectron2Backend(FaceDetectionBackend):
    @property
    def name(self) -> str:
        return "detectron2"

    def available(self) -> bool:
        return is_detectron2_available()

    @lru_cache(maxsize=1)
    def _get_predictor(self):
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        paths = resolve_detectron2_model()
        cfg = get_cfg()
        cfg.merge_from_file(paths.config)
        cfg.MODEL.WEIGHTS = paths.weights
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = DETECTRON2_SCORE_THRESH
        device = resolve_detectron2_device()
        cfg.MODEL.DEVICE = device
        return DefaultPredictor(cfg), paths, device

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        self.ensure_available()
        predictor, _paths, _device = self._get_predictor()
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        detections: list[dict[str, Any]] = []

        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()

        keypoints = None
        if instances.has("pred_keypoints"):
            keypoints = instances.pred_keypoints.numpy()

        pred_classes = None
        if instances.has("pred_classes"):
            pred_classes = instances.pred_classes.numpy()

        kept = 0
        for idx in range(len(boxes)):
            if pred_classes is not None and DETECTRON2_CLASS_FILTER is not None:
                if int(pred_classes[idx]) not in DETECTRON2_CLASS_FILTER:
                    continue
            x1, y1, x2, y2 = boxes[idx].tolist()
            det: dict[str, Any] = {
                "face_index": kept,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "det_score": float(scores[idx]),
                "bbox_width": float(x2 - x1),
                "bbox_height": float(y2 - y1),
            }
            if pred_classes is not None:
                det["class_id"] = int(pred_classes[idx])
            if keypoints is not None:
                kps = keypoints[idx]
                det["keypoints"] = json_safe(kps.tolist())
                det["landmarks"] = [[float(x), float(y)] for x, y in kps[:, :2]]
                if kps.shape[1] >= 3:
                    det["keypoint_visibility"] = [float(v) for v in kps[:, 2]]
            detections.append(det)
            kept += 1

        return detections

    def detectron2_to_sidecar_payload(
        self,
        image: np.ndarray,
        detections: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """All detectron2 tool outputs for face.detectron2.* sidecar keys."""
        if detections is None:
            detections = self.detect(image)
        h_img, w_img = image.shape[:2]
        _predictor, paths, device = self._get_predictor()
        class_filter = (
            sorted(DETECTRON2_CLASS_FILTER) if DETECTRON2_CLASS_FILTER is not None else None
        )
        return json_safe(
            {
                "faces": self.to_records(detections),
                "face_count": len(detections),
                "image_size": [int(w_img), int(h_img)],
                "score_thresh": DETECTRON2_SCORE_THRESH,
                "config_path": paths.config,
                "weights_path": paths.weights,
                "model_zoo": paths.model_zoo,
                "class_filter": class_filter,
                "device": device,
            }
        )

    def ensure_available(self) -> None:
        if self.available():
            return
        from meta_face.deps import detectron2_install_message, detectron2_weights_message

        try:
            import detectron2  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            raise RuntimeError(detectron2_install_message()) from None
        raise RuntimeError(detectron2_weights_message())
