"""Detectron2 face detection backend."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from meta_face.backends.base import FaceDetectionBackend
from meta_face.config import (
    DETECTRON2_CONFIG_PATH,
    DETECTRON2_DEVICE,
    DETECTRON2_SCORE_THRESH,
    DETECTRON2_WEIGHTS_PATH,
)
from meta_face.models import is_detectron2_available


class Detectron2Backend(FaceDetectionBackend):
    @property
    def name(self) -> str:
        return "detectron2"

    def available(self) -> bool:
        try:
            import detectron2  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False
        return is_detectron2_available()

    @lru_cache(maxsize=1)
    def _get_predictor(self):
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        cfg = get_cfg()
        cfg.merge_from_file(str(DETECTRON2_CONFIG_PATH))
        cfg.MODEL.WEIGHTS = str(DETECTRON2_WEIGHTS_PATH)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECTRON2_SCORE_THRESH
        cfg.MODEL.DEVICE = DETECTRON2_DEVICE
        return DefaultPredictor(cfg)

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        self.ensure_available()
        predictor = self._get_predictor()
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        detections: list[dict[str, Any]] = []

        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()

        keypoints = None
        if instances.has("pred_keypoints"):
            keypoints = instances.pred_keypoints.numpy()

        for idx in range(len(boxes)):
            x1, y1, x2, y2 = boxes[idx].tolist()
            det: dict[str, Any] = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "det_score": float(scores[idx]),
            }
            if keypoints is not None:
                kps = keypoints[idx]
                det["landmarks"] = [[float(x), float(y)] for x, y in kps]
            detections.append(det)

        return detections

    def ensure_available(self) -> None:
        if self.available():
            return
        try:
            import detectron2  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "Detectron2 backend is not installed. Install optional extras:\n"
                "  pip install -e \".[detectron2]\"\n"
                "Then install detectron2 wheels for your CUDA/torch version:\n"
                "  https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md"
            ) from None
        raise RuntimeError(
            "Detectron2 model files are missing. Download weights:\n"
            "  mf download --backend detectron2"
        )
