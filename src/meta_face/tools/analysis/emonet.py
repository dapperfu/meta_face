"""EmoNet valence/arousal prediction (optional dependency)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import torch

from meta_face.tools.analysis.base import FaceContext, face_results_payload

TOOL_NAME = "emonet"
TOOL_VERSION = "1.0.0"
MODEL_NAME = "emonet_8"


@lru_cache(maxsize=1)
def _get_model():
    from emonet.models import EmoNet

    model = EmoNet(n_expression=8)
    model.eval()
    return model


def availability() -> str | None:
    try:
        import emonet  # noqa: F401
    except ImportError:
        return (
            "EmoNet is not installed. Install optional extras: "
            "pip install -e '.[emotion]'  # requires emonet package"
        )
    try:
        _get_model()
    except Exception as exc:
        return f"EmoNet failed to initialize: {exc}"
    return None


def _preprocess(crop_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0)


def analyze_faces(
    image_bgr: Any,
    faces: list[FaceContext],
) -> dict[str, Any]:
    del image_bgr
    model = _get_model()
    per_face: list[dict[str, Any]] = []
    with torch.no_grad():
        for ctx in faces:
            output = model(_preprocess(ctx.crop_bgr))
            record: dict[str, Any] = {"face_index": ctx.face_index}
            if isinstance(output, dict):
                if "valence" in output:
                    record["valence"] = float(output["valence"].reshape(-1)[0])
                if "arousal" in output:
                    record["arousal"] = float(output["arousal"].reshape(-1)[0])
                expr = output.get("expression")
                if expr is not None:
                    record["expression_logits"] = [
                        float(v) for v in expr.reshape(-1).tolist()
                    ]
            per_face.append(record)
    return face_results_payload(per_face, model=MODEL_NAME)
