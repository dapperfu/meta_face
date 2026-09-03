"""Complete Py-Feat image results, supporting legacy Detector and Detectorv1/v2."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import numpy as np

from meta_face.sdk import SDKSession, encode_result, provider_issue, provider_options, provider_version
from meta_face.tools.analysis.base import FaceContext, face_results_payload

TOOL_NAME = "py_feat"
TOOL_VERSION = "2.0.0"
MODEL_NAME = "py-feat"


@lru_cache(maxsize=1)
def _get_detector(config_json: str = "{}") -> Any:
    options = json.loads(config_json)
    sdk = SDKSession(TOOL_NAME)
    name = options.get("detector_class")
    if name is None:
        name = "Detectorv1" if hasattr(sdk.module, "Detectorv1") else "Detector"
    return sdk.call(name, **sdk.resolve(options.get("detector", {})))


def availability() -> str | None:
    return provider_issue(TOOL_NAME)


def analyze_faces(image_bgr: Any, faces: list[FaceContext]) -> dict[str, Any]:
    options = provider_options(TOOL_NAME)
    unknown = set(options) - {"detector_class", "detector", "detect"}
    if unknown:
        raise ValueError(f"Unknown Py-Feat options: {sorted(unknown)}")
    detector = _get_detector(json.dumps(options, sort_keys=True))
    # The legacy API accepts filenames, not an RGB ndarray. A lossless temporary
    # image works with both current Detector classes and keeps image coordinates.
    with TemporaryDirectory(prefix="meta-face-feat-") as directory:
        path = Path(directory) / "input.png"
        if not cv2.imwrite(str(path), image_bgr):
            raise OSError("Could not prepare Py-Feat input image")
        kwargs = dict(options.get("detect", {}))
        if kwargs.get("output_size") is not None:
            raise ValueError("Py-Feat scan requires original-size coordinates; use mf sdk for resizing")
        if hasattr(detector, "detect"):
            kwargs["data_type"] = "image"
            fex = detector.detect([str(path)], **kwargs)
        else:
            fex = detector.detect_image([str(path)], **kwargs)

    records: list[dict[str, Any]] = []
    non_face_rows: list[dict[str, Any]] = []
    column_groups: dict[str, Any] = {}
    if fex is not None:
        for name in ("au_columns", "emotion_columns", "facebox_columns", "landmark_columns",
                     "facepose_columns", "gaze_columns", "identity_columns", "blendshape_columns"):
            columns = getattr(fex, name, None)
            if columns is not None:
                column_groups[name] = encode_result(columns)
        for raw in fex.to_dict(orient="records"):
            raw.pop("input", None)  # Temporary filename is not durable provenance.
            row = encode_result(raw)
            box = [row.get(c) for c in ("FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight")]
            if any(v is None for v in box) or not np.isfinite(box).all() or min(box[2:]) <= 0:
                non_face_rows.append(row)
                continue
            x, y, w, h = box
            record = {"face_index": len(records), "bbox": [x, y, x + w, y + h], "raw": row}
            for field, group in (("action_units", "au_columns"), ("emotion_scores", "emotion_columns"),
                                 ("landmarks", "landmark_columns"), ("pose", "facepose_columns"),
                                 ("gaze", "gaze_columns"), ("identity", "identity_columns"),
                                 ("blendshapes", "blendshape_columns")):
                record[field] = {c: row[c] for c in column_groups.get(group, []) if c in row}
            records.append(record)
    info = getattr(detector, "info", {})
    return face_results_payload(records, model=MODEL_NAME, extra={
        "sdk_version": provider_version(TOOL_NAME), "options": options,
        "detector_class": type(detector).__name__, "model_info": encode_result(info),
        "face_index_source": TOOL_NAME, "column_groups": column_groups,
        "non_face_rows": non_face_rows,
    })
