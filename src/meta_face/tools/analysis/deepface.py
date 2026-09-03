"""DeepFace detection, representation, attributes and optional anti-spoofing."""

from __future__ import annotations

import gc
import logging
import os
from typing import Any

from meta_face.coordinates import to_normalized
from meta_face.sdk import SDKSession, encode_result, provider_issue, provider_options, provider_version
from meta_face.tools.analysis.base import FaceContext, face_results_payload

TOOL_NAME = "deepface"
TOOL_VERSION = "2.0.0"
MODEL_NAME = "deepface"

logger = logging.getLogger(__name__)


def availability() -> str | None:
    return provider_issue(TOOL_NAME)


def _configure_tensorflow_gpu() -> None:
    """Keep TF from grabbing the whole GPU before DeepFace builds its graphs."""
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    try:
        import tensorflow as tf
    except ImportError:
        return
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except (RuntimeError, ValueError):
        return


def _is_gpu_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "failed to allocate memory",
            "out of memory",
            "resource exhausted",
            "oom when allocating",
        )
    )


def _release_gpu() -> None:
    gc.collect()
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except (ImportError, AttributeError, RuntimeError):
        pass
    try:
        import torch

        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except (ImportError, AttributeError, RuntimeError):
        pass


def _call_with_oom_retry(operation: Any) -> Any:
    try:
        return operation()
    except Exception as exc:
        if not _is_gpu_oom(exc):
            raise
        logger.warning("DeepFace GPU OOM; releasing GPU memory and retrying once")
        _release_gpu()
        return operation()


def _crop_coordinates(result: dict[str, Any], crop: Any) -> dict[str, Any]:
    """The aligned crop has its own frame; do not treat its region as image pixels."""
    h, w = crop.shape[:2]
    result = to_normalized(encode_result(result), (w, h))
    result["coordinates"]["space"] = "aligned_crop"
    return result


def analyze_faces(image_bgr: Any, faces: list[FaceContext]) -> dict[str, Any]:
    """Detect independently; align once and reuse each detected face for all heads."""
    options = provider_options(TOOL_NAME)
    unknown = set(options) - {"operations", "extract_faces", "represent", "analyze"}
    if unknown:
        raise ValueError(f"Unknown DeepFace options: {sorted(unknown)}")
    operations = options.get("operations", ["detect", "represent", "analyze"])
    if set(operations) - {"detect", "represent", "analyze", "liveness"}:
        raise ValueError("DeepFace photo operations: detect, represent, analyze, liveness")
    _configure_tensorflow_gpu()
    sdk = SDKSession(TOOL_NAME)
    detection = dict(options.get("extract_faces", {}))
    detection.update(color_face="bgr", normalize_face=False)
    detection.setdefault("enforce_detection", True)
    if "liveness" in operations:
        detection["anti_spoofing"] = True
    try:
        extracted = _call_with_oom_retry(
            lambda: sdk.call("extract_faces", img_path=image_bgr, **detection)
        )
    except ValueError as exc:
        if "Face could not be detected" not in str(exc):
            raise
        extracted = []

    records = []
    for idx, detected in enumerate(extracted):
        crop = detected["face"]
        area = detected["facial_area"]
        record = {k: v for k, v in detected.items() if k != "face"}
        record.update(
            face_index=idx,
            bbox=[area["x"], area["y"], area["x"] + area["w"], area["y"] + area["h"]],
        )
        if "represent" in operations:
            kwargs = dict(options.get("represent", {}))
            kwargs.update(detector_backend="skip", align=False)
            represented = _call_with_oom_retry(
                lambda: sdk.call("represent", img_path=crop, **kwargs)
            )
            record["representations"] = [_crop_coordinates(item, crop) for item in represented]
            if represented:
                record["embedding"] = represented[0]["embedding"]
        if "analyze" in operations:
            kwargs = dict(options.get("analyze", {}))
            kwargs.update(detector_backend="skip", align=False)
            kwargs.setdefault("silent", True)
            analyzed = _call_with_oom_retry(
                lambda: sdk.call("analyze", img_path=crop, **kwargs)
            )
            attributes = analyzed[0] if isinstance(analyzed, list) and analyzed else analyzed
            record["attributes"] = _crop_coordinates(attributes, crop) if isinstance(attributes, dict) else attributes
            if isinstance(attributes, dict):
                aliases = {"dominant_emotion": "emotion_label", "emotion": "emotion_scores",
                           "age": "age", "dominant_gender": "gender", "gender": "gender_scores",
                           "dominant_race": "race", "race": "race_scores"}
                for key, alias in aliases.items():
                    if key in attributes:
                        record[alias] = attributes[key]
        records.append(record)
    return face_results_payload(encode_result(records), model=MODEL_NAME, extra={
        "sdk_version": provider_version(TOOL_NAME), "options": options,
        "face_index_source": TOOL_NAME, "operations": operations,
        "recognition_model": options.get("represent", {}).get("model_name", "VGG-Face"),
    })
