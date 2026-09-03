"""UniFace 4 detection, recognition and all static facial analysis heads.

Tracking, stores, matting and anonymization are available through mf sdk.

Analysis heads are constructed one at a time so a GPU OOM in HeadPose does not
drop the detector, embeddings, or the other heads. Failed heads are stored as
``{"error": "..."}`` and the sidecar is still written.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

from meta_face.bbox import crop_face
from meta_face.config import ONNX_PROVIDERS
from meta_face.sdk import (
    SDKSession,
    encode_result,
    provider_issue,
    provider_options,
    provider_version,
)
from meta_face.tools.analysis.base import FaceContext, face_results_payload

TOOL_NAME = "uniface"
TOOL_VERSION = "2.0.0"
MODEL_NAME = "uniface"
PHOTO_ANALYSES = (
    "AgeGender", "FairFace", "Emotion", "FaceAttribNet", "Landmark106", "PIPNet",
    "FaceMesh", "MobileGaze", "HeadPose", "EDifFIQA", "MiniFASNet", "BiSeNet", "XSeg",
)

logger = logging.getLogger(__name__)


def _is_gpu_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "failed to allocate memory",
            "bfcarena",
            "out of memory",
            "cuda_error_out_of_memory",
            "cublas_status_alloc_failed",
        )
    )


def _inference_error_types() -> tuple[type[BaseException], ...]:
    types: tuple[type[BaseException], ...] = (RuntimeError, ValueError, OSError)
    try:
        from onnxruntime.capi.onnxruntime_pybind11_state import Fail
    except ImportError:
        return types
    return (*types, Fail)


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except (ImportError, AttributeError, RuntimeError):
        return


def _drop(model: Any) -> None:
    if hasattr(model, "session"):
        model.session = None
    _release_cuda()


def _error_payload(exc: BaseException) -> dict[str, str]:
    return {"error": f"{type(exc).__name__}: {exc}"}


def _build_component(sdk: SDKSession, spec: str | dict[str, Any]) -> Any:
    """Construct a UniFace class on CUDA only (no CPUExecutionProvider fallback)."""
    if isinstance(spec, str):
        return sdk.call(spec, providers=list(ONNX_PROVIDERS))
    kwargs = dict(sdk.resolve(spec.get("kwargs", {})))
    kwargs.setdefault("providers", list(ONNX_PROVIDERS))
    return sdk.call(spec["class"], **kwargs)


def _run_named_analysis(name: str, model: Any, image_bgr: Any, face: Any, calls: dict[str, Any]) -> Any:
    kwargs = calls.get(name, {})
    if name in {"AgeGender", "FairFace", "Emotion", "FaceAttribNet"}:
        return model.predict(image_bgr, face, **kwargs)
    if name in {"Landmark106", "PIPNet"}:
        return model.get_landmarks(image_bgr, face.bbox, **kwargs)
    if name == "FaceMesh":
        return model.predict(image_bgr, faces=[face], **kwargs)
    if name in {"MobileGaze", "HeadPose"}:
        try:
            crop = crop_face(image_bgr, face.bbox, buffer_pct=0)
        except ValueError as exc:
            raise ValueError(f"{name} face crop is empty") from exc
        if crop.size == 0 or min(crop.shape[:2]) < 2:
            raise ValueError(f"{name} face crop is empty")
        return model.estimate(crop, **kwargs)
    if name == "EDifFIQA":
        return model.predict(image_bgr, face.landmarks, **kwargs)
    if name == "MiniFASNet":
        return model.predict(image_bgr, face.bbox, **kwargs)
    return model.parse(image_bgr, landmarks=face.landmarks, **kwargs)


def _call_with_oom_retry(operation: Any) -> Any:
    try:
        return operation()
    except _inference_error_types() as exc:
        if not _is_gpu_oom(exc):
            raise
        logger.warning("UniFace GPU OOM; releasing CUDA memory and retrying once")
        _release_cuda()
        return operation()


def availability() -> str | None:
    issue = provider_issue(TOOL_NAME)
    installed = provider_version(TOOL_NAME)
    if issue is None and installed and int(installed.split(".")[0]) < 4:
        return "UniFace 4 is required. Install: pip install -e '.[dev]'"
    return issue


def analyze_faces(image_bgr: Any, faces: list[FaceContext]) -> dict[str, Any]:
    options = provider_options(TOOL_NAME)
    unknown = set(options) - {"detector", "recognizer", "analyses", "models", "calls"}
    if unknown:
        raise ValueError(f"Unknown UniFace options: {sorted(unknown)}")
    sdk = SDKSession(TOOL_NAME)
    calls = options.get("calls", {})
    names = options.get("analyses", list(PHOTO_ANALYSES))
    if set(names) - set(PHOTO_ANALYSES):
        raise ValueError(f"Unknown UniFace photo analysis: {set(names) - set(PHOTO_ANALYSES)}")
    configs = options.get("models", {})

    detector = _call_with_oom_retry(lambda: _build_component(sdk, options.get("detector", "SCRFD")))
    detector_class = type(detector).__name__
    detections = detector.detect(image_bgr, **calls.get("detect", {}))
    _drop(detector)
    detector = None

    records: list[dict[str, Any]] = []
    for idx, face in enumerate(detections):
        record = encode_result(face)
        record.update(face_index=idx, analyses={})
        records.append(record)

    spec = options.get("recognizer", "ArcFace")
    recognizer_class = None
    if spec is not None:
        recognizer = None
        try:
            recognizer = _call_with_oom_retry(lambda: _build_component(sdk, spec))
            recognizer_class = type(recognizer).__name__
            for idx, face in enumerate(detections):
                try:
                    embedding = _call_with_oom_retry(
                        lambda f=face: recognizer.get_normalized_embedding(image_bgr, f.landmarks)
                    )
                    records[idx]["embedding"] = encode_result(embedding)
                except _inference_error_types() as exc:
                    logger.warning("UniFace embedding failed for face %s: %s", idx, exc)
                    records[idx]["embedding"] = _error_payload(exc)
        except _inference_error_types() as exc:
            logger.warning("UniFace recognizer failed: %s", exc)
            for record in records:
                record["embedding"] = _error_payload(exc)
        finally:
            if recognizer is not None:
                _drop(recognizer)
                recognizer = None

    failed_analyses: list[str] = []
    for name in names:
        model = None
        try:
            model = _call_with_oom_retry(
                lambda n=name: _build_component(sdk, {"class": n, "kwargs": configs.get(n, {})})
            )
            for idx, face in enumerate(detections):
                try:
                    result = _call_with_oom_retry(
                        lambda n=name, m=model, f=face: _run_named_analysis(n, m, image_bgr, f, calls)
                    )
                    records[idx]["analyses"][name] = encode_result(result)
                except _inference_error_types() as exc:
                    logger.warning("UniFace %s failed for face %s: %s", name, idx, exc)
                    records[idx]["analyses"][name] = _error_payload(exc)
                    if name not in failed_analyses:
                        failed_analyses.append(name)
        except _inference_error_types() as exc:
            logger.warning("UniFace %s failed to load: %s", name, exc)
            payload = _error_payload(exc)
            for record in records:
                record["analyses"][name] = payload
            failed_analyses.append(name)
        finally:
            if model is not None:
                _drop(model)
                model = None

    extra: dict[str, Any] = {
        "sdk_version": provider_version(TOOL_NAME),
        "options": options,
        "face_index_source": TOOL_NAME,
        "analyses": list(names),
        "detector_class": detector_class,
        "recognizer_class": recognizer_class,
    }
    if failed_analyses:
        extra["failed_analyses"] = failed_analyses
    return face_results_payload(records, model=MODEL_NAME, extra=extra)
