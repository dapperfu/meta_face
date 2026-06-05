"""Download ONNX and MediaPipe weights for analysis tools."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

from meta_face.config import (
    analysis_models_dir,
    bisenet_model_path,
    face_antispoof_onnx_model_path,
    fairface_model_path,
    fer_plus_model_path,
    mediapipe_model_path,
    opencv_fer_model_path,
    yakhyo_gaze_model_path,
)

logger = logging.getLogger(__name__)

# Public model URLs (best-effort; may require manual download if URLs change).
_MODEL_URLS: dict[str, tuple[Path, str]] = {
    "opencv_fer": (
        opencv_fer_model_path(),
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "facial_expression_recognition/"
        "facial_expression_recognition_mobilefacenet_2022july.onnx",
    ),
    "fer_plus": (
        fer_plus_model_path(),
        "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/"
        "emotion_ferplus/model/emotion-ferplus-8.onnx",
    ),
    "mediapipe": (
        mediapipe_model_path(),
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
    ),
    "fairface": (
        fairface_model_path(),
        "https://github.com/yakhyo/fairface-onnx/releases/download/weights/fairface.onnx",
    ),
    "bisenet": (
        bisenet_model_path(),
        "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx",
    ),
    "yakhyo_gaze": (
        yakhyo_gaze_model_path(),
        "https://github.com/yakhyo/gaze-estimation/releases/download/weights/"
        "resnet34_gaze.onnx",
    ),
    "face_antispoof_onnx": (
        face_antispoof_onnx_model_path(),
        "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/"
        "MiniFASNetV2.onnx",
    ),
}


class AnalysisModelDownloadError(RuntimeError):
    """Raised when one or more analysis model downloads fail."""

    def __init__(self, failures: list[str], paths: dict[str, Path]) -> None:
        self.failures = failures
        self.paths = paths
        summary = "\n".join(f"  - {item}" for item in failures)
        super().__init__(f"Failed to download {len(failures)} analysis model(s):\n{summary}")


def _download_url(url: str, dest: Path, *, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not force:
        return dest
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {dest.name} from {url}. "
            f"Place the file manually at {dest}. Error: {exc}"
        ) from exc
    return dest


def download_analysis_model(name: str, *, force: bool = False) -> Path:
    """Download a single analysis model by backend key."""
    key = name.strip().lower()
    if key not in _MODEL_URLS:
        known = ", ".join(sorted(_MODEL_URLS))
        raise ValueError(f"Unknown analysis model '{name}'. Known: {known}")
    dest, url = _MODEL_URLS[key]
    return _download_url(url, dest, force=force)


def download_all_analysis_models(*, force: bool = False) -> dict[str, Path]:
    """Download all registered analysis ONNX/MediaPipe models.

    Continues after individual failures and raises ``AnalysisModelDownloadError``
    at the end with a summary when any download fails. Successfully downloaded
    models are returned in ``exc.paths`` on error.
    """
    paths: dict[str, Path] = {}
    failures: list[str] = []
    for key in sorted(_MODEL_URLS):
        try:
            paths[key] = download_analysis_model(key, force=force)
        except RuntimeError as exc:
            logger.warning("Failed to download analysis model %s: %s", key, exc)
            failures.append(f"{key}: {exc}")
    if failures:
        raise AnalysisModelDownloadError(failures, paths)
    return paths


def is_analysis_model_available(name: str) -> bool:
    key = name.strip().lower()
    if key not in _MODEL_URLS:
        return False
    dest, _ = _MODEL_URLS[key]
    return dest.is_file()


def analysis_models_root() -> Path:
    return analysis_models_dir()
