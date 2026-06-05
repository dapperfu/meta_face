"""Download ONNX and MediaPipe weights for analysis tools."""

from __future__ import annotations

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

# Public model URLs (best-effort; may require manual download if URLs change).
_MODEL_URLS: dict[str, tuple[Path, str]] = {
    "opencv_fer": (
        opencv_fer_model_path(),
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
        "facial_expression_recognition/facial_expression_recognition.onnx",
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
        "https://github.com/dchen236/FairFace/raw/master/fairface.onnx",
    ),
    "bisenet": (
        bisenet_model_path(),
        "https://github.com/yakhyo/face-parsing/releases/download/v1.0/face_parsing.onnx",
    ),
    "yakhyo_gaze": (
        yakhyo_gaze_model_path(),
        "https://github.com/yakhyo/gaze-estimation/releases/download/v1.0/gaze.onnx",
    ),
    "face_antispoof_onnx": (
        face_antispoof_onnx_model_path(),
        "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/releases/download/v1.0/"
        "anti_spoof.onnx",
    ),
}


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
    """Download all registered analysis ONNX/MediaPipe models."""
    paths: dict[str, Path] = {}
    for key in sorted(_MODEL_URLS):
        paths[key] = download_analysis_model(key, force=force)
    return paths


def is_analysis_model_available(name: str) -> bool:
    key = name.strip().lower()
    if key not in _MODEL_URLS:
        return False
    dest, _ = _MODEL_URLS[key]
    return dest.is_file()


def analysis_models_root() -> Path:
    return analysis_models_dir()
