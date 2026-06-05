"""Face model weight management (insightface, dlib, and detectron2).

Downloading weights is a standalone step so model packs can be fetched
explicitly (for example before starting GPU workers) instead of lazily on the
first inference call.
"""

from __future__ import annotations

import shutil
import urllib.request
from importlib.resources import files as pkg_files
from pathlib import Path

from meta_face.config import (
    DLIB_ROOT,
    DETECTRON2_CONFIG_PATH,
    DETECTRON2_WEIGHTS_PATH,
    INSIGHTFACE_MODEL,
    INSIGHTFACE_ROOT,
    detectron2_dir,
    insightface_model_dir,
)

# WIDER FACE RetinaNet R50 weights (Detectron2). Override via META_FACE_DETECTRON2_WEIGHTS.
DETECTRON2_WEIGHTS_URL = (
    "https://github.com/akanametov/face-detection-detectron2/"
    "releases/download/v0.1.0/model_final.pth"
)


def model_dir(name: str | None = None) -> Path:
    """Return the directory where the given insightface model pack lives."""
    return insightface_model_dir(name)


def is_available(name: str | None = None) -> bool:
    """True when the insightface model pack directory exists with ONNX weights."""
    path = model_dir(name)
    return path.is_dir() and any(path.glob("*.onnx"))


def download(name: str | None = None, *, force: bool = False) -> Path:
    """Download and unzip an insightface model pack; return its directory."""
    from insightface.utils import storage

    pack = name or INSIGHTFACE_MODEL
    dir_path = storage.download("models", pack, force=force, root=INSIGHTFACE_ROOT)
    return Path(dir_path)


def dlib_model_dir() -> Path:
    """Directory for optional dlib model file copies."""
    return Path(DLIB_ROOT)


def is_dlib_available() -> bool:
    """True when face_recognition_models provides required .dat files."""
    try:
        import face_recognition_models
    except ImportError:
        return False

    pkg_dir = Path(face_recognition_models.__path__[0])
    required = (
        "shape_predictor_68_face_landmarks.dat",
        "dlib_face_recognition_resnet_model_v1.dat",
    )
    return all((pkg_dir / name).is_file() for name in required)


def download_dlib_models(*, force: bool = False) -> Path:
    """Verify dlib models via face_recognition_models; optionally mirror to DLIB_ROOT."""
    if not is_dlib_available():
        raise RuntimeError(
            "face_recognition_models is missing or incomplete. "
            "Reinstall: pip install --force-reinstall face_recognition"
        )

    import face_recognition_models

    dest = dlib_model_dir()
    if force or not dest.is_dir() or not any(dest.glob("*.dat")):
        dest.mkdir(parents=True, exist_ok=True)
        pkg_dir = Path(face_recognition_models.__path__[0])
        for dat_file in pkg_dir.glob("*.dat"):
            link = dest / dat_file.name
            if link.exists() and not force:
                continue
            if link.exists():
                link.unlink()
            link.symlink_to(dat_file)

    return dest


def bundled_detectron2_config() -> Path:
    """Packaged RetinaNet config shipped with meta_face."""
    return Path(pkg_files("meta_face.data.detectron2") / "retinanet_wider_face.yaml")


def is_detectron2_available() -> bool:
    """True when Detectron2 config and weight files exist."""
    return DETECTRON2_CONFIG_PATH.is_file() and DETECTRON2_WEIGHTS_PATH.is_file()


def download_detectron2_weights(*, force: bool = False) -> Path:
    """Install config under META_FACE_DATA and download model_final.pth."""
    detectron2_dir()
    config_dest = DETECTRON2_CONFIG_PATH
    weights_dest = DETECTRON2_WEIGHTS_PATH

    if force or not config_dest.is_file():
        bundled = bundled_detectron2_config()
        shutil.copy2(bundled, config_dest)

    if force or not weights_dest.is_file():
        weights_dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(DETECTRON2_WEIGHTS_URL, weights_dest)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to download Detectron2 weights from {DETECTRON2_WEIGHTS_URL}. "
                f"Place model_final.pth at {weights_dest} or set META_FACE_DETECTRON2_WEIGHTS. "
                f"Original error: {exc}"
            ) from exc

    if not weights_dest.is_file() or weights_dest.stat().st_size < 1024:
        raise RuntimeError(
            f"Detectron2 weights at {weights_dest} are missing or too small. "
            "Download manually or retry: mf download --backend detectron2"
        )

    return weights_dest


def download_all(*, insightface_model: str | None = None, force: bool = False) -> dict[str, Path]:
    """Download/verify all backend model weights."""
    paths: dict[str, Path] = {
        "insightface": download(insightface_model, force=force),
        "dlib": download_dlib_models(force=force),
    }
    try:
        import detectron2  # noqa: F401
    except ImportError:
        return paths
    try:
        paths["detectron2"] = download_detectron2_weights(force=force)
    except RuntimeError:
        pass
    return paths
