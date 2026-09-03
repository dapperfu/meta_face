"""Face model weight management (insightface and dlib).

Downloading weights is a standalone step so model packs can be fetched
explicitly (for example before starting GPU workers) instead of lazily on the
first inference call.
"""

from __future__ import annotations

from pathlib import Path

from meta_face.config import (
    DLIB_ROOT,
    INSIGHTFACE_MODEL,
    INSIGHTFACE_ROOT,
    insightface_model_dir,
)

__all__ = [
    "download",
    "download_all",
    "download_dlib_models",
    "is_available",
    "is_dlib_available",
    "model_dir",
]


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


def download_all(*, insightface_model: str | None = None, force: bool = False) -> dict[str, Path]:
    """Download/verify all backend model weights."""
    return {
        "insightface": download(insightface_model, force=force),
        "dlib": download_dlib_models(force=force),
    }
