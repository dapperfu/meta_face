"""Configuration for meta-face."""

from __future__ import annotations

import os
from pathlib import Path

# Tool versions written into sidecar face.<tool>.version keys.
TOOL_VERSIONS: dict[str, str] = {
    "scrfd": "1.0.0",
    "arcface": "1.0.0",
    "cluster": "1.0.0",
}

# Per-image tools vs collection-level aggregate tools.
PER_IMAGE_TOOLS: frozenset[str] = frozenset({"scrfd", "arcface"})
AGGREGATE_TOOLS: frozenset[str] = frozenset({"hdbscan", "cluster"})
ALL_TOOLS: frozenset[str] = PER_IMAGE_TOOLS | AGGREGATE_TOOLS
DEFAULT_TOOLS: tuple[str, ...] = ("scrfd", "arcface")

# insightface model pack (SCRFD + ArcFace).
INSIGHTFACE_MODEL: str = os.environ.get("META_FACE_MODEL", "buffalo_l")
INSIGHTFACE_CTX_ID: int = int(os.environ.get("META_FACE_GPU_ID", "0"))
# Root for downloaded model packs; shared by the downloader and inference.
INSIGHTFACE_ROOT: str = os.environ.get(
    "META_FACE_INSIGHTFACE_ROOT",
    str(Path.home() / ".insightface"),
)

# Redis / RQ
REDIS_HOST: str = os.environ.get("META_FACE_REDIS_HOST", "127.0.0.1")
REDIS_PORT: int = int(os.environ.get("META_FACE_REDIS_PORT", "26379"))
REDIS_DB: int = int(os.environ.get("META_FACE_REDIS_DB", "0"))
REDIS_URL: str = os.environ.get(
    "META_FACE_REDIS_URL",
    f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
)
RQ_QUEUE_NAME: str = os.environ.get("META_FACE_QUEUE", "meta-face")
RQ_CLUSTER_QUEUE_NAME: str = os.environ.get("META_FACE_CLUSTER_QUEUE", "meta-face-cluster")
RQ_JOB_TIMEOUT: int = int(os.environ.get("META_FACE_JOB_TIMEOUT", "3600"))

# Local data directory for FAISS index and metadata sidecar files.
DATA_DIR: Path = Path(os.environ.get("META_FACE_DATA", Path.home() / ".meta_face"))
FAISS_INDEX_PATH: Path = DATA_DIR / "faces.faiss"
FAISS_META_PATH: Path = DATA_DIR / "faces.faiss.meta"

# Supported image extensions (lowercase, with leading dot).
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".heic",
        ".heif",
    }
)

# Sidecar key prefixes
FACE_KEY_PREFIX: str = "face."


def ensure_data_dir() -> Path:
    """Create the data directory if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def insightface_model_dir(name: str | None = None) -> Path:
    """Directory where an insightface model pack lives (may not yet exist)."""
    root = Path(os.path.expanduser(INSIGHTFACE_ROOT))
    return root / "models" / (name or INSIGHTFACE_MODEL)


def tool_version_key(tool: str) -> str:
    return f"{FACE_KEY_PREFIX}{tool}.version"


def tool_processed_at_key(tool: str) -> str:
    return f"{FACE_KEY_PREFIX}{tool}.processed_at"


def tool_data_key(tool: str, field: str) -> str:
    return f"{FACE_KEY_PREFIX}{tool}.{field}"
