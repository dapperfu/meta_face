"""Configuration for meta-face."""

from __future__ import annotations

import os
from pathlib import Path

# Tool versions written into sidecar face.<tool>.version keys.
TOOL_VERSIONS: dict[str, str] = {
    "scrfd": "1.0.0",
    "arcface": "1.0.0",
    "detectron2": "1.0.0",
    "dlib_detect": "1.0.0",
    "dlib_embed": "1.0.0",
    "cluster": "1.0.0",
    "cluster_dlib": "1.0.0",
}

# Per-image tools vs collection-level aggregate tools.
PER_IMAGE_TOOLS: frozenset[str] = frozenset(
    {"scrfd", "arcface", "detectron2", "dlib_detect", "dlib_embed"}
)
EMBEDDING_TOOLS: frozenset[str] = frozenset({"arcface", "dlib_embed"})
AGGREGATE_TOOLS: frozenset[str] = frozenset({"cluster", "cluster_dlib"})
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

# face_recognition / dlib (CPU-oriented, deprecated but supported).
DLIB_MODEL: str = os.environ.get("META_FACE_DLIB_MODEL", "hog")
DLIB_ROOT: str = os.environ.get(
    "META_FACE_DLIB_ROOT",
    str(Path.home() / ".meta_face" / "dlib"),
)

# Detectron2 (optional extra; face RetinaNet on WIDER FACE).
DETECTRON2_SCORE_THRESH: float = float(
    os.environ.get("META_FACE_DETECTRON2_SCORE_THRESH", "0.5")
)
DETECTRON2_DEVICE: str = os.environ.get("META_FACE_DETECTRON2_DEVICE", "cuda:0")

# Redis / RQ
REDIS_HOST: str = os.environ.get("META_FACE_REDIS_HOST", "127.0.0.1")
REDIS_PORT: int = int(os.environ.get("META_FACE_REDIS_PORT", "26379"))
REDIS_DB: int = int(os.environ.get("META_FACE_REDIS_DB", "0"))
REDIS_URL: str = os.environ.get(
    "META_FACE_REDIS_URL",
    f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
)
RQ_QUEUE_NAME: str = os.environ.get("META_FACE_QUEUE", "meta-face")
RQ_SCAN_QUEUE_NAME: str = os.environ.get("META_FACE_SCAN_QUEUE", "meta-face-scan")
RQ_CLUSTER_QUEUE_NAME: str = os.environ.get("META_FACE_CLUSTER_QUEUE", "meta-face-cluster")
RQ_JOB_TIMEOUT: int = int(os.environ.get("META_FACE_JOB_TIMEOUT", "3600"))

# Local data directory for FAISS index and metadata sidecar files.
DATA_DIR: Path = Path(os.environ.get("META_FACE_DATA", Path.home() / ".meta_face"))
# Legacy single-index paths (arcface); prefer faiss_index_path() for new code.
FAISS_INDEX_PATH: Path = DATA_DIR / "faces.arcface.faiss"
FAISS_META_PATH: Path = DATA_DIR / "faces.arcface.faiss.meta"

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

# Annotated image output suffix (photo.jpg -> photo_scrfd.jpg).
ANNOTATE_OUTPUT_SUFFIX: str = "scrfd"

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


def faiss_index_path(embedding_tool: str) -> Path:
    """FAISS index path for a given embedding tool (arcface or dlib_embed)."""
    return DATA_DIR / f"faces.{embedding_tool}.faiss"


def faiss_meta_path(embedding_tool: str) -> Path:
    """FAISS metadata JSON path for a given embedding tool."""
    return DATA_DIR / f"faces.{embedding_tool}.faiss.meta"


def cluster_tool_for_embedding(embedding_tool: str) -> str:
    """Sidecar cluster tool name for an embedding source."""
    if embedding_tool == "arcface":
        return "cluster"
    if embedding_tool == "dlib_embed":
        return "cluster_dlib"
    raise ValueError(f"Unknown embedding tool: {embedding_tool}")


def detectron2_dir() -> Path:
    """Directory for Detectron2 config and downloaded weights."""
    path = ensure_data_dir() / "detectron2"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_detectron2_config_path() -> Path:
    return detectron2_dir() / "retinanet_wider_face.yaml"


def default_detectron2_weights_path() -> Path:
    return detectron2_dir() / "model_final.pth"


DETECTRON2_CONFIG_PATH: Path = Path(
    os.environ.get("META_FACE_DETECTRON2_CONFIG", str(default_detectron2_config_path()))
)
DETECTRON2_WEIGHTS_PATH: Path = Path(
    os.environ.get("META_FACE_DETECTRON2_WEIGHTS", str(default_detectron2_weights_path()))
)


def normalize_embedding_tool(name: str) -> str:
    """Validate and normalize an embedding tool name."""
    key = name.strip().lower()
    if key not in EMBEDDING_TOOLS:
        valid = ", ".join(sorted(EMBEDDING_TOOLS))
        raise ValueError(f"Unknown embedding tool: {name}. Valid: {valid}")
    return key
