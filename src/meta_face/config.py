"""Configuration for meta-face."""

from __future__ import annotations

import os
from pathlib import Path

# Tool versions written into sidecar face.<tool>.version keys.
TOOL_VERSIONS: dict[str, str] = {
    "scrfd": "2.0.0",
    "arcface": "1.1.0",
    "dlib_detect": "2.1.0",
    "dlib_embed": "1.2.0",
    "cluster": "1.1.0",
    "cluster_dlib": "1.1.0",
    # Phase 1: emotion / expression (ONNX, low friction)
    "emotiefflib": "1.1.0",
    "opencv_fer": "1.1.0",
    "mediapipe_blendshapes": "1.1.0",
    "fer_plus": "1.1.0",
    # Phase 2: AU / gaze SDKs
    "libreface": "1.0.0",
    "openface3": "1.0.0",
    "yakhyo_gaze": "1.1.0",
    "l2cs_net": "1.0.0",
    # Phase 3: attributes, parsing, liveness, UniFace
    "fairface": "1.0.0",
    "bisenet": "1.1.0",
    "face_antispoof_onnx": "1.1.0",
    "face_anti_spoofing": "1.0.0",
    "uniface": "2.0.0",
    # Phase 4: heavier optional wrappers
    "py_feat": "2.0.0",
    "emonet": "1.0.0",
    "deepface": "2.0.0",
    "inspireface": "1.0.0",
}

# Crop-based analysis tools (require scrfd face detections).
ANALYSIS_TOOLS: frozenset[str] = frozenset(
    {
        "emotiefflib",
        "opencv_fer",
        "mediapipe_blendshapes",
        "fer_plus",
        "libreface",
        "openface3",
        "py_feat",
        "emonet",
        "deepface",
        "yakhyo_gaze",
        "l2cs_net",
        "fairface",
        "bisenet",
        "face_antispoof_onnx",
        "face_anti_spoofing",
        "uniface",
        "inspireface",
    }
)

# These SDKs detect their own faces; their face indices belong to their own namespace.
INDEPENDENT_ANALYSIS_TOOLS = frozenset({"deepface", "uniface", "py_feat"})
CROP_ANALYSIS_TOOLS = ANALYSIS_TOOLS - INDEPENDENT_ANALYSIS_TOOLS

# Per-image tools vs collection-level aggregate tools.
DETECTION_TOOLS: frozenset[str] = frozenset(
    {"scrfd", "arcface", "dlib_detect", "dlib_embed"}
)
PER_IMAGE_TOOL_ORDER: tuple[str, ...] = (
    "scrfd",
    "arcface",
    "dlib_detect",
    "dlib_embed",
) + tuple(sorted(ANALYSIS_TOOLS))
PER_IMAGE_TOOLS: frozenset[str] = frozenset(PER_IMAGE_TOOL_ORDER)
EMBEDDING_TOOLS: frozenset[str] = frozenset({"arcface", "dlib_embed"})
AGGREGATE_TOOLS: frozenset[str] = frozenset({"cluster", "cluster_dlib"})
ALL_TOOLS: frozenset[str] = PER_IMAGE_TOOLS | AGGREGATE_TOOLS

# One-line scan purpose for `mf tools` / --tools help.
TOOL_SCANS_FOR: dict[str, str] = {
    "scrfd": "Face boxes and 5-point landmarks (InsightFace)",
    "arcface": "512-d identity embeddings for SCRFD faces",
    "dlib_detect": "Face boxes and 68-point landmarks (dlib HOG/CNN)",
    "dlib_embed": "128-d identity embeddings for dlib faces",
    "cluster": "HDBSCAN groups over ArcFace embeddings (whole collection)",
    "cluster_dlib": "HDBSCAN groups over dlib embeddings (whole collection)",
    "emotiefflib": "7-class emotion on SCRFD crops",
    "opencv_fer": "7-class expression on SCRFD crops",
    "mediapipe_blendshapes": "Face mesh and 52 ARKit blendshapes on SCRFD crops",
    "fer_plus": "8-class FER+ emotion on SCRFD crops",
    "libreface": "Action units, emotion, and gaze on SCRFD crops",
    "openface3": "Action units, emotion, and gaze on SCRFD crops",
    "py_feat": "Own detect plus AU / emotion / identity columns",
    "emonet": "Valence and arousal on SCRFD crops",
    "deepface": "Own detect plus embeddings, attributes, optional liveness",
    "yakhyo_gaze": "Gaze yaw and pitch on SCRFD crops",
    "l2cs_net": "Gaze yaw and pitch (L2CS-Net) on SCRFD crops",
    "fairface": "Race / age / gender scores on SCRFD crops",
    "bisenet": "Face-part mask (hair, skin, eyes, mouth) on SCRFD crops",
    "face_antispoof_onnx": "Live vs spoof scores on SCRFD crops",
    "face_anti_spoofing": "Silent-Face live vs spoof scores on SCRFD crops",
    "uniface": "Own detect plus attributes, gaze, parsing, liveness",
    "inspireface": "Own detect plus optional attributes",
}
# Meta-tools for `mf scan` when --tools is omitted (expanded via tools/registry).
DEFAULT_SCAN_META_TOOLS: tuple[str, ...] = (
    "insightface",
    "face_recognition",
)
DEFAULT_TOOLS: tuple[str, ...] = (
    "scrfd",
    "arcface",
    "dlib_detect",
    "dlib_embed",
)

# insightface model pack (SCRFD + ArcFace).
INSIGHTFACE_MODEL: str = os.environ.get("META_FACE_MODEL", "buffalo_l")
INSIGHTFACE_CTX_ID: int = int(os.environ.get("META_FACE_GPU_ID", "0"))
# Root for downloaded model packs; shared by the downloader and inference.
INSIGHTFACE_ROOT: str = os.environ.get(
    "META_FACE_INSIGHTFACE_ROOT",
    str(Path.home() / ".insightface"),
)
# ONNX Runtime: CUDA only. Never list CPUExecutionProvider or the CPU pip package.
ONNX_PROVIDERS: tuple[str, ...] = ("CUDAExecutionProvider",)

# face_recognition / dlib (CPU-oriented, deprecated but supported).
DLIB_MODEL: str = os.environ.get("META_FACE_DLIB_MODEL", "hog")
DLIB_ROOT: str = os.environ.get(
    "META_FACE_DLIB_ROOT",
    str(Path.home() / ".meta_face" / "dlib"),
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
RQ_SCAN_QUEUE_NAME: str = os.environ.get("META_FACE_SCAN_QUEUE", "meta-face-scan")
RQ_CLUSTER_QUEUE_NAME: str = os.environ.get("META_FACE_CLUSTER_QUEUE", "meta-face-cluster")
RQ_JOB_TIMEOUT: int = int(os.environ.get("META_FACE_JOB_TIMEOUT", "3600"))
RQ_DETECT_JOB_TIMEOUT: int = int(os.environ.get("META_FACE_DETECT_JOB_TIMEOUT", "600"))
RQ_ANALYSIS_JOB_TIMEOUT: int = int(os.environ.get("META_FACE_ANALYSIS_JOB_TIMEOUT", "900"))
RQ_MEDIAPIPE_JOB_TIMEOUT: int = int(os.environ.get("META_FACE_MEDIAPIPE_JOB_TIMEOUT", "1800"))
# How long a writer waits for `{sidecar}.lock` before failing. The lock file
# exists only while a write is in progress (sidecar-rs >= 0.2.4).
SIDECAR_LOCK_TIMEOUT_SECS: float = float(os.environ.get("META_FACE_SIDECAR_LOCK_TIMEOUT", "10"))


def rq_job_timeout(backend_key: str) -> int:
    """Per-pipeline RQ timeout so one slow tool does not share another job's budget."""
    key = backend_key.strip().lower()
    if key in {"insightface", "face_recognition", "scrfd", "arcface", "dlib_detect", "dlib_embed", "annotate"}:
        return RQ_DETECT_JOB_TIMEOUT
    if key in {"mediapipe_blendshapes", "mediapipe"}:
        return RQ_MEDIAPIPE_JOB_TIMEOUT
    if key in ANALYSIS_TOOLS or key == "sdk":
        return RQ_ANALYSIS_JOB_TIMEOUT
    return RQ_JOB_TIMEOUT

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


def normalize_embedding_tool(name: str) -> str:
    """Validate and normalize an embedding tool name."""
    key = name.strip().lower()
    if key not in EMBEDDING_TOOLS:
        valid = ", ".join(sorted(EMBEDDING_TOOLS))
        raise ValueError(f"Unknown embedding tool: {name}. Valid: {valid}")
    return key


def analysis_models_dir() -> Path:
    """Directory for downloaded ONNX / MediaPipe analysis model weights."""
    return DATA_DIR / "analysis_models"


def opencv_fer_model_path() -> Path:
    return analysis_models_dir() / "opencv_facial_expression_recognition.onnx"


def fer_plus_model_path() -> Path:
    return analysis_models_dir() / "emotion-ferplus-8.onnx"


def mediapipe_model_path() -> Path:
    return analysis_models_dir() / "face_landmarker.task"


def fairface_model_path() -> Path:
    return analysis_models_dir() / "fairface.onnx"


def bisenet_model_path() -> Path:
    return analysis_models_dir() / "bisenet_face_parsing.onnx"


def yakhyo_gaze_model_path() -> Path:
    return analysis_models_dir() / "yakhyo_gaze.onnx"


def face_antispoof_onnx_model_path() -> Path:
    return analysis_models_dir() / "face_antispoof.onnx"
