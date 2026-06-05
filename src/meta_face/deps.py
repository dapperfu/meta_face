"""Runtime dependency checks with actionable error messages."""

from __future__ import annotations

from meta_face.tools.registry import (
    detectron2_tools_requested,
    dlib_tools_requested,
    insightface_tools_requested,
)


class PipelineDependencyError(RuntimeError):
    """A required pipeline dependency is missing or broken."""


def require_insightface_runtime() -> None:
    """Ensure onnxruntime and insightface are importable before InsightFace jobs run."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise PipelineDependencyError(
            "onnxruntime is not installed. Install the project with GPU extras: "
            "pip install -e '.[dev]' or pip install onnxruntime-gpu>=1.17"
        ) from None

    if not hasattr(ort, "InferenceSession"):
        module_path = getattr(ort, "__file__", "unknown")
        raise PipelineDependencyError(
            "onnxruntime is installed but unusable (missing InferenceSession). "
            f"Loaded module: {module_path}. "
            "Check for a local file named onnxruntime.py shadowing the package, "
            "then reinstall: pip uninstall -y onnxruntime onnxruntime-gpu && "
            "pip install onnxruntime-gpu>=1.17"
        ) from None

    try:
        import insightface  # noqa: F401
    except ImportError:
        raise PipelineDependencyError(
            "insightface is not installed. Install the project dependencies: "
            "pip install -e ."
        ) from None
    except AttributeError as exc:
        raise PipelineDependencyError(
            "insightface failed to import because onnxruntime is broken. "
            f"Original error: {exc}"
        ) from None


def require_dlib_runtime() -> None:
    """Ensure face_recognition (dlib) is importable before dlib pipeline jobs run."""
    try:
        import face_recognition  # noqa: F401
    except ImportError:
        raise PipelineDependencyError(
            "face_recognition is not installed. Install the project dependencies: "
            "pip install -e .\n"
            "If dlib failed to build, install system packages first: "
            "sudo apt install cmake build-essential"
        ) from None
    except Exception as exc:
        raise PipelineDependencyError(
            "face_recognition failed to import (dlib may not be compiled). "
            f"Original error: {exc}\n"
            "Install build tools: sudo apt install cmake build-essential, then "
            "pip install --force-reinstall face_recognition"
        ) from None


def require_detectron2_runtime() -> None:
    """Ensure detectron2, torch, and model weights are available."""
    try:
        import detectron2  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        raise PipelineDependencyError(
            "detectron2 is not installed. Install optional extras:\n"
            "  pip install -e \".[detectron2]\"\n"
            "Then install detectron2 for your CUDA/torch build:\n"
            "  https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md"
        ) from None

    from meta_face.models import is_detectron2_available

    if not is_detectron2_available():
        raise PipelineDependencyError(
            "Detectron2 model files are missing. Download weights:\n"
            "  mf download --backend detectron2"
        )


def require_inference_runtime(tools: list[str] | None = None) -> None:
    """Ensure backends needed for the requested per-image tools are available."""
    if tools is None:
        require_insightface_runtime()
        require_dlib_runtime()
        return

    if insightface_tools_requested(tools):
        require_insightface_runtime()
    if dlib_tools_requested(tools):
        require_dlib_runtime()
    if detectron2_tools_requested(tools):
        require_detectron2_runtime()


def require_cluster_runtime() -> None:
    """Ensure FAISS and HDBSCAN are importable before cluster jobs run."""
    try:
        import faiss  # noqa: F401
    except ImportError:
        raise PipelineDependencyError(
            "faiss is not installed. Install the project dependencies: "
            "pip install -e . (requires faiss-gpu-cu12)"
        ) from None

    try:
        import hdbscan  # noqa: F401
    except ImportError:
        raise PipelineDependencyError(
            "hdbscan is not installed. Install the project dependencies: "
            "pip install -e ."
        ) from None
