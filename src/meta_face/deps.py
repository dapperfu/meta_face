"""Runtime dependency checks with actionable error messages."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import time
from importlib.metadata import distribution

from meta_face.tools.registry import (
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
            "onnxruntime-gpu is not installed. Install: pip install -e '.[dev]' "
            "or pip install 'onnxruntime-gpu[cuda,cudnn]>=1.17'"
        ) from None

    if not hasattr(ort, "InferenceSession"):
        module_path = getattr(ort, "__file__", "unknown")
        raise PipelineDependencyError(
            "onnxruntime-gpu is installed but unusable (missing InferenceSession). "
            f"Loaded module: {module_path}. "
            "Check for a local file named onnxruntime.py shadowing the package, "
            "then reinstall: pip uninstall -y onnxruntime && "
            "pip install 'onnxruntime-gpu[cuda,cudnn]>=1.17'"
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
    """Ensure dlib HOG models load without importing face_recognition's CUDA CNN."""
    try:
        import dlib  # noqa: F401
        import face_recognition_models  # noqa: F401
    except ImportError:
        raise PipelineDependencyError(
            "dlib/face_recognition_models are not installed. Install the project "
            "dependencies: pip install -e ."
        ) from None
    from meta_face.config import DLIB_MODEL

    if DLIB_MODEL != "cnn":
        return
    try:
        import face_recognition  # noqa: F401
    except ImportError:
        raise PipelineDependencyError(
            "face_recognition is required when META_FACE_DLIB_MODEL=cnn. "
            "Install the project dependencies: pip install -e ."
        ) from None


def adjust_per_image_tools_for_runtime(
    per_image_tools: list[str],
    *,
    analysis_explicit: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Drop unavailable tools when not explicitly requested; warn otherwise."""
    warnings: list[str] = []
    result = list(per_image_tools)
    analysis_explicit = analysis_explicit or set()

    from meta_face.config import ANALYSIS_TOOLS
    from meta_face.tools.analysis.registry import tool_availability

    for tool in list(result):
        if tool not in ANALYSIS_TOOLS:
            continue
        issue = tool_availability(tool)
        if issue is None:
            continue
        if tool in analysis_explicit:
            raise PipelineDependencyError(issue)
        warnings.append(f"Skipping {tool} (not available):\n{issue}")
        result = [name for name in result if name != tool]

    return result, warnings


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
    from meta_face.config import CROP_ANALYSIS_TOOLS
    from meta_face.tools.registry import validate_tools

    if set(validate_tools(tools)) & CROP_ANALYSIS_TOOLS:
        require_insightface_runtime()


def _debug_log_faiss(hypothesis_id: str, message: str, data: dict) -> None:
    # #region agent log
    try:
        with open(
            "/projects/spring_photography/meta_face/.cursor/debug-d5505a.log",
            "a",
            encoding="utf-8",
        ) as fh:
            fh.write(
                json.dumps(
                    {
                        "sessionId": "d5505a",
                        "runId": "pre-fix",
                        "hypothesisId": hypothesis_id,
                        "location": "deps.py:require_cluster_runtime",
                        "message": message,
                        "data": data,
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except OSError:
        pass
    # #endregion


def _cublaslt_has_env_mode_symbol(lib_path: str) -> bool | None:
    try:
        out = subprocess.check_output(["nm", "-D", lib_path], stderr=subprocess.STDOUT, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return "cublasLtGetEnvironmentMode" in out


def _preload_faiss_cublaslt() -> str | None:
    """Load venv libcublasLt before faiss so system CUDA in LD_LIBRARY_PATH cannot shadow it."""
    for file in distribution("nvidia-cublas-cu12").files:
        if file.name == "libcublasLt.so.12":
            lib_path = str(file.locate())
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            return lib_path
    return None


def require_cluster_runtime() -> None:
    """Ensure FAISS and HDBSCAN are importable before cluster jobs run."""
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    system_cublaslt = "/usr/local/cuda/lib64/libcublasLt.so.12"
    # #region agent log
    _debug_log_faiss(
        "A",
        "LD_LIBRARY_PATH before faiss import",
        {
            "ld_library_path": ld_library_path,
            "system_cublaslt_exists": os.path.exists(system_cublaslt),
            "system_has_env_mode_symbol": _cublaslt_has_env_mode_symbol(system_cublaslt),
        },
    )
    # #endregion
    preloaded_lt: str | None = None
    try:
        preloaded_lt = _preload_faiss_cublaslt()
        # #region agent log
        _debug_log_faiss(
            "C",
            "preloaded venv libcublasLt before faiss",
            {
                "preloaded_lt": preloaded_lt,
                "venv_has_env_mode_symbol": (
                    _cublaslt_has_env_mode_symbol(preloaded_lt) if preloaded_lt else None
                ),
            },
        )
        # #endregion
        import faiss  # noqa: F401
        # #region agent log
        _debug_log_faiss("C", "faiss import succeeded", {"faiss_version": faiss.__version__})
        # #endregion
    except ImportError:
        raise PipelineDependencyError(
            "faiss is not installed. Install the project dependencies: "
            "pip install -e . (requires faiss-gpu-cu12)"
        ) from None
    except OSError as exc:
        # #region agent log
        _debug_log_faiss(
            "B",
            "faiss import failed with OSError",
            {
                "exc_type": type(exc).__name__,
                "exc_msg": str(exc),
                "preloaded_lt": preloaded_lt,
                "ld_library_path": ld_library_path,
            },
        )
        # #endregion
        raise PipelineDependencyError(
            "faiss failed to load CUDA libraries (libcublas/cublasLt mismatch). "
            f"Original error: {exc}\n"
            "If LD_LIBRARY_PATH points at an older system CUDA, unset it or upgrade "
            "system CUDA to match the pip nvidia-cublas-cu12 version."
        ) from None

    try:
        import hdbscan  # noqa: F401
    except ImportError:
        raise PipelineDependencyError(
            "hdbscan is not installed. Install the project dependencies: "
            "pip install -e ."
        ) from None
