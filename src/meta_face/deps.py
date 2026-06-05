"""Runtime dependency checks with actionable error messages."""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import subprocess
import time
from importlib.metadata import distribution

from meta_face.tools.registry import (
    analysis_tools_requested,
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


def _debug_log_dlib(hypothesis_id: str, message: str, data: dict) -> None:
    # #region agent log
    try:
        with open(
            "/projects/spring_photography/meta_face/.cursor/debug-8b7781.log",
            "a",
            encoding="utf-8",
        ) as fh:
            fh.write(
                json.dumps(
                    {
                        "sessionId": "8b7781",
                        "runId": "pre-fix",
                        "hypothesisId": hypothesis_id,
                        "location": "deps.py:require_dlib_runtime",
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


def require_dlib_runtime() -> None:
    """Ensure face_recognition (dlib) is importable before dlib pipeline jobs run."""
    pkg_spec = importlib.util.find_spec("face_recognition")
    dlib_spec = importlib.util.find_spec("dlib")
    # #region agent log
    _debug_log_dlib(
        "A",
        "package discovery before import",
        {
            "face_recognition_found": pkg_spec is not None,
            "face_recognition_origin": getattr(pkg_spec, "origin", None),
            "dlib_found": dlib_spec is not None,
            "dlib_origin": getattr(dlib_spec, "origin", None),
        },
    )
    # #endregion
    try:
        import face_recognition  # noqa: F401
        # #region agent log
        _debug_log_dlib("D", "face_recognition import succeeded", {})
        # #endregion
    except ImportError as exc:
        # #region agent log
        _debug_log_dlib(
            "D",
            "ImportError during face_recognition import",
            {"exc_type": type(exc).__name__, "exc_msg": str(exc)},
        )
        # #endregion
        raise PipelineDependencyError(
            "face_recognition is not installed. Install the project dependencies: "
            "pip install -e .\n"
            "If dlib failed to build, install system packages first: "
            "sudo apt install cmake build-essential"
        ) from None
    except Exception as exc:
        # #region agent log
        _debug_log_dlib(
            "C",
            "non-ImportError during face_recognition import",
            {"exc_type": type(exc).__name__, "exc_msg": str(exc)},
        )
        # #endregion
        raise PipelineDependencyError(
            "face_recognition failed to import (dlib may not be compiled). "
            f"Original error: {exc}\n"
            "Install build tools: sudo apt install cmake build-essential, then "
            "pip install --force-reinstall face_recognition"
        ) from None


def detectron2_install_message() -> str:
    """Actionable steps to install the detectron2 Python package."""
    return (
        "detectron2 Python package is not installed.\n"
        "Note: pip install -e \".[detectron2]\" only installs torch/torchvision.\n"
        "Build detectron2 against the same CUDA as PyTorch:\n"
        "  pip install -e \".[detectron2]\"\n"
        "  CUDA_HOME=/usr/local/cuda-13.0 pip install --no-build-isolation \\\n"
        "    'git+https://github.com/facebookresearch/detectron2.git'\n"
        "Set CUDA_HOME to match torch.version.cuda "
        "(python -c \"import torch; print(torch.version.cuda)\").\n"
        "See https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md"
    )


def detectron2_weights_message() -> str:
    """Actionable steps when detectron2 weights/config are missing."""
    from meta_face.config import DETECTRON2_MODEL_ZOO, DETECTRON2_WEIGHTS_PATH
    from meta_face.detectron2_model import uses_custom_detectron2_paths

    if uses_custom_detectron2_paths():
        from meta_face.config import DETECTRON2_CONFIG_PATH, DETECTRON2_WEIGHTS_PATH

        return (
            "Detectron2 custom model files are missing.\n"
            f"  config:  {DETECTRON2_CONFIG_PATH}\n"
            f"  weights: {DETECTRON2_WEIGHTS_PATH}"
        )
    return (
        "Detectron2 is not ready.\n"
        "  mf download --backend detectron2\n"
        f"Default model zoo: {DETECTRON2_MODEL_ZOO}"
    )


def detectron2_runtime_issue() -> str | None:
    """Return an error message when detectron2 cannot run, else None."""
    try:
        import detectron2  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return detectron2_install_message()

    from meta_face.detectron2_model import is_detectron2_available

    if not is_detectron2_available():
        return detectron2_weights_message()
    return None


def adjust_per_image_tools_for_runtime(
    per_image_tools: list[str],
    *,
    detectron2_explicit: bool,
    analysis_explicit: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Drop unavailable tools when not explicitly requested; warn otherwise."""
    warnings: list[str] = []
    result = list(per_image_tools)
    analysis_explicit = analysis_explicit or set()

    if "detectron2" in result:
        issue = detectron2_runtime_issue()
        if issue is not None:
            if detectron2_explicit:
                raise PipelineDependencyError(issue)
            warnings.append(
                "Skipping detectron2 (not fully available):\n"
                f"{issue}\nScan continues with the remaining tools."
            )
            result = [tool for tool in result if tool != "detectron2"]

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


def require_detectron2_runtime() -> None:
    """Ensure detectron2, torch, and model weights are available."""
    issue = detectron2_runtime_issue()
    if issue is not None:
        raise PipelineDependencyError(issue) from None


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
    if analysis_tools_requested(tools):
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
