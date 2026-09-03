"""ONNX Runtime GPU-only session helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from meta_face.config import ONNX_PROVIDERS


def inference_session(model_path: Path | str, **kwargs: Any) -> Any:
    """Load an ONNX model on CUDA. CPUExecutionProvider is never requested."""
    import onnxruntime as ort

    available = list(ort.get_available_providers())
    missing = [name for name in ONNX_PROVIDERS if name not in available]
    if missing:
        raise RuntimeError(
            "onnxruntime-gpu CUDA provider is unavailable. "
            f"Need {list(ONNX_PROVIDERS)}, have {available}. "
            "Uninstall the CPU package `onnxruntime` and install `onnxruntime-gpu` only."
        )
    return ort.InferenceSession(
        str(model_path),
        providers=list(ONNX_PROVIDERS),
        **kwargs,
    )
