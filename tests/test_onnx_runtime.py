"""ONNX Runtime is GPU-only."""

from __future__ import annotations

from pathlib import Path

from meta_face.config import ONNX_PROVIDERS


def test_onnx_providers_list_cuda_only() -> None:
    assert ONNX_PROVIDERS == ("CUDAExecutionProvider",)
    assert "CPUExecutionProvider" not in ONNX_PROVIDERS


def test_pyproject_lists_onnxruntime_gpu_only() -> None:
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert '"onnxruntime-gpu[cuda,cudnn]>=1.17"' in text
    assert '"onnxruntime>=' not in text
    assert '"onnxruntime"' not in text
