#!/usr/bin/env bash
# Build and install detectron2 into the active venv.
# CUDA_HOME must match the CUDA version PyTorch was built with.
set -euo pipefail

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "PyTorch is not installed. Run: pip install -e \".[detectron2]\"" >&2
  exit 1
fi

TORCH_CUDA="$(python -c "import torch; print(torch.version.cuda or '')")"
if [[ -z "${TORCH_CUDA}" ]]; then
  echo "PyTorch was built without CUDA; detectron2 GPU build may not apply." >&2
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in "/usr/local/cuda-${TORCH_CUDA}" "/usr/local/cuda"; do
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="${candidate}"
      break
    fi
  done
fi

if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "Set CUDA_HOME to an nvcc matching PyTorch CUDA ${TORCH_CUDA}." >&2
  exit 1
fi

echo "Using CUDA_HOME=${CUDA_HOME} for PyTorch CUDA ${TORCH_CUDA}"
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
echo "detectron2 installed. Download weights with: mf download --backend detectron2"
