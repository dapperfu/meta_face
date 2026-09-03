#!/usr/bin/env bash
# Probe common reasons NVIDIA CUDA is missing or unused.
# Read-only. No sudo. Does not install packages.
#
# Usage:
#   bash scripts/debug_nvidia_cuda.sh
#   bash scripts/debug_nvidia_cuda.sh /path/to/python

set -o pipefail

PASS=0
FAIL=0
WARN=0

pass() { PASS=$((PASS + 1)); printf '  [PASS] %s\n' "$*"; }
fail() { FAIL=$((FAIL + 1)); printf '  [FAIL] %s\n' "$*"; }
warn() { WARN=$((WARN + 1)); printf '  [WARN] %s\n' "$*"; }
info() { printf '  [info] %s\n' "$*"; }
section() { printf '\n== %s ==\n' "$*"; }

have() { command -v "$1" >/dev/null 2>&1; }

print_cmd() {
  local label="$1"
  shift
  if have "$1"; then
    info "${label}: $(command -v "$1")"
    "$@" 2>&1 | sed 's/^/         /' || warn "${label} exited $?"
  else
    warn "${label}: ${1} not on PATH"
  fi
}

PYTHON_BIN="${1:-}"
if [ -z "${PYTHON_BIN}" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif have python3; then
    PYTHON_BIN="$(command -v python3)"
  elif have python; then
    PYTHON_BIN="$(command -v python)"
  fi
fi

printf 'NVIDIA / CUDA debug\n'
printf 'host=%s  kernel=%s  date=%s\n' "$(hostname)" "$(uname -r)" "$(date -Iseconds)"
info "USER=${USER:-?}  SHELL=${SHELL:-?}  PWD=${PWD}"
info "VIRTUAL_ENV=${VIRTUAL_ENV:-<unset>}"
info "python=${PYTHON_BIN:-<none>}"

# --- GPU present ---
section "PCI NVIDIA devices"
if have lspci; then
  nvidia_pci="$(lspci -nn 2>/dev/null | grep -i 'nvidia' || true)"
  if [ -n "${nvidia_pci}" ]; then
    pass "lspci sees NVIDIA hardware"
    printf '%s\n' "${nvidia_pci}" | sed 's/^/         /'
  else
    fail "lspci lists no NVIDIA device (wrong machine, passthrough, or no GPU)"
  fi
else
  warn "lspci not installed"
fi

# --- Conflicting driver ---
section "Kernel modules"
if have lsmod; then
  if lsmod | grep -q '^nouveau'; then
    fail "nouveau is loaded; it conflicts with the NVIDIA proprietary driver"
    lsmod | grep nouveau | sed 's/^/         /'
  else
    pass "nouveau is not loaded"
  fi
  nvidia_mods="$(lsmod | awk '/^nvidia/ {print}')"
  if [ -d /sys/module/nvidia ] || [ -n "${nvidia_mods}" ]; then
    pass "nvidia kernel module is loaded"
    if [ -n "${nvidia_mods}" ]; then
      printf '%s\n' "${nvidia_mods}" | sed 's/^/         /'
    else
      info "/sys/module/nvidia exists (lsmod may hide modules in this namespace)"
    fi
  else
    fail "nvidia kernel module is not loaded"
  fi
else
  warn "lsmod not available"
  if [ -d /sys/module/nvidia ]; then
    pass "/sys/module/nvidia exists"
  fi
fi

if [ -r /proc/driver/nvidia/version ]; then
  pass "/proc/driver/nvidia/version is readable"
  sed 's/^/         /' /proc/driver/nvidia/version
else
  fail "/proc/driver/nvidia/version missing (driver not loaded)"
fi

# --- Device nodes ---
section "Device nodes"
if [ -e /dev/nvidia0 ] || [ -e /dev/nvidiactl ]; then
  pass "NVIDIA device nodes exist"
  ls -l /dev/nvidia* 2>/dev/null | sed 's/^/         /' || true
else
  fail "no /dev/nvidia* nodes (driver, container device, or cgroup issue)"
fi
if [ -e /dev/nvidia-uvm ]; then
  pass "/dev/nvidia-uvm present (needed for CUDA unified memory)"
else
  warn "/dev/nvidia-uvm missing; CUDA may still work for some apps"
fi

# --- Driver userland ---
section "nvidia-smi / driver"
if have nvidia-smi; then
  pass "nvidia-smi is on PATH ($(command -v nvidia-smi))"
  if nvidia-smi >/tmp/meta_face_nvidia_smi.out 2>/tmp/meta_face_nvidia_smi.err; then
    pass "nvidia-smi ran"
    sed 's/^/         /' /tmp/meta_face_nvidia_smi.out
    drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
    cuda_from_smi="$(nvidia-smi | awk -F'CUDA Version:' 'NF>1 {gsub(/^[ \t]+/,"",$2); split($2,a," "); print a[1]; exit}')"
    name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ',' -)"
    info "GPU name(s): ${name:-unknown}"
    info "Driver: ${drv:-unknown}  CUDA (max advertised by driver): ${cuda_from_smi:-unknown}"
  else
    fail "nvidia-smi failed"
    sed 's/^/         /' /tmp/meta_face_nvidia_smi.err
  fi
  rm -f /tmp/meta_face_nvidia_smi.out /tmp/meta_face_nvidia_smi.err
else
  fail "nvidia-smi not on PATH (NVIDIA userland not installed or PATH incomplete)"
fi

# --- Env that hides GPUs ---
section "Environment that can hide CUDA"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if [ "${CUDA_VISIBLE_DEVICES}" = "-1" ]; then
    fail "CUDA_VISIBLE_DEVICES=-1 hides every GPU"
  else
    info "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  fi
else
  pass "CUDA_VISIBLE_DEVICES is unset (all GPUs visible to CUDA apps)"
fi
info "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
info "CUDA_HOME=${CUDA_HOME:-<unset>}"
info "CUDA_PATH=${CUDA_PATH:-<unset>}"
info "CUDA_ROOT=${CUDA_ROOT:-<unset>}"
info "PATH=${PATH}"
info "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

# --- Toolkit ---
section "CUDA toolkit (nvcc /usr/local/cuda)"
if have nvcc; then
  pass "nvcc is on PATH ($(command -v nvcc))"
  nvcc --version 2>&1 | sed 's/^/         /'
else
  warn "nvcc not on PATH (toolkit optional if the app ships CUDA via pip wheels)"
fi
if [ -d /usr/local/cuda ]; then
  pass "/usr/local/cuda exists"
  if [ -L /usr/local/cuda ]; then
    info "/usr/local/cuda -> $(readlink -f /usr/local/cuda)"
  fi
  if [ -x /usr/local/cuda/bin/nvcc ]; then
    /usr/local/cuda/bin/nvcc --version 2>&1 | sed 's/^/         /'
  fi
else
  warn "/usr/local/cuda missing"
fi
ls -d /usr/local/cuda-* 2>/dev/null | sed 's/^/         /' || info "no /usr/local/cuda-* directories"

# --- Shared libraries ---
section "Driver and CUDA libraries"
check_so() {
  local name="$1"
  local hits
  hits="$(ldconfig -p 2>/dev/null | grep -F "${name}" || true)"
  if [ -n "${hits}" ]; then
    pass "ldconfig knows ${name}"
    printf '%s\n' "${hits}" | head -n 8 | sed 's/^/         /'
  else
    warn "ldconfig does not list ${name}"
  fi
}

if have ldconfig; then
  check_so "libcuda.so"
  check_so "libcudart.so"
  check_so "libcublas.so"
  check_so "libcudnn.so"
  check_so "libnvinfer.so"
  check_so "libcupti.so"
else
  warn "ldconfig not available"
fi

shopt -s nullglob
for dir in \
  /usr/lib/x86_64-linux-gnu \
  /usr/lib64 \
  /usr/local/cuda/lib64 \
  /usr/local/cuda/targets/x86_64-linux/lib
do
  [ -d "${dir}" ] || continue
  hits="$(ls -1 "${dir}"/libcuda.so* "${dir}"/libcudart.so* "${dir}"/libcudnn.so* 2>/dev/null | head -n 8 || true)"
  if [ -n "${hits}" ]; then
    info "${dir}:"
    printf '%s\n' "${hits}" | sed 's/^/         /'
  fi
done
shopt -u nullglob

# --- Secure Boot (unsigned NVIDIA modules) ---
section "Secure Boot"
if have mokutil; then
  sb="$(mokutil --sb-state 2>/dev/null || true)"
  if printf '%s\n' "${sb}" | grep -qi 'enabled'; then
    warn "Secure Boot is enabled; unsigned NVIDIA modules will not load"
    printf '%s\n' "${sb}" | sed 's/^/         /'
  else
    info "${sb:-mokutil returned nothing}"
  fi
else
  info "mokutil not installed (skip Secure Boot check)"
fi

# --- Container / WSL ---
section "Container and WSL"
if [ -f /.dockerenv ]; then
  warn "running inside Docker; GPU needs --gpus / nvidia-container-toolkit"
else
  info "no /.dockerenv"
fi
if [ -f /proc/1/cgroup ] && grep -qE 'docker|containerd|kubepods' /proc/1/cgroup 2>/dev/null; then
  warn "PID 1 cgroup looks like a container"
fi
if have nvidia-container-cli; then
  info "nvidia-container-cli present"
  nvidia-container-cli info 2>&1 | head -n 20 | sed 's/^/         /' || true
fi
if grep -qi microsoft /proc/version 2>/dev/null; then
  warn "WSL kernel; CUDA needs NVIDIA WSL driver on Windows plus CUDA in the distro"
fi

# --- Python / this project ---
section "Python CUDA stacks (onnxruntime, torch, faiss)"
if [ -z "${PYTHON_BIN}" ] || [ ! -x "${PYTHON_BIN}" ]; then
  fail "no python interpreter to probe"
else
  info "using ${PYTHON_BIN}"
  "${PYTHON_BIN}" -V 2>&1 | sed 's/^/         /'
  if "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m pip show onnxruntime onnxruntime-gpu torch faiss-gpu-cu12 faiss-cpu 2>/dev/null \
      | awk '/^(Name|Version|Location|Requires):/ {print "         " $0}'
    has_ort=0
    has_ort_gpu=0
    "${PYTHON_BIN}" -m pip show onnxruntime >/dev/null 2>&1 && has_ort=1
    "${PYTHON_BIN}" -m pip show onnxruntime-gpu >/dev/null 2>&1 && has_ort_gpu=1
    if [ "${has_ort}" -eq 1 ] && [ "${has_ort_gpu}" -eq 1 ]; then
      fail "both onnxruntime and onnxruntime-gpu are installed; CPU package often wins and CUDA EP disappears"
      info "fix: ${PYTHON_BIN} -m pip uninstall -y onnxruntime && ${PYTHON_BIN} -m pip install --force-reinstall onnxruntime-gpu"
    elif [ "${has_ort_gpu}" -eq 1 ]; then
      pass "onnxruntime-gpu is installed (no CPU onnxruntime package)"
    elif [ "${has_ort}" -eq 1 ]; then
      warn "pip package is onnxruntime (CPU). meta_face wants onnxruntime-gpu."
      info "fix: ${PYTHON_BIN} -m pip uninstall -y onnxruntime && ${PYTHON_BIN} -m pip install onnxruntime-gpu"
    else
      warn "neither onnxruntime nor onnxruntime-gpu is installed on this python"
    fi
  else
    warn "pip module not available on this python"
  fi

  "${PYTHON_BIN}" - <<'PY' || warn "python CUDA probe failed"
import sys

def banner(msg):
    print("         " + msg)

try:
    import onnxruntime as ort
except Exception as exc:
    banner("onnxruntime import failed: %s" % exc)
else:
    banner("onnxruntime %s  file=%s" % (ort.__version__, getattr(ort, "__file__", "?")))
    try:
        providers = ort.get_available_providers()
    except Exception as exc:
        banner("get_available_providers failed: %s" % exc)
        providers = []
    banner("providers=%s" % providers)
    if "CUDAExecutionProvider" in providers:
        banner("CUDAExecutionProvider is available")
    else:
        banner("CUDAExecutionProvider MISSING (CPU wheel, or GPU wheel cannot see libcudart/cudnn/libcuda)")
        if "CPUExecutionProvider" in providers:
            banner("CPUExecutionProvider is present; this matches InsightFace/ONNX falling back to CPU")

try:
    import torch
except Exception as exc:
    banner("torch not importable: %s" % exc)
else:
    banner("torch %s  cuda_built=%s  cuda_available=%s" % (
        torch.__version__,
        getattr(torch.version, "cuda", None),
        torch.cuda.is_available(),
    ))
    if torch.cuda.is_available():
        banner("torch device0=%s  count=%s" % (torch.cuda.get_device_name(0), torch.cuda.device_count()))
    else:
        banner("torch cannot see CUDA (CPU torch wheel, or driver/toolkit mismatch)")

try:
    import faiss
except Exception as exc:
    banner("faiss not importable: %s" % exc)
else:
    banner("faiss %s" % getattr(faiss, "__version__", "?"))
    gpu = getattr(faiss, "get_num_gpus", None)
    if callable(gpu):
        try:
            banner("faiss.get_num_gpus()=%s" % gpu())
        except Exception as exc:
            banner("faiss.get_num_gpus failed: %s" % exc)
PY
fi

# --- Tiny driver CUDA probe (optional) ---
section "Optional: libcudart presence for current python"
if [ -n "${PYTHON_BIN}" ] && [ -x "${PYTHON_BIN}" ]; then
  "${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
import ctypes, glob, os, sys
banner = lambda m: print("         " + m)
candidates = []
for key in ("LD_LIBRARY_PATH",):
    val = os.environ.get(key) or ""
    for part in val.split(":"):
        if part:
            candidates.extend(glob.glob(os.path.join(part, "libcudart.so*")))
for path in (
    "/usr/local/cuda/lib64/libcudart.so",
    "/usr/lib/x86_64-linux-gnu/libcudart.so.12",
    "/usr/lib/x86_64-linux-gnu/libcudart.so.11",
):
    if os.path.exists(path):
        candidates.append(path)
seen = []
for path in candidates:
    if path not in seen:
        seen.append(path)
if not seen:
    banner("no libcudart.so found on common paths")
    sys.exit(0)
for path in seen[:6]:
    try:
        ctypes.CDLL(path)
        banner("loaded %s" % path)
    except OSError as exc:
        banner("could not load %s: %s" % (path, exc))
try:
    ctypes.CDLL("libcuda.so.1")
    banner("loaded libcuda.so.1 (driver)")
except OSError as exc:
    banner("could not load libcuda.so.1: %s" % exc)
PY
fi

section "Summary"
printf '  pass=%s  fail=%s  warn=%s\n' "${PASS}" "${FAIL}" "${WARN}"
if [ "${FAIL}" -gt 0 ]; then
  printf '  CUDA is not ready, or userland cannot see the GPU.\n'
  printf '  Typical meta_face fix: uninstall CPU onnxruntime, install onnxruntime-gpu, keep NVIDIA driver loaded.\n'
  exit 1
fi
printf '  Driver side looks usable. If ONNX still warns, the Python wheel is CPU-only or cannot load CUDA libs.\n'
exit 0
