#!/usr/bin/env bash
set -euo pipefail

# Expects CC, CXX, TORCH_CUDA_ARCH_LIST, MAX_JOBS from environment (e.g. direnv).
# See ~/build_cuda.sh sourced by .envrc.
# Skips already-installed packages. Pass --force to reinstall.

force=false
[[ "${1:-}" == "--force" ]] && force=true

echo "CC=$CC  CXX=$CXX  CUDA_ARCH=$TORCH_CUDA_ARCH_LIST  MAX_JOBS=$MAX_JOBS"

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}.html"

install_if_missing() {
    local module="$1"; shift
    # Check that the module both imports and has real content (not an empty namespace package).
    if ! $force && python -c "import $module; assert hasattr($module, '__file__') and $module.__file__ is not None" 2>/dev/null; then
        echo "Skipping $module (already installed). Use --force to reinstall."
        return
    fi
    uv pip install -v --no-build-isolation "$@"
}

# torch-scatter — scatter ops for grid-based models (prebuilt wheel, no compilation)
if ! $force && python -c "import torch_scatter" 2>/dev/null; then
    echo "Skipping torch_scatter (already installed). Use --force to reinstall."
else
    uv pip install torch-scatter --find-links "$PYG_WHEEL_URL"
fi

# torch-cluster — FPS sampling for 3DShape2VecSet (prebuilt wheel, no compilation)
if ! $force && python -c "import torch_cluster" 2>/dev/null; then
    echo "Skipping torch_cluster (already installed). Use --force to reinstall."
else
    uv pip install torch-cluster --find-links "$PYG_WHEEL_URL"
fi

# Detectron2 — Mask R-CNN, PointRend point sampling
install_if_missing detectron2 \
    "detectron2@git+https://github.com/facebookresearch/detectron2.git"

# PyTorch3D — heterogeneous batching, 3D ops
install_if_missing pytorch3d \
    "pytorch3d@git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"

# tiny-cuda-nn — NeRF-like positional encoding (tcnn backend)
install_if_missing tinycudann \
    "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

# xFormers — memory-efficient attention
# Prebuilt wheels available for torch >=2.10: uv pip install xformers
# For torch <2.10, uncomment to build from source:
# install_if_missing xformers \
#     "xformers@git+https://github.com/facebookresearch/xformers.git@main"
