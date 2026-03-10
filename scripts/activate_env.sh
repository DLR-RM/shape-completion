# Activate Python environment. Sourced by SLURM/runner scripts.
# Priority: .venv (uv) > pyenv > conda/micromamba
#
# On SLURM nodes direnv doesn't run, so we also source ~/build_cuda.sh
# for CUDA env vars (CC, CXX, TORCH_CUDA_ARCH_LIST, etc.).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# CUDA environment (direnv doesn't run on SLURM nodes)
if [[ -f "$HOME/build_cuda.sh" ]]; then
    source "$HOME/build_cuda.sh"
fi

# Micromamba-installed system libraries (assimp, opencv, eigen3, CGAL, etc.)
if [[ -n "${MAMBA_ROOT_PREFIX:-}" ]]; then
    export CPLUS_INCLUDE_PATH="$MAMBA_ROOT_PREFIX/include:$MAMBA_ROOT_PREFIX/include/opencv4${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export LIBRARY_PATH="$MAMBA_ROOT_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$MAMBA_ROOT_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Python environment (skip if one is already active)
if [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Using existing env: ${VIRTUAL_ENV:-$CONDA_DEFAULT_ENV} ($(python --version))"
elif [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    source "$REPO_ROOT/.venv/bin/activate"
    echo "Using venv: $REPO_ROOT/.venv ($(python --version))"
elif command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init -)"
    if ! pyenv which python | grep -q "completion"; then
        pyenv activate completion
    fi
    echo "Using pyenv: $(pyenv version-name)"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate completion
    echo "Using conda: completion"
elif command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate completion
    echo "Using micromamba: completion"
else
    echo "warning: no Python environment found" >&2
fi
