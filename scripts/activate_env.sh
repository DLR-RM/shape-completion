# Activate Python environment. Sourced by SLURM/runner scripts.
# Priority: .venv (uv) > pyenv > conda/micromamba
#
# On SLURM nodes direnv doesn't run, so we also source ~/build_cuda.sh
# for CUDA env vars (CC, CXX, TORCH_CUDA_ARCH_LIST, etc.).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

path_prepend_once() {
    local var_name="$1"
    local path="$2"
    local current="${!var_name:-}"

    [[ -z "$path" ]] && return
    case ":$current:" in
        *":$path:"*) ;;
        *) export "$var_name=$path${current:+:$current}" ;;
    esac
}

path_remove() {
    local var_name="$1"
    local path="$2"
    local current="${!var_name:-}"
    local old_ifs="$IFS"
    local part
    local next=""

    [[ -z "$current" || -z "$path" ]] && return
    IFS=:
    for part in $current; do
        [[ -z "$part" || "$part" == "$path" ]] && continue
        next="${next:+$next:}$part"
    done
    IFS="$old_ifs"
    export "$var_name=$next"
}

path_remove_prefix() {
    local var_name="$1"
    local prefix="$2"
    local current="${!var_name:-}"
    local old_ifs="$IFS"
    local part
    local next=""

    [[ -z "$current" || -z "$prefix" ]] && return
    IFS=:
    for part in $current; do
        [[ -z "$part" || "$part" == "$prefix"* ]] && continue
        next="${next:+$next:}$part"
    done
    IFS="$old_ifs"
    export "$var_name=$next"
}

# CUDA environment (direnv doesn't run on SLURM nodes)
if [[ -f "$HOME/build_cuda.sh" ]]; then
    source "$HOME/build_cuda.sh"
fi

if [[ -z "${CC:-}" ]] || ! command -v "$CC" >/dev/null 2>&1; then
    if command -v gcc-12 >/dev/null 2>&1; then
        export CC=gcc-12
    else
        export CC=gcc
    fi
fi

if [[ -z "${CXX:-}" ]] || ! command -v "$CXX" >/dev/null 2>&1; then
    if command -v g++-12 >/dev/null 2>&1; then
        export CXX=g++-12
    else
        export CXX=g++
    fi
fi

if [[ -z "${CUDAHOSTCXX:-}" ]] || ! command -v "$CUDAHOSTCXX" >/dev/null 2>&1; then
    export CUDAHOSTCXX="$CXX"
fi

shape_completion_cache_dir="${SHAPE_COMPLETION_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/shape-completion}"
if ! mkdir -p "$shape_completion_cache_dir" 2>/dev/null || [[ ! -w "$shape_completion_cache_dir" ]]; then
    shape_completion_cache_dir="/tmp/shape-completion-${USER:-user}"
    mkdir -p "$shape_completion_cache_dir"
fi

if [[ -z "${TMPDIR:-}" || ! -w "${TMPDIR:-}" || "${TMPDIR:-}" == /home_local/* ]]; then
    export TMPDIR="$shape_completion_cache_dir/tmp"
    mkdir -p "$TMPDIR"
fi

export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$shape_completion_cache_dir/torch_extensions}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

# OSL Python runtime libraries can shadow pyenv/uv Python libraries.
path_remove_prefix LD_LIBRARY_PATH "/opt/python/"

# Micromamba libraries are needed for building libkinect and similar C++
# extensions, but its libudev breaks Open3D on older glibc nodes.
if [[ -n "${MAMBA_ROOT_PREFIX:-}" && "${SHAPE_COMPLETION_USE_MAMBA_LIBS:-0}" == "1" ]]; then
    path_prepend_once CPLUS_INCLUDE_PATH "$MAMBA_ROOT_PREFIX/include/opencv4"
    path_prepend_once CPLUS_INCLUDE_PATH "$MAMBA_ROOT_PREFIX/include"
    path_prepend_once LIBRARY_PATH "$MAMBA_ROOT_PREFIX/lib"
    path_prepend_once LD_LIBRARY_PATH "$MAMBA_ROOT_PREFIX/lib"
elif [[ -n "${MAMBA_ROOT_PREFIX:-}" ]]; then
    path_remove LD_LIBRARY_PATH "$MAMBA_ROOT_PREFIX/lib"
fi

# Python environment
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    source "$REPO_ROOT/.venv/bin/activate"
    echo "Using venv: $REPO_ROOT/.venv ($(python --version))"
elif [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Using existing env: ${VIRTUAL_ENV:-$CONDA_DEFAULT_ENV} ($(python --version))"
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
