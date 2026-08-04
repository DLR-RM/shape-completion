#!/usr/bin/env bash

# Configure SLURM auto requeue
#SBATCH --signal=SIGUSR1@90

# Activate Python + CUDA environment. Slurm may execute a spool copy of this
# script, and the submission directory is not necessarily the repository.
_activate_env=""
for _candidate in \
	"${SHAPE_COMPLETION_ROOT:+$SHAPE_COMPLETION_ROOT/scripts/activate_env.sh}" \
	"${SLURM_SUBMIT_DIR:+$SLURM_SUBMIT_DIR/scripts/activate_env.sh}" \
	"$PWD/scripts/activate_env.sh" \
	"$HOME/USERDIR/git/shape-completion/scripts/activate_env.sh" \
	"$(dirname "${BASH_SOURCE[0]}")/activate_env.sh"; do
	if [ -n "$_candidate" ] && [ -f "$_candidate" ]; then
		_activate_env="$_candidate"
		break
	fi
done
if [ -z "$_activate_env" ]; then
	echo "Could not locate scripts/activate_env.sh from submit dir, working dir, or script dir." >&2
	exit 1
fi
source "$_activate_env"

# Job Information Output
echo "============================="
echo "         JOB INFOS           "
echo "============================="
echo "Node List: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Submit host: $SLURM_SUBMIT_HOST"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_TASKS_PER_NODE"
echo "In the directory: $(pwd)"
echo "As the user: $(whoami)"
if command -v python >/dev/null 2>&1; then
    echo "Python version: $(python -c 'import sys; print(sys.version)')"
fi
if command -v uv >/dev/null 2>&1; then
    echo "uv version: $(uv --version)"
elif command -v pip >/dev/null 2>&1; then
    echo "pip version: $(pip --version)"
fi

nvidia-smi

start_time=$(date +%s)
echo "Job started on $(date)"

echo "============================="
echo "         JOB OUTPUT          "
echo "============================="

# Disable multi-threading for multi-processing
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TBB_NUM_THREADS=1

# debugging
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export USE_EXTERNAL_BPY_MODULE=1

# Set default directory and HDF5 usage
DIRS="default"
USE_HDF5="False"
if [ "$USER" == "humt_ma" ]; then
    DIRS="dlr"
    CACHE="${SHAPE_COMPLETION_CACHE_DIR:-/home_local/humt_ma/.cache}"
    CACHE_TEST_FILE="$CACHE/.shape_completion_write_test"
    if ! mkdir -p "$CACHE" 2>/dev/null || ! touch "$CACHE_TEST_FILE" 2>/dev/null; then
        CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/shape-completion"
        mkdir -p "$CACHE"
    else
        rm -f "$CACHE_TEST_FILE"
    fi
    mkdir -p "$CACHE/tmp" "$CACHE/torch_extensions"
    export TMPDIR="$CACHE/tmp"
    export XDG_CACHE_HOME="$CACHE"
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE/torch_extensions}"
elif [ "$USER" == "di35xol" ]; then
    DIRS="lrz"
    USE_HDF5="True"
fi

RAW_COMMAND="False"
if [ "${1:-}" = "--raw" ]; then
    RAW_COMMAND="True"
    shift
fi
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [--raw] COMMAND [ARG ...]" >&2
    exit 2
fi

COMMAND=("$@")
if [ "$RAW_COMMAND" = "False" ]; then
    COMMAND+=("dirs=$DIRS" "load.hdf5=$USE_HDF5")
fi

COMMAND_STATUS=0
if [ "$SLURM_JOB_NAME" = "interactive" ]; then
    if [ "$RAW_COMMAND" = "False" ]; then
        COMMAND+=("data.cache=False" "log.wandb=False" "log.progress=rich")
    fi
    echo "Running INTERACTIVELY with COMMAND:" "${COMMAND[@]}"
    echo ""
    "${COMMAND[@]}" || COMMAND_STATUS=$?
elif [ "$SLURM_JOB_NAME" = "" ]; then
    if [ "$RAW_COMMAND" = "False" ]; then
        COMMAND+=("data.cache=False" "log.wandb=False" "log.progress=rich")
    fi
    echo "Running LOCALLY with COMMAND:" "${COMMAND[@]}"
    echo ""
    "${COMMAND[@]}" || COMMAND_STATUS=$?
else
    if [ "$RAW_COMMAND" = "False" ]; then
        COMMAND+=("log.progress=False")
    fi
    echo "Running on SLURM with COMMAND: srun" "${COMMAND[@]}"
    echo ""
    srun "${COMMAND[@]}" || COMMAND_STATUS=$?
fi

echo "Job ended on $(date)"
end_time=$(date +%s)
total_time=$((end_time - start_time))
echo "Job execution took ${total_time} s"
exit "$COMMAND_STATUS"
