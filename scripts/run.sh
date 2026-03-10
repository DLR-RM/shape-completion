#!/usr/bin/env bash

# Configure SLURM auto requeue
#SBATCH --signal=SIGUSR1@90

# Activate Python + CUDA environment
source "$(dirname "${BASH_SOURCE[0]}")/activate_env.sh"

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
# To use a custom dirs config, set DIRS to your config name (without .yaml):
# DIRS="my_config"

COMMAND=("$@")
COMMAND+=("dirs=$DIRS" "load.hdf5=$USE_HDF5")

if [ "$SLURM_JOB_NAME" = "interactive" ]; then
    COMMAND+=("data.cache=False" "log.wandb=False" "log.progress=rich")
    echo "Running INTERACTIVELY with COMMAND:" "${COMMAND[@]}"
    echo ""
    "${COMMAND[@]}"
elif [ "$SLURM_JOB_NAME" = "" ]; then
    COMMAND+=("data.cache=False" "log.wandb=False" "log.progress=rich")
    echo "Running LOCALLY with COMMAND:" "${COMMAND[@]}"
    echo ""
    "${COMMAND[@]}"
else
    COMMAND+=("log.progress=False")
    echo "Running on SLURM with COMMAND: srun" "${COMMAND[@]}"
    echo ""
    srun "${COMMAND[@]}"
fi

echo "Job ended on $(date)"
end_time=$(date +%s)
total_time=$((end_time - start_time))
echo "Job execution took ${total_time} s"
