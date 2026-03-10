#!/usr/bin/env bash
#
# SLURM wrapper for render_tabletop.sh. All arguments are forwarded.
# Requires GIT_ROOT to be set (e.g. in .bashrc/.zshrc).
#
# Usage:
#   sbatch -p RMC-C01-BATCH --mem-per-cpu=5G --gres=gpu:1 --cpus-per-task=8 -t 1-0 \
#     -w rmc-gpu12 scripts/render_slurm.sh <SPLIT> <RUNS> <VIEWS> <SHARD> [OUT_DIR] [SCENE] [OPTIONS]
#
# Examples:
#   # 100 train scenes, 10 views, shard 0 (default out_dir=tabletop.v3)
#   sbatch ... scripts/render_slurm.sh train 100 10 0
#
#   # 500 train scenes, 24 views, shard 3, explicit output dir
#   sbatch ... scripts/render_slurm.sh train 500 24 3 tabletop.v3
#
#   # Validation split
#   sbatch ... scripts/render_slurm.sh val 500 24 0
#
#   # Pile scene preset with passthrough options
#   sbatch ... scripts/render_slurm.sh train 100 10 0 tabletop.pile pile --placement tower

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

srun "$GIT_ROOT/shape-completion/scripts/render_tabletop.sh" "$@"

echo "Job ended on $(date)"
end_time=$(date +%s)
total_time=$((end_time - start_time))
echo "Job execution took ${total_time} s"
