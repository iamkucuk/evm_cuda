#!/bin/bash -l
#SBATCH --job-name=evm_fp16
#SBATCH --partition=palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/arf/scratch/fkucuk/logs/evm_cuda/%j.out
#SBATCH --error=/arf/scratch/fkucuk/logs/evm_cuda/%j.err
#
# Build the EVM CUDA extension, run the full test suite, then run the
# comprehensive profiler: Python CPU vs CUDA FP32 vs CUDA FP16 for both
# the color and motion pipelines, with per-stage breakdown.
#
# Submit from $PROJECT_ROOT with:   sbatch deploy/submit_profile.slurm
# Tail output with:                 tail -f /arf/scratch/fkucuk/logs/evm_cuda/<JOBID>.out

set -euo pipefail

export TERM="${TERM:-dumb}"

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
echo "[sbatch] PROJECT_ROOT=$PROJECT_ROOT"
echo "[sbatch] node=$(hostname)  slurm_job_id=$SLURM_JOB_ID"
echo "[sbatch] GPUs assigned:"; nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv 2>/dev/null || true

# --- 1. Load toolchain + Python deps --------------------------------------
# shellcheck disable=SC1091
source "$PROJECT_ROOT/deploy/truba_env.sh"

# --- 2. Build the extension -----------------------------------------------
echo "[sbatch] building _evm_cuda.so ..."
bash "$PROJECT_ROOT/deploy/build.sh"

# --- 3. Run tests ----------------------------------------------------------
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/cuda:${PYTHONPATH:-}"

echo "[sbatch] === Python baseline (oracle) ==="
pytest tests/ -q --no-header

echo "[sbatch] === CUDA kernels vs Python baseline ==="
pytest tests/cuda/ -v --no-header

# --- 4. Run the comprehensive profiler -------------------------------------
echo "[sbatch] === CPU vs FP32 vs FP16 comparison ==="
python "$PROJECT_ROOT/scripts/profile_full_comparison.py" 2>&1 | tee "$PROJECT_ROOT/comparison_output.txt"

echo "[sbatch] === done ==="
