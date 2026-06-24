#!/bin/bash
#SBATCH --job-name=evm-prof
#SBATCH --partition=palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/arf/scratch/fkucuk/logs/%j.out
#SBATCH --error=/arf/scratch/fkucuk/logs/%j.err

# Re-run all EVM CUDA pipelines (Color/Motion x FP32/FP16) on TRUBA and
# confirm timings against the docs (A100-80GB: color 72/84ms, motion 209/172ms).

set -euo pipefail

export TERM=xterm

# Source environment-modules so `module load` works under SLURM's non-login shell.
if [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
fi

PROJ=/arf/scratch/fkucuk/projects/evm_cuda
export PYTHONPATH="$PROJ/cuda"

module purge
module load lib/cuda/12.6

cd "$PROJ"

echo "============================================================"
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Date: $(date)"
echo "============================================================"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
           --format=csv
echo

# Sanity: extension loads and reports a CUDA device.
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "cuda")
from evm_cuda import _evm_cuda
print("Extension loaded OK:", _evm_cuda.__file__)
PY

echo
echo "============================================================"
echo "Running profile_full_comparison.py (all 4 GPU pipelines + CPU)"
echo "============================================================"
.venv/bin/python scripts/profile_full_comparison.py

echo
echo "============================================================"
echo "Job $SLURM_JOB_ID DONE: $(date)"
echo "============================================================"
