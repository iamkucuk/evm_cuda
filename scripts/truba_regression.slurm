#!/bin/bash
#SBATCH --job-name=evm-regress
#SBATCH --partition=palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/arf/scratch/fkucuk/logs/%j.out
#SBATCH --error=/arf/scratch/fkucuk/logs/%j.err

# Regression gate: run the CUDA test suite (RMSE vs CPU oracle) + the full
# pipeline comparison on TRUBA. Exits non-zero if any test fails.

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
echo "Regression job $SLURM_JOB_ID on $(hostname)  $(date)"
echo "============================================================"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

# Sanity: extension loads.
.venv/bin/python -c "import sys; sys.path.insert(0,'cuda'); from evm_cuda import _evm_cuda; print('Extension OK:', _evm_cuda.__file__)"

echo
echo "============================================================"
echo "[1/2] CUDA test suite (RMSE vs Python baseline oracle)"
echo "============================================================"
.venv/bin/python -m pytest tests/ tests/cuda/ -q

echo
echo "============================================================"
echo "[2/2] Full pipeline comparison (CPU vs FP32 vs FP16)"
echo "============================================================"
.venv/bin/python scripts/profile_full_comparison.py

echo
echo "============================================================"
echo "Regression job $SLURM_JOB_ID DONE: $(date)"
echo "============================================================"
