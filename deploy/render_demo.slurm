#!/bin/bash -l
#SBATCH --job-name=evm_render
#SBATCH --partition=palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/arf/scratch/fkucuk/logs/evm_cuda/%j.out
#SBATCH --error=/arf/scratch/fkucuk/logs/evm_cuda/%j.err
#
# Re-run the CUDA test suite (fresh confirmation) AND render the comparison
# output videos for face.mp4 and baby.mp4. Outputs land under
# $PROJECT_ROOT/output/cuda_render/ for later rsync back to the laptop.

set -euo pipefail
export TERM="${TERM:-dumb}"

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/cuda:${PYTHONPATH:-}"

echo "[render] node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || true

# Load toolchain + venv (build is incremental, ~seconds since the .so is cached).
# shellcheck disable=SC1091
source deploy/truba_env.sh
bash deploy/build.sh

echo
echo "[render] === 30 CUDA tests vs Python baseline ==="
pytest tests/cuda/ -q --no-header

echo
echo "[render] === render output videos (CUDA vs Python) ==="
python scripts/render_cuda_videos.py

echo
echo "[render] === done ==="
