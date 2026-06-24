#!/bin/bash -l
#SBATCH --job-name=evm_prof
#SBATCH --partition=palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/arf/scratch/fkucuk/logs/evm_cuda/%j.out
#SBATCH --error=/arf/scratch/fkucuk/logs/evm_cuda/%j.err

set -euo pipefail
export TERM="${TERM:-dumb}"

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/cuda:${PYTHONPATH:-}"

# shellcheck disable=SC1091
source deploy/truba_env.sh
bash deploy/build.sh >/dev/null 2>&1 || bash deploy/build.sh

nvidia-smi --query-gpu=name,memory.total,memory.bandwidth --format=csv,noheader

python scripts/profile_full_comparison.py
