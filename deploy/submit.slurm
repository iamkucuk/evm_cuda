#!/bin/bash -l
#SBATCH --job-name=evm_cuda
#SBATCH --partition=palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/arf/scratch/fkucuk/logs/evm_cuda/%j.out
#SBATCH --error=/arf/scratch/fkucuk/logs/evm_cuda/%j.err
#
# Build the EVM CUDA extension and run the full test suite on a TRUBA GPU.
#
# Submit from $PROJECT_ROOT with:   sbatch deploy/submit.slurm
# Tail output with:                 tail -f /arf/scratch/fkucuk/logs/evm_cuda/<JOBID>.out
# Switch to Hopper:                 --partition=kolyoz-cuda --gres=gpu:h100:1
#
# Logs and outputs go under /arf/scratch (WEKA, visible from ARF-ACC compute).
# Never put outputs under /truba/home/ — not mounted on compute nodes.
#
# The `#!/bin/bash -l` shebang makes SLURM run us in a login shell, which
# sources /etc/profile and so defines the `module` function (Tcl Environment
# Modules). Without -l, `module load ...` in truba_env.sh would fail.

set -euo pipefail

# Suppress Tcl Modules' "TERM environment variable needs set" noise.
export TERM="${TERM:-dumb}"

# Where this job was submitted from.
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

# --- 3. Pipeline integration tests need the MIT samples; warn if missing
# (the login node has internet to repopulate, compute nodes don't).
if [[ ! -f "$PROJECT_ROOT/data/face.mp4" ]]; then
    echo "[sbatch] WARNING: data/face.mp4 missing — pipeline tests will skip."
    echo "         Run on the login node: python scripts/download_samples.py face baby --with-references"
fi

# --- 4. Run the test suites -----------------------------------------------
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/cuda:${PYTHONPATH:-}"

echo "[sbatch] === Python baseline (oracle) ==="
pytest tests/ -q --no-header

echo "[sbatch] === CUDA kernels vs Python baseline ==="
pytest tests/cuda/ -v --no-header

echo "[sbatch] === done ==="
