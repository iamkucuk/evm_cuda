#!/bin/bash
#SBATCH --job-name=evm-regress
#SBATCH --partition=kolyoz-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/arf/scratch/fkucuk/logs/%j.out
#SBATCH --error=/arf/scratch/fkucuk/logs/%j.err

# Regression gate: run the CUDA test suite (RMSE vs CPU oracle) + the full
# pipeline comparison on TRUBA (kolyoz-cuda: H100/H200). Exits non-zero if any
# test fails.

set -uo pipefail   # NOTE: no -e; we want to reach the profiler even if a
                   # non-critical test fails, and report the combined result.

export TERM=xterm
PROJ=/arf/scratch/fkucuk/projects/evm_cuda
export PYTHONPATH="$PROJ/cuda"

module purge
if [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
fi
module load lib/cuda/12.6

cd "$PROJ"

echo "============================================================"
echo "Regression job $SLURM_JOB_ID on $(hostname)  $(date)"
echo "============================================================"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

# Sanity: extension loads (rebuild for THIS node's GPU arch if it was built
# for a different one — the .so must match the device's compute capability).
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.')
BUILT_SM=$(grep "CMAKE_CUDA_ARCHITECTURES:STRING" cuda/build/CMakeCache.txt 2>/dev/null | cut -d= -f2)
echo "GPU sm_${SM}; .so built for sm_${BUILT_SM:-none}"
if [ "$SM" != "$BUILT_SM" ]; then
    echo "Rebuilding the extension for sm_${SM}..."
    .venv/bin/pip install -q ninja >/dev/null 2>&1
    export pybind11_DIR=$(.venv/bin/python -c "import pybind11; print(pybind11.get_cmake_dir())")
    rm -rf cuda/build && mkdir -p cuda/build
    cmake -S cuda -B cuda/build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=$SM -G Ninja -DCMAKE_MAKE_PROGRAM=$(.venv/bin/python -c "import ninja; print(ninja.BIN_DIR+'/ninja')") >/dev/null 2>&1 || \
    cmake -S cuda -B cuda/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$SM
    cmake --build cuda/build --config Release -j
fi
.venv/bin/python -c "import sys; sys.path.insert(0,'cuda'); from evm_cuda import _evm_cuda; print('Extension OK:', _evm_cuda.__file__)"

echo
echo "============================================================"
echo "[1/2] CUDA test suite (RMSE vs Python baseline oracle)"
echo "============================================================"
.venv/bin/python -m pytest tests/ tests/cuda/ -q
TEST_RC=$?

echo
echo "============================================================"
echo "[2/2] Full pipeline comparison (CPU vs FP32 vs FP16)"
echo "============================================================"
.venv/bin/python scripts/profile_full_comparison.py
PROF_RC=$?

echo
echo "============================================================"
echo "Regression job $SLURM_JOB_ID DONE: $(date)"
echo "pytest exit=$TEST_RC  profiler exit=$PROF_RC"
echo "============================================================"
exit $TEST_RC
