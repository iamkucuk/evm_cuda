#!/bin/bash
# Build the _evm_cuda.so for the current node's GPU arch. Run on a TRUBA
# compute/login node with the CUDA module loaded.
set -euo pipefail
PROJ=/arf/scratch/fkucuk/projects/evm_cuda
cd "$PROJ"

source /usr/share/Modules/init/bash
module load lib/cuda/12.6

SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' .')
echo "Building _evm_cuda.so for sm_${SM}..."

VENV=$PROJ/.venv
# pip-install the build tools so they're on the venv's PATH regardless of node.
$VENV/bin/pip install -q cmake ninja pybind11 >/dev/null 2>&1
export PATH="$VENV/bin:$PATH"

# Put the pip-installed ninja on PATH so cmake's -G Ninja can find it.
NINJA_BIN=$($VENV/bin/python -c "import ninja, os; print(os.path.join(os.path.dirname(ninja.__file__), 'data', 'bin'))")
export PATH="$NINJA_BIN:$PATH"
export pybind11_DIR=$($VENV/bin/python -c "import pybind11; print(pybind11.get_cmake_dir())")

rm -rf cuda/build
cmake -S cuda -B cuda/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=$SM -G Ninja
cmake --build cuda/build --config Release -j

echo "--- built: ---"
ls -la cuda/evm_cuda/_evm_cuda*.so

# Verify the binding exists.
$VENV/bin/python -c "import sys; sys.path.insert(0,'cuda'); from evm_cuda import _evm_cuda; print('gpu_mem_info present:', hasattr(_evm_cuda, 'gpu_mem_info'))"
echo "BUILD_OK"
