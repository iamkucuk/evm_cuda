#!/bin/bash
# Build the _evm_cuda.so for the current node's GPU arch. Run on a TRUBA
# compute node with the CUDA module loaded (or after module load).
set -euo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ"

if [[ -f /usr/share/Modules/init/bash ]]; then
  # shellcheck disable=SC1091
  source /usr/share/Modules/init/bash
  module load lib/cuda/12.6 2>/dev/null || true
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; run this on a GPU node" >&2
  exit 1
fi

SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
echo "Building _evm_cuda.so for sm_${SM}..."

VENV=$PROJ/.venv
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q cmake ninja pybind11 numpy scipy opencv-python-headless av >/dev/null

export PATH="$VENV/bin:$PATH"
NINJA_BIN=$(python -c "import ninja, os; print(os.path.join(os.path.dirname(ninja.__file__), 'data', 'bin'))")
export PATH="$NINJA_BIN:$PATH"
export pybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

rm -rf cuda/build
cmake -S cuda -B cuda/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="$SM" -G Ninja
cmake --build cuda/build --config Release -j

echo "--- built: ---"
ls -la cuda/evm_cuda/_evm_cuda*.so
python -c "import sys; sys.path.insert(0,'cuda'); from evm_cuda import _evm_cuda; print('import ok')"
echo "BUILD_OK"
