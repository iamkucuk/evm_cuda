#!/usr/bin/env bash
# build.sh — build the _evm_cuda pybind11 extension inside a SLURM job.
#
# Run AFTER sourcing deploy/truba_env.sh. Produces cuda/evm_cuda/_evm_cuda.so
# which is importable as `from evm_cuda import _evm_cuda`.
#
# Idempotent: re-runs CMake incrementally. Pass CLEAN=1 to wipe the build dir
# and rebuild from scratch (useful after changing CUDA_ARCHITECTURES).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$SCRIPT_DIR")}"
CUDA_DIR="$PROJECT_ROOT/cuda"
BUILD_DIR="$CUDA_DIR/build"

# Sanity: env must have been sourced.
if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not on PATH. Did you 'source deploy/truba_env.sh'?" >&2
    exit 1
fi

if [[ "${CLEAN:-0}" == "1" && -d "$BUILD_DIR" ]]; then
    echo "[build] CLEAN=1: removing $BUILD_DIR" >&2
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure. CMAKE_CUDA_ARCHITECTURES defaults to "60 70 80 89 90" in
# CMakeLists.txt; override here if you want to target only the assigned node.
cmake -S "$CUDA_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
    "${CMAKE_GENERATOR_ARG[@]:-}"

# Build with all available cores.
cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"

# Verify the .so landed where the Python wrapper expects it. CMake/pybind11
# may emit an ABI-tagged name like _evm_cuda.cpython-312-x86_64-linux-gnu.so
# (PEP 3149) — Python finds either transparently, so we glob.
SO_DIR="$CUDA_DIR/evm_cuda"
SO=$(ls "$SO_DIR"/_evm_cuda*.so 2>/dev/null | head -1)
if [[ -z "$SO" ]]; then
    echo "ERROR: no _evm_cuda*.so under $SO_DIR — check the CMake output above." >&2
    exit 1
fi
echo "[build] OK: $SO ($(du -h "$SO" | cut -f1))"

# Smoke import: verifies the .so links cleanly against the runtime Python.
cd "$PROJECT_ROOT"
PYTHONPATH="$CUDA_DIR:${PYTHONPATH:-}" python -c "
import sys; sys.path.insert(0, 'cuda')
import evm_cuda
print('evm_cuda.have_cuda =', evm_cuda.have_cuda)
from evm_cuda import _evm_cuda
print('binom5 =', _evm_cuda.binom5())
print('drop_last =', _evm_cuda.drop_last)
"
