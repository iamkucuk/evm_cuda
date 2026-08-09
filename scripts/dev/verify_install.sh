#!/usr/bin/env bash
#
# verify_install.sh — prove that `pip install .` produces a working package.
#
# Plan step 1.15. This is the check that judges every packaging change:
# cuda/CMakeLists.txt, pyproject.toml and the Makefile are all "tested" by
# running this script, per .claude/rules/development-practices.md.
#
# What it does:
#   1. builds a throwaway venv outside the repository
#   2. installs THIS working tree into it with a plain `pip install .`
#   3. imports the installed package from a directory that is not the repo,
#      so a source-tree import cannot masquerade as a successful install
#   4. exercises the pure-Python public API and reports the CUDA state
#   5. runs the test suite against the installed package
#
# It is expected to pass on a machine with NO CUDA: the extension is optional,
# `have_cuda` is then False, and require_cuda() must raise a named error rather
# than fall back silently. On a machine WITH nvcc the same run additionally
# proves the extension compiled and imported.
#
# Usage:  bash scripts/dev/verify_install.sh
# Exit:   0 = everything passed. Any non-zero = stop and read the output.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$(mktemp -d "${TMPDIR:-/tmp}/evm_verify_install.XXXXXX")/venv"
# Import checks run from here, never from $REPO: with the packages under src/
# a repo-root import would fail anyway, but running elsewhere makes that
# guarantee explicit instead of incidental.
NEUTRAL="$(dirname "$VENV")"

cleanup() { rm -rf "$(dirname "$VENV")"; }
trap cleanup EXIT

step() { printf '\n=== %s ===\n' "$1"; }

step "1/5  fresh venv at $VENV"
"${PYTHON:-python3}" -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -V

step "2/5  pip install . (from $REPO)"
# No --editable and no --no-build-isolation: this is the path a stranger gets
# from PyPI, backend and CMake fetched by pip itself.
"$PY" -m pip install "$REPO"

step "3/5  import checks (cwd=$NEUTRAL, not the repo)"
cd "$NEUTRAL"
"$PY" - <<'PYCHECK'
import inspect
import pathlib
import sys

import evm

where = pathlib.Path(inspect.getfile(evm)).resolve()
print(f"evm            {evm.__version__}")
print(f"loaded from    {where}")
if "site-packages" not in where.parts:
    sys.exit(f"FAIL: evm was imported from {where}, not from the install")

missing = [n for n in evm.__all__ if not hasattr(evm, n)]
if missing:
    sys.exit(f"FAIL: public names missing from the installed package: {missing}")
print(f"public API     {len(evm.__all__)} names, all present")

# The deprecated top-level shim must keep working and must say it is deprecated.
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import evm_cuda

    evm_cuda.have_cuda
if not any(issubclass(w.category, DeprecationWarning) for w in caught):
    sys.exit("FAIL: evm_cuda shim did not emit a DeprecationWarning")
print("evm_cuda shim  imports and warns")
PYCHECK

step "4/5  CPU pipeline + CUDA state"
"$PY" - <<'PYRUN'
import sys

import numpy as np

import evm

# Pure-Python surface must work with no GPU and no video files.
rng = np.random.default_rng(0)
frame = rng.random((32, 32, 3))
if not np.allclose(evm.yiq_to_rgb(evm.rgb_to_yiq(frame)), frame, atol=1e-6):
    sys.exit("FAIL: rgb_to_yiq/yiq_to_rgb round trip broken")
pyr, ind = evm.build_lpyr(frame[:, :, 0])
if not np.allclose(evm.recon_lpyr(pyr, ind), frame[:, :, 0], atol=1e-10):
    sys.exit("FAIL: Laplacian pyramid round trip broken")
print("cpu pipeline   colour + pyramid round trips OK")

# CUDA is optional. Whatever its state, it has to be reported, never guessed.
import evm.cuda as gpu

print(f"have_cuda      {gpu.have_cuda}")
if gpu.have_cuda:
    gpu.require_cuda()
    print("require_cuda   returned (extension is importable)")
else:
    print(f"import_error   {type(gpu.import_error).__name__}: {gpu.import_error}")
    try:
        gpu.require_cuda()
    except Exception as exc:  # the loud, named failure is the requirement
        print(f"require_cuda   raised {type(exc).__name__} as required")
    else:
        sys.exit("FAIL: require_cuda() returned quietly with no extension")
PYRUN

step "5/5  pytest against the installed package"
cd "$REPO"
"$PY" -m pip install --quiet pytest
# -p no:randomly keeps the order comparable with the recorded baselines.
"$PY" -m pytest tests/ -q -p no:randomly

printf '\n=== PASS: pip install . produces a working package ===\n'
