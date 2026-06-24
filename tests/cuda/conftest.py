"""Shared fixtures + skip gate for the CUDA test suite.

Every test module in tests/cuda/ is gated on the ``have_cuda`` marker: if
the compiled extension isn't importable (e.g. on the Mac dev host, or on a
build without nvcc), the whole suite skips cleanly. When the extension IS
present (i.e. after `make build`), the tests run and
compare each kernel's output to the Python baseline ``evm/`` within the
tolerances documented in AGENTS.md §2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make both the evm/ baseline and the cuda/evm_cuda/ wrapper importable
# regardless of pytest's invocation directory.
ROOT = Path(__file__).resolve().parents[2]
CUDA_DIR = ROOT / "cuda"
for p in (str(ROOT), str(CUDA_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import evm  # noqa: E402  (Python baseline — the oracle)

try:
    import evm_cuda  # noqa: E402
    from evm_cuda import _evm_cuda  # noqa: E402
    have_cuda = True
    cuda_import_error: Exception | None = None
except Exception as e:  # pragma: no cover - exercised on Mac dev host
    have_cuda = False
    cuda_import_error = e

# Skip marker usable as @pytest.mark.skipif(not have_cuda, ...).
skip_no_cuda = pytest.mark.skipif(
    not have_cuda,
    reason=f"evm_cuda._evm_cuda not built ({cuda_import_error!r})",
)

# Per-stage tolerances (AGENTS.md §2). Centralized so tests stay DRY.
TOL = {
    "color_cvt":       1e-6,
    "corr_dn":         1e-5,
    "up_conv":         1e-5,
    "lpyr_roundtrip":  1e-5,   # FP32 vs Python's FP64 1e-9
    "blur_dn":         1e-5,
    "iir":             1e-5,
    "butter":          1e-5,
    "ideal":           1e-4,
    "amplify_render":  1e-6,
    "end_to_end_rmse": 1e-2,   # 0.01
}

# Constant arrays shared across tests, copied from the bindings (which get
# them from evm_common.cuh). Tests assert these match the Python baseline.
if have_cuda:
    BINOM5_CUDA = np.array(_evm_cuda.binom5(), dtype=np.float32)
    BINOM5_SUM1_CUDA = np.array(_evm_cuda.binom5_sum1(), dtype=np.float32)
else:  # pragma: no cover
    BINOM5_CUDA = np.array([0.08838834764831843, 0.35355339059327373,
                            0.5303300858899106, 0.35355339059327373,
                            0.08838834764831843], dtype=np.float32)
    BINOM5_SUM1_CUDA = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625],
                                dtype=np.float32)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """Relative mean abs error, the workhorse for kernel comparisons."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).mean() / (np.abs(b).mean() + 1e-12))


def abs_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float64)
                        - np.asarray(b, dtype=np.float64)).max())
