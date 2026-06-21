"""Laplacian pyramid round-trip + per-level match vs evm.pyramids."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CUDA_DIR = ROOT / "cuda"
for p in (str(ROOT), str(CUDA_DIR), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evm.pyramids import build_lpyr, recon_lpyr, max_pyr_ht, BINOM5  # noqa: E402
from conftest import TOL, abs_err, have_cuda, skip_no_cuda, BINOM5_CUDA  # noqa: E402

if have_cuda:
    from evm_cuda import _evm_cuda  # noqa: E402


@skip_no_cuda
@pytest.mark.parametrize("h,w", [(64, 64), (96, 64)])
def test_lpyr_roundtrip_cuda(h, w):
    """build then recon, FP32, must hit < 1e-5 (vs Python FP64 < 1e-9)."""
    rng = np.random.default_rng(0)
    img = rng.random((h, w)).astype(np.float32)
    levels = 1 + max_pyr_ht((h, w), 5)

    bands_list, _ = _evm_cuda.lpyr_build(img, levels, BINOM5_CUDA)
    bands = [np.ascontiguousarray(b, dtype=np.float32) for b in bands_list]
    recon = _evm_cuda.lpyr_recon(bands, BINOM5_CUDA)
    assert recon.shape == img.shape
    assert abs_err(recon, img) < TOL["lpyr_roundtrip"]


@skip_no_cuda
@pytest.mark.parametrize("h,w", [(64, 64), (96, 64)])
def test_lpyr_bands_match_baseline(h, w):
    """Each band of the CUDA pyramid matches the Python baseline's band."""
    rng = np.random.default_rng(0)
    img = rng.random((h, w)).astype(np.float32)
    levels = 1 + max_pyr_ht((h, w), 5)

    # Python baseline (FP64).
    pyr, pind = build_lpyr(img.astype(np.float64), "auto")
    # Split into per-level bands.
    offsets = [0]
    for l in range(levels):
        offsets.append(offsets[-1] + int(pind[l, 0] * pind[l, 1]))

    cuda_bands, _ = _evm_cuda.lpyr_build(img, levels, BINOM5_CUDA)
    for l in range(levels):
        lh, lw = int(pind[l, 0]), int(pind[l, 1])
        py_band = pyr[offsets[l]:offsets[l + 1]].reshape(lh, lw).astype(np.float32)
        cu_band = np.ascontiguousarray(cuda_bands[l], dtype=np.float32)
        assert abs_err(cu_band, py_band) < TOL["corr_dn"]


@skip_no_cuda
@pytest.mark.parametrize("h,w", [(64, 64), (45, 33)])
def test_blur_dn_matches_baseline(h, w):
    """blur_dn downsampled output matches Python baseline at each nlevs."""
    from evm.pyramids import blur_dn, BINOM5_SUM1
    rng = np.random.default_rng(1)
    img = rng.random((h, w)).astype(np.float32)
    from conftest import BINOM5_SUM1_CUDA
    for nlevs in (1, 2, 3):
        # Python baseline (FP64).
        py = blur_dn(img.astype(np.float64), nlevs).astype(np.float32)
        cu = _evm_cuda.blur_dn(img, nlevs, BINOM5_SUM1_CUDA)
        assert cu.shape == py.shape
        assert abs_err(cu, py) < TOL["blur_dn"]
