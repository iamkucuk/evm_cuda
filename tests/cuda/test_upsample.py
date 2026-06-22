"""Bilinear upsample kernel test — vs cv2.resize(INTER_LINEAR).

The CUDA kernel must match cv2's coordinate convention exactly:
half-pixel centers + replicate (clamp-to-edge) border.
"""

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

from conftest import have_cuda, skip_no_cuda  # noqa: E402

if have_cuda:
    import cv2  # noqa: E402
    from evm_cuda.batched import DeviceBuffer  # noqa: E402
    from evm_cuda import _evm_cuda  # noqa: E402


@skip_no_cuda
def test_bilinear_upsample_matches_cv2_power_of_2():
    """2x upsample (in_H/in_W even) must match cv2 bit-closely."""
    rng = np.random.default_rng(42)
    M, in_H, in_W = 3, 37, 29
    src = rng.uniform(-1, 1, size=(M, in_H, in_W, 3)).astype(np.float32)

    # cv2 reference
    expected = np.stack([
        cv2.resize(src[i], (in_W * 2, in_H * 2), interpolation=cv2.INTER_LINEAR)
        for i in range(M)], axis=0)

    # CUDA kernel
    d_in = DeviceBuffer.from_array(src)
    d_out = DeviceBuffer(M * in_H * 2 * in_W * 2 * 3 * 4)
    _evm_cuda.batched_bilinear_upsample_3ch(
        d_in.ptr, d_out.ptr, M, in_H, in_W, in_H * 2, in_W * 2)
    got = d_out.download_f32(M * in_H * 2 * in_W * 2 * 3).reshape(
        M, in_H * 2, in_W * 2, 3)

    assert got.shape == expected.shape
    # cv2 and our kernel use the same FP32 math; tiny rounding diffs in the
    # fused multiply-add. <1e-5 is well within the pipeline tolerance budget.
    assert np.abs(got - expected).max() < 1e-5


@skip_no_cuda
def test_bilinear_upsample_matches_cv2_odd_ratio():
    """Odd upsample ratio (not power of 2) — stresses the half-pixel formula."""
    rng = np.random.default_rng(7)
    M, in_H, in_W = 2, 5, 7
    src = rng.uniform(0, 1, size=(M, in_H, in_W, 3)).astype(np.float32)

    out_H, out_W = 11, 13  # odd ratio (5->11, 7->13)
    expected = np.stack([
        cv2.resize(src[i], (out_W, out_H), interpolation=cv2.INTER_LINEAR)
        for i in range(M)], axis=0)

    d_in = DeviceBuffer.from_array(src)
    d_out = DeviceBuffer(M * out_H * out_W * 3 * 4)
    _evm_cuda.batched_bilinear_upsample_3ch(
        d_in.ptr, d_out.ptr, M, in_H, in_W, out_H, out_W)
    got = d_out.download_f32(M * out_H * out_W * 3).reshape(M, out_H, out_W, 3)

    assert got.shape == expected.shape
    assert np.abs(got - expected).max() < 1e-5
