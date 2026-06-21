"""color_cvt kernel tests — vs evm.rgb_to_yiq / evm.yiq_to_rgb."""

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

import evm  # noqa: E402
from conftest import (  # noqa: E402
    TOL, abs_err, have_cuda, skip_no_cuda,
    BINOM5_CUDA, BINOM5_SUM1_CUDA,
)

if have_cuda:
    from evm_cuda import _evm_cuda  # noqa: E402


@skip_no_cuda
def test_binom5_constants_match_baseline():
    from evm.pyramids import BINOM5, BINOM5_SUM1
    assert abs_err(BINOM5_CUDA, BINOM5.astype(np.float32)) < 1e-7
    assert abs_err(BINOM5_SUM1_CUDA, BINOM5_SUM1.astype(np.float32)) < 1e-7


@skip_no_cuda
def test_bgr_u8_to_ntsc_matches_baseline():
    rng = np.random.default_rng(0)
    r = rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8)
    rgb = r[:, :, ::-1].astype(np.float64) / 255.0
    expected = evm.rgb_to_yiq(rgb).astype(np.float32)

    got = _evm_cuda.bgr_u8_to_ntsc_f32(r)
    assert got.shape == expected.shape == (64, 48, 3)
    assert abs_err(got, expected) < TOL["color_cvt"]


@skip_no_cuda
def test_ntsc_f32_to_bgr_u8_matches_baseline():
    rng = np.random.default_rng(1)
    yiq = rng.uniform(-0.5, 0.5, size=(64, 48, 3)).astype(np.float32)
    yiq[..., 0] += 0.5

    rgb = evm.yiq_to_rgb(yiq.astype(np.float64))
    rgb = np.clip(rgb, 0.0, 1.0)
    expected = np.round(rgb * 255.0).astype(np.uint8)[:, :, ::-1]

    got = _evm_cuda.ntsc_f32_to_bgr_u8(yiq)
    assert got.shape == expected.shape == (64, 48, 3)
    # Up to 1 LSB of rounding difference between CUDA rintf and numpy.
    assert (got.astype(int) - expected.astype(int)).max() <= 1
