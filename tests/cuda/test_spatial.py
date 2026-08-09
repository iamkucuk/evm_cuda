"""Spatial primitives tests — corr_dn and up_conv vs evm.cpu.pyramids."""

from __future__ import annotations

import numpy as np
import pytest

from evm.cpu.pyramids import corr_dn_axis, up_conv_axis, BINOM5
from conftest import TOL, abs_err, have_cuda, skip_no_cuda

if have_cuda:
    from evm.cuda import _evm_cuda


@skip_no_cuda
@pytest.mark.parametrize("h,w", [(64, 64), (45, 33), (32, 96)])
def test_corr_dn_rows_matches_baseline(h, w):
    rng = np.random.default_rng(0)
    img = rng.random((h, w)).astype(np.float32)
    expected = corr_dn_axis(img.astype(np.float64), BINOM5, axis=0).astype(np.float32)
    got = _evm_cuda.corr_dn_rows(img, BINOM5.astype(np.float32))
    assert got.shape == expected.shape
    assert abs_err(got, expected) < TOL["corr_dn"]


@skip_no_cuda
@pytest.mark.parametrize("h,w", [(64, 64), (45, 33), (32, 96)])
def test_corr_dn_cols_matches_baseline(h, w):
    rng = np.random.default_rng(1)
    img = rng.random((h, w)).astype(np.float32)
    expected = corr_dn_axis(img.astype(np.float64), BINOM5, axis=1).astype(np.float32)
    got = _evm_cuda.corr_dn_cols(img, BINOM5.astype(np.float32))
    assert got.shape == expected.shape
    assert abs_err(got, expected) < TOL["corr_dn"]


@skip_no_cuda
@pytest.mark.parametrize("h,w", [(64, 64), (45, 33)])
def test_up_conv_rows_matches_baseline(h, w):
    rng = np.random.default_rng(2)
    small = corr_dn_axis(rng.random((h, w)), BINOM5, axis=0)  # (h/2, w)
    expected = up_conv_axis(small, BINOM5, axis=0, out_size=h).astype(np.float32)
    got = _evm_cuda.up_conv_rows(
        small.astype(np.float32), h, BINOM5.astype(np.float32))
    assert got.shape == expected.shape == (h, w)
    assert abs_err(got, expected) < TOL["up_conv"]


@skip_no_cuda
@pytest.mark.parametrize("h,w", [(64, 64), (45, 33)])
def test_up_conv_cols_matches_baseline(h, w):
    rng = np.random.default_rng(3)
    small = corr_dn_axis(rng.random((h, w)), BINOM5, axis=1)  # (h, w/2)
    expected = up_conv_axis(small, BINOM5, axis=1, out_size=w).astype(np.float32)
    got = _evm_cuda.up_conv_cols(
        small.astype(np.float32), w, BINOM5.astype(np.float32))
    assert got.shape == expected.shape == (h, w)
    assert abs_err(got, expected) < TOL["up_conv"]
