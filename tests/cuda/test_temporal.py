"""Temporal filter kernel tests — iir / butter / ideal vs evm.filters."""

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

from evm.filters import (  # noqa: E402
    iir_bandpass, butter_bandpass, ideal_bandpass,
)
from evm_cuda.runtime import butter_bandpass_coeffs  # noqa: E402
from conftest import TOL, abs_err, have_cuda, skip_no_cuda  # noqa: E402

if have_cuda:
    from evm_cuda import _evm_cuda  # noqa: E402

FPS = 30.0
T = 300


def _sine(freq, amp=1.0):
    n = np.arange(T)
    return (amp * np.sin(2 * np.pi * freq * n / FPS)).astype(np.float32)


@skip_no_cuda
@pytest.mark.parametrize("freq", [0.5, 1.0, 3.0])
def test_iir_matches_baseline(freq):
    sig = _sine(freq).reshape(T, 1, 1, 1)  # (T, 1, 1, 1)
    expected = iir_bandpass(sig.astype(np.float64), 0.4, 0.05, axis=0) \
        .reshape(T).astype(np.float32)
    nt = np.ascontiguousarray(sig.reshape(T, 1).T)  # (N=1, T)
    got = _evm_cuda.iir_bandpass(nt, 0.4, 0.05).reshape(T)
    assert abs_err(got, expected) < TOL["iir"]


@skip_no_cuda
def test_iir_dc_goes_to_zero():
    """Matches the Python baseline's DC-suppression property."""
    dc = np.ones((1, T), dtype=np.float32)
    out = _evm_cuda.iir_bandpass(dc, 0.4, 0.05).reshape(T)
    # The Python baseline's steady-state output for DC is exactly 0.
    assert np.abs(out[T // 2:]).max() < TOL["iir"]


@skip_no_cuda
def test_butter_matches_baseline():
    sig = _sine(1.0)
    expected = butter_bandpass(sig.astype(np.float64), 0.5, 2.0, FPS, order=1) \
        .astype(np.float32)
    (b0h, b1h, a1h), (b0l, b1l, a1l) = butter_bandpass_coeffs(0.5, 2.0, FPS)
    nt = np.ascontiguousarray(sig.reshape(T, 1).T)  # (1, T)
    got = _evm_cuda.butter_bandpass(nt, b0h, b1h, a1h, b0l, b1l, a1l).reshape(T)
    assert abs_err(got, expected) < TOL["butter"]


@skip_no_cuda
def test_ideal_in_band_matches_baseline():
    sig = _sine(0.9)  # inside (0.83, 0.99)
    expected = ideal_bandpass(sig.astype(np.float64), 0.83, 0.99, FPS) \
        .astype(np.float32)
    nt = np.ascontiguousarray(sig.reshape(T, 1).T)
    got = _evm_cuda.ideal_bandpass(nt, 0.83, 0.99, FPS).reshape(T)
    assert abs_err(got, expected) < TOL["ideal"]


@skip_no_cuda
def test_ideal_out_of_band_matches_baseline():
    sig = _sine(5.0)  # well outside the band
    expected = ideal_bandpass(sig.astype(np.float64), 0.83, 0.99, FPS) \
        .astype(np.float32)
    nt = np.ascontiguousarray(sig.reshape(T, 1).T)
    got = _evm_cuda.ideal_bandpass(nt, 0.83, 0.99, FPS).reshape(T)
    # Both should be ~0; assert the absolute error is small.
    assert abs_err(got, expected) < TOL["ideal"]
