"""Temporal filter tests, matched to the MATLAB reference behaviour.

Three filters, each tested with an in-band and out-of-band pure sinusoid:

* ``ideal_bandpass`` uses MATLAB's one-sided mask with *strict* inequalities,
  so the band edge itself is rejected.
* ``butter_bandpass`` is order-1; the in-band tone survives, the far-out band
  is suppressed.
* ``iir_bandpass`` runs the direct r1/r2 recursion; the steady-state response
  peaks inside (r2, r1).
"""

from __future__ import annotations

import numpy as np
import pytest

from evm.filters import butter_bandpass, ideal_bandpass, iir_bandpass

FPS = 30.0
T = 300


def _sine(freq: float, amp: float = 1.0) -> np.ndarray:
    n = np.arange(T)
    return (amp * np.sin(2 * np.pi * freq * n / FPS)).astype(np.float64)


def _peak_amplitude(filtered: np.ndarray) -> float:
    return float(np.abs(np.fft.rfft(filtered - filtered.mean())).max())


def test_ideal_strict_inequalities_at_band_edge() -> None:
    # wl=0.83, wh=1.0: a 1.0 Hz tone sits ON the upper edge -> rejected (strict <)
    edge = ideal_bandpass(_sine(1.0), 0.83, 1.0, FPS)
    assert _peak_amplitude(edge) < 1.0


def test_ideal_passes_in_band_and_rejects_out_of_band() -> None:
    passed = ideal_bandpass(_sine(0.9), 0.83, 0.99, FPS)
    rejected = ideal_bandpass(_sine(5.0), 0.83, 0.99, FPS)
    assert _peak_amplitude(passed) > 50.0
    assert _peak_amplitude(rejected) < 1.0


def test_ideal_preserves_shape() -> None:
    sig = _sine(0.9).reshape(T, 1, 1, 1)
    out = ideal_bandpass(sig, 0.83, 0.99, FPS)
    assert out.shape == sig.shape


def test_butter_attenuates_out_of_band() -> None:
    passed = butter_bandpass(_sine(1.0), 0.5, 2.0, FPS, order=1)
    rejected = butter_bandpass(_sine(8.0), 0.5, 2.0, FPS, order=1)
    assert _peak_amplitude(passed) > _peak_amplitude(rejected) * 3


def test_iir_rejects_dc_and_high_freq() -> None:
    # r1=0.4, r2=0.05 -> band roughly in the low Hz range
    low = iir_bandpass(_sine(0.5), 0.4, 0.05)[T // 2 :]
    high = iir_bandpass(_sine(10.0), 0.4, 0.05)[T // 2 :]
    assert _peak_amplitude(low) > _peak_amplitude(high)


def test_iir_requires_r1_gt_r2() -> None:
    with pytest.raises(ValueError):
        iir_bandpass(_sine(1.0), 0.05, 0.4)


def test_iir_dc_input_goes_to_zero() -> None:
    # Constant input -> both lowpass states converge to the constant -> diff = 0
    dc = np.ones(T)
    out = iir_bandpass(dc, 0.4, 0.05)
    assert np.abs(out[T // 2 :]).max() < 1e-9
