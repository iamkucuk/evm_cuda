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

import warnings

import numpy as np
import pytest

from vidmag.cpu.filters import butter_bandpass, ideal_bandpass, iir_bandpass

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


def test_ideal_bandpass_warns_when_the_band_selects_no_bins():
    """A band narrower than the frequency resolution silently returns zeros.

    A short clip resolves frequencies only in steps of ``sampling_rate / n``, so
    a narrow band can fall entirely between two bins. The filter is then a
    correct no-op, the pipeline hands back its input unamplified, and nothing
    looks wrong. That is the failure this warning exists to make visible: the
    shipped ``pulse`` preset (0.833-1.0 Hz) selects nothing at all below roughly
    181 frames at 30 fps, which is most short test clips.
    """
    sig = np.random.default_rng(0).standard_normal((90, 4)).astype(np.float64)

    with pytest.warns(RuntimeWarning, match="selected no frequency bins"):
        out = ideal_bandpass(sig, 50 / 60, 1.0, 30.0)
    assert np.array_equal(out, np.zeros_like(out)), "no bins kept must give zeros"

    # The same band over a long enough signal keeps at least one bin and must
    # not warn.
    long_sig = np.random.default_rng(0).standard_normal((300, 4)).astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        kept = ideal_bandpass(long_sig, 50 / 60, 1.0, 30.0)
    assert np.abs(kept).max() > 0.0, "a band with bins in it must pass signal"
