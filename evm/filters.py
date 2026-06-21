"""Temporal bandpass filters, faithful to the MIT MATLAB reference.

Three filters, one per MATLAB amplification function:

* :func:`ideal_bandpass` — brick-wall bandpass via FFT. Matches
  ``ideal_bandpassing.m``: a one-sided frequency mask
  ``Freq = (0..n-1)/n*sr``, ``mask = (Freq > wl) & (Freq < wh)`` applied to the
  raw FFT output bins. Because the mask covers both the positive band and its
  aliased negative counterpart, the in-band gain matches the reference.
* :func:`butter_bandpass` — subtraction of two first-order Butterworth
  lowpass filters. Matches ``amplify_spatial_lpyr_temporal_butter.m``, which
  calls MATLAB ``butter(1, Wn, 'low')`` (== :func:`scipy.signal.butter`) and
  applies the resulting second-order-section via a per-frame recursion. We use
  :func:`scipy.signal.lfilter` for the same result in batch form.
* :func:`iir_bandpass` — direct ``r1``/``r2`` coefficients. Matches
  ``amplify_spatial_lpyr_temporal_iir.m``: ``y1[n]=(1-r1)*y1[n-1]+r1*x[n]``,
  ``y2[n]=(1-r2)*y2[n-1]+r2*x[n]``, ``out = y1 - y2``, with the first-frame
  state initialised to ``x[0]``.

All three operate along ``axis=0`` of an arbitrary-trailing-shape array.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter


# ---------------------------------------------------------------------------
# Ideal (FFT brick-wall)
# ---------------------------------------------------------------------------


def ideal_bandpass(
    signal: np.ndarray, wl: float, wh: float, sampling_rate: float, axis: int = 0
) -> np.ndarray:
    """MATLAB ``ideal_bandpassing`` — FFT, zero bins outside (wl, wh), ifft.

    The mask uses one-sided frequencies ``Freq = (0..n-1)/n*sr`` and *strict*
    inequalities, matching the reference exactly. The ``real(ifft(...))`` is
    taken at the end.
    """
    n = signal.shape[axis]
    freqs = np.arange(n) / n * sampling_rate
    mask = ((freqs > wl) & (freqs < wh)).astype(signal.dtype)

    spectrum = np.fft.fft(signal, axis=axis)
    shape = [1] * signal.ndim
    shape[axis] = n
    spectrum *= mask.reshape(shape)
    return np.real(np.fft.ifft(spectrum, axis=axis))


# ---------------------------------------------------------------------------
# Butterworth (subtraction of two 1st-order lowpass filters)
# ---------------------------------------------------------------------------


def _butter_lowpass_coeffs(order: int, wn: float):
    """MATLAB ``butter(order, wn, 'low')`` == ``scipy.signal.butter``."""
    return butter(order, wn, btype="low")


def butter_bandpass(
    signal: np.ndarray,
    fl: float,
    fh: float,
    sampling_rate: float,
    order: int = 1,
    axis: int = 0,
) -> np.ndarray:
    """Difference of two first-order Butterworth lowpass filters.

    ``fh`` sets the high-pass cutoff (via the faster lowpass ``low_a/high_a``)
    and ``fl`` the low-pass cutoff; ``filtered = lowpass(fh) - lowpass(fl)``,
    as in ``amplify_spatial_lpyr_temporal_butter.m``. Cutoffs are normalised
    to Nyquist as MATLAB does (``butter`` takes ``Wn`` in ``[0, 1]`` where 1
    is Nyquist = sampling_rate/2).
    """
    nyq = sampling_rate / 2.0
    high_b, high_a = _butter_lowpass_coeffs(order, fh / nyq)
    low_b, low_a = _butter_lowpass_coeffs(order, fl / nyq)
    lp_high = lfilter(high_b, high_a, signal, axis=axis, zi=None)
    lp_low = lfilter(low_b, low_a, signal, axis=axis, zi=None)
    return lp_high - lp_low


# ---------------------------------------------------------------------------
# IIR (direct r1/r2 coefficients)
# ---------------------------------------------------------------------------


def iir_bandpass(
    signal: np.ndarray, r1: float, r2: float, axis: int = 0
) -> np.ndarray:
    """Direct-coefficient IIR bandpass from
    ``amplify_spatial_lpyr_temporal_iir.m``.

        y1[n] = (1-r1)*y1[n-1] + r1*x[n]
        y2[n] = (1-r2)*y2[n-1] + r2*x[n]
        out[n] = y1[n] - y2[n]      (r1 > r2)

    Initial state ``y1[0] = y2[0] = x[0]``. Runs causal, one frame at a time
    along ``axis``; the rest of the shape is filtered independently per
    location (a CUDA port can do this with one thread per spatial pixel).
    """
    if r1 <= r2:
        raise ValueError(f"require r1 > r2; got r1={r1}, r2={r2}")

    # Move time axis to front for a simple sequential loop.
    x = np.moveaxis(signal, axis, 0)
    n = x.shape[0]
    y1 = x[0].astype(np.float64).copy()
    y2 = x[0].astype(np.float64).copy()
    out = np.empty_like(x, dtype=np.float64)
    out[0] = y1 - y2
    for i in range(1, n):
        y1 = (1.0 - r1) * y1 + r1 * x[i]
        y2 = (1.0 - r2) * y2 + r2 * x[i]
        out[i] = y1 - y2
    return np.moveaxis(out, 0, axis).astype(signal.dtype, copy=False)
