"""Tests for the device-resident color bandpass (item 1, section A).

Validates the kernels that collapse the color pipeline's Stage 2b/3a/3b/3c/4a
host round-trips into a single device-resident sequence:

    d_gdown_planar (n*3, hl, wl)
      -> batched_planar_to_interleaved_3ch   -> (n, hl, wl, 3)
      -> batched_thwc_to_nt                  -> (N=hl*wl*3, T=n)
      -> batched_ideal_bandpass              -> (N, T)
      -> batched_nt_to_thwc_scaled (scale=1) -> (n, hl, wl, 3)
      -> batched_apply_channel_gain          -> (n, hl, wl, 3) with Y*alpha, I/Q*alpha*chromAtt

The whole sequence must match the per-channel host reference within
``TOL["ideal"]`` (cuFFT FP32 vs numpy FP64 — see DESIGN.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import TOL, abs_err, have_cuda, skip_no_cuda

if have_cuda:
    from vidmag.cuda import _vidmag_cuda
    from vidmag.cuda.batched import DeviceBuffer
    from vidmag.cpu.filters import ideal_bandpass


# ---------------------------------------------------------------------------
# batched_planar_to_interleaved_3ch: inverse of batched_to_planar_3ch
# ---------------------------------------------------------------------------

@skip_no_cuda
@pytest.mark.parametrize("n,hl,wl", [(2, 16, 12), (1, 33, 25), (3, 8, 8)])
def test_planar_to_interleaved_round_trip(n, hl, wl):
    """planar->interleaved->planar is bit-exact (pure layout transform)."""
    rng = np.random.default_rng(11)
    orig = rng.random((n, hl, wl, 3)).astype(np.float32)  # interleaved

    # interleaved -> planar (existing binding)
    planar = np.empty((n * 3, hl, wl), dtype=np.float32)
    d_interleaved = DeviceBuffer.from_array(np.ascontiguousarray(orig))
    d_planar = DeviceBuffer(n * 3 * hl * wl * 4)
    _vidmag_cuda.batched_to_planar_3ch(d_interleaved.ptr, d_planar.ptr, n, hl, wl)
    planar = d_planar.download_f32(n * 3 * hl * wl).reshape(n * 3, hl, wl)

    # planar -> interleaved (NEW binding under test)
    d_planar2 = DeviceBuffer.from_array(np.ascontiguousarray(planar))
    d_back = DeviceBuffer(n * hl * wl * 3 * 4)
    _vidmag_cuda.batched_planar_to_interleaved_3ch(
        d_planar2.ptr, d_back.ptr, n, hl, wl)
    back = d_back.download_f32(n * hl * wl * 3).reshape(n, hl, wl, 3)

    assert np.array_equal(back, orig), \
        f"round-trip mismatch: {(back != orig).sum()} elements differ"


# ---------------------------------------------------------------------------
# batched_apply_channel_gain: per-channel multiply over (n, H, W, 3)
# ---------------------------------------------------------------------------

@skip_no_cuda
@pytest.mark.parametrize("n,h,w", [(2, 16, 12), (3, 8, 8)])
def test_apply_channel_gain_batched(n, h, w):
    """Y*alpha, I*alpha*chromAtt, Q*alpha*chromAtt on (n,H,W,3)."""
    rng = np.random.default_rng(22)
    sig = rng.random((n, h, w, 3)).astype(np.float32)
    alpha, chrom_att = 50.0, 1.0
    g = [alpha, alpha * chrom_att, alpha * chrom_att]

    ref = sig * np.array(g, dtype=np.float32)  # numpy broadcast reference

    d_sig = DeviceBuffer.from_array(np.ascontiguousarray(sig))
    _vidmag_cuda.batched_apply_channel_gain(
        d_sig.ptr, n, h, w, g[0], g[1], g[2])
    out = d_sig.download_f32(n * h * w * 3).reshape(n, h, w, 3)

    assert abs_err(out, ref) < 1e-6, f"err={abs_err(out, ref):.2e}"


# ---------------------------------------------------------------------------
# End-to-end device-resident bandpass sequence vs per-channel host reference
# ---------------------------------------------------------------------------

@skip_no_cuda
def test_device_resident_bandpass_matches_per_channel_host():
    """The unified (N=hl*wl*3, T) device sequence == 3 per-channel host calls.

    This is the core correctness assertion for item 1: collapsing 4 host
    round-trips into 0 must produce output within TOL["ideal"] of the old
    per-channel host-reshape reference.
    """
    rng = np.random.default_rng(33)
    n, hl, wl = 40, 12, 10
    # Band [0.5, 2.0] rather than [0.83, 1.0]. With 40 frames at 30 fps the
    # frequency bins sit 0.75 Hz apart, so the narrow band fell between two of
    # them: the filter kept nothing and BOTH sides of this comparison were
    # all-zero arrays. The assertion held no matter what the device did. The
    # wider band keeps the bins at 0.75 and 1.5 Hz, so the two sides now carry
    # real signal and the comparison means something.
    fl, fh, sampling_rate = 0.5, 2.0, 30.0

    # Start from the same planar layout the pipeline hands us post-blur_dn.
    planar = rng.random((n * 3, hl, wl)).astype(np.float32)

    # --- Host reference: per-channel ideal_bandpass (the OLD Stage 2b-4a) ---
    gdown = planar.reshape(n, 3, hl, wl).transpose(0, 2, 3, 1)  # (n,hl,wl,3)
    gdown = np.ascontiguousarray(gdown)
    ref = np.empty_like(gdown)
    for c in range(3):
        sig = np.ascontiguousarray(gdown[..., c].reshape(n, hl * wl).T)  # (N,T)
        # axis=1 is the time axis of this (pixels, frames) layout, and time is
        # what the device kernel filters. The default axis=0 filtered across
        # pixels instead; that went unnoticed only because the old band made
        # both sides zero.
        out = ideal_bandpass(sig.astype(np.float64), fl, fh, sampling_rate, axis=1)
        ref[..., c] = np.ascontiguousarray(out.T).reshape(n, hl, wl)

    alpha, chrom_att = 50.0, 1.0
    gain = np.array([alpha, alpha * chrom_att, alpha * chrom_att],
                    dtype=np.float32)
    ref = ref * gain

    # Guard against this comparison quietly becoming vacuous again. If the band
    # and the clip length ever stop overlapping a frequency bin, the reference
    # collapses to zeros and the assertion below would pass against anything.
    assert np.abs(ref).max() > 1e-3, (
        "reference is all zeros: the band selects no frequency bins for this "
        "clip length, so this test would pass no matter what the device did"
    )

    # --- Device-resident sequence (the NEW code path) ---
    N = hl * wl * 3  # unified batch over all 3 channels
    d_planar = DeviceBuffer.from_array(np.ascontiguousarray(planar))
    d_thwc = DeviceBuffer(n * hl * wl * 3 * 4)
    _vidmag_cuda.batched_planar_to_interleaved_3ch(
        d_planar.ptr, d_thwc.ptr, n, hl, wl)

    d_sig = DeviceBuffer(N * n * 4)
    _vidmag_cuda.batched_thwc_to_nt(d_thwc.ptr, d_sig.ptr, n, N)

    d_filt = DeviceBuffer(N * n * 4)
    _vidmag_cuda.batched_ideal_bandpass(
        d_sig.ptr, d_filt.ptr, n, N, fl, fh, sampling_rate)

    d_filt_thwc = DeviceBuffer(n * hl * wl * 3 * 4)
    _vidmag_cuda.batched_nt_to_thwc_scaled(d_filt.ptr, d_filt_thwc.ptr, n, N, 1.0)

    _vidmag_cuda.batched_apply_channel_gain(
        d_filt_thwc.ptr, n, hl, wl, gain[0], gain[1], gain[2])

    out = d_filt_thwc.download_f32(n * hl * wl * 3).reshape(n, hl, wl, 3)

    # cuFFT FP32 vs numpy FP64 — the same tolerance the OLD per-channel path
    # was held to (DESIGN.md:130-134 justifies the 1e-4 slack).
    assert out.shape == ref.shape
    err = abs_err(out, ref.astype(np.float32))
    assert err < TOL["ideal"], f"err={err:.2e} (tol={TOL['ideal']:.0e})"
