"""color_cvt kernel tests — vs vidmag.rgb_to_yiq / vidmag.yiq_to_rgb."""

from __future__ import annotations

import numpy as np

import vidmag
from conftest import (
    TOL, abs_err, have_cuda, skip_no_cuda,
    BINOM5_CUDA, BINOM5_SUM1_CUDA,
)

if have_cuda:
    from vidmag.cuda import _vidmag_cuda


@skip_no_cuda
def test_binom5_constants_match_baseline():
    from vidmag.cpu.pyramids import BINOM5, BINOM5_SUM1
    assert abs_err(BINOM5_CUDA, BINOM5.astype(np.float32)) < 1e-7
    assert abs_err(BINOM5_SUM1_CUDA, BINOM5_SUM1.astype(np.float32)) < 1e-7


@skip_no_cuda
def test_bgr_u8_to_ntsc_matches_baseline():
    rng = np.random.default_rng(0)
    r = rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8)
    rgb = r[:, :, ::-1].astype(np.float64) / 255.0
    expected = vidmag.rgb_to_yiq(rgb).astype(np.float32)

    got = _vidmag_cuda.bgr_u8_to_ntsc_f32(r)
    assert got.shape == expected.shape == (64, 48, 3)
    assert abs_err(got, expected) < TOL["color_cvt"]


@skip_no_cuda
def test_ntsc_f32_to_bgr_u8_matches_baseline():
    rng = np.random.default_rng(1)
    yiq = rng.uniform(-0.5, 0.5, size=(64, 48, 3)).astype(np.float32)
    yiq[..., 0] += 0.5

    rgb = vidmag.yiq_to_rgb(yiq.astype(np.float64))
    rgb = np.clip(rgb, 0.0, 1.0)
    expected = np.round(rgb * 255.0).astype(np.uint8)[:, :, ::-1]

    got = _vidmag_cuda.ntsc_f32_to_bgr_u8(yiq)
    assert got.shape == expected.shape == (64, 48, 3)
    # Up to 1 LSB of rounding difference between CUDA rintf and numpy.
    assert (got.astype(int) - expected.astype(int)).max() <= 1


@skip_no_cuda
def test_batched_bgr_u8_to_ntsc_f16_near_f32():
    """Fused u8→half NTSC matches float NTSC after promote (half rounding)."""
    from vidmag.cuda.batched import DeviceBuffer

    rng = np.random.default_rng(3)
    T, H, W = 2, 32, 24
    bgr = rng.integers(0, 256, size=(T, H, W, 3), dtype=np.uint8)
    d_in = DeviceBuffer.from_array(np.ascontiguousarray(bgr))

    d_f32 = DeviceBuffer(T * H * W * 3 * 4)
    _vidmag_cuda.batched_bgr_u8_to_ntsc_f32(d_in.ptr, d_f32.ptr, T, H, W)
    ref = d_f32.download_f32(T * H * W * 3)

    d_f16 = DeviceBuffer(T * H * W * 3 * 2)
    _vidmag_cuda.batched_bgr_u8_to_ntsc_f16(d_in.ptr, d_f16.ptr, T, H, W)
    d_prom = DeviceBuffer(T * H * W * 3 * 4)
    _vidmag_cuda.f16_to_f32(d_f16.ptr, d_prom.ptr, T * H * W * 3)
    got = d_prom.download_f32(T * H * W * 3)

    err = abs_err(got, ref)
    assert err < 2e-3, f"fused half NTSC vs float NTSC err={err:.2e}"
