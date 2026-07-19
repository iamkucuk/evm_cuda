"""Direct tests for the batched device-resident kernels.

These exercise the batched bindings (batched_lpyr_build, batched_lpyr_recon,
batched_blur_dn_color, batched_add_planar_quantize, batched_upsample_add_quantize)
which internally use the batched spatial kernels (grid.z = M) and the
scatter/gather kernels for channel-major band layout.

The batched spatial kernels are mathematically identical to the single-slice
kernels tested in test_spatial.py / test_lpyr.py — the only difference is the
z-dimension. These tests verify the stride arithmetic, scatter/gather offsets,
and fused render kernels produce correct results.
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

from evm.pyramids import build_lpyr, recon_lpyr, blur_dn, max_pyr_ht, BINOM5, BINOM5_SUM1  # noqa: E402
from conftest import TOL, abs_err, have_cuda, skip_no_cuda, BINOM5_CUDA, BINOM5_SUM1_CUDA  # noqa: E402

if have_cuda:
    from evm_cuda import _evm_cuda  # noqa: E402
    from evm_cuda.batched import DeviceBuffer, _d_binom5, _d_binom5_sum1  # noqa: E402


@skip_no_cuda
@pytest.mark.parametrize("h,w", [(64, 64), (45, 33)])
def test_batched_lpyr_build_matches_single_slice(h, w):
    """batched_lpyr_build (M=n*3 slices at once) == per-slice lpyr_build.

    The binding always uses M = n_frames * 3 (3 channels per frame).
    Input: (n*3, H, W) planar float32. Output: channel-major band layout
    (level, chan, n_frames, spatial) with slice_off = chan * n_frames + frame.
    """
    rng = np.random.default_rng(42)
    n_frames = 3  # small batch for fast testing
    levels = 1 + max_pyr_ht((h, w), 5)
    M = n_frames * 3

    # Input: (M, H, W) — n_frames frames × 3 channels, planar.
    imgs = rng.random((M, h, w)).astype(np.float32)

    # --- Reference: per-slice lpyr_build (slice m = frame m//3, chan m%3) ---
    ref_bands = []  # ref_bands[m][level]
    for m in range(M):
        bands_list, _ = _evm_cuda.lpyr_build(
            np.ascontiguousarray(imgs[m]), levels, BINOM5_CUDA)
        ref_bands.append([np.ascontiguousarray(b, dtype=np.float32) for b in bands_list])

    # --- Batched: all M slices at once ---
    d_in = DeviceBuffer.from_array(imgs)
    # Compute output layout (must match batched_lpyr_build in bindings.cpp).
    level_sizes = []
    ch, cw = h, w
    level_hw = []
    for _ in range(levels):
        level_sizes.append(ch * cw)
        level_hw.append((ch, cw))
        ch = (ch + 1) // 2
        cw = (cw + 1) // 2
    total_floats = sum(s * M for s in level_sizes)
    d_out = DeviceBuffer(total_floats * 4)

    _evm_cuda.batched_lpyr_build(
        d_in.ptr, d_out.ptr, n_frames, h, w, levels, _d_binom5(), 5)

    # Download and compare. Channel-major layout: for slice m (frame=m//3, chan=m%3),
    # offset within level l = level_offset + (chan*n_frames + frame) * level_size.
    out = d_out.download_f32(total_floats).copy()
    level_offset = 0
    for l in range(levels):
        sz = level_sizes[l]
        lh, lw = level_hw[l]
        for m in range(M):
            frame = m // 3
            chan = m % 3
            slice_off = chan * n_frames + frame
            band_start = level_offset + slice_off * sz
            batched_band = out[band_start : band_start + sz].reshape(lh, lw)
            ref_band = ref_bands[m][l]
            assert abs_err(batched_band, ref_band) < TOL["corr_dn"], \
                f"level {l}, slice {m}: err={abs_err(batched_band, ref_band):.2e}"
        level_offset += sz * M


@skip_no_cuda
@pytest.mark.parametrize("h,w", [(64, 64), (45, 33)])
def test_batched_lpyr_recon_matches_single_slice(h, w):
    """batched_lpyr_recon (M=n*3 slices) == per-slice lpyr_recon.

    Reads channel-major band layout, outputs frame-major (M, H, W).
    """
    rng = np.random.default_rng(7)
    n_frames = 3
    levels = 1 + max_pyr_ht((h, w), 5)
    M = n_frames * 3

    imgs = rng.random((M, h, w)).astype(np.float32)

    # Build reference bands using per-slice API.
    ref_bands = []
    for m in range(M):
        bands_list, _ = _evm_cuda.lpyr_build(
            np.ascontiguousarray(imgs[m]), levels, BINOM5_CUDA)
        ref_bands.append([np.ascontiguousarray(b, dtype=np.float32) for b in bands_list])

    # Pack bands into channel-major layout (matches batched_lpyr_recon's read).
    level_sizes = []
    ch, cw = h, w
    for _ in range(levels):
        level_sizes.append(ch * cw)
        ch = (ch + 1) // 2
        cw = (cw + 1) // 2
    total_floats = sum(s * M for s in level_sizes)

    band_flat = np.empty(total_floats, dtype=np.float32)
    level_offset = 0
    for l in range(levels):
        sz = level_sizes[l]
        for m in range(M):
            frame = m // 3
            chan = m % 3
            slice_off = chan * n_frames + frame
            band_flat[level_offset + slice_off * sz :
                      level_offset + (slice_off + 1) * sz] = ref_bands[m][l].ravel()
        level_offset += sz * M

    d_bands = DeviceBuffer.from_array(np.ascontiguousarray(band_flat))
    d_out = DeviceBuffer(M * h * w * 4)

    _evm_cuda.batched_lpyr_recon(
        d_bands.ptr, d_out.ptr, n_frames, h, w, levels, _d_binom5(), 5)

    out = d_out.download_f32(M * h * w).reshape(M, h, w)

    # Compare against per-slice recon.
    for m in range(M):
        ref_recon = _evm_cuda.lpyr_recon(ref_bands[m], BINOM5_CUDA)
        assert abs_err(out[m], ref_recon) < TOL["lpyr_roundtrip"], \
            f"slice {m}: err={abs_err(out[m], ref_recon):.2e}"


@skip_no_cuda
@pytest.mark.parametrize("h,w", [(64, 64), (45, 33)])
def test_batched_blur_dn_color_matches_single_slice(h, w):
    """batched_blur_dn_color (M slices) == per-slice blur_dn.

    Note: batched_blur_dn_color takes M directly (no ×3), unlike lpyr.
    """
    rng = np.random.default_rng(99)
    M = 4  # M slices, passed directly (not multiplied by 3)
    nlevs = 3

    imgs = rng.random((M, h, w)).astype(np.float32)

    # Reference: per-slice.
    ref = []
    for m in range(M):
        cu = _evm_cuda.blur_dn(
            np.ascontiguousarray(imgs[m]), nlevs, BINOM5_SUM1_CUDA)
        ref.append(cu)

    # Batched (M passed directly as the batch count).
    hl, wl = h, w
    for _ in range(nlevs):
        hl = (hl + 1) // 2
        wl = (wl + 1) // 2

    d_in = DeviceBuffer.from_array(imgs)
    d_out = DeviceBuffer(M * hl * wl * 4)
    _evm_cuda.batched_blur_dn_color(
        d_in.ptr, d_out.ptr, M, h, w, nlevs,
        _d_binom5_sum1(), 5)

    out = d_out.download_f32(M * hl * wl).reshape(M, hl, wl)

    for m in range(M):
        assert abs_err(out[m], ref[m]) < TOL["blur_dn"], \
            f"slice {m}: err={abs_err(out[m], ref[m]):.2e}"


@skip_no_cuda
def test_batched_add_planar_quantize_matches_add_and_quantize():
    """Fused planar-delta add+quantize == separate transpose + add_and_quantize."""
    rng = np.random.default_rng(123)
    n, h, w = 3, 16, 12

    # NTSC frames (n, H, W, 3) interleaved.
    ntsc = rng.random((n, h, w, 3)).astype(np.float32)
    # Delta in planar layout (n*3, H, W).
    delta_planar = rng.random((n * 3, h, w)).astype(np.float32)
    chrom_att = 0.1

    # Reference: transpose planar -> interleaved, then add_and_quantize per-frame.
    delta_interleaved = np.empty((n, h, w, 3), dtype=np.float32)
    for f in range(n):
        for c in range(3):
            delta_interleaved[f, :, :, c] = delta_planar[f * 3 + c]

    ref_out = np.empty((n, h, w, 3), dtype=np.uint8)
    for f in range(n):
        # Apply chrom_att to I,Q channels of delta.
        d = delta_interleaved[f].copy()
        d[:, :, 1] *= chrom_att
        d[:, :, 2] *= chrom_att
        ref_out[f] = _evm_cuda.add_and_quantize(ntsc[f], d)

    # Batched fused kernel.
    d_ntsc = DeviceBuffer.from_array(np.ascontiguousarray(ntsc))
    d_delta = DeviceBuffer.from_array(np.ascontiguousarray(delta_planar))
    d_out = DeviceBuffer(n * h * w * 3)
    _evm_cuda.batched_add_planar_quantize(
        d_ntsc.ptr, d_delta.ptr, d_out.ptr, n, h, w, chrom_att)

    out = d_out.download_u8(n * h * w * 3).reshape(n, h, w, 3)

    assert np.array_equal(out, ref_out), \
        f"Mismatch: {(out != ref_out).sum()} pixels differ"


@skip_no_cuda
def test_batched_upsample_add_quantize_matches_two_kernel():
    """Fused upsample+add+quantize == separate upsample + add_and_quantize."""
    rng = np.random.default_rng(456)
    n = 3
    in_h, in_w = 8, 6
    out_h, out_w = 16, 12

    ntsc = rng.random((n, out_h, out_w, 3)).astype(np.float32)
    filt = rng.random((n, in_h, in_w, 3)).astype(np.float32)

    # Reference: upsample then add+quantize (chrom_att=1.0 for color).
    d_filt = DeviceBuffer.from_array(np.ascontiguousarray(filt))
    d_upsampled = DeviceBuffer(n * out_h * out_w * 3 * 4)
    _evm_cuda.batched_bilinear_upsample_3ch(
        d_filt.ptr, d_upsampled.ptr, n, in_h, in_w, out_h, out_w)
    ref_up = d_upsampled.download_f32(n * out_h * out_w * 3).reshape(n, out_h, out_w, 3)

    ref_out = np.empty((n, out_h, out_w, 3), dtype=np.uint8)
    for f in range(n):
        ref_out[f] = _evm_cuda.add_and_quantize(ntsc[f], ref_up[f])

    # Fused kernel.
    d_ntsc = DeviceBuffer.from_array(np.ascontiguousarray(ntsc))
    d_out = DeviceBuffer(n * out_h * out_w * 3)
    _evm_cuda.batched_upsample_add_quantize(
        d_ntsc.ptr, d_filt.ptr, d_out.ptr,
        n, in_h, in_w, out_h, out_w, 1.0)

    out = d_out.download_u8(n * out_h * out_w * 3).reshape(n, out_h, out_w, 3)

    assert np.array_equal(out, ref_out), \
        f"Mismatch: {(out != ref_out).sum()} pixels differ"


# ---------------------------------------------------------------------------
# FP16 color pipeline kernel tests
# ---------------------------------------------------------------------------

@skip_no_cuda
@pytest.mark.parametrize("h,w", [(64, 64), (45, 33)])
def test_batched_blur_dn_color_f16_matches_fp32(h, w):
    """FP16 blur_dn_color reads __half input, writes FP32 output.

    Must match the FP32 blur_dn_color within FP16 round-off (< 1e-2 abs).
    """
    rng = np.random.default_rng(99)
    M = 4
    nlevs = 3

    imgs_f32 = rng.random((M, h, w)).astype(np.float32)

    # FP32 reference.
    hl, wl = h, w
    for _ in range(nlevs):
        hl = (hl + 1) // 2
        wl = (wl + 1) // 2
    d_in_f32 = DeviceBuffer.from_array(imgs_f32)
    d_out_f32 = DeviceBuffer(M * hl * wl * 4)
    _evm_cuda.batched_blur_dn_color(
        d_in_f32.ptr, d_out_f32.ptr, M, h, w, nlevs,
        _d_binom5_sum1(), 5)
    ref = d_out_f32.download_f32(M * hl * wl).reshape(M, hl, wl)

    # FP16 path: convert input to __half, run f16 blur_dn_color.
    # upload() is raw-bytes, so np.float16 transfers correctly (2 bytes/element).
    imgs_f16 = np.ascontiguousarray(imgs_f32.astype(np.float16))
    d_in_f16 = DeviceBuffer.from_array(imgs_f16)
    d_out = DeviceBuffer(M * hl * wl * 4)
    _evm_cuda.batched_blur_dn_color_f16(
        d_in_f16.ptr, d_out.ptr, M, h, w, nlevs,
        _d_binom5_sum1(), 5)
    out = d_out.download_f32(M * hl * wl).reshape(M, hl, wl)

    for m in range(M):
        err = abs_err(out[m], ref[m])
        assert err < TOL["blur_dn"] * 100, \
            f"slice {m}: err={err:.2e} (FP16 round-off)"


@skip_no_cuda
def test_batched_upsample_add_quantize_f16_matches_fp32():
    """FP16 upsample_add_quantize reads __half NTSC, must match FP32 output.

    NTSC values are in [0, 1]; FP16 has ~3 decimal digits of precision there,
    so the uint8 quantized output should be bit-identical or within 1 ULP.
    """
    rng = np.random.default_rng(789)
    n = 3
    in_h, in_w = 8, 6
    out_h, out_w = 16, 12

    ntsc_f32 = rng.random((n, out_h, out_w, 3)).astype(np.float32)
    filt = rng.random((n, in_h, in_w, 3)).astype(np.float32)

    # FP32 reference.
    d_ntsc_f32 = DeviceBuffer.from_array(np.ascontiguousarray(ntsc_f32))
    d_filt = DeviceBuffer.from_array(np.ascontiguousarray(filt))
    d_out_f32 = DeviceBuffer(n * out_h * out_w * 3)
    _evm_cuda.batched_upsample_add_quantize(
        d_ntsc_f32.ptr, d_filt.ptr, d_out_f32.ptr,
        n, in_h, in_w, out_h, out_w, 1.0)
    ref = d_out_f32.download_u8(n * out_h * out_w * 3).reshape(n, out_h, out_w, 3)

    # FP16 path: NTSC stored as __half.
    ntsc_f16 = np.ascontiguousarray(ntsc_f32.astype(np.float16))
    d_ntsc_f16 = DeviceBuffer.from_array(ntsc_f16)
    d_out = DeviceBuffer(n * out_h * out_w * 3)
    _evm_cuda.batched_upsample_add_quantize_f16(
        d_ntsc_f16.ptr, d_filt.ptr, d_out.ptr,
        n, in_h, in_w, out_h, out_w, 1.0)
    out = d_out.download_u8(n * out_h * out_w * 3).reshape(n, out_h, out_w, 3)

    # Allow up to 2 ULP difference per channel (FP16 rounding in the YIQ->RGB
    # matrix multiply can flip the final rintf by 1).
    diff = np.abs(out.astype(np.int16) - ref.astype(np.int16))
    assert diff.max() <= 2, f"FP16 vs FP32 output differs by {diff.max()} (max allowed: 2)"
