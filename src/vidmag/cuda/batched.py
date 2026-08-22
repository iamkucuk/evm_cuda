"""Batched (device-resident) EVM pipelines.

The numpy-in/numpy-out wrappers in `_vidmag_cuda` each do cudaMalloc + H2D +
kernel + D2H + cudaFree per call. Early profiling
showed >95% of wall time is that overhead.

Design principle: minimize host<->device transfers.
  - Motion pipeline: ONE upload at entry + ONE download at exit. Everything
    in between (NTSC, pyramid, IIR, recon, render) stays on-device.
  - Color pipeline: ONE upload at entry + ONE download at exit. The bandpass
    (Stage 2b) is fully device-resident — the per-channel ideal filter runs as
    a single unified cuFFT batch over (N=hl*wl*3, T=n), so no host reshape or
    intermediate transfer is needed.
Each transfer is measured as its own stage (see the benchmark module).

This is harder to read than pipelines.py (explicit buffer management) but
the profiler justifies it: the old per-frame API did ~1773 binding calls;
these pipelines do ~15 batched calls with zero per-call transfers.

The spatial kernels (blur_dn, lpyr_build/recon) use batched variants that
process all n*3 slices per launch via grid.z = M, collapsing ~35k launches
into ~50. See bindings.cpp batched_lpyr_build / batched_blur_dn_color.

Two layers, as in ``vidmag.cpu.magnify``
-------------------------------------
* the **array cores** — ``color_gdown_ideal_core``, ``motion_lpyr_iir_core``
  and their ``_fp16_`` twins — take decoded frames already in memory,
  ``(T, H, W, 3)`` uint8 BGR, and return the magnified frames in the same shape
  and dtype. They touch no files. These are this backend's *override* of the
  Pipelines protocol (`docs/dev/PLAN.md` section 3c): the whole point of the
  module is that the stages are fused and device-resident, which op-by-op
  execution through the Ops protocol would throw away.
* the **path wrappers** — ``magnify_color_gdown_ideal`` and friends — decode
  (dropping the last ``_vidmag_cuda.drop_last`` frames, as the MATLAB reference
  does), call the core, write the output file when ``out_path`` is non-empty,
  and return float32 in [0, 1]. Signatures unchanged by the split.

The ``on_stage`` profiling hook belongs to the core, because the stages it
names are the core's. ``vidmag.cuda.benchmark`` still drives it through the path
wrapper, so nothing about the measured code path moved.

FP32 and FP16 bodies stay duplicated on purpose (section 3d rule 4): merging
them risks numeric drift, and README.md's accuracy figures are load-bearing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from . import _vidmag_cuda
from ._common import figure6_alpha_schedule, read_frames as _read_frames
# One frame validator for every backend: a wrong shape or dtype must produce the
# same message whether the user picked "cpu" or "cuda", so the CPU oracle's
# checker is imported rather than reimplemented.
from ..cpu.magnify import _as_frames


class DeviceBuffer:
    """Thin wrapper over pool-backed `_vidmag_cuda.DeviceBuffer` (free-list by size; no hot-path cudaFree)."""
    def __init__(self, nbytes: int):
        self._buf = _vidmag_cuda.DeviceBuffer(nbytes)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "DeviceBuffer":
        b = cls(arr.nbytes)
        b.upload(arr)
        return b

    def upload(self, arr: np.ndarray) -> None:
        self._buf.upload(arr)

    def download_f32(self, count: int) -> np.ndarray:
        return self._buf.download_f32(count)

    def download_u8(self, count: int) -> np.ndarray:
        return self._buf.download_u8(count)

    @property
    def ptr(self) -> int:
        return self._buf.ptr

    def ptr_at(self, float_offset: int) -> int:
        """Device pointer offset by float_offset elements (assumes 4-byte float)."""
        return self._buf.ptr + float_offset * 4

    def ptr_at_half(self, half_offset: int) -> int:
        """Device pointer offset by half_offset elements (2-byte __half)."""
        return self._buf.ptr + half_offset * 2

    @property
    def nbytes(self) -> int:
        return self._buf.nbytes


# Lazy-cached device-side filter pointers.
_D_BINOM5 = None
_D_BINOM5_SUM1 = None

def _d_binom5() -> int:
    global _D_BINOM5
    if _D_BINOM5 is None:
        _D_BINOM5 = _vidmag_cuda.d_binom5_ptr()
    return _D_BINOM5

def _d_binom5_sum1() -> int:
    global _D_BINOM5_SUM1
    if _D_BINOM5_SUM1 is None:
        _D_BINOM5_SUM1 = _vidmag_cuda.d_binom5_sum1_ptr()
    return _D_BINOM5_SUM1


# ---------------------------------------------------------------------------
# Shared host-side helpers (frame I/O, Figure-6 schedule)
# ---------------------------------------------------------------------------

def _warmup_gpu_pool():
    """Pre-touch the CUDA driver's memory pool so the first large cudaMalloc
    in the pipeline is instant.

    Without this, the first cudaMalloc(~100MB+) takes ~1s because the driver
    lazily sets up page tables on first large allocation. A quick alloc+free
    of 1GB warms the pool; all subsequent allocations (even larger ones) are
    then O(1). Measured: 1.0s -> 0.0s on H200."""
    # Allocate 1GB, free immediately. The driver retains the virtual->physical
    # mapping in its pool, so the next cudaMalloc reuses it.
    _vidmag_cuda.warmup_device_pool(1024 * 1024 * 1024)


def _warmup_gpu_pool_motion(n: int, h: int, w: int, levels: int):
    """Motion pipeline allocates larger buffers (up to ~2.5GB for baby.mp4).
    Warm up a pool big enough to cover the largest single allocation."""
    # Largest single alloc: band data = sum of level_sizes * n * 3 floats
    ch, cw = h, w
    total_per_slice = 0
    for _ in range(levels):
        total_per_slice += ch * cw
        ch, cw = (ch + 1) // 2, (cw + 1) // 2
    largest = total_per_slice * n * 3 * 4  # bytes
    # Round up to next GB
    nbytes = max(1024 * 1024 * 1024, ((largest + 1024*1024*1024 - 1) // (1024*1024*1024)) * (1024*1024*1024))
    _vidmag_cuda.warmup_device_pool(nbytes)

def _write(out_path: str | Path, frames_uint8: np.ndarray, fps: float) -> None:
    """Write a ``(T, H, W, 3)`` uint8 BGR frame array to an H.264 MP4.

    Delegates to the shared encoder in ``vidmag.io.h264`` (browser/VSCode-playable
    H.264 ``yuv420p`` +faststart via PyAV) so the CPU baseline (``vidmag.cpu``) and
    this CUDA port share one encode implementation.
    """
    from ..io.h264 import encode_h264
    encode_h264(frames_uint8, out_path, fps)


# ---------------------------------------------------------------------------
# Color pipeline (Gaussian downsample + ideal bandpass)
# ---------------------------------------------------------------------------
#
# Host<->device transfers (only 2 pipeline-level + 3 small bandpass round-trips):
#   1 H2D (whole clip u8 at entry)
#   1 D2H (final uint8 output at exit)
#   Stage 2b: 1 D2H of the Gaussian pyramid + 3 H2D/D2H for per-channel bandpass
#
# Everything else (color_cvt, blur_dn, upsample, render) is fully device-resident.
# The Stage 2b host round-trip is the remaining transfer bottleneck — a
# device-resident ideal_bandpass would eliminate it.

def color_gdown_ideal_core(
    frames_bgr_u8: np.ndarray,
    fps: float,
    *,
    alpha: float,
    level: int,
    fl: float,
    fh: float,
    chrom_attenuation: float = 1.0,
    sampling_rate: float | None = None,
    on_stage: "Callable[[str, Callable[[], object]], object] | None" = None,
) -> np.ndarray:
    """``amplify_spatial_Gdown_temporal_ideal.m``, array in / array out, fused.

    ``frames_bgr_u8`` is ``(T, H, W, 3)`` uint8 BGR and so is the result. One
    H2D at entry, one D2H at exit; every stage in between is device-resident.
    ``sampling_rate`` defaults to ``fps``.
    """
    def _stage(name, body):
        return body() if on_stage is None else on_stage(name, body)

    frames = _as_frames(frames_bgr_u8)
    if sampling_rate is None:
        sampling_rate = fps
    n = len(frames)
    h, w = frames[0].shape[:2]

    # (n, h, w, 3) uint8 BGR — the upload needs it C-contiguous.
    clip_u8 = np.ascontiguousarray(frames)

    _warmup_gpu_pool()  # first cudaMalloc is ~1s without this; ~0s with

    hl, wl = h, w
    for _ in range(level):
        hl = (hl + 1) // 2
        wl = (wl + 1) // 2
    N_band = hl * wl * 3

    # Preallocate outside timed stages so stage medians measure kernels.
    d_clip = _stage("0) H2D: clip", lambda: DeviceBuffer.from_array(clip_u8))

    d_ntsc = DeviceBuffer(n * h * w * 3 * 4)
    def _s1():
        _vidmag_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, d_ntsc.ptr, n, h, w)
        return None
    _stage("1) color_cvt", _s1)
    del d_clip

    d_ntsc_planar = DeviceBuffer(n * 3 * h * w * 4)
    d_gdown_planar = DeviceBuffer(n * 3 * hl * wl * 4)
    def _s2():
        _vidmag_cuda.batched_to_planar_3ch(d_ntsc.ptr, d_ntsc_planar.ptr, n, h, w)
        _vidmag_cuda.batched_blur_dn_color(
            d_ntsc_planar.ptr, d_gdown_planar.ptr, n * 3, h, w, level,
            _d_binom5_sum1(), 5)
        return None
    _stage("2) blur_dn", _s2)
    del d_ntsc_planar

    d_gdown_thwc = DeviceBuffer(n * hl * wl * 3 * 4)
    d_sig = DeviceBuffer(N_band * n * 4)
    d_filt = DeviceBuffer(N_band * n * 4)
    d_filt_thwc = DeviceBuffer(n * hl * wl * 3 * 4)
    def _s_bandpass():
        _vidmag_cuda.batched_planar_to_interleaved_3ch(
            d_gdown_planar.ptr, d_gdown_thwc.ptr, n, hl, wl)
        _vidmag_cuda.batched_thwc_to_nt(d_gdown_thwc.ptr, d_sig.ptr, n, N_band)
        _vidmag_cuda.batched_ideal_bandpass(
            d_sig.ptr, d_filt.ptr, n, N_band, fl, fh, sampling_rate)
        _vidmag_cuda.batched_nt_to_thwc_scaled(
            d_filt.ptr, d_filt_thwc.ptr, n, N_band, 1.0)
        gain_y = alpha
        gain_iq = alpha * chrom_attenuation
        _vidmag_cuda.batched_apply_channel_gain(
            d_filt_thwc.ptr, n, hl, wl, gain_y, gain_iq, gain_iq)
        return None
    _stage("2b) bandpass (device-resident)", _s_bandpass)
    del d_gdown_planar, d_gdown_thwc, d_sig, d_filt

    d_out_u8 = DeviceBuffer(n * h * w * 3)
    def _s4b():
        _vidmag_cuda.batched_upsample_add_quantize(
            d_ntsc.ptr, d_filt_thwc.ptr, d_out_u8.ptr,
            n, hl, wl, h, w, 1.0)
        return None
    _stage("4b) render", _s4b)

    out = _stage(
        "4c) D2H: output",
        lambda: d_out_u8.download_u8(n * h * w * 3).reshape(n, h, w, 3),
    )

    return out


def magnify_color_gdown_ideal(
    vid_path: str | Path,
    out_path: str | Path,
    *,
    alpha: float,
    level: int,
    fl: float,
    fh: float,
    chrom_attenuation: float = 1.0,
    sampling_rate: float | None = None,
    on_stage: "Callable[[str, Callable[[], object]], object] | None" = None,
) -> np.ndarray:
    """Run :func:`color_gdown_ideal_core` on ``vid_path``, write ``out_path``.

    Reads the clip (dropping the last ``_vidmag_cuda.drop_last`` frames, as the
    MATLAB reference does) and returns the magnified frames as float32 in
    [0, 1]. An empty ``out_path`` skips the file write.
    """
    frames, fps = _read_frames(vid_path)
    out = color_gdown_ideal_core(
        np.stack(frames, axis=0), fps,
        alpha=alpha, level=level, fl=fl, fh=fh,
        chrom_attenuation=chrom_attenuation,
        sampling_rate=sampling_rate,
        on_stage=on_stage,
    )
    if out_path:
        _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0


def color_gdown_ideal_fp16_core(
    frames_bgr_u8: np.ndarray,
    fps: float,
    *,
    alpha: float,
    level: int,
    fl: float,
    fh: float,
    chrom_attenuation: float = 1.0,
    sampling_rate: float | None = None,
    on_stage: "Callable[[str, Callable[[], object]], object] | None" = None,
) -> np.ndarray:
    """Color pipeline with FP16 NTSC storage, array in / array out.

    NTSC (the dominant persistent buffer, read by render) is stored as __half.
    All other buffers keep the FP32 layout of the FP32 pipeline: the Gaussian
    downsample output goes to FP32 for the cuFFT bandpass, and the filt signal
    (FFT output) stays FP32. Only the NTSC buffer read by the fused render
    kernel is halved, which is where the bandwidth win lands (render is ~73%
    of GPU time in the FP32 color pipeline).

    Deliberately a separate body from :func:`color_gdown_ideal_core` rather
    than a ``dtype`` parameter over one body — see the module docstring.
    """
    def _stage(name, body):
        return body() if on_stage is None else on_stage(name, body)

    frames = _as_frames(frames_bgr_u8)
    if sampling_rate is None:
        sampling_rate = fps
    n = len(frames)
    h, w = frames[0].shape[:2]

    clip_u8 = np.ascontiguousarray(frames)

    _warmup_gpu_pool()

    ntsc_floats = n * h * w * 3
    hl, wl = h, w
    for _ in range(level):
        hl = (hl + 1) // 2
        wl = (wl + 1) // 2
    N_band = hl * wl * 3

    # Preallocate outside timed stages (same lesson as motion f16): stage
    # medians should measure kernels, not DeviceBuffer/pool noise.
    d_clip = _stage("0) H2D: clip", lambda: DeviceBuffer.from_array(clip_u8))

    d_ntsc = DeviceBuffer(ntsc_floats * 2)
    def _s1():
        _vidmag_cuda.batched_bgr_u8_to_ntsc_f16(d_clip.ptr, d_ntsc.ptr, n, h, w)
        return None
    _stage("1) color_cvt", _s1)
    del d_clip

    d_ntsc_planar = DeviceBuffer(n * 3 * h * w * 2)  # __half
    d_gdown_planar = DeviceBuffer(n * 3 * hl * wl * 4)  # FP32 for FFT
    def _s2():
        _vidmag_cuda.batched_to_planar_3ch_f16(d_ntsc.ptr, d_ntsc_planar.ptr, n, h, w)
        _vidmag_cuda.batched_blur_dn_color_f16(
            d_ntsc_planar.ptr, d_gdown_planar.ptr, n * 3, h, w, level,
            _d_binom5_sum1(), 5)
        return None
    _stage("2) blur_dn", _s2)
    del d_ntsc_planar

    d_gdown_thwc = DeviceBuffer(n * hl * wl * 3 * 4)
    d_sig = DeviceBuffer(N_band * n * 4)
    d_filt = DeviceBuffer(N_band * n * 4)
    d_filt_thwc = DeviceBuffer(n * hl * wl * 3 * 4)
    def _s_bandpass():
        _vidmag_cuda.batched_planar_to_interleaved_3ch(
            d_gdown_planar.ptr, d_gdown_thwc.ptr, n, hl, wl)
        _vidmag_cuda.batched_thwc_to_nt(d_gdown_thwc.ptr, d_sig.ptr, n, N_band)
        _vidmag_cuda.batched_ideal_bandpass(
            d_sig.ptr, d_filt.ptr, n, N_band, fl, fh, sampling_rate)
        _vidmag_cuda.batched_nt_to_thwc_scaled(
            d_filt.ptr, d_filt_thwc.ptr, n, N_band, 1.0)
        gain_y = alpha
        gain_iq = alpha * chrom_attenuation
        _vidmag_cuda.batched_apply_channel_gain(
            d_filt_thwc.ptr, n, hl, wl, gain_y, gain_iq, gain_iq)
        return None
    _stage("2b) bandpass (device-resident)", _s_bandpass)
    del d_gdown_planar, d_gdown_thwc, d_sig, d_filt

    d_out_u8 = DeviceBuffer(n * h * w * 3)
    def _s4b():
        _vidmag_cuda.batched_upsample_add_quantize_f16(
            d_ntsc.ptr, d_filt_thwc.ptr, d_out_u8.ptr,
            n, hl, wl, h, w, 1.0)
        return None
    _stage("4b) render", _s4b)

    out = _stage(
        "4c) D2H: output",
        lambda: d_out_u8.download_u8(n * h * w * 3).reshape(n, h, w, 3),
    )

    return out


def magnify_color_gdown_ideal_fp16(
    vid_path: str | Path,
    out_path: str | Path,
    *,
    alpha: float,
    level: int,
    fl: float,
    fh: float,
    chrom_attenuation: float = 1.0,
    sampling_rate: float | None = None,
    on_stage: "Callable[[str, Callable[[], object]], object] | None" = None,
) -> np.ndarray:
    """Run :func:`color_gdown_ideal_fp16_core` on ``vid_path``, write ``out_path``."""
    frames, fps = _read_frames(vid_path)
    out = color_gdown_ideal_fp16_core(
        np.stack(frames, axis=0), fps,
        alpha=alpha, level=level, fl=fl, fh=fh,
        chrom_attenuation=chrom_attenuation,
        sampling_rate=sampling_rate,
        on_stage=on_stage,
    )
    if out_path:
        _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Motion pipeline (Laplacian pyramid + IIR bandpass)
# ---------------------------------------------------------------------------

def motion_lpyr_iir_core(
    frames_bgr_u8: np.ndarray,
    fps: float,
    *,
    alpha: float,
    lambda_c: float,
    r1: float,
    r2: float,
    chrom_attenuation: float = 0.1,
    exaggeration_factor: float = _vidmag_cuda.exaggeration_factor,
    on_stage: "Callable[[str, Callable[[], object]], object] | None" = None,
) -> np.ndarray:
    """``amplify_spatial_lpyr_temporal_iir.m``, array in / array out, fused.

    ``frames_bgr_u8`` is ``(T, H, W, 3)`` uint8 BGR and so is the result.

    ``fps`` is accepted and ignored, exactly as in the CPU core
    (``vidmag/cpu/magnify.py:431``): the r1/r2 IIR is a recursion on frame index
    with no sampling rate in it. It stays in the signature so all four cores
    share one call shape.
    """
    def _stage(name, body):
        return body() if on_stage is None else on_stage(name, body)

    frames = _as_frames(frames_bgr_u8)
    n = len(frames)
    h, w = frames[0].shape[:2]

    levels = 1
    hh, ww = h, w
    while hh >= 5 and ww >= 5:
        levels += 1; hh = (hh + 1) // 2; ww = (ww + 1) // 2

    alpha_sched = figure6_alpha_schedule(
        levels, alpha, lambda_c, h, w, exaggeration_factor=exaggeration_factor)

    level_sizes = []
    ch, cw = h, w
    for _ in range(levels):
        level_sizes.append((ch, cw))
        ch = (ch + 1) // 2; cw = (cw + 1) // 2

    clip_u8 = np.ascontiguousarray(frames)

    _warmup_gpu_pool_motion(n, h, w, levels)  # motion uses larger buffers

    lvl_sizes = [s[0] * s[1] for s in level_sizes]
    total_band_floats = sum(s * (n * 3) for s in lvl_sizes)
    level_offsets = []
    offset = 0
    for sz in lvl_sizes:
        level_offsets.append(offset)
        offset += sz * n * 3

    # DeviceBuffer is pool-backed (free-list by size). Allocate sequentially
    # so peak VRAM matches the old pipeline. Benchmark does an untimed warmup
    # call that fills the pool; timed stages then hit reuse.
    d_clip = _stage("0) H2D: clip", lambda: DeviceBuffer.from_array(clip_u8))

    d_ntsc = DeviceBuffer(n * h * w * 3 * 4)

    def _sA():
        _vidmag_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, d_ntsc.ptr, n, h, w)
        return None
    _stage("A) NTSC", _sA)
    del d_clip

    d_ntsc_planar = DeviceBuffer(n * 3 * h * w * 4)
    d_bands = DeviceBuffer(total_band_floats * 4)

    def _sB():
        _vidmag_cuda.batched_to_planar_3ch_chan_outer(
            d_ntsc.ptr, d_ntsc_planar.ptr, n, h, w)
        _vidmag_cuda.batched_lpyr_build(
            d_ntsc_planar.ptr, d_bands.ptr, n, h, w, levels, _d_binom5(), 5)
        return None
    _stage("B) lpyr_build", _sB)
    del d_ntsc_planar

    d_filtered = DeviceBuffer(total_band_floats * 4)

    def _sC():
        for l in range(levels):
            sz = lvl_sizes[l]
            a = float(alpha_sched[l])
            for c in range(3):
                sig_off = level_offsets[l] + c * n * sz
                _vidmag_cuda.batched_iir_bandpass_tn(
                    d_bands.ptr_at(sig_off),
                    d_filtered.ptr_at(sig_off),
                    n, sz, r1, r2, a)
        return None
    _stage("C) IIR", _sC)
    del d_bands

    d_delta_planar = DeviceBuffer(n * 3 * h * w * 4)

    def _sD1():
        _vidmag_cuda.batched_lpyr_recon(
            d_filtered.ptr, d_delta_planar.ptr, n, h, w, levels, _d_binom5(), 5)
        return None
    _stage("D1) recon", _sD1)
    del d_filtered

    d_out_u8 = DeviceBuffer(n * h * w * 3)

    def _sD2():
        _vidmag_cuda.batched_add_planar_quantize(
            d_ntsc.ptr, d_delta_planar.ptr, d_out_u8.ptr,
            n, h, w, chrom_attenuation)
        return None
    _stage("D2) render", _sD2)

    # --- Stage D2H: output frames download -----------------------------------
    out = _stage("D2H) output",
                 lambda: d_out_u8.download_u8(n * h * w * 3).reshape(n, h, w, 3))

    return out


def magnify_motion_lpyr_iir(
    vid_path: str | Path,
    out_path: str | Path,
    *,
    alpha: float,
    lambda_c: float,
    r1: float,
    r2: float,
    chrom_attenuation: float = 0.1,
    exaggeration_factor: float = _vidmag_cuda.exaggeration_factor,
    on_stage: "Callable[[str, Callable[[], object]], object] | None" = None,
) -> np.ndarray:
    """Run :func:`motion_lpyr_iir_core` on ``vid_path``, write ``out_path``.

    Reads the clip (dropping the last ``_vidmag_cuda.drop_last`` frames) and
    returns float32 in [0, 1]. An empty ``out_path`` skips the file write.
    """
    frames, fps = _read_frames(vid_path)
    out = motion_lpyr_iir_core(
        np.stack(frames, axis=0), fps,
        alpha=alpha, lambda_c=lambda_c, r1=r1, r2=r2,
        chrom_attenuation=chrom_attenuation,
        exaggeration_factor=exaggeration_factor,
        on_stage=on_stage,
    )
    if out_path:
        _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0


def motion_lpyr_iir_fp16_core(
    frames_bgr_u8: np.ndarray,
    fps: float,
    *,
    alpha: float,
    lambda_c: float,
    r1: float,
    r2: float,
    chrom_attenuation: float = 0.1,
    exaggeration_factor: float = _vidmag_cuda.exaggeration_factor,
    on_stage: "Callable[[str, Callable[[], object]], object] | None" = None,
) -> np.ndarray:
    """Motion pipeline with FP16 storage for large intermediates, array in/out.

    NTSC, planar, Laplacian bands, filtered bands, and delta use ``__half``.
    Spatial: half storage, float accumulate (cvt_in/out). IIR: TN + FP64 state.
    Bands half end-to-end; Stage A fused u8→half NTSC.
    Optional half-math spatial: batched_lpyr_*_f16_halfacc (slower, less accurate).

    Peak VRAM is lower than FP32: measured 8.4 GB on baby.mp4 (301 frames,
    544x960) against 16.3 GB for the FP32 path, so this one fits a 16 GB card
    and FP32 motion does not.

    ``fps`` is accepted and ignored (see :func:`motion_lpyr_iir_core`).
    Deliberately a separate body from the FP32 core — see the module docstring.
    """
    def _stage(name, body):
        return body() if on_stage is None else on_stage(name, body)

    frames = _as_frames(frames_bgr_u8)
    n = len(frames)
    h, w = frames[0].shape[:2]

    levels = 1
    hh, ww = h, w
    while hh >= 5 and ww >= 5:
        levels += 1; hh = (hh + 1) // 2; ww = (ww + 1) // 2

    alpha_sched = figure6_alpha_schedule(
        levels, alpha, lambda_c, h, w, exaggeration_factor=exaggeration_factor)

    level_sizes = []
    ch, cw = h, w
    for _ in range(levels):
        level_sizes.append((ch, cw))
        ch = (ch + 1) // 2; cw = (cw + 1) // 2

    clip_u8 = np.ascontiguousarray(frames)
    _warmup_gpu_pool_motion(n, h, w, levels)

    ntsc_floats = n * h * w * 3
    planar_floats = n * 3 * h * w
    lvl_sizes = [s[0] * s[1] for s in level_sizes]
    total_band_floats = sum(s * (n * 3) for s in lvl_sizes)
    level_offsets = []
    offset = 0
    for sz in lvl_sizes:
        level_offsets.append(offset)
        offset += sz * n * 3

    # Match FP32 packaging: allocate outside timed stages so stage medians
    # measure kernels, not pool/alloc noise (was hiding the f16 density win).
    d_clip = _stage("0) H2D: clip", lambda: DeviceBuffer.from_array(clip_u8))

    d_ntsc = DeviceBuffer(ntsc_floats * 2)
    def _sA():
        _vidmag_cuda.batched_bgr_u8_to_ntsc_f16(d_clip.ptr, d_ntsc.ptr, n, h, w)
        return None
    _stage("A) NTSC", _sA)
    del d_clip

    d_ntsc_planar = DeviceBuffer(planar_floats * 2)
    d_bands = DeviceBuffer(total_band_floats * 2)
    def _sB():
        _vidmag_cuda.batched_to_planar_3ch_chan_outer_f16(
            d_ntsc.ptr, d_ntsc_planar.ptr, n, h, w)
        _vidmag_cuda.batched_lpyr_build_f16(
            d_ntsc_planar.ptr, d_bands.ptr, n, h, w, levels,
            _d_binom5(), 5)
        return None
    _stage("B) lpyr_build", _sB)
    del d_ntsc_planar

    d_filtered = DeviceBuffer(total_band_floats * 2)
    def _sC():
        for l in range(levels):
            sz = lvl_sizes[l]
            a = float(alpha_sched[l])
            for c in range(3):
                sig_off = level_offsets[l] + c * n * sz
                _vidmag_cuda.batched_iir_bandpass_tn_f16(
                    d_bands.ptr_at_half(sig_off),
                    d_filtered.ptr_at_half(sig_off),
                    n, sz, r1, r2, a)
        return None
    _stage("C) IIR", _sC)
    del d_bands

    d_delta = DeviceBuffer(n * 3 * h * w * 2)
    def _sD1():
        _vidmag_cuda.batched_lpyr_recon_f16(
            d_filtered.ptr, d_delta.ptr, n, h, w, levels, _d_binom5(), 5)
        return None
    _stage("D1) recon", _sD1)
    del d_filtered

    d_out_u8 = DeviceBuffer(n * h * w * 3)
    def _sD2():
        _vidmag_cuda.batched_add_planar_quantize_f16(
            d_ntsc.ptr, d_delta.ptr, d_out_u8.ptr,
            n, h, w, chrom_attenuation)
        return None
    _stage("D2) render", _sD2)

    out = _stage(
        "D2H) output",
        lambda: d_out_u8.download_u8(n * h * w * 3).reshape(n, h, w, 3),
    )

    return out


def magnify_motion_lpyr_iir_fp16(
    vid_path: str | Path,
    out_path: str | Path,
    *,
    alpha: float,
    lambda_c: float,
    r1: float,
    r2: float,
    chrom_attenuation: float = 0.1,
    exaggeration_factor: float = _vidmag_cuda.exaggeration_factor,
    on_stage: "Callable[[str, Callable[[], object]], object] | None" = None,
) -> np.ndarray:
    """Run :func:`motion_lpyr_iir_fp16_core` on ``vid_path``, write ``out_path``."""
    frames, fps = _read_frames(vid_path)
    out = motion_lpyr_iir_fp16_core(
        np.stack(frames, axis=0), fps,
        alpha=alpha, lambda_c=lambda_c, r1=r1, r2=r2,
        chrom_attenuation=chrom_attenuation,
        exaggeration_factor=exaggeration_factor,
        on_stage=on_stage,
    )
    if out_path:
        _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0
