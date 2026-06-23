"""Batched (device-resident) EVM pipelines.

The numpy-in/numpy-out wrappers in `_evm_cuda` each do cudaMalloc + H2D +
kernel + D2H + cudaFree per call. The profiler (docs/profile_baseline.txt)
showed >95% of wall time is that overhead.

Design principle: the ONLY host<->device transfers are:
  1. ONE upload of the input clip at pipeline entry.
  2. ONE download of the final uint8 output at pipeline exit.
Everything in between stays on-device via DeviceBuffer pointers.

This is harder to read than pipelines.py (explicit buffer management) but
the profiler justifies it: the old per-frame API did ~1773 binding calls;
these pipelines do ~15 batched calls with zero per-call transfers.

The spatial kernels (blur_dn, lpyr_build/recon) use batched variants that
process all n*3 slices per launch via grid.z = M, collapsing ~35k launches
into ~50. See bindings.cpp batched_lpyr_build / batched_blur_dn_color.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import _evm_cuda


class DeviceBuffer:
    """Thin wrapper over _evm_cuda.DeviceBuffer (RAII cudaMalloc'd region)."""
    def __init__(self, nbytes: int):
        self._buf = _evm_cuda.DeviceBuffer(nbytes)

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
        """Device pointer offset by float_offset elements (byte-safe)."""
        return self._buf.ptr + float_offset * 4

    @property
    def nbytes(self) -> int:
        return self._buf.nbytes


# Lazy-cached device-side filter pointers.
_D_BINOM5 = None
_D_BINOM5_SUM1 = None

def _d_binom5() -> int:
    global _D_BINOM5
    if _D_BINOM5 is None:
        _D_BINOM5 = _evm_cuda.d_binom5_ptr()
    return _D_BINOM5

def _d_binom5_sum1() -> int:
    global _D_BINOM5_SUM1
    if _D_BINOM5_SUM1 is None:
        _D_BINOM5_SUM1 = _evm_cuda.d_binom5_sum1_ptr()
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
    _evm_cuda.warmup_device_pool(1024 * 1024 * 1024)


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
    _evm_cuda.warmup_device_pool(nbytes)

def _read_frames(path: str | Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if len(frames) > _evm_cuda.drop_last:
        frames = frames[: len(frames) - _evm_cuda.drop_last]
    return frames, float(fps)


def _write(out_path: str | Path, frames_uint8: np.ndarray, fps: float) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _ = frames_uint8.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open for {out_path!r}")
    try:
        for i in range(t):
            writer.write(frames_uint8[i])
    finally:
        writer.release()


def figure6_alpha_schedule(
    n_levels: int, alpha: float, lambda_c: float,
    vid_h: int, vid_w: int,
    *, exaggeration_factor: float = _evm_cuda.exaggeration_factor,
) -> list[float]:
    delta = lambda_c / 8.0 / (1.0 + alpha)
    lam = (vid_h ** 2 + vid_w ** 2) ** 0.5 / 3.0
    coarse_first: list[float] = []
    for l in range(n_levels, 0, -1):
        if l == n_levels or l == 1:
            a = 0.0
        else:
            curr = (lam / delta / 8.0 - 1.0) * exaggeration_factor
            a = min(curr, alpha) if curr > alpha else curr
        coarse_first.append(a)
        lam /= 2.0
    return list(reversed(coarse_first))


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
# The Stage 2b host round-trip is the remaining transfer bottleneck — see the
# HANDOFF for the device-resident ideal_bandpass optimization opportunity.

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
) -> np.ndarray:
    frames, fps = _read_frames(vid_path)
    if sampling_rate is None:
        sampling_rate = fps
    n = len(frames)
    h, w = frames[0].shape[:2]

    clip_u8 = np.stack(frames, axis=0)  # (n, h, w, 3) uint8 BGR, C-contiguous

    _warmup_gpu_pool()  # first cudaMalloc is ~1s without this; ~0s with

    # --- Stage 1: batched color convert (whole clip, 1 kernel launch) ------
    # ntsc STAYS ON DEVICE — we never download it. The add-back in stage 4
    # happens on-device via batched_upsample_add_quantize, so the ntsc
    # buffer never crosses PCIe. Only the final uint8 output comes down.
    d_clip = DeviceBuffer.from_array(clip_u8)
    d_ntsc = DeviceBuffer(n * h * w * 3 * 4)
    _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, d_ntsc.ptr, n, h, w)

    # --- Stage 2: batched blur_dn downsample (whole clip on-device) ---------
    # Was: 873-call Python loop (291 frames * 3 channels), each call doing
    # cudaMalloc*5 + H2D*2 + kernel + D2H + cudaFree*5 = the color pipeline's
    # biggest hotspot (55% of time per the profiler).
    #
    # Now: 1 planar transpose kernel + 1 batched C++ loop over n*3 contiguous
    # slices (scratch allocated once). Data stays on-device the whole time.
    hl, wl = h, w
    for _ in range(level):
        hl = (hl + 1) // 2
        wl = (wl + 1) // 2

    d_ntsc_planar = DeviceBuffer(n * 3 * h * w * 4)
    _evm_cuda.batched_to_planar_3ch(d_ntsc.ptr, d_ntsc_planar.ptr, n, h, w)

    d_gdown_planar = DeviceBuffer(n * 3 * hl * wl * 4)
    _evm_cuda.batched_blur_dn_color(
        d_ntsc_planar.ptr, d_gdown_planar.ptr, n * 3, h, w, level,
        _d_binom5_sum1(), 5)

    # D2H once: reshape planar (n,3,hl,wl) -> interleaved (n,hl,wl,3) for the
    # bandpass stage, which expects channel as the last axis.
    gdown = d_gdown_planar.download_f32(n * 3 * hl * wl).reshape(n, 3, hl, wl)
    gdown = np.ascontiguousarray(gdown.transpose(0, 2, 3, 1))

    # --- Stage 3: ideal bandpass per channel (batched over all pixels) ------
    # ONE H2D + ONE kernel + ONE D2H per channel (3 total), vs n*1 in the old
    # path. This is where the big win is for the temporal stage.
    filt = np.empty_like(gdown)
    for c in range(3):
        sig = np.ascontiguousarray(gdown[..., c].reshape(n, hl * wl).T)
        d_sig = DeviceBuffer.from_array(sig)
        d_out = DeviceBuffer(n * hl * wl * 4)
        _evm_cuda.batched_ideal_bandpass(
            d_sig.ptr, d_out.ptr, n, hl * wl, fl, fh, sampling_rate)
        filt[..., c] = d_out.download_f32(n * hl * wl).reshape(hl * wl, n).T.reshape(n, hl, wl)

    # --- Stage 4: gain + upsample + add + quantize (ALL on-device) ----------
    # Was: 291x host cv2.resize(INTER_LINEAR) calls + np.stack + host add +
    # upload. Now: upload the small filt once, GPU bilinear upsample (matches
    # cv2 half-pixel + replicate convention bit-exactly), fused add+quantize.
    # The only host<->device transfer in this stage is the small filt upload
    # and the final uint8 output download.
    gain = np.array([alpha, alpha * chrom_attenuation, alpha * chrom_attenuation],
                    dtype=np.float32)
    filt = filt * gain

    d_filt = DeviceBuffer.from_array(np.ascontiguousarray(filt))
    # Fused upsample + add + quantize: reads the small filtered signal + the
    # full-res NTSC frame, interpolates inline, writes uint8 output directly.
    # Eliminates the n*h*w*3*4 float32 intermediate buffer + 1 kernel launch.
    d_out_u8 = DeviceBuffer(n * h * w * 3)
    _evm_cuda.batched_upsample_add_quantize(
        d_ntsc.ptr, d_filt.ptr, d_out_u8.ptr,
        n, hl, wl, h, w, 1.0)
    out = d_out_u8.download_u8(n * h * w * 3).reshape(n, h, w, 3)

    _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Motion pipeline (Laplacian pyramid + IIR bandpass)
# ---------------------------------------------------------------------------

def magnify_motion_lpyr_iir(
    vid_path: str | Path,
    out_path: str | Path,
    *,
    alpha: float,
    lambda_c: float,
    r1: float,
    r2: float,
    chrom_attenuation: float = 0.1,
    exaggeration_factor: float = _evm_cuda.exaggeration_factor,
) -> np.ndarray:
    frames, fps = _read_frames(vid_path)
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

    clip_u8 = np.stack(frames, axis=0)

    _warmup_gpu_pool_motion(n, h, w, levels)  # motion uses larger buffers

    # --- Stage A: batched NTSC convert (whole clip, 1 launch) --------------
    # ntsc STAYS ON DEVICE — used by Stage D's add_and_quantize (device-resident).
    d_clip = DeviceBuffer.from_array(clip_u8)
    d_ntsc = DeviceBuffer(n * h * w * 3 * 4)
    _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, d_ntsc.ptr, n, h, w)

    # --- Stage B: batched Laplacian pyramid build (whole clip on-device) ----
    # Bands stay on-device through Stage C (IIR) and Stage D (recon). The ONLY
    # host<->device transfers in Stages B-D are the alpha upload and the final
    # delta download. The channel-major output layout makes each (level,
    # channel) a contiguous (T=n, N=lh*lw) block for the temporal filter.
    d_ntsc_planar = DeviceBuffer(n * 3 * h * w * 4)
    _evm_cuda.batched_to_planar_3ch(d_ntsc.ptr, d_ntsc_planar.ptr, n, h, w)

    lvl_sizes = [s[0] * s[1] for s in level_sizes]
    total_band_floats = sum(s * (n * 3) for s in lvl_sizes)
    d_bands = DeviceBuffer(total_band_floats * 4)
    _evm_cuda.batched_lpyr_build(
        d_ntsc_planar.ptr, d_bands.ptr, n, h, w, levels,
        _d_binom5(), 5)

    # Per-level offset table (must match the C++ binding's layout).
    level_offsets = []
    offset = 0
    for sz in lvl_sizes:
        level_offsets.append(offset)
        offset += sz * n * 3

    # --- Stage C: temporal IIR (fully on-device, no host round-trip) ---------
    # Was: 27 H2D+kernel+D2H cycles (~5.8s, 52% of pipeline).
    # Now: per (level, channel), the n frames are a contiguous (T,N) block in
    # d_bands. Transpose to (N,T) on-device, run IIR, transpose back with
    # per-level alpha folded into the transpose (no separate scale pass).
    # All device-to-device — zero host transfers.
    d_filtered = DeviceBuffer(total_band_floats * 4)
    for l in range(levels):
        sz = lvl_sizes[l]
        for c in range(3):
            # Source: channel c's n frames at this level (T=n, N=sz), contiguous
            sig_off = level_offsets[l] + c * n * sz
            # Temp buffer for transpose (N,T)
            d_nt = DeviceBuffer(n * sz * 4)
            _evm_cuda.batched_thwc_to_nt(
                d_bands.ptr_at(sig_off), d_nt.ptr, n, sz)
            # IIR on (N,T)
            d_filt_nt = DeviceBuffer(n * sz * 4)
            _evm_cuda.batched_iir_bandpass(
                d_nt.ptr, d_filt_nt.ptr, n, sz, r1, r2)
            # Transpose back (N,T) -> (T,N) with alpha_sched[l] folded in —
            # replaces a separate scale_inplace launch.
            dst_off = level_offsets[l] + c * n * sz
            _evm_cuda.batched_nt_to_thwc_scaled(
                d_filt_nt.ptr, d_filtered.ptr_at(dst_off), n, sz, alpha_sched[l])

    # --- Stage D: batched recon + device-resident render --------------------
    # lpyr_recon output stays on-device in planar (n*3, H, W) layout. The render
    # kernel reads delta directly from planar layout (folding the transpose
    # inline), adds to NTSC with chromAtt, and writes uint8 BGR. No intermediate
    # interleaved buffer, no separate transpose pass.
    d_delta_planar = DeviceBuffer(n * 3 * h * w * 4)
    _evm_cuda.batched_lpyr_recon(
        d_filtered.ptr, d_delta_planar.ptr, n, h, w, levels, _d_binom5(), 5)

    # Fused planar-delta add + quantize (keep d_ntsc from Stage A).
    d_out_u8 = DeviceBuffer(n * h * w * 3)
    _evm_cuda.batched_add_planar_quantize(
        d_ntsc.ptr, d_delta_planar.ptr, d_out_u8.ptr,
        n, h, w, chrom_attenuation)
    out = d_out_u8.download_u8(n * h * w * 3).reshape(n, h, w, 3)

    _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0


def magnify_motion_lpyr_iir_fp16(
    vid_path: str | Path,
    out_path: str | Path,
    *,
    alpha: float,
    lambda_c: float,
    r1: float,
    r2: float,
    chrom_attenuation: float = 0.1,
    exaggeration_factor: float = _evm_cuda.exaggeration_factor,
) -> np.ndarray:
    """Motion pipeline with FP16 intermediate storage.

    Same algorithm as magnify_motion_lpyr_iir, but stores the large buffers
    (NTSC, bands) in FP16 to halve VRAM usage (23 GB -> 12 GB for baby.mp4).
    All compute kernels run in FP32. FP16 to FP32 conversion happens at
    buffer boundaries.

    The IIR accumulator stays FP64 (already is). The question is whether the
    FP16 quantization steps accumulate enough error to break the <0.01 RMSE
    end-to-end tolerance.
    """
    frames, fps = _read_frames(vid_path)
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

    clip_u8 = np.stack(frames, axis=0)
    _warmup_gpu_pool_motion(n, h, w, levels)

    ntsc_floats = n * h * w * 3

    # --- Stage A: NTSC convert (FP32 compute), store as FP16 ----------------
    d_clip = DeviceBuffer.from_array(clip_u8)
    d_ntsc_f32 = DeviceBuffer(ntsc_floats * 4)
    d_ntsc_f16 = DeviceBuffer(ntsc_floats * 2)
    _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, d_ntsc_f32.ptr, n, h, w)
    _evm_cuda.f32_to_f16(d_ntsc_f32.ptr, d_ntsc_f16.ptr, ntsc_floats)
    del d_ntsc_f32

    # --- Stage B: lpyr_build with FP16 scratch -------------------------------
    # Convert FP16 NTSC to FP32, transpose to planar, convert planar to FP16.
    # The batched_lpyr_build_f16 allocates __half scratch (3.6 GB vs 7.3 GB).
    d_ntsc_f32 = DeviceBuffer(ntsc_floats * 4)
    _evm_cuda.f16_to_f32(d_ntsc_f16.ptr, d_ntsc_f32.ptr, ntsc_floats)

    planar_floats = n * 3 * h * w
    d_ntsc_planar = DeviceBuffer(planar_floats * 4)
    _evm_cuda.batched_to_planar_3ch(d_ntsc_f32.ptr, d_ntsc_planar.ptr, n, h, w)
    del d_ntsc_f32

    d_ntsc_planar_f16 = DeviceBuffer(planar_floats * 2)
    _evm_cuda.f32_to_f16(d_ntsc_planar.ptr, d_ntsc_planar_f16.ptr, planar_floats)
    del d_ntsc_planar

    lvl_sizes = [s[0] * s[1] for s in level_sizes]
    total_band_floats = sum(s * (n * 3) for s in lvl_sizes)
    d_bands = DeviceBuffer(total_band_floats * 4)
    _evm_cuda.batched_lpyr_build_f16(
        d_ntsc_planar_f16.ptr, d_bands.ptr, n, h, w, levels,
        _d_binom5(), 5)
    del d_ntsc_planar_f16

    level_offsets = []
    offset = 0
    for sz in lvl_sizes:
        level_offsets.append(offset)
        offset += sz * n * 3

    # --- Stage C: temporal IIR (FP32 compute) -------------------------------
    d_filtered = DeviceBuffer(total_band_floats * 4)
    for l in range(levels):
        sz = lvl_sizes[l]
        for c in range(3):
            sig_off = level_offsets[l] + c * n * sz
            d_nt = DeviceBuffer(n * sz * 4)
            _evm_cuda.batched_thwc_to_nt(
                d_bands.ptr_at(sig_off), d_nt.ptr, n, sz)
            d_filt_nt = DeviceBuffer(n * sz * 4)
            _evm_cuda.batched_iir_bandpass(
                d_nt.ptr, d_filt_nt.ptr, n, sz, r1, r2)
            dst_off = level_offsets[l] + c * n * sz
            _evm_cuda.batched_nt_to_thwc_scaled(
                d_filt_nt.ptr, d_filtered.ptr_at(dst_off), n, sz, alpha_sched[l])
    del d_bands

    # --- Stage D: recon + render (FP32 compute from FP16-stored NTSC) -------
    d_ntsc_f32 = DeviceBuffer(ntsc_floats * 4)
    _evm_cuda.f16_to_f32(d_ntsc_f16.ptr, d_ntsc_f32.ptr, ntsc_floats)
    del d_ntsc_f16

    d_delta_planar = DeviceBuffer(n * 3 * h * w * 4)
    _evm_cuda.batched_lpyr_recon(
        d_filtered.ptr, d_delta_planar.ptr, n, h, w, levels, _d_binom5(), 5)
    del d_filtered

    d_out_u8 = DeviceBuffer(n * h * w * 3)
    _evm_cuda.batched_add_planar_quantize(
        d_ntsc_f32.ptr, d_delta_planar.ptr, d_out_u8.ptr,
        n, h, w, chrom_attenuation)
    out = d_out_u8.download_u8(n * h * w * 3).reshape(n, h, w, 3)

    _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0
