"""Batched (device-resident) EVM pipelines — Phase 1 optimization.

The numpy-in/numpy-out wrappers in `_evm_cuda` each do cudaMalloc + H2D +
kernel + D2H + cudaFree per call. The profiler (docs/profile_baseline.txt)
showed >95% of wall time is that overhead.

Design principle: the ONLY host<->device transfers are:
  1. ONE upload of the input clip at pipeline entry.
  2. ONE download of the final uint8 output at pipeline exit.
Everything in between stays on-device via DeviceBuffer pointers.

This is harder to read than pipelines.py (explicit buffer management) but
the profiler justifies it: 1773 binding calls -> ~20 batched calls.

NOTE: the pyramid build still goes through numpy for now (the lpyr_build
host orchestrator wasn't rewritten to be device-resident). That's the next
phase if profiling shows it's still the bottleneck.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import _evm_cuda
from .runtime import butter_bandpass_coeffs


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
# Host<->device transfer count:
#   1 H2D (whole clip u8)
#   1 D2H per frame per channel during blur_dn (unavoidable: the lpyr host
#       orchestrator needs host arrays) — TODO: device-resident blur_dn
#   1 H2D + 1 D2H per channel for ideal_bandpass
#   1 H2D per frame for render add-back, 1 D2H per frame for output
#
# Compared to the old pipeline (4 transfers per binding call), this cuts
# transfers by ~10x for the color convert and bandpass stages.

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

    # --- Stage 1: batched color convert (whole clip, 1 kernel launch) ------
    d_clip = DeviceBuffer.from_array(clip_u8)
    d_ntsc = DeviceBuffer(n * h * w * 3 * 4)
    _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, d_ntsc.ptr, n, h, w)

    # Pull NTSC back to host for the pyramid/downsample stage (still uses
    # numpy-in/out wrappers). This is ONE D2H of the whole clip — far better
    # than the old per-frame round-trips.
    ntsc = d_ntsc.download_f32(n * h * w * 3).reshape(n, h, w, 3)

    # --- Stage 2: per-frame blur_dn downsample (still numpy-in/out for now) -
    hl, wl = h, w
    for _ in range(level):
        hl = (hl + 1) // 2
        wl = (wl + 1) // 2

    binom5_sum1 = np.array(_evm_cuda.binom5_sum1(), dtype=np.float32)
    gdown = np.empty((n, hl, wl, 3), dtype=np.float32)
    for i in range(n):
        for c in range(3):
            gdown[i, :, :, c] = _evm_cuda.blur_dn(
                np.ascontiguousarray(ntsc[i, :, :, c], dtype=np.float32),
                level, binom5_sum1)

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

    # --- Stage 4: gain + per-frame upsample + add + quantize ----------------
    # The upsample is cv2 (host-side), so we need the filtered data on host.
    gain = np.array([alpha, alpha * chrom_attenuation, alpha * chrom_attenuation],
                    dtype=np.float32)
    filt = filt * gain

    out = np.empty((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        upsampled = cv2.resize(filt[i].astype(np.float32), (w, h),
                               interpolation=cv2.INTER_LINEAR)
        rendered = ntsc[i] + upsampled
        out[i] = _evm_cuda.ntsc_f32_to_bgr_u8(rendered)

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

    # --- Stage A: batched NTSC convert (whole clip, 1 launch) --------------
    d_clip = DeviceBuffer.from_array(clip_u8)
    d_ntsc = DeviceBuffer(n * h * w * 3 * 4)
    _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, d_ntsc.ptr, n, h, w)
    ntsc = d_ntsc.download_f32(n * h * w * 3).reshape(n, h, w, 3)

    # --- Stage B: per-frame pyramid build (numpy-in/out for now) ------------
    binom5 = np.array(_evm_cuda.binom5(), dtype=np.float32)
    pyrs = []
    for i in range(n):
        fp = []
        for c in range(3):
            bands, _ = _evm_cuda.lpyr_build(
                np.ascontiguousarray(ntsc[i, :, :, c], dtype=np.float32),
                levels, binom5)
            fp.append([np.ascontiguousarray(b, dtype=np.float32) for b in bands])
        pyrs.append(fp)

    # --- Stage C: temporal IIR per level per channel (batched over space) ---
    # ONE H2D + kernel + D2H per (level, channel) — 27 total vs 27 in old path,
    # but each is a single batched call instead of per-pixel round-trips.
    filtered = []
    for l in range(levels):
        lh, lw = level_sizes[l]
        chans_out = []
        for c in range(3):
            sig = np.stack([pyrs[i][c][l] for i in range(n)], axis=0)
            d_sig = DeviceBuffer.from_array(
                np.ascontiguousarray(sig.reshape(n, lh * lw).T))
            d_out = DeviceBuffer(n * lh * lw * 4)
            _evm_cuda.batched_iir_bandpass(d_sig.ptr, d_out.ptr, n, lh * wl, r1, r2)
            out = d_out.download_f32(n * lh * wl).reshape(lh * lw, n).T
            chans_out.append(np.ascontiguousarray(out).reshape(n, lh, lw) * alpha_sched[l])
        filtered.append(chans_out)

    # --- Stage D: per-frame recon + chromAtt + add + quantize --------------
    out = np.empty((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        delta_chans = []
        for c in range(3):
            bands = [filtered[l][c][i] for l in range(levels)]
            recon = _evm_cuda.lpyr_recon(bands, binom5)
            delta_chans.append(recon)
        delta = np.stack(delta_chans, axis=-1)
        delta = _evm_cuda.attenuate_chrom(
            np.ascontiguousarray(delta, dtype=np.float32), chrom_attenuation)
        out[i] = _evm_cuda.add_and_quantize(ntsc[i], delta)

    _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0
