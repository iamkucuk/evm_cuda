#!/usr/bin/env python3
"""Stage-by-stage profiler for the batched color pipeline.

Measures steady-state per-stage GPU time over N timed iterations (default 5),
with a warmup iteration excluded from timing. Reports median + min/max.

Design principles (Harris methodology):
  - Measure FIRST, attack biggest bottleneck, one change at a time.
  - cudaDeviceSynchronize after every stage so perf_counter captures GPU time
    (batched_* wrappers are fire-and-forget — they queue on stream 0).
  - Pre-allocate ALL device buffers before timing so cudaMalloc doesn't
    contaminate kernel measurements.
  - Warmup run primes kernel JIT/binary load + cuFFT plan cache.
  - N timed iterations with median reporting excludes one-time costs.
  - Video decode (_read_frames) and encode (_write) are NOT timed —
    this measures the GPU pipeline only.

Usage:
  PYTHONPATH=$PWD/cuda python scripts/profile_color.py [N_ITERATIONS]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = ROOT / "cuda"
for p in (str(ROOT), str(CUDA_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from evm_cuda.batched import (  # noqa: E402
    DeviceBuffer, _read_frames, _d_binom5_sum1, _warmup_gpu_pool,
)
from evm_cuda import _evm_cuda  # noqa: E402

DATA = ROOT / "data"
VID = str(DATA / "face.mp4")
ALPHA = 50; LEVEL = 4; FL = 50/60; FH = 60/60; CHROM_ATT = 1.0; SR = 30.0

# Number of timed iterations. Override via command-line arg.
N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def sync():
    """Block until all queued GPU work finishes.

    The batched_* wrappers queue work on stream 0 and return immediately.
    Without sync, perf_counter only captures host launch overhead."""
    _evm_cuda.device_synchronize()


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


def main():
    # --- Load + decode video (NOT timed) ------------------------------------
    frames, fps = _read_frames(VID)
    n = len(frames)
    h, w = frames[0].shape[:2]
    print(f"clip: {n} frames, {h}x{w}, fps={fps:.1f}, timed iterations: {N_ITER}")

    hl, wl = h, w
    for _ in range(LEVEL):
        hl = (hl + 1) // 2
        wl = (wl + 1) // 2

    gain = np.array([ALPHA, ALPHA * CHROM_ATT, ALPHA * CHROM_ATT], dtype=np.float32)
    clip_u8 = np.stack(frames, axis=0)

    # --- Warmup memory pool (one-time, not timed) ---------------------------
    _warmup_gpu_pool()
    sync()

    # --- Pre-allocate ALL device buffers (not timed) ------------------------
    # Input clip upload happens once; reused across all iterations.
    d_clip = DeviceBuffer.from_array(clip_u8)
    d_ntsc = DeviceBuffer(n * h * w * 3 * 4)
    d_ntsc_planar = DeviceBuffer(n * 3 * h * w * 4)
    d_gdown_planar = DeviceBuffer(n * 3 * hl * wl * 4)
    # Stage 3 per-channel buffers (reused across 3 channels within each run).
    d_sig = DeviceBuffer(n * hl * wl * 4)
    d_bpass_out = DeviceBuffer(n * hl * wl * 4)
    d_filt = DeviceBuffer(n * hl * wl * 3 * 4)
    d_out_u8 = DeviceBuffer(n * h * w * 3)

    # Pre-allocate numpy buffers for the Stage 2b host round-trip.
    gdown_planar = np.empty(n * 3 * hl * wl, dtype=np.float32)
    filt = np.empty((n, hl, wl, 3), dtype=np.float32)

    # --- Define one pipeline run as a function returning per-stage times ----
    def run_once() -> dict[str, float]:
        st = {}

        # Stage 1: NTSC convert
        t0 = time.perf_counter()
        _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, d_ntsc.ptr, n, h, w)
        sync()
        st["1) color_cvt"] = time.perf_counter() - t0

        # Stage 2: planar transpose + blur_dn
        t0 = time.perf_counter()
        _evm_cuda.batched_to_planar_3ch(d_ntsc.ptr, d_ntsc_planar.ptr, n, h, w)
        _evm_cuda.batched_blur_dn_color(
            d_ntsc_planar.ptr, d_gdown_planar.ptr, n * 3, h, w, LEVEL,
            _d_binom5_sum1(), 5)
        sync()
        st["2) blur_dn"] = time.perf_counter() - t0

        # Stage 2b: D2H + reshape (host round-trip — known bottleneck #7)
        t0 = time.perf_counter()
        gdown = d_gdown_planar.download_f32(n * 3 * hl * wl).reshape(n, 3, hl, wl)
        gdown = np.ascontiguousarray(gdown.transpose(0, 2, 3, 1))
        st["2b) D2H + reshape"] = time.perf_counter() - t0

        # Stage 3: ideal bandpass per channel
        t0 = time.perf_counter()
        for c in range(3):
            sig = np.ascontiguousarray(gdown[..., c].reshape(n, hl * wl).T)
            d_sig.upload(sig)
            _evm_cuda.batched_ideal_bandpass(
                d_sig.ptr, d_bpass_out.ptr, n, hl * wl, FL, FH, SR)
            filt[..., c] = d_bpass_out.download_f32(n * hl * wl).reshape(
                hl * wl, n).T.reshape(n, hl, wl)
        st["3) ideal_bandpass"] = time.perf_counter() - t0

        # Stage 4: gain + upload + fused upsample+add+quantize + D2H
        t0 = time.perf_counter()
        filt_gained = np.ascontiguousarray(filt * gain)
        d_filt.upload(filt_gained)
        _evm_cuda.batched_upsample_add_quantize(
            d_ntsc.ptr, d_filt.ptr, d_out_u8.ptr,
            n, hl, wl, h, w, 1.0)
        out = d_out_u8.download_u8(n * h * w * 3).reshape(n, h, w, 3)
        st["4) upsample + render"] = time.perf_counter() - t0

        return st

    # --- Warmup run (primes kernel JIT, cuFFT cache; not timed) -------------
    run_once()

    # --- Timed runs ---------------------------------------------------------
    all_runs = [run_once() for _ in range(N_ITER)]

    # --- Report median + min/max per stage ----------------------------------
    stage_keys = list(all_runs[0].keys())
    print(f"\n{'Stage':<28s} {'median':>8s} {'min':>8s} {'max':>8s} {'%':>6s}")
    print("-" * 63)
    medians = {k: median([r[k] for r in all_runs]) for k in stage_keys}
    total_med = sum(medians.values())
    for k in stage_keys:
        vals = [r[k] for r in all_runs]
        pct = medians[k] / total_med * 100
        print(f"{k:<28s} {medians[k]:>7.4f}s {min(vals):>7.4f}s {max(vals):>7.4f}s {pct:>5.1f}%")
    print("-" * 63)
    print(f"{'Pipeline total (median)':.<28s} {total_med:>7.4f}s")


if __name__ == "__main__":
    main()
