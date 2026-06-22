#!/usr/bin/env python3
"""Stage-by-stage profiler for the batched motion pipeline.

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
  PYTHONPATH=$PWD/cuda python scripts/profile_motion.py [N_ITERATIONS]
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
    DeviceBuffer, _read_frames, _d_binom5,
    _warmup_gpu_pool_motion, figure6_alpha_schedule,
)
from evm_cuda import _evm_cuda  # noqa: E402

DATA = ROOT / "data"
VID = str(DATA / "baby.mp4")
ALPHA = 10; LAMBDA_C = 16; R1 = 0.4; R2 = 0.05; CHROM_ATT = 0.1

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
    print(f"clip: {n} frames, {h}x{w}, fps={fps:.1f}")

    levels = 1; hh, ww = h, w
    while hh >= 5 and ww >= 5:
        levels += 1; hh = (hh + 1) // 2; ww = (ww + 1) // 2
    print(f"pyramid levels: {levels}, timed iterations: {N_ITER}")

    alpha_sched = figure6_alpha_schedule(levels, ALPHA, LAMBDA_C, h, w)
    level_sizes = []
    ch, cw = h, w
    for _ in range(levels):
        level_sizes.append((ch, cw))
        ch = (ch + 1) // 2; cw = (cw + 1) // 2
    lvl_sizes = [s[0] * s[1] for s in level_sizes]
    total_band_floats = sum(s * (n * 3) for s in lvl_sizes)

    clip_u8 = np.stack(frames, axis=0)

    # --- Warmup memory pool (one-time, not timed) ---------------------------
    _warmup_gpu_pool_motion(n, h, w, levels)
    sync()

    # --- Pre-allocate ALL device buffers (not timed) ------------------------
    # Input clip upload happens once; reused across all iterations.
    d_clip = DeviceBuffer.from_array(clip_u8)
    d_ntsc = DeviceBuffer(n * h * w * 3 * 4)
    d_ntsc_planar = DeviceBuffer(n * 3 * h * w * 4)
    d_bands = DeviceBuffer(total_band_floats * 4)
    d_filtered = DeviceBuffer(total_band_floats * 4)
    d_delta_planar = DeviceBuffer(n * 3 * h * w * 4)
    d_out_u8 = DeviceBuffer(n * h * w * 3)
    # Stage C temp buffers (max level size covers all levels).
    max_sz = max(lvl_sizes)
    d_nt = DeviceBuffer(n * max_sz * 4)
    d_filt_nt = DeviceBuffer(n * max_sz * 4)

    # Per-level offset table for Stage C.
    level_offsets = []
    off = 0
    for sz in lvl_sizes:
        level_offsets.append(off)
        off += sz * n * 3

    # --- Define one pipeline run as a function returning per-stage times ----
    def run_once() -> dict[str, float]:
        st = {}

        # Stage A: NTSC convert
        t0 = time.perf_counter()
        _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, d_ntsc.ptr, n, h, w)
        sync()
        st["A) NTSC convert"] = time.perf_counter() - t0

        # Stage B: planar transpose + lpyr_build
        t0 = time.perf_counter()
        _evm_cuda.batched_to_planar_3ch(d_ntsc.ptr, d_ntsc_planar.ptr, n, h, w)
        _evm_cuda.batched_lpyr_build(
            d_ntsc_planar.ptr, d_bands.ptr, n, h, w, levels, _d_binom5(), 5)
        sync()
        st["B) lpyr_build"] = time.perf_counter() - t0

        # Stage C: temporal IIR
        t0 = time.perf_counter()
        for l in range(levels):
            sz = lvl_sizes[l]
            for c in range(3):
                sig_off = level_offsets[l] + c * n * sz
                _evm_cuda.batched_thwc_to_nt(
                    d_bands.ptr_at(sig_off), d_nt.ptr, n, sz)
                _evm_cuda.batched_iir_bandpass(
                    d_nt.ptr, d_filt_nt.ptr, n, sz, R1, R2)
                dst_off = level_offsets[l] + c * n * sz
                _evm_cuda.batched_nt_to_thwc_scaled(
                    d_filt_nt.ptr, d_filtered.ptr_at(dst_off),
                    n, sz, alpha_sched[l])
        sync()
        st["C) temporal IIR"] = time.perf_counter() - t0

        # Stage D1: lpyr_recon
        t0 = time.perf_counter()
        _evm_cuda.batched_lpyr_recon(
            d_filtered.ptr, d_delta_planar.ptr, n, h, w, levels, _d_binom5(), 5)
        sync()
        st["D1) lpyr_recon"] = time.perf_counter() - t0

        # Stage D2: render (fused planar-delta add+quantize + D2H download)
        t0 = time.perf_counter()
        _evm_cuda.batched_add_planar_quantize(
            d_ntsc.ptr, d_delta_planar.ptr, d_out_u8.ptr,
            n, h, w, CHROM_ATT)
        out = d_out_u8.download_u8(n * h * w * 3).reshape(n, h, w, 3)
        st["D2) render"] = time.perf_counter() - t0

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
