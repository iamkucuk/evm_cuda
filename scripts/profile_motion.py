#!/usr/bin/env python3
"""Stage-by-stage profiler for the batched motion pipeline (Phase 3).

Mirrors the exact code path of batched.magnify_motion_lpyr_iir, instrumenting
each stage with wall-clock timing. Run on a GPU node.

Follows the Harris methodology: measure first, attack the biggest bottleneck,
one change at a time. This script gives the per-stage breakdown needed to
pick the next optimization target.
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


def main():
    frames, fps = _read_frames(VID)
    n = len(frames)
    h, w = frames[0].shape[:2]
    print(f"clip: {n} frames, {h}x{w}, fps={fps:.1f}")

    levels = 1; hh, ww = h, w
    while hh >= 5 and ww >= 5:
        levels += 1; hh = (hh + 1) // 2; ww = (ww + 1) // 2
    print(f"pyramid levels: {levels}")

    alpha_sched = figure6_alpha_schedule(levels, ALPHA, LAMBDA_C, h, w)
    level_sizes = []
    ch, cw = h, w
    for _ in range(levels):
        level_sizes.append((ch, cw))
        ch = (ch + 1) // 2; cw = (cw + 1) // 2

    clip_u8 = np.stack(frames, axis=0)

    def t():
        return time.perf_counter()

    timings = {}

    # Warmup (one-time, excluded from total)
    t0 = t()
    _warmup_gpu_pool_motion(n, h, w, levels)
    timings["warmup (one-time)"] = t() - t0

    # Stage A: NTSC convert + download
    t0 = t()
    d_clip = DeviceBuffer.from_array(clip_u8)
    d_ntsc = DeviceBuffer(n * h * w * 3 * 4)
    _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, d_ntsc.ptr, n, h, w)
    ntsc = d_ntsc.download_f32(n * h * w * 3).reshape(n, h, w, 3)
    timings["A) NTSC convert + D2H"] = t() - t0

    # Stage B: planar transpose + lpyr_build (on-device)
    t0 = t()
    d_ntsc_planar = DeviceBuffer(n * 3 * h * w * 4)
    _evm_cuda.batched_to_planar_3ch(d_ntsc.ptr, d_ntsc_planar.ptr, n, h, w)
    lvl_sizes = [s[0] * s[1] for s in level_sizes]
    total_band_floats = sum(s * (n * 3) for s in lvl_sizes)
    d_bands = DeviceBuffer(total_band_floats * 4)
    _evm_cuda.batched_lpyr_build(
        d_ntsc_planar.ptr, d_bands.ptr, n, h, w, levels, _d_binom5(), 5)
    timings["B) lpyr_build (on-device)"] = t() - t0

    # Stage C: temporal IIR (on-device, no host transfers)
    t0 = t()
    level_offsets = []
    off = 0
    for sz in lvl_sizes:
        level_offsets.append(off)
        off += sz * n * 3
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
                d_nt.ptr, d_filt_nt.ptr, n, sz, R1, R2)
            _evm_cuda.batched_scale_inplace(d_filt_nt.ptr, n * sz, alpha_sched[l])
            dst_off = level_offsets[l] + c * n * sz
            _evm_cuda.batched_nt_to_thwc(
                d_filt_nt.ptr, d_filtered.ptr_at(dst_off), n, sz)
    timings["C) temporal IIR (on-device)"] = t() - t0

    # Stage D: recon + download + render
    t0 = t()
    d_delta_planar = DeviceBuffer(n * 3 * h * w * 4)
    _evm_cuda.batched_lpyr_recon(
        d_filtered.ptr, d_delta_planar.ptr, n, h, w, levels, _d_binom5(), 5)
    delta_planar = d_delta_planar.download_f32(n * 3 * h * w).reshape(n, 3, h, w)
    delta = np.ascontiguousarray(delta_planar.transpose(0, 2, 3, 1))
    out = np.empty((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        d = _evm_cuda.attenuate_chrom(
            np.ascontiguousarray(delta[i], dtype=np.float32), CHROM_ATT)
        out[i] = _evm_cuda.add_and_quantize(ntsc[i], d)
    timings["D) recon + render"] = t() - t0

    # Report
    pipeline_keys = [k for k in timings if not k.startswith("warmup")]
    total = sum(timings[k] for k in pipeline_keys)
    print(f"\n{'Stage':<35s} {'Time':>8s} {'%':>6s}")
    print("-" * 52)
    for k in pipeline_keys:
        pct = timings[k] / total * 100
        print(f"{k:<35s} {timings[k]:>7.3f}s {pct:>5.1f}%")
    print("-" * 52)
    print(f"{'Pipeline total':.<35s} {total:>7.3f}s")
    print(f"{'(+ one-time warmup)':.<35s} {timings['warmup (one-time)']:>7.3f}s")


if __name__ == "__main__":
    main()
