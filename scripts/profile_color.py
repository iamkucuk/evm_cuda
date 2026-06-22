#!/usr/bin/env python3
"""Stage-by-stage profiler for the batched color pipeline (Phase 1h).

Mirrors the exact code path of batched.magnify_color_gdown_ideal, instrumenting
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
import cv2  # noqa: E402

from evm_cuda.batched import (  # noqa: E402
    DeviceBuffer, _read_frames, _d_binom5_sum1, _warmup_gpu_pool,
)
from evm_cuda import _evm_cuda  # noqa: E402

DATA = ROOT / "data"
VID = str(DATA / "face.mp4")
ALPHA = 50; LEVEL = 4; FL = 50/60; FH = 60/60; CHROM_ATT = 1.0; SR = 30.0


def main():
    frames, fps = _read_frames(VID)
    n = len(frames)
    h, w = frames[0].shape[:2]
    print(f"clip: {n} frames, {h}x{w}, fps={fps:.1f}")

    hl, wl = h, w
    for _ in range(LEVEL):
        hl = (hl + 1) // 2
        wl = (wl + 1) // 2

    gain = np.array([ALPHA, ALPHA * CHROM_ATT, ALPHA * CHROM_ATT], dtype=np.float32)
    clip_u8 = np.stack(frames, axis=0)

    def t():
        return time.perf_counter()

    timings = {}

    # Warmup (one-time, excluded from total)
    t0 = t()
    _warmup_gpu_pool()
    timings["warmup (one-time)"] = t() - t0

    # Stage 1: upload + color convert (ntsc stays on device)
    t0 = t()
    d_clip = DeviceBuffer.from_array(clip_u8)
    d_ntsc = DeviceBuffer(n * h * w * 3 * 4)
    _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, d_ntsc.ptr, n, h, w)
    timings["1) upload + color_cvt"] = t() - t0

    # Stage 2: planar transpose + blur_dn (on-device)
    t0 = t()
    d_ntsc_planar = DeviceBuffer(n * 3 * h * w * 4)
    _evm_cuda.batched_to_planar_3ch(d_ntsc.ptr, d_ntsc_planar.ptr, n, h, w)
    d_gdown_planar = DeviceBuffer(n * 3 * hl * wl * 4)
    _evm_cuda.batched_blur_dn_color(
        d_ntsc_planar.ptr, d_gdown_planar.ptr, n * 3, h, w, LEVEL,
        _d_binom5_sum1(), 5)
    timings["2) blur_dn (on-device)"] = t() - t0

    # Stage 2b: D2H + reshape (needed for bandpass)
    t0 = t()
    gdown = d_gdown_planar.download_f32(n * 3 * hl * wl).reshape(n, 3, hl, wl)
    gdown = np.ascontiguousarray(gdown.transpose(0, 2, 3, 1))
    timings["2b) D2H + reshape"] = t() - t0

    # Stage 3: ideal bandpass per channel
    t0 = t()
    filt = np.empty_like(gdown)
    for c in range(3):
        sig = np.ascontiguousarray(gdown[..., c].reshape(n, hl * wl).T)
        d_sig = DeviceBuffer.from_array(sig)
        d_out = DeviceBuffer(n * hl * wl * 4)
        _evm_cuda.batched_ideal_bandpass(
            d_sig.ptr, d_out.ptr, n, hl * wl, FL, FH, SR)
        filt[..., c] = d_out.download_f32(n * hl * wl).reshape(
            hl * wl, n).T.reshape(n, hl, wl)
    timings["3) ideal_bandpass"] = t() - t0

    # Stage 4: gain + upload + upsample + add+quantize + D2H
    t0 = t()
    filt = filt * gain
    d_filt = DeviceBuffer.from_array(np.ascontiguousarray(filt))
    d_upsampled = DeviceBuffer(n * h * w * 3 * 4)
    _evm_cuda.batched_bilinear_upsample_3ch(
        d_filt.ptr, d_upsampled.ptr, n, hl, wl, h, w)
    d_out_u8 = DeviceBuffer(n * h * w * 3)
    _evm_cuda.batched_add_and_quantize(
        d_ntsc.ptr, d_upsampled.ptr, d_out_u8.ptr, n * h, w)
    out = d_out_u8.download_u8(n * h * w * 3).reshape(n, h, w, 3)
    timings["4) upsample + render"] = t() - t0

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
