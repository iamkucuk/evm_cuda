#!/usr/bin/env python3
"""Stage-by-stage comparison: Python CPU vs CUDA FP32 vs CUDA FP16.

Mirrors the motion pipeline's exact stages for all three paths, measuring
GPU/CPU time per stage with sync boundaries. Decode/encode excluded.
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

import numpy as np
import cv2

DATA = ROOT / "data"
VID = str(DATA / "baby.mp4")
ALPHA = 10; LAMBDA_C = 16; R1 = 0.4; R2 = 0.05; CHROM_ATT = 0.1
N_ITER = 5


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n//2] if n % 2 == 1 else (s[n//2-1] + s[n//2]) / 2


def read_frames():
    cap = cv2.VideoCapture(VID)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames[:len(frames)-10], fps


def main():
    frames, fps = read_frames()
    n = len(frames)
    h, w = frames[0].shape[:2]
    clip_u8 = np.stack(frames, axis=0)
    print(f"clip: {n} frames, {h}x{w}, fps={fps:.1f}, iterations: {N_ITER}\n")

    levels = 1; hh, ww = h, w
    while hh >= 5 and ww >= 5:
        levels += 1; hh = (hh + 1) // 2; ww = (ww + 1) // 2

    level_sizes = []
    ch, cw = h, w
    for _ in range(levels):
        level_sizes.append((ch, cw))
        ch = (ch + 1) // 2; cw = (cw + 1) // 2

    # ================================================================
    # 1. PYTHON BASELINE (CPU) — total only (per-stage not exposed)
    # ================================================================
    print("Measuring Python baseline (CPU)...")
    import evm

    def run_python():
        t0 = time.perf_counter()
        evm.magnify_motion_lpyr_iir(
            VID, "/tmp/_py_motion.mp4",
            alpha=ALPHA, lambda_c=LAMBDA_C, r1=R1, r2=R2,
            chrom_attenuation=CHROM_ATT)
        return time.perf_counter() - t0

    run_python()  # warmup
    py_times = [run_python() for _ in range(3)]
    py_total = median(py_times)
    print(f"  CPU: {py_total:.3f}s (total, decode+encode excluded by evm internals)\n")

    # ================================================================
    # 2. CUDA FP32
    # ================================================================
    print("Measuring CUDA FP32...")
    from evm_cuda.batched import (DeviceBuffer, _read_frames, _d_binom5,
                                   _warmup_gpu_pool_motion, figure6_alpha_schedule)
    from evm_cuda import _evm_cuda

    def sync():
        _evm_cuda.device_synchronize()

    alpha_sched = figure6_alpha_schedule(levels, ALPHA, LAMBDA_C, h, w)
    lvl_sizes = [s[0]*s[1] for s in level_sizes]
    total_band_floats = sum(s * (n*3) for s in lvl_sizes)

    _warmup_gpu_pool_motion(n, h, w, levels)
    sync()

    level_offsets = []
    off = 0
    for sz in lvl_sizes:
        level_offsets.append(off)
        off += sz * n * 3

    d_clip = DeviceBuffer.from_array(clip_u8)
    d_out_u8 = DeviceBuffer(n * h * w * 3)
    max_sz = max(lvl_sizes)

    def run_fp32():
        st = {}
        # Stage A
        t0 = time.perf_counter()
        ntsc = DeviceBuffer(n*h*w*3*4)
        _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, ntsc.ptr, n, h, w)
        sync()
        st["A) NTSC convert"] = time.perf_counter() - t0

        # Stage B
        t0 = time.perf_counter()
        planar = DeviceBuffer(n*3*h*w*4)
        _evm_cuda.batched_to_planar_3ch(ntsc.ptr, planar.ptr, n, h, w)
        bands = DeviceBuffer(total_band_floats*4)
        _evm_cuda.batched_lpyr_build(planar.ptr, bands.ptr, n, h, w, levels, _d_binom5(), 5)
        sync()
        st["B) lpyr_build"] = time.perf_counter() - t0
        del planar

        # Stage C
        t0 = time.perf_counter()
        filtered = DeviceBuffer(total_band_floats*4)
        nt_buf = DeviceBuffer(n*max_sz*4)
        filt_buf = DeviceBuffer(n*max_sz*4)
        for l in range(levels):
            sz = lvl_sizes[l]
            for c in range(3):
                sig_off = level_offsets[l] + c*n*sz
                _evm_cuda.batched_thwc_to_nt(bands.ptr_at(sig_off), nt_buf.ptr, n, sz)
                _evm_cuda.batched_iir_bandpass(nt_buf.ptr, filt_buf.ptr, n, sz, R1, R2)
                dst_off = level_offsets[l] + c*n*sz
                _evm_cuda.batched_nt_to_thwc_scaled(filt_buf.ptr, filtered.ptr_at(dst_off), n, sz, alpha_sched[l])
        sync()
        st["C) temporal IIR"] = time.perf_counter() - t0
        del bands, nt_buf, filt_buf

        # Stage D
        t0 = time.perf_counter()
        delta = DeviceBuffer(n*3*h*w*4)
        _evm_cuda.batched_lpyr_recon(filtered.ptr, delta.ptr, n, h, w, levels, _d_binom5(), 5)
        _evm_cuda.batched_add_planar_quantize(ntsc.ptr, delta.ptr, d_out_u8.ptr, n, h, w, CHROM_ATT)
        sync()
        st["D) recon + render"] = time.perf_counter() - t0
        del ntsc, filtered, delta
        return st

    run_fp32()
    fp32_runs = [run_fp32() for _ in range(N_ITER)]
    fp32_med = {k: median([r[k] for r in fp32_runs]) for k in fp32_runs[0]}
    print(f"  FP32: {sum(fp32_med.values()):.3f}s\n")

    # ================================================================
    # 3. CUDA FP16
    # ================================================================
    print("Measuring CUDA FP16...")
    ntsc_floats = n*h*w*3
    planar_floats = n*3*h*w

    def run_fp16():
        st = {}
        # Stage A: NTSC convert + f32->f16
        t0 = time.perf_counter()
        ntsc_f32 = DeviceBuffer(ntsc_floats*4)
        ntsc_f16 = DeviceBuffer(ntsc_floats*2)
        _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, ntsc_f32.ptr, n, h, w)
        _evm_cuda.f32_to_f16(ntsc_f32.ptr, ntsc_f16.ptr, ntsc_floats)
        sync()
        st["A) NTSC convert"] = time.perf_counter() - t0
        del ntsc_f32

        # Stage B: f16->f32 + planar + f32->f16 + lpyr_build_f16
        t0 = time.perf_counter()
        ntsc_f32 = DeviceBuffer(ntsc_floats*4)
        _evm_cuda.f16_to_f32(ntsc_f16.ptr, ntsc_f32.ptr, ntsc_floats)
        planar = DeviceBuffer(planar_floats*4)
        _evm_cuda.batched_to_planar_3ch(ntsc_f32.ptr, planar.ptr, n, h, w)
        del ntsc_f32
        planar_f16 = DeviceBuffer(planar_floats*2)
        _evm_cuda.f32_to_f16(planar.ptr, planar_f16.ptr, planar_floats)
        del planar
        bands = DeviceBuffer(total_band_floats*4)
        _evm_cuda.batched_lpyr_build_f16(planar_f16.ptr, bands.ptr, n, h, w, levels, _d_binom5(), 5)
        del planar_f16
        sync()
        st["B) lpyr_build"] = time.perf_counter() - t0

        # Stage C: temporal IIR (FP32)
        t0 = time.perf_counter()
        filtered = DeviceBuffer(total_band_floats*4)
        nt_buf = DeviceBuffer(n*max_sz*4)
        filt_buf = DeviceBuffer(n*max_sz*4)
        for l in range(levels):
            sz = lvl_sizes[l]
            for c in range(3):
                sig_off = level_offsets[l] + c*n*sz
                _evm_cuda.batched_thwc_to_nt(bands.ptr_at(sig_off), nt_buf.ptr, n, sz)
                _evm_cuda.batched_iir_bandpass(nt_buf.ptr, filt_buf.ptr, n, sz, R1, R2)
                dst_off = level_offsets[l] + c*n*sz
                _evm_cuda.batched_nt_to_thwc_scaled(filt_buf.ptr, filtered.ptr_at(dst_off), n, sz, alpha_sched[l])
        sync()
        st["C) temporal IIR"] = time.perf_counter() - t0
        del bands, nt_buf, filt_buf

        # Stage D: recon + f16->f32 + render
        t0 = time.perf_counter()
        delta = DeviceBuffer(n*3*h*w*4)
        _evm_cuda.batched_lpyr_recon(filtered.ptr, delta.ptr, n, h, w, levels, _d_binom5(), 5)
        del filtered
        ntsc_f32 = DeviceBuffer(ntsc_floats*4)
        _evm_cuda.f16_to_f32(ntsc_f16.ptr, ntsc_f32.ptr, ntsc_floats)
        _evm_cuda.batched_add_planar_quantize(ntsc_f32.ptr, delta.ptr, d_out_u8.ptr, n, h, w, CHROM_ATT)
        sync()
        st["D) recon + render"] = time.perf_counter() - t0
        del ntsc_f32, ntsc_f16, delta
        return st

    run_fp16()
    fp16_runs = [run_fp16() for _ in range(N_ITER)]
    fp16_med = {k: median([r[k] for r in fp16_runs]) for k in fp16_runs[0]}
    print(f"  FP16: {sum(fp16_med.values()):.3f}s\n")

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    stage_keys = list(fp32_med.keys())
    f32_total = sum(fp32_med.values())
    f16_total = sum(fp16_med.values())

    print("=" * 75)
    print(f"{'Stage':<22s} {'CUDA FP32':>12s} {'CUDA FP16':>12s} {'FP16/FP32':>10s}")
    print("-" * 60)
    for k in stage_keys:
        f32v = fp32_med[k]; f16v = fp16_med[k]
        ratio16 = f"{(1-f16v/f32v)*100:+.1f}%" if f32v > 0 else "-"
        print(f"{k:<22s} {f32v*1000:>10.1f}ms {f16v*1000:>10.1f}ms {ratio16:>9s}")
    print("-" * 60)
    print(f"{'GPU pipeline total':<22s} {f32_total*1000:>10.1f}ms {f16_total*1000:>10.1f}ms {(1-f16_total/f32_total)*100:>+9.1f}%")
    print()
    print(f"{'Python CPU total':<22s} {py_total*1000:>10.1f}ms")
    print(f"{'Speedup (CPU→FP32)':<22s} {py_total/f32_total:>10.1f}x")
    print(f"{'Speedup (CPU→FP16)':<22s} {py_total/f16_total:>10.1f}x")
    print("=" * 75)


if __name__ == "__main__":
    main()
