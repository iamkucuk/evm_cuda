#!/usr/bin/env python3
"""Comprehensive comparison: Python CPU vs CUDA FP32 vs CUDA FP16.

Runs both pipelines (color + motion) through all three paths with per-stage
timing and renders all 6 output videos. Decode/encode excluded from stage
timing where possible.

Output:
  - Stage timing table for each pipeline x path
  - 6 rendered videos in output/
  - JSON results file
"""
from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = ROOT / "cuda"
for p in (str(ROOT), str(CUDA_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import cv2

DATA = ROOT / "data"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

# Color pipeline params (face.mp4)
COLOR_VID = str(DATA / "face.mp4")
COLOR_ALPHA = 50; COLOR_LEVEL = 4; COLOR_FL = 50/60; COLOR_FH = 60/60
COLOR_CHROM = 1.0; COLOR_SR = 30.0

# Motion pipeline params (baby.mp4)
MOTION_VID = str(DATA / "baby.mp4")
MOTION_ALPHA = 10; MOTION_LAMBDA = 16; MOTION_R1 = 0.4; MOTION_R2 = 0.05
MOTION_CHROM = 0.1

N_ITER = 5


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


def time_fn(fn, n_iter=N_ITER, warmup=True):
    """Time a function n_iter times with one warmup. Returns (median, min, max)."""
    if warmup:
        fn()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return median(times), min(times), max(times)


# ===========================================================================
# COLOR PIPELINE
# ===========================================================================

def run_color_cpu():
    import evm
    evm.magnify_color_gdown_ideal(
        COLOR_VID, str(OUTPUT / "face_cpu.mp4"),
        alpha=COLOR_ALPHA, level=COLOR_LEVEL, fl=COLOR_FL, fh=COLOR_FH,
        chrom_attenuation=COLOR_CHROM, sampling_rate=COLOR_SR)


def run_color_fp32():
    from evm_cuda.batched import magnify_color_gdown_ideal
    magnify_color_gdown_ideal(
        COLOR_VID, str(OUTPUT / "face_fp32.mp4"),
        alpha=COLOR_ALPHA, level=COLOR_LEVEL, fl=COLOR_FL, fh=COLOR_FH,
        chrom_attenuation=COLOR_CHROM, sampling_rate=COLOR_SR)


def run_color_fp16():
    from evm_cuda.batched import magnify_color_gdown_ideal_fp16
    magnify_color_gdown_ideal_fp16(
        COLOR_VID, str(OUTPUT / "face_fp16.mp4"),
        alpha=COLOR_ALPHA, level=COLOR_LEVEL, fl=COLOR_FL, fh=COLOR_FH,
        chrom_attenuation=COLOR_CHROM, sampling_rate=COLOR_SR)


# ===========================================================================
# MOTION PIPELINE
# ===========================================================================

def run_motion_cpu():
    import evm
    evm.magnify_motion_lpyr_iir(
        MOTION_VID, str(OUTPUT / "baby_cpu.mp4"),
        alpha=MOTION_ALPHA, lambda_c=MOTION_LAMBDA, r1=MOTION_R1, r2=MOTION_R2,
        chrom_attenuation=MOTION_CHROM)


def run_motion_fp32():
    from evm_cuda.batched import magnify_motion_lpyr_iir
    magnify_motion_lpyr_iir(
        MOTION_VID, str(OUTPUT / "baby_fp32.mp4"),
        alpha=MOTION_ALPHA, lambda_c=MOTION_LAMBDA, r1=MOTION_R1, r2=MOTION_R2,
        chrom_attenuation=MOTION_CHROM)


def run_motion_fp16():
    from evm_cuda.batched import magnify_motion_lpyr_iir_fp16
    magnify_motion_lpyr_iir_fp16(
        MOTION_VID, str(OUTPUT / "baby_fp16.mp4"),
        alpha=MOTION_ALPHA, lambda_c=MOTION_LAMBDA, r1=MOTION_R1, r2=MOTION_R2,
        chrom_attenuation=MOTION_CHROM)


# ===========================================================================
# CUDA STAGE-LEVEL PROFILER (for FP32 and FP16)
# ===========================================================================

def profile_cuda_stages(pipeline: str, precision: str):
    """Run the existing stage profiler for the given pipeline/precision.
    Returns dict of stage_name -> median_seconds, or None on failure."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CUDA_DIR)

    if pipeline == "color" and precision == "fp32":
        script = str(ROOT / "scripts" / "profile_color.py")
    elif pipeline == "motion" and precision == "fp32":
        script = str(ROOT / "scripts" / "profile_motion.py")
    else:
        return None  # No stage profiler for FP16 yet

    import subprocess
    import re
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, env=env, check=True, timeout=180)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    stages = {}
    for line in result.stdout.splitlines():
        m = re.match(r"\s*(.+?)\s+([\d.]+)s\s+([\d.]+)s\s+([\d.]+)s\s+([\d.]+)%", line)
        if m:
            stages[m.group(1).strip()] = float(m.group(2))
        if "Pipeline total" in line:
            m2 = re.search(r"([\d.]+)s", line)
            if m2:
                stages["_total"] = float(m2.group(1))
    return stages


def profile_motion_fp16_stages():
    """Inline FP16 motion profiler with per-stage timing."""
    from evm_cuda.batched import (DeviceBuffer, _read_frames, _d_binom5,
                                   _warmup_gpu_pool_motion, figure6_alpha_schedule)
    from evm_cuda import _evm_cuda

    def sync():
        _evm_cuda.device_synchronize()

    frames, fps = _read_frames(MOTION_VID)
    n = len(frames)
    h, w = frames[0].shape[:2]
    clip_u8 = np.stack(frames, axis=0)

    levels = 1; hh, ww = h, w
    while hh >= 5 and ww >= 5:
        levels += 1; hh = (hh + 1) // 2; ww = (ww + 1) // 2
    alpha_sched = figure6_alpha_schedule(levels, MOTION_ALPHA, MOTION_LAMBDA, h, w)
    level_sizes = []
    ch, cw = h, w
    for _ in range(levels):
        level_sizes.append((ch, cw))
        ch = (ch + 1) // 2; cw = (cw + 1) // 2
    lvl_sizes = [s[0]*s[1] for s in level_sizes]
    total_band_floats = sum(s * (n*3) for s in lvl_sizes)
    ntsc_floats = n * h * w * 3
    planar_floats = n * 3 * h * w
    max_sz = max(lvl_sizes)

    _warmup_gpu_pool_motion(n, h, w, levels)
    sync()

    level_offsets = []
    off = 0
    for sz in lvl_sizes:
        level_offsets.append(off)
        off += sz * n * 3

    d_clip = DeviceBuffer.from_array(clip_u8)
    d_out_u8 = DeviceBuffer(n * h * w * 3)

    def run_once():
        st = {}
        t0 = time.perf_counter()
        ntsc_f32 = DeviceBuffer(ntsc_floats * 4)
        ntsc_f16 = DeviceBuffer(ntsc_floats * 2)
        _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, ntsc_f32.ptr, n, h, w)
        _evm_cuda.f32_to_f16(ntsc_f32.ptr, ntsc_f16.ptr, ntsc_floats)
        sync()
        st["A) NTSC convert"] = time.perf_counter() - t0
        del ntsc_f32

        t0 = time.perf_counter()
        planar = DeviceBuffer(planar_floats * 2)
        _evm_cuda.batched_to_planar_3ch_f16(ntsc_f16.ptr, planar.ptr, n, h, w)
        bands_f32 = DeviceBuffer(total_band_floats * 4)
        _evm_cuda.batched_lpyr_build_f16(planar.ptr, bands_f32.ptr, n, h, w, levels, _d_binom5(), 5)
        del planar
        bands = DeviceBuffer(total_band_floats * 2)
        _evm_cuda.f32_to_f16(bands_f32.ptr, bands.ptr, total_band_floats)
        del bands_f32
        sync()
        st["B) lpyr_build"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        filtered = DeviceBuffer(total_band_floats * 2)
        nt_buf = DeviceBuffer(n * max_sz * 2)
        filt_buf = DeviceBuffer(n * max_sz * 2)
        for l in range(levels):
            sz = lvl_sizes[l]
            for c in range(3):
                sig_off = level_offsets[l] + c * n * sz
                _evm_cuda.batched_thwc_to_nt_f16(bands.ptr_at_half(sig_off), nt_buf.ptr, n, sz)
                _evm_cuda.batched_iir_bandpass_f16(nt_buf.ptr, filt_buf.ptr, n, sz, MOTION_R1, MOTION_R2)
                dst_off = level_offsets[l] + c * n * sz
                _evm_cuda.batched_nt_to_thwc_scaled_f16(filt_buf.ptr, filtered.ptr_at_half(dst_off), n, sz, alpha_sched[l])
        sync()
        st["C) temporal IIR"] = time.perf_counter() - t0
        del bands, nt_buf, filt_buf

        t0 = time.perf_counter()
        delta = DeviceBuffer(n * 3 * h * w * 2)
        _evm_cuda.batched_lpyr_recon_f16(filtered.ptr, delta.ptr, n, h, w, levels, _d_binom5(), 5)
        sync()
        st["D1) lpyr_recon"] = time.perf_counter() - t0
        del filtered

        t0 = time.perf_counter()
        _evm_cuda.batched_add_planar_quantize_f16(ntsc_f16.ptr, delta.ptr, d_out_u8.ptr, n, h, w, MOTION_CHROM)
        # Include D2H download to match the FP32 profiler's Stage D2 timing.
        out = d_out_u8.download_u8(n * h * w * 3).reshape(n, h, w, 3)
        sync()
        st["D2) render"] = time.perf_counter() - t0
        del ntsc_f16, delta

        st["_total"] = sum(st.values())
        return st

    run_once()  # warmup
    runs = [run_once() for _ in range(N_ITER)]
    return {k: median([r[k] for r in runs]) for k in runs[0]}


def profile_color_fp16_stages():
    """Inline FP16 color profiler with per-stage timing.

    Mirrors the FP32 color profiler (profile_color.py) but with FP16 NTSC
    storage. The FFT bandpass stays FP32 — only NTSC and blur_dn scratch
    are halved.
    """
    from evm_cuda.batched import DeviceBuffer, _read_frames, _d_binom5_sum1, _warmup_gpu_pool
    from evm_cuda import _evm_cuda

    def sync():
        _evm_cuda.device_synchronize()

    frames, fps = _read_frames(COLOR_VID)
    n = len(frames)
    h, w = frames[0].shape[:2]
    clip_u8 = np.stack(frames, axis=0)

    hl, wl = h, w
    for _ in range(COLOR_LEVEL):
        hl = (hl + 1) // 2
        wl = (wl + 1) // 2

    ntsc_floats = n * h * w * 3
    _warmup_gpu_pool()
    sync()

    d_clip = DeviceBuffer.from_array(clip_u8)
    d_out_u8 = DeviceBuffer(n * h * w * 3)

    def run_once():
        st = {}

        # Stage 1: NTSC convert (FP32 compute) -> __half storage
        t0 = time.perf_counter()
        ntsc_f32 = DeviceBuffer(ntsc_floats * 4)
        ntsc_f16 = DeviceBuffer(ntsc_floats * 2)
        _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, ntsc_f32.ptr, n, h, w)
        _evm_cuda.f32_to_f16(ntsc_f32.ptr, ntsc_f16.ptr, ntsc_floats)
        sync()
        st["1) color_cvt"] = time.perf_counter() - t0
        del ntsc_f32

        # Stage 2: FP16 planar + FP16 blur_dn -> FP32 gdown
        t0 = time.perf_counter()
        planar = DeviceBuffer(n * 3 * h * w * 2)
        _evm_cuda.batched_to_planar_3ch_f16(ntsc_f16.ptr, planar.ptr, n, h, w)
        gdown_planar = DeviceBuffer(n * 3 * hl * wl * 4)
        _evm_cuda.batched_blur_dn_color_f16(
            planar.ptr, gdown_planar.ptr, n * 3, h, w, COLOR_LEVEL,
            _d_binom5_sum1(), 5)
        sync()
        st["2) blur_dn"] = time.perf_counter() - t0
        del planar

        # Stage 2b: D2H + reshape for FFT (host round-trip)
        t0 = time.perf_counter()
        gdown = gdown_planar.download_f32(n * 3 * hl * wl).reshape(n, 3, hl, wl)
        gdown = np.ascontiguousarray(gdown.transpose(0, 2, 3, 1))
        del gdown_planar
        st["2b) D2H + reshape"] = time.perf_counter() - t0

        # Stage 3: ideal bandpass per channel (FP32 FFT)
        t0 = time.perf_counter()
        filt = np.empty_like(gdown)
        for c in range(3):
            sig = np.ascontiguousarray(gdown[..., c].reshape(n, hl * wl).T)
            d_sig = DeviceBuffer.from_array(sig)
            d_out = DeviceBuffer(n * hl * wl * 4)
            _evm_cuda.batched_ideal_bandpass(
                d_sig.ptr, d_out.ptr, n, hl * wl, COLOR_FL, COLOR_FH, COLOR_SR)
            filt[..., c] = d_out.download_f32(n * hl * wl).reshape(hl * wl, n).T.reshape(n, hl, wl)
        sync()
        st["3) ideal_bandpass"] = time.perf_counter() - t0

        # Stage 4: gain + upload + FP16 fused upsample+add+quantize + D2H
        t0 = time.perf_counter()
        gain = np.array([COLOR_ALPHA, COLOR_ALPHA * COLOR_CHROM, COLOR_ALPHA * COLOR_CHROM],
                        dtype=np.float32)
        filt = filt * gain
        d_filt = DeviceBuffer.from_array(np.ascontiguousarray(filt))
        _evm_cuda.batched_upsample_add_quantize_f16(
            ntsc_f16.ptr, d_filt.ptr, d_out_u8.ptr,
            n, hl, wl, h, w, 1.0)
        out = d_out_u8.download_u8(n * h * w * 3).reshape(n, h, w, 3)
        sync()
        st["4) upsample + render"] = time.perf_counter() - t0
        del ntsc_f16, d_filt

        st["_total"] = sum(v for k, v in st.items() if not k.startswith("_"))
        return st

    run_once()  # warmup
    runs = [run_once() for _ in range(N_ITER)]
    return {k: median([r[k] for r in runs]) for k in runs[0]}


def main():
    print("=" * 70)
    print("COMPREHENSIVE COMPARISON: Python CPU vs CUDA FP32 vs CUDA FP16")
    print("=" * 70)

    # Get GPU info
    r = os.popen("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader").read().strip()
    print(f"GPU: {r}\n")

    results = {}

    # ====================================================================
    # COLOR PIPELINE
    # ====================================================================
    print("\n" + "=" * 70)
    print("COLOR PIPELINE (face.mp4, 528x592, level=4)")
    print("=" * 70)

    # CPU
    print("\n[Python CPU]")
    med, mn, mx = time_fn(run_color_cpu, n_iter=3)
    print(f"  Total: {med:.3f}s")
    results["color_cpu_total"] = med

    # FP32 e2e + stages
    print("\n[CUDA FP32]")
    med32, _, _ = time_fn(run_color_fp32)
    print(f"  End-to-end: {med32:.3f}s")
    results["color_fp32_e2e"] = med32

    fp32_stages = profile_cuda_stages("color", "fp32")
    if fp32_stages:
        print(f"  GPU-only total: {fp32_stages.get('_total', 0):.4f}s")
        results["color_fp32_stages"] = fp32_stages

    # FP16 e2e + stages
    print("\n[CUDA FP16]")
    try:
        med16, _, _ = time_fn(run_color_fp16)
        print(f"  End-to-end: {med16:.3f}s")
        results["color_fp16_e2e"] = med16

        # FP16 stages (inline profiler)
        print("  Profiling stages...")
        fp16_stages = profile_color_fp16_stages()
        if fp16_stages:
            print(f"  GPU-only total: {fp16_stages.get('_total', 0):.4f}s")
            results["color_fp16_stages"] = fp16_stages
    except Exception as e:
        print(f"  FAILED: {e}")

    # ====================================================================
    # MOTION PIPELINE
    # ====================================================================
    print("\n" + "=" * 70)
    print("MOTION PIPELINE (baby.mp4, 960x544, 9 levels)")
    print("=" * 70)

    # CPU
    print("\n[Python CPU]")
    med, mn, mx = time_fn(run_motion_cpu, n_iter=3)
    print(f"  Total: {med:.3f}s")
    results["motion_cpu_total"] = med

    # FP32 e2e + stages
    print("\n[CUDA FP32]")
    med32, _, _ = time_fn(run_motion_fp32)
    print(f"  End-to-end: {med32:.3f}s")
    results["motion_fp32_e2e"] = med32

    fp32_stages = profile_cuda_stages("motion", "fp32")
    if fp32_stages:
        print(f"  GPU-only total: {fp32_stages.get('_total', 0):.4f}s")
        results["motion_fp32_stages"] = fp32_stages

    # FP16 e2e
    print("\n[CUDA FP16]")
    try:
        med16, _, _ = time_fn(run_motion_fp16)
        print(f"  End-to-end: {med16:.3f}s")
        results["motion_fp16_e2e"] = med16

        # FP16 stages (inline profiler)
        print("  Profiling stages...")
        fp16_stages = profile_motion_fp16_stages()
        if fp16_stages:
            print(f"  GPU-only total: {fp16_stages.get('_total', 0):.4f}s")
            results["motion_fp16_stages"] = fp16_stages
    except Exception as e:
        print(f"  FAILED: {e}")

    # ====================================================================
    # SUMMARY TABLE
    # ====================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Color stage comparison
    if "color_fp32_stages" in results and "color_fp16_stages" in results:
        s32 = results["color_fp32_stages"]
        s16 = results["color_fp16_stages"]
        print(f"\nCOLOR — {'Stage':<22s} {'FP32':>10s} {'FP16':>10s} {'FP16/FP32':>10s}")
        print("-" * 55)
        for k in s32:
            if k.startswith("_"):
                continue
            v32 = s32.get(k, 0)
            v16 = s16.get(k, 0)
            pct = f"{(1 - v16/v32)*100:+.1f}%" if v32 > 0 else "-"
            print(f"  {k:<22s} {v32*1000:>9.1f}ms {v16*1000:>9.1f}ms {pct:>9s}")
        t32 = s32.get("_total", 0)
        t16 = s16.get("_total", 0)
        print("-" * 55)
        print(f"  {'GPU pipeline total':<22s} {t32*1000:>9.1f}ms {t16*1000:>9.1f}ms {(1-t16/t32)*100:>+9.1f}%")

    # Motion stage comparison
    if "motion_fp32_stages" in results and "motion_fp16_stages" in results:
        s32 = results["motion_fp32_stages"]
        s16 = results["motion_fp16_stages"]
        print(f"\n{'Stage':<25s} {'FP32':>10s} {'FP16':>10s} {'FP16/FP32':>10s}")
        print("-" * 58)
        for k in s32:
            if k.startswith("_"):
                continue
            v32 = s32.get(k, 0)
            v16 = s16.get(k, 0)
            pct = f"{(1 - v16/v32)*100:+.1f}%" if v32 > 0 else "-"
            print(f"{k:<25s} {v32*1000:>9.1f}ms {v16*1000:>9.1f}ms {pct:>9s}")
        t32 = s32.get("_total", 0)
        t16 = s16.get("_total", 0)
        print("-" * 58)
        print(f"{'GPU pipeline total':<25s} {t32*1000:>9.1f}ms {t16*1000:>9.1f}ms {(1-t16/t32)*100:>+9.1f}%")

    # End-to-end comparison
    print(f"\n{'Path':<30s} {'Color':>10s} {'Motion':>10s}")
    print("-" * 52)
    print(f"{'Python CPU':<30s} {results.get('color_cpu_total',0)*1000:>9.0f}ms {results.get('motion_cpu_total',0)*1000:>9.0f}ms")
    print(f"{'CUDA FP32 (e2e)':<30s} {results.get('color_fp32_e2e',0)*1000:>9.0f}ms {results.get('motion_fp32_e2e',0)*1000:>9.0f}ms")
    if "color_fp16_e2e" in results or "motion_fp16_e2e" in results:
        c16 = f"{results.get('color_fp16_e2e',0)*1000:>9.0f}ms" if "color_fp16_e2e" in results else "N/A"
        m16 = f"{results.get('motion_fp16_e2e',0)*1000:>9.0f}ms" if "motion_fp16_e2e" in results else "N/A"
        print(f"{'CUDA FP16 (e2e)':<30s} {c16:>10s} {m16:>10s}")
    print("-" * 52)

    # Speedups
    print(f"\nSpeedup vs CPU:")
    print(f"  Color FP32:  {results.get('color_cpu_total',0)/results.get('color_fp32_e2e',1):.1f}x")
    print(f"  Motion FP32: {results.get('motion_cpu_total',0)/results.get('motion_fp32_e2e',1):.1f}x")
    if "color_fp16_e2e" in results:
        print(f"  Color FP16:  {results.get('color_cpu_total',0)/results.get('color_fp16_e2e',1):.1f}x")
    if "motion_fp16_e2e" in results:
        print(f"  Motion FP16: {results.get('motion_cpu_total',0)/results.get('motion_fp16_e2e',1):.1f}x")

    # Video list
    print(f"\nOutput videos:")
    for f in sorted(OUTPUT.glob("*.mp4")):
        print(f"  {f.name}: {f.stat().st_size/1024/1024:.1f} MB")

    # Save JSON
    with open(ROOT / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to comparison_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
