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


# ===========================================================================
# MOTION PIPELINE
# ===========================================================================

def run_motion_cpu():
    import evm
    evm.magnify_motion_lpyr_iir(
        MOTION_VID, str(OUTPUT / "baby_cpu.mp4"),
        alpha=MOTION_ALPHA, lambda_c=MOTION_LAMBDA, r1=MOTION_R1, r2=MOTION_R2,
        chrom_attenuation=MOTION_CHROM)


# ===========================================================================
# CPU STAGE-LEVEL PROFILER (mirrors the GPU stage boundaries)
# ===========================================================================

def profile_color_cpu_stages():
    """Per-stage CPU timing for the color pipeline.

    Re-implements the four stages of evm.magnify_color_gdown_ideal with
    perf_counter boundaries that match the GPU profiler's stage names, so the
    comparison table can show CPU vs FP32 vs FP16 for every stage.
    """
    import evm
    from evm.pyramids import blur_dn_clr
    from evm.filters import ideal_bandpass

    frames, fps = evm.magnify._read_frames(COLOR_VID)
    n = len(frames)
    h, w = frames[0].shape[:2]

    def run_once():
        st = {}

        # Stage 1: NTSC color convert (per-frame, same as GPU)
        t0 = time.perf_counter()
        ntsc_frames = [evm.magnify._rgb_frame_to_ntsc(fr) for fr in frames]
        st["1) color_cvt"] = time.perf_counter() - t0

        # Stage 2: Gaussian downsample (blur_dn_clr per frame)
        t0 = time.perf_counter()
        gdown = np.stack([blur_dn_clr(ntsc, COLOR_LEVEL) for ntsc in ntsc_frames], axis=0)
        st["2) blur_dn"] = time.perf_counter() - t0

        # Stage 2b: no host round-trip on CPU (data is already in host memory)
        st["2b) D2H + reshape"] = 0.0

        # Stage 3: ideal bandpass per channel
        t0 = time.perf_counter()
        filtered = np.stack(
            [ideal_bandpass(gdown[..., c].astype(np.float64), COLOR_FL, COLOR_FH, COLOR_SR)
             for c in range(3)], axis=-1)
        st["3) ideal_bandpass"] = time.perf_counter() - t0

        # Stage 4: upsample + add + quantize (render)
        t0 = time.perf_counter()
        gain = np.array([COLOR_ALPHA, COLOR_ALPHA * COLOR_CHROM, COLOR_ALPHA * COLOR_CHROM])
        filtered = filtered * gain
        out = np.empty((n, h, w, 3), dtype=np.uint8)
        for i in range(n):
            upsampled = cv2.resize(filtered[i], (w, h), interpolation=cv2.INTER_LINEAR)
            rendered = ntsc_frames[i] + upsampled
            out[i] = evm.magnify._ntsc_to_bgr_uint8(rendered)
        st["4) upsample + render"] = time.perf_counter() - t0

        st["_total"] = sum(v for k, v in st.items() if not k.startswith("_"))
        return st

    # No warmup, single run — minimize RAM pressure on low-memory hosts.
    return run_once()


def profile_motion_cpu_stages():
    """Per-stage CPU timing for the motion pipeline.

    Re-implements _streaming_lpyr_motion with perf_counter boundaries matching
    the GPU profiler's stage names.
    """
    import evm
    from evm.pyramids import laplacian_pyramid_channels, reconstruct_from_channels
    from evm.filters import iir_bandpass

    frames, fps = evm.magnify._read_frames(MOTION_VID)
    n = len(frames)
    h, w = frames[0].shape[:2]

    def run_once():
        st = {}

        # Stage A: NTSC color convert
        t0 = time.perf_counter()
        ntsc_frames = [evm.magnify._rgb_frame_to_ntsc(fr) for fr in frames]
        st["A) NTSC convert"] = time.perf_counter() - t0

        # Stage B: Laplacian pyramid build (all frames)
        t0 = time.perf_counter()
        pyrs = [laplacian_pyramid_channels(f, "auto") for f in ntsc_frames]
        n_levels = pyrs[0][1].shape[0]
        pind = pyrs[0][1]
        n_coeffs = sum(int(pind[l, 0] * pind[l, 1]) for l in range(n_levels))
        series = np.empty((n, n_coeffs, 3), dtype=np.float64)
        for i in range(n):
            for l in range(n_levels):
                sl = evm.magnify._level_slice(l, pind)
                series[i, sl, :] = pyrs[i][0][l].reshape(-1, 3)
        st["B) lpyr_build"] = time.perf_counter() - t0

        # Stage C: temporal IIR filter
        t0 = time.perf_counter()
        filtered = iir_bandpass(series, MOTION_R1, MOTION_R2, axis=0)
        st["C) temporal IIR"] = time.perf_counter() - t0

        # Stage D1: reconstruct pyramid + apply alpha schedule
        t0 = time.perf_counter()
        alpha_sched = evm.magnify.figure6_alpha_schedule(
            n_levels, MOTION_ALPHA, MOTION_LAMBDA, h, w)
        filtered_per_frame = []
        for i in range(n):
            bands = []
            for l in range(n_levels):
                sl = evm.magnify._level_slice(l, pind)
                lh, lw = int(pind[l, 0]), int(pind[l, 1])
                bands.append(filtered[i, sl, :].reshape(lh, lw, 3) * alpha_sched[l])
            filtered_per_frame.append(bands)
        rendered_ntsc = evm.magnify._amplify_lpyr_stack(
            ntsc_frames, filtered_per_frame, pind, MOTION_CHROM)
        st["D1) lpyr_recon"] = time.perf_counter() - t0

        # Stage D2: YIQ->RGB + quantize (render)
        t0 = time.perf_counter()
        out = np.stack([evm.magnify._ntsc_to_bgr_uint8(x) for x in rendered_ntsc], axis=0)
        st["D2) render"] = time.perf_counter() - t0

        st["_total"] = sum(v for k, v in st.items() if not k.startswith("_"))
        return st

    # No warmup, single run — minimize RAM pressure on low-memory hosts.
    return run_once()


# ===========================================================================
# GPU stage-level profiling is delegated to evm_cuda.benchmark.run() — the same
# code path the Colab notebook uses — so CPU/Colab/TRUBA can never drift apart
# on methodology. The benchmark module owns the warmup/sync/median discipline
# and now reports H2D/D2H transfers as their own stages.
# ===========================================================================
# GPU stage-level profiling is delegated to evm_cuda.benchmark.run() — the same
# code path the Colab notebook uses — so CPU/Colab/TRUBA can never drift apart
# on methodology. The benchmark module owns the warmup/sync/median discipline
# and now reports H2D/D2H transfers as their own stages.
# ===========================================================================


def _bench_gpu(pipeline, precision, params, label):
    """Run one (pipeline, precision) config via benchmark.run and print it."""
    from evm_cuda import benchmark
    print(f"\n[CUDA {precision.upper()}]")
    r = benchmark.run(pipeline, precision, params,
                      out_path=str(OUTPUT / f"{pipeline}_{precision}.mp4"))
    print(r)
    return r


def main():
    print("=" * 70)
    print("COMPREHENSIVE COMPARISON: Python CPU vs CUDA FP32 vs CUDA FP16")
    print("=" * 70)

    # Get GPU info
    r = os.popen("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader").read().strip()
    print(f"GPU: {r}\n")

    # --- Color pipeline params (face.mp4) ---
    COLOR_PARAMS = dict(alpha=COLOR_ALPHA, level=COLOR_LEVEL, fl=COLOR_FL,
                        fh=COLOR_FH, chrom_attenuation=COLOR_CHROM,
                        sampling_rate=COLOR_SR)

    # ====================================================================
    # COLOR PIPELINE
    # ====================================================================
    print("\n" + "=" * 70)
    print("COLOR PIPELINE (face.mp4)")
    print("=" * 70)

    print("\n[Python CPU]")
    try:
        med, _, _ = time_fn(run_color_cpu, n_iter=3)
        print(f"  Total: {med:.3f}s")
        cpu_color = profile_color_cpu_stages()
        if cpu_color:
            print(f"  Compute-only total: {cpu_color.get('_total', 0):.4f}s")
    except Exception as e:
        print(f"  FAILED: {e}")
        cpu_color = {}

    color_params = dict(vid=COLOR_VID, **COLOR_PARAMS)
    color_fp32 = _bench_gpu("color", "fp32", color_params, "color")
    color_fp16 = _bench_gpu("color", "fp16", color_params, "color")

    # ====================================================================
    # MOTION PIPELINE
    # ====================================================================
    print("\n" + "=" * 70)
    print("MOTION PIPELINE (baby.mp4)")
    print("=" * 70)

    print("\n[Python CPU]")
    try:
        # n_iter=1 + warmup=False: the motion CPU pipeline builds 301 FP64
        # Laplacian pyramids; repeating OOM-kills on ~13 GB systems.
        med, _, _ = time_fn(run_motion_cpu, n_iter=1, warmup=False)
        print(f"  Total: {med:.3f}s")
        cpu_motion = profile_motion_cpu_stages()
        if cpu_motion:
            print(f"  Compute-only total: {cpu_motion.get('_total', 0):.4f}s")
    except Exception as e:
        print(f"  FAILED: {e}")
        cpu_motion = {}

    MOTION_PARAMS = dict(alpha=MOTION_ALPHA, lambda_c=MOTION_LAMBDA,
                         r1=MOTION_R1, r2=MOTION_R2,
                         chrom_attenuation=MOTION_CHROM)
    motion_params = dict(vid=MOTION_VID, **MOTION_PARAMS)
    motion_fp32 = _bench_gpu("motion", "fp32", motion_params, "motion")
    motion_fp16 = _bench_gpu("motion", "fp16", motion_params, "motion")

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # GPU comparison table (compute + transfer breakdown).
    from evm_cuda import benchmark
    print(benchmark.summarize(
        [color_fp32, color_fp16, motion_fp32, motion_fp16], n_iter=5))

    # CPU vs GPU speedup (compute-only).
    print("\nSpeedup vs CPU (compute-only):")
    for pipe, cpu_stages, fp32, fp16 in [
        ("color", cpu_color, color_fp32, color_fp16),
        ("motion", cpu_motion, motion_fp32, motion_fp16),
    ]:
        cpu_total = cpu_stages.get("_total", 0) * 1000 if cpu_stages else 0
        for prec, res in [("FP32", fp32), ("FP16", fp16)]:
            if cpu_total and res and res.measured:
                print(f"  {pipe} {prec}: {cpu_total / res.compute_ms:.0f}x "
                      f"({res.compute_ms:.0f}ms vs {cpu_total:.0f}ms CPU)")

    # Output videos
    print(f"\nOutput videos:")
    for f in sorted(OUTPUT.glob("*.mp4")):
        print(f"  {f.name}: {f.stat().st_size/1024/1024:.1f} MB")

    # Save JSON
    payload = {
        "gpu": r,
        "cpu_color_total": cpu_color.get("_total", 0) if cpu_color else 0,
        "cpu_motion_total": cpu_motion.get("_total", 0) if cpu_motion else 0,
        "results": [
            {"pipeline": r.pipeline, "precision": r.precision,
             "compute_ms": r.compute_ms, "transfer_ms": r.transfer_ms,
             "total_ms": r.total_ms, "notes": r.notes}
            for r in [color_fp32, color_fp16, motion_fp32, motion_fp16]
        ],
    }
    with open(ROOT / "comparison_results.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults saved to comparison_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
