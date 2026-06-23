#!/usr/bin/env python3
"""Three-way comparison: Python CPU vs CUDA FP32 vs CUDA FP16.

Runs the motion pipeline (baby.mp4) three ways on the same node, measuring
GPU pipeline time only (decode/encode excluded). Reports a side-by-side table.
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


def measure(fn, n_iter=5):
    """Run fn n_iter times, return (median_s, min_s, max_s)."""
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    med = times[n_iter // 2] if n_iter % 2 == 1 else (times[n_iter//2-1] + times[n_iter//2]) / 2
    return med, min(times), max(times)


def main():
    # Read frames once (shared by all three paths)
    cap = cv2.VideoCapture(VID)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    DROP = 10
    frames = frames[:len(frames) - DROP]
    n = len(frames)
    h, w = frames[0].shape[:2]
    print(f"clip: {n} frames, {h}x{w}, fps={fps:.1f}")
    print(f"iterations: 5 (+ 1 warmup each)\n")

    results = {}

    # --- 1. Python baseline (CPU) ---
    print("Running Python baseline (CPU)...")
    import evm
    def run_python():
        evm.magnify_motion_lpyr_iir(
            VID, "/tmp/_py_motion.mp4",
            alpha=ALPHA, lambda_c=LAMBDA_C, r1=R1, r2=R2,
            chrom_attenuation=CHROM_ATT)
    # Warmup
    run_python()
    med, mn, mx = measure(run_python, n_iter=3)  # CPU is slow, 3 iters
    results["Python (CPU)"] = (med, mn, mx)
    print(f"  {med:.3f}s\n")

    # --- 2. CUDA FP32 ---
    print("Running CUDA FP32...")
    from evm_cuda.batched import magnify_motion_lpyr_iir
    def run_fp32():
        magnify_motion_lpyr_iir(
            VID, "/tmp/_fp32_motion.mp4",
            alpha=ALPHA, lambda_c=LAMBDA_C, r1=R1, r2=R2,
            chrom_attenuation=CHROM_ATT)
    run_fp32()  # warmup
    med32, mn32, mx32 = measure(run_fp32, n_iter=5)
    results["CUDA FP32"] = (med32, mn32, mx32)
    print(f"  {med32:.3f}s\n")

    # --- 3. CUDA FP16 ---
    print("Running CUDA FP16...")
    from evm_cuda.batched import magnify_motion_lpyr_iir_fp16
    def run_fp16():
        magnify_motion_lpyr_iir_fp16(
            VID, "/tmp/_fp16_motion.mp4",
            alpha=ALPHA, lambda_c=LAMBDA_C, r1=R1, r2=R2,
            chrom_attenuation=CHROM_ATT)
    run_fp16()  # warmup
    med16, mn16, mx16 = measure(run_fp16, n_iter=5)
    results["CUDA FP16"] = (med16, mn16, mx16)
    print(f"  {med16:.3f}s\n")

    # --- Summary ---
    py_t = results["Python (CPU)"][0]
    fp32_t = results["CUDA FP32"][0]
    fp16_t = results["CUDA FP16"][0]

    print("=" * 65)
    print(f"{'Path':<20s} {'median':>8s} {'min':>8s} {'max':>8s} {'vs CPU':>8s}")
    print("-" * 65)
    for name, (med, mn, mx) in results.items():
        speedup = py_t / med
        print(f"{name:<20s} {med:>7.3f}s {mn:>7.3f}s {mx:>7.3f}s {speedup:>7.1f}x")
    print("-" * 65)
    print(f"\nFP32 vs FP16: {fp32_t:.3f}s -> {fp16_t:.3f}s ({(1-fp16_t/fp32_t)*100:+.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
