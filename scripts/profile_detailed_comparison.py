#!/usr/bin/env python3
"""Stage-by-stage comparison: Python CPU vs CUDA FP32 vs CUDA FP16.

Uses the real pipeline functions (which handle alloc/free correctly) and
measures GPU-only time by excluding decode/encode.
"""
from __future__ import annotations

import sys
import os
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


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n//2] if n % 2 == 1 else (s[n//2-1] + s[n//2]) / 2


def main():
    print("Motion pipeline comparison: Python CPU vs CUDA FP32 vs CUDA FP16")
    print(f"clip: baby.mp4, 291 frames, 544x960, 9 levels\n")

    results = {}

    # --- 1. Python baseline ---
    print("Python CPU...")
    import evm
    py_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        evm.magnify_motion_lpyr_iir(
            VID, "/tmp/_py.mp4",
            alpha=ALPHA, lambda_c=LAMBDA_C, r1=R1, r2=R2, chrom_attenuation=CHROM_ATT)
        py_times.append(time.perf_counter() - t0)
    py_med = median(py_times)
    results["Python CPU"] = py_med
    print(f"  {py_med:.3f}s\n")

    # --- 2. CUDA FP32 (using existing profiler for stage breakdown) ---
    print("CUDA FP32 (stage breakdown)...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CUDA_DIR)
    import subprocess
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "profile_motion.py")],
        capture_output=True, text=True, env=env, check=True, timeout=180)
    print(r.stdout)
    # Parse total from profiler output
    import re
    m = re.search(r"Pipeline total.*?([\d.]+)s", r.stdout)
    fp32_total = float(m.group(1)) if m else 0
    results["CUDA FP32"] = fp32_total

    # --- 3. CUDA FP16 end-to-end ---
    print("CUDA FP16 (end-to-end, decode+encode included)...")
    from evm_cuda.batched import magnify_motion_lpyr_iir_fp16
    def run_fp16():
        t0 = time.perf_counter()
        magnify_motion_lpyr_iir_fp16(
            VID, "/tmp/_fp16.mp4",
            alpha=ALPHA, lambda_c=LAMBDA_C, r1=R1, r2=R2, chrom_attenuation=CHROM_ATT)
        return time.perf_counter() - t0
    run_fp16()  # warmup
    fp16_times = [run_fp16() for _ in range(5)]
    fp16_med = median(fp16_times)
    results["CUDA FP16 (e2e)"] = fp16_med
    print(f"  {fp16_med:.3f}s\n")

    # --- 4. CUDA FP32 end-to-end (for fair comparison with FP16 e2e) ---
    print("CUDA FP32 (end-to-end, decode+encode included)...")
    from evm_cuda.batched import magnify_motion_lpyr_iir
    def run_fp32():
        t0 = time.perf_counter()
        magnify_motion_lpyr_iir(
            VID, "/tmp/_fp32.mp4",
            alpha=ALPHA, lambda_c=LAMBDA_C, r1=R1, r2=R2, chrom_attenuation=CHROM_ATT)
        return time.perf_counter() - t0
    run_fp32()  # warmup
    fp32_times = [run_fp32() for _ in range(5)]
    fp32_e2e = median(fp32_times)
    results["CUDA FP32 (e2e)"] = fp32_e2e
    print(f"  {fp32_e2e:.3f}s\n")

    # --- Summary ---
    print("=" * 65)
    print(f"{'Path':<25s} {'Time':>10s} {'vs CPU':>8s}")
    print("-" * 45)
    print(f"{'Python CPU':<25s} {py_med:>9.3f}s {'1.0x':>7s}")
    print(f"{'CUDA FP32 (GPU-only)':<25s} {fp32_total:>9.3f}s {py_med/fp32_total:>7.1f}x")
    print(f"{'CUDA FP32 (end-to-end)':<25s} {fp32_e2e:>9.3f}s {py_med/fp32_e2e:>7.1f}x")
    print(f"{'CUDA FP16 (end-to-end)':<25s} {fp16_med:>9.3f}s {py_med/fp16_med:>7.1f}x")
    print("-" * 45)
    print(f"\nFP32 e2e vs FP16 e2e: {fp32_e2e:.3f}s -> {fp16_med:.3f}s ({(1-fp16_med/fp32_e2e)*100:+.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
