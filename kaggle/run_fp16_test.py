#!/usr/bin/env python3
"""FP16 precision test for EVM CUDA motion pipeline.

Builds the extension, runs both FP32 and FP16 motion pipelines on baby.mp4,
and compares RMSE to check if FP16 storage holds the <0.01 tolerance.
"""
from __future__ import annotations
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

REPO_URL = "https://github.com/iamkucuk/evm_cuda.git"
BRANCH = "feature/kernel-optimization"
REPO_DIR = Path("evm_cuda")


def run(cmd, **kw):
    print(f"$ {' '.join(cmd)}")
    kw.setdefault("check", True)
    return subprocess.run(cmd, **kw)


def detect_gpu():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, check=True)
    parts = r.stdout.strip().split(", ")
    return parts[0], parts[1].replace(".", "")


def main():
    gpu_name, cuda_arch = detect_gpu()
    print(f"GPU: {gpu_name}, sm_{cuda_arch}\n")

    # Clone
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(REPO_DIR)])
    os.chdir(REPO_DIR)

    # Build deps
    run([sys.executable, "-m", "pip", "install", "-q",
         "cmake", "ninja", "pybind11", "numpy", "scipy",
         "opencv-python", "requests"])

    # Build
    r = subprocess.run(
        [sys.executable, "-c", "import pybind11; print(pybind11.get_cmake_dir())"],
        capture_output=True, text=True, check=True)
    os.environ["pybind11_DIR"] = r.stdout.strip()

    build_dir = Path("cuda/build")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    run(["cmake", "-S", "cuda", "-B", str(build_dir),
         "-DCMAKE_BUILD_TYPE=Release",
         f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}", "-G", "Ninja"])
    run(["cmake", "--build", str(build_dir), "--config", "Release", "-j"])
    print("Build complete.\n")

    # Download data
    run([sys.executable, "scripts/download_samples.py", "baby"])

    # Run FP16 vs FP32 test
    sys.path.insert(0, str(Path("cuda").resolve()))
    import numpy as np
    from evm_cuda.batched import magnify_motion_lpyr_iir, magnify_motion_lpyr_iir_fp16

    os.makedirs("output", exist_ok=True)

    print("Running FP32 motion pipeline...")
    t0 = time.time()
    out32 = magnify_motion_lpyr_iir(
        "data/baby.mp4", "output/baby_fp32.mp4",
        alpha=10, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)
    t32 = time.time() - t0
    print(f"  Done in {t32:.2f}s")

    print("\nRunning FP16 motion pipeline...")
    t0 = time.time()
    try:
        out16 = magnify_motion_lpyr_iir_fp16(
            "data/baby.mp4", "output/baby_fp16.mp4",
            alpha=10, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)
        t16 = time.time() - t0
        print(f"  Done in {t16:.2f}s")
    except Exception as e:
        print(f"  FAILED: {e}")
        return

    rmse = float(np.sqrt(np.mean((out32 - out16) ** 2)))
    max_err = float(np.max(np.abs(out32 - out16)))
    mae = float(np.mean(np.abs(out32 - out16)))

    print(f"\n{'='*50}")
    print(f"FP32 vs FP16 PRECISION COMPARISON")
    print(f"{'='*50}")
    print(f"GPU:      {gpu_name}")
    print(f"RMSE:     {rmse:.6f}")
    print(f"Max err:  {max_err:.6f}")
    print(f"MAE:      {mae:.6f}")
    print(f"Tolerance: <0.01 RMSE")
    print(f"PASS:     {rmse < 0.01}")
    print(f"\nTiming: FP32={t32:.2f}s  FP16={t16:.2f}s  speedup={t32/t16:.2f}x")
    print(f"{'='*50}")

    with open("fp16_results.json", "w") as f:
        json.dump({"gpu": gpu_name, "rmse": rmse, "max_err": max_err,
                    "mae": mae, "pass": rmse < 0.01,
                    "t32": t32, "t16": t16}, f, indent=2)


if __name__ == "__main__":
    main()
