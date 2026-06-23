#!/usr/bin/env python3
"""Full CPU vs FP32 vs FP16 comparison on Kaggle GPU.

Clones the repo, builds for the detected GPU, runs the full comparison
profiler (CPU + FP32 + FP16, both pipelines, per-stage breakdown), and
renders all 6 output videos (face/baby × cpu/fp32/fp16).

Push:
    kaggle kernels push -p kaggle/
Status:
    kaggle kernels status furkankucuk/evm-cuda-full-comparison
Pull:
    kaggle kernels output furkankucuk/evm-cuda-full-comparison -p ./results_full
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO_URL = "https://github.com/iamkucuk/evm_cuda.git"
BRANCH = "feature/kernel-optimization"
REPO_DIR = Path("evm_cuda")


def run(cmd, **kw):
    print(f"$ {' '.join(cmd)}")
    kw.setdefault("check", True)
    return subprocess.run(cmd, **kw)


def main():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True)
    parts = r.stdout.strip().split(", ")
    gpu_name, cuda_arch, vram = parts[0], parts[1].replace(".", ""), parts[2]
    print(f"GPU: {gpu_name}, sm_{cuda_arch}, VRAM: {vram}\n")

    # --- Clone + build ---
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(REPO_DIR)])
    os.chdir(REPO_DIR)

    run([sys.executable, "-m", "pip", "install", "-q",
         "cmake", "ninja", "pybind11", "numpy", "scipy",
         "opencv-python", "requests"])

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

    # --- Download samples ---
    run([sys.executable, "scripts/download_samples.py", "face", "baby", "--with-references"])

    # --- Run the full comparison profiler ---
    # This runs CPU + FP32 + FP16 for both pipelines with per-stage breakdown
    # and renders all 6 output videos. The script is self-contained.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("cuda").resolve())

    print("\n" + "=" * 60)
    print("RUNNING FULL COMPARISON: CPU vs FP32 vs FP16")
    print("=" * 60)

    # Stream output directly so Kaggle's log captures it line-by-line.
    # check=False so an OOM on FP32 motion (16GB GPUs) doesn't kill the run.
    result = subprocess.run(
        [sys.executable, "-u", "scripts/profile_full_comparison.py"],
        env=env, timeout=1800,
    )

    # Copy JSON results if they exist (profiler writes it)
    results = {}
    if Path("comparison_results.json").exists():
        with open("comparison_results.json") as f:
            results = json.load(f)

    # --- List output videos ---
    print("\n" + "=" * 60)
    print("OUTPUT VIDEOS")
    print("=" * 60)
    outdir = Path("output")
    if outdir.exists():
        for f in sorted(outdir.glob("*.mp4")):
            size = f.stat().st_size / 1024 / 1024
            print(f"  {f.name}: {size:.1f} MB")

    # GPU info for the summary
    results["gpu"] = gpu_name
    results["arch"] = cuda_arch
    results["vram"] = vram

    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nGPU: {gpu_name} (sm_{cuda_arch}, {vram})")
    print("Results saved to comparison_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
