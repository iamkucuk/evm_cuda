#!/usr/bin/env python3
"""GPU-only profiler: FP32 + FP16 for both pipelines on any GPU.

The CPU baseline is the same Python code regardless of GPU, so we use the
A100 measurements as reference. This script focuses on measuring how the GPU
pipelines scale across different hardware (P100, T4, A100, etc).

All measurement goes through ``evm_cuda.benchmark.run`` / ``summarize`` — the
SAME code the Colab notebook uses — so Kaggle and Colab can never drift apart
on methodology. Configs that don't fit in VRAM are skipped gracefully.

Push:
    kaggle kernels push -p kaggle/
Status:
    kaggle kernels status furkankucuk/evm-cuda-gpu-comparison
Pull:
    kaggle kernels output furkankucuk/evm-cuda-gpu-comparison -p ./results_gpu
"""
from __future__ import annotations
import gc, json, os, shutil, subprocess, sys
from pathlib import Path

REPO_URL = "https://github.com/iamkucuk/eulerian-video-magnification-cuda.git"
BRANCH = "main"   # was 'feature/kernel-optimization' — now merged
REPO_DIR = Path("evm_cuda")

# CPU reference (A100 run, 2026-06-24). GPU-independent, for speedup ratios.
CPU_REF_MS = {"color": 10350, "motion": 46255}

# Pipeline parameters (MIT face/baby samples, reproduceResults.m).
COLOR = dict(alpha=50, level=4, fl=50/60, fh=60/60,
             chrom_attenuation=1.0, sampling_rate=30.0)
MOTION = dict(alpha=10, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)


def run(cmd, **kw):
    print(f"$ {' '.join(cmd)}")
    kw.setdefault("check", True)
    return subprocess.run(cmd, **kw)


def main():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True)
    gpu_name, cuda_arch, vram = (s.strip() for s in r.stdout.strip().split(","))
    print(f"GPU: {gpu_name}, sm_{cuda_arch}, VRAM: {vram}\n")

    # --- Clone + build ---
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(REPO_DIR)])
    os.chdir(REPO_DIR)

    run([sys.executable, "-m", "pip", "install", "-q",
         "cmake", "ninja", "pybind11", "numpy", "scipy",
         "opencv-python", "requests", "av"])
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
         f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch.replace('.', '')}", "-G", "Ninja"])
    run(["cmake", "--build", str(build_dir), "--config", "Release", "-j"])
    print("Build complete.\n")

    run([sys.executable, "scripts/download_samples.py", "face", "baby"])
    sys.path.insert(0, str(Path("cuda").resolve()))

    # --- Run all 4 configs via the shared benchmark API ---
    from evm_cuda import benchmark

    os.makedirs("output", exist_ok=True)
    results = []
    for pipeline, precision, params, out in [
        ("color", "fp32", COLOR, "output/face_fp32.mp4"),
        ("color", "fp16", COLOR, "output/face_fp16.mp4"),
        ("motion", "fp32", MOTION, "output/baby_fp32.mp4"),
        ("motion", "fp16", MOTION, "output/baby_fp16.mp4"),
    ]:
        print("\n" + "=" * 60)
        print(f"{pipeline.upper()} — {precision.upper()}")
        print("=" * 60)
        vid = "data/face.mp4" if pipeline == "color" else "data/baby.mp4"
        res = benchmark.run(pipeline, precision, dict(vid=vid, **params),
                            out_path=out, n_iter=5)
        print(res)
        results.append(res)
        gc.collect()

    # --- Comparison table (FP32 vs FP16) ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(benchmark.summarize(results, n_iter=5))

    # --- Speedup vs the CPU baseline ---
    print("\nSpeedup vs CPU baseline (A100 reference):")
    by = {(r.pipeline, r.precision): r for r in results}
    for pipe, cpu_ms in CPU_REF_MS.items():
        for prec in ("fp32", "fp16"):
            res = by.get((pipe, prec))
            if res and res.measured:
                print(f"  {pipe} {prec}: {cpu_ms / res.total_ms:.0f}x "
                      f"({res.total_ms:.0f} ms vs {cpu_ms} ms CPU)")

    # --- Output videos ---
    outdir = Path("output")
    if outdir.exists():
        print("\nOutput videos:")
        for f in sorted(outdir.glob("*.mp4")):
            print(f"  {f.name}: {f.stat().st_size/1024/1024:.1f} MB")

    # --- Persist raw results as JSON ---
    payload = {
        "gpu": gpu_name, "arch": cuda_arch, "vram": vram,
        "cpu_ref_ms": CPU_REF_MS,
        "results": [
            {"pipeline": r.pipeline, "precision": r.precision,
             "total_ms": r.total_ms, "gpu": r.gpu, "notes": r.notes,
             "stages": [{"name": s.name, "median_ms": s.median_ms}
                        for s in r.stages]}
            for r in results
        ],
    }
    with open("gpu_comparison_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("\nResults saved to gpu_comparison_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
