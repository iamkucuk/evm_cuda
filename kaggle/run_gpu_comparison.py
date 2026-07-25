#!/usr/bin/env python3
"""GPU profiler: color+motion × FP32+FP16 on Kaggle (P100 or T4).

Uses the same ``evm_cuda.benchmark.run`` harness as local/osiris/H100:
1 untimed warmup + median of ``N_ITER`` timed runs, sync per stage.

Push:
    kaggle kernels push -p kaggle/
Status:
    kaggle kernels status furkankucuk/evm-cuda-gpu-comparison
Pull:
    kaggle kernels output furkankucuk/evm-cuda-gpu-comparison -p ./kaggle/results_gpu
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/iamkucuk/eulerian-video-magnification-cuda.git"
# Measure the production path that owns the half-smem / true half-band work.
BRANCH = "feature/true-fp16-motion"
REPO_DIR = Path("evm_cuda")
N_ITER = 7

# CPU baselines used in README / blog_speedup (Python/NumPy).
CPU_REF_MS = {"color": 11194.0, "motion": 44190.0}

# Match remeasure scripts on osiris/TRUBA.
COLOR = dict(
    alpha=50.0,
    level=4,
    fl=50 / 60,
    fh=60 / 60,
    chrom_attenuation=1.0,
    sampling_rate=30.0,
)
MOTION = dict(
    alpha=20.0,
    lambda_c=16.0,
    r1=0.4,
    r2=0.05,
    chrom_attenuation=0.1,
)


def run(cmd, **kw):
    print(f"$ {' '.join(cmd)}", flush=True)
    kw.setdefault("check", True)
    return subprocess.run(cmd, **kw)


def pack(r):
    return {
        "pipeline": r.pipeline,
        "precision": r.precision,
        "compute_ms": r.compute_ms,
        "transfer_ms": r.transfer_ms,
        "total_ms": r.total_ms,
        "gpu": r.gpu,
        "notes": r.notes,
        "stages": [
            {
                "name": s.name,
                "median_ms": s.median_ms,
                "min_ms": s.min_ms,
                "max_ms": s.max_ms,
            }
            for s in r.stages
        ],
    }


def main():
    r = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    gpu_name, cuda_arch, vram = (s.strip() for s in r.stdout.strip().split(","))
    print(f"GPU: {gpu_name}, sm_{cuda_arch}, VRAM: {vram}\n", flush=True)

    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "-b",
            BRANCH,
            REPO_URL,
            str(REPO_DIR),
        ]
    )
    os.chdir(REPO_DIR)

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "cmake",
            "ninja",
            "pybind11",
            "numpy",
            "scipy",
            "opencv-python",
            "requests",
            "av",
        ]
    )
    r = subprocess.run(
        [sys.executable, "-c", "import pybind11; print(pybind11.get_cmake_dir())"],
        capture_output=True,
        text=True,
        check=True,
    )
    os.environ["pybind11_DIR"] = r.stdout.strip()

    build_dir = Path("cuda/build")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    sm = cuda_arch.replace(".", "")
    run(
        [
            "cmake",
            "-S",
            "cuda",
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CUDA_ARCHITECTURES={sm}",
            "-G",
            "Ninja",
        ]
    )
    run(["cmake", "--build", str(build_dir), "--config", "Release", "-j"])
    print("Build complete.\n", flush=True)

    run([sys.executable, "scripts/download_samples.py", "face", "baby"])
    # Repo root for shared.h264 encode path; cuda/ for _evm_cuda package.
    sys.path.insert(0, str(Path(".").resolve()))
    sys.path.insert(0, str(Path("cuda").resolve()))

    from evm_cuda import benchmark

    os.makedirs("output", exist_ok=True)
    results = []
    packed = []
    for pipeline, precision, params, out in [
        ("color", "fp32", COLOR, "output/face_fp32.mp4"),
        ("color", "fp16", COLOR, "output/face_fp16.mp4"),
        ("motion", "fp32", MOTION, "output/baby_fp32.mp4"),
        ("motion", "fp16", MOTION, "output/baby_fp16.mp4"),
    ]:
        print("\n" + "=" * 60, flush=True)
        print(f"{pipeline.upper()} — {precision.upper()}", flush=True)
        print("=" * 60, flush=True)
        vid = "data/face.mp4" if pipeline == "color" else "data/baby.mp4"
        res = benchmark.run(
            pipeline,
            precision,
            dict(vid=vid, **params),
            out_path=out,
            n_iter=N_ITER,
        )
        print(res, flush=True)
        results.append(res)
        packed.append(pack(res))
        gc.collect()

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(benchmark.summarize(results, n_iter=N_ITER), flush=True)

    print("\nSpeedup vs CPU baseline (README CPU numbers):", flush=True)
    by = {(r.pipeline, r.precision): r for r in results}
    for pipe, cpu_ms in CPU_REF_MS.items():
        for prec in ("fp32", "fp16"):
            res = by.get((pipe, prec))
            if res and res.measured and res.compute_ms > 0:
                print(
                    f"  {pipe} {prec}: compute {cpu_ms / res.compute_ms:.0f}x "
                    f"({res.compute_ms:.1f} ms vs {cpu_ms} ms CPU); "
                    f"full {cpu_ms / res.total_ms:.0f}x ({res.total_ms:.1f} ms)",
                    flush=True,
                )
            elif res:
                print(f"  {pipe} {prec}: skipped ({res.notes})", flush=True)

    outdir = Path("output")
    if outdir.exists():
        print("\nOutput videos:", flush=True)
        for f in sorted(outdir.glob("*.mp4")):
            print(f"  {f.name}: {f.stat().st_size / 1024 / 1024:.1f} MB", flush=True)

    payload = {
        "gpu": gpu_name,
        "arch": cuda_arch,
        "vram": vram,
        "branch": BRANCH,
        "n_iter": N_ITER,
        "cpu_ref_ms": CPU_REF_MS,
        "params": {"color": COLOR, "motion": MOTION},
        "results": packed,
    }
    with open("gpu_comparison_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("\nResults saved to gpu_comparison_results.json", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
