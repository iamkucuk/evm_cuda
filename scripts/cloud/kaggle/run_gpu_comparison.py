#!/usr/bin/env python3
"""GPU profiler: color+motion × FP32+FP16 on Kaggle (P100 or T4).

Uses the same ``vidmag.cuda.benchmark.run`` harness as local/H100:
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
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/iamkucuk/eulerian-video-magnification-cuda.git"
# The branch carrying the current kernels. Change back to "main" once this
# branch is merged there.
BRANCH = "library-restructure"
REPO_DIR = Path("evm_cuda")
N_ITER = 7

# Match remeasure scripts across hosts.
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


def measure_cpu_baseline() -> dict:
    """Time the NumPy reference on this machine, for both clips.

    Returned in milliseconds, keyed by pipeline. Any failure returns an empty
    mapping rather than raising: the graphics measurements are the point of this
    run, and losing them because a reference timing failed would be the wrong
    trade.
    """
    import time

    import numpy as np

    import vidmag
    from vidmag.cuda._common import read_frames

    out = {}
    for pipeline, clip, preset in (
        ("color", "data/face.mp4", "pulse"),
        ("motion", "data/baby.mp4", "motion"),
    ):
        try:
            frame_list, fps = read_frames(clip)
            frames = np.asarray(frame_list)
            t0 = time.perf_counter()
            vidmag.magnify(frames, preset=preset, fps=fps, backend="cpu")
            out[pipeline] = (time.perf_counter() - t0) * 1e3
            print(
                f"  processor baseline, {pipeline}: {out[pipeline]:,.0f} ms", flush=True
            )
        except Exception as exc:  # reported, never fatal — see the docstring
            print(f"  processor baseline, {pipeline}: FAILED ({exc})", flush=True)
    return out


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

    # Which commit these numbers describe. A shallow clone of a moving branch
    # is not enough to identify it later, and the A100 and H100 records in
    # benches/ are unusable for exactly that reason.
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print(f"Commit: {commit or 'unknown'}\n", flush=True)

    # One command installs the runtime dependencies and compiles the CUDA
    # extension for this GPU: the build defaults to
    # CMAKE_CUDA_ARCHITECTURES=native, and pip fetches the build backend plus
    # cmake/ninja itself, so no toolchain has to be provisioned first.
    run([sys.executable, "-m", "pip", "install", "-q", "."])
    print("Build complete.\n", flush=True)

    run([sys.executable, "scripts/download_samples.py", "face", "baby"])

    # The package was installed after this interpreter started, so the import
    # system's cached listing of site-packages has to be dropped before `vidmag`
    # can be found.
    importlib.invalidate_caches()
    from vidmag.cuda import benchmark

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

    # The processor baseline is measured HERE, on the machine that ran the
    # kernels. It used to be two numbers copied from the README, measured on a
    # different machine entirely; dividing this machine's graphics time by that
    # machine's processor time produces a ratio describing neither. If the
    # baseline cannot be measured the ratios are simply omitted, because a
    # missing ratio is better than a meaningless one.
    cpu_ref_ms = measure_cpu_baseline()

    print("\nSpeedup against this machine's own processor baseline:", flush=True)
    by = {(r.pipeline, r.precision): r for r in results}
    for pipe, cpu_ms in cpu_ref_ms.items():
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
        "commit": commit or "unknown",
        "n_iter": N_ITER,
        "cpu_ref_ms": cpu_ref_ms,
        "params": {"color": COLOR, "motion": MOTION},
        "results": packed,
    }
    with open("gpu_comparison_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("\nResults saved to gpu_comparison_results.json", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
