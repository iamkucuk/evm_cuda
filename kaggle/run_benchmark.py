#!/usr/bin/env python3
"""EVM CUDA benchmark script for Kaggle Kernels.

Pushes to Kaggle's free GPU (T4 or P100), builds the CUDA extension from
source, runs both pipeline profilers, renders output videos, and saves a
results summary.

Usage on Kaggle:
    kaggle kernels push -p kaggle/

Usage standalone (any machine with nvcc + GPU):
    python kaggle/run_benchmark.py

The script auto-detects the GPU's compute capability and builds for that
single architecture (fast compile).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_URL = "https://github.com/iamkucuk/evm_cuda.git"
BRANCH = "feature/kernel-optimization"
REPO_DIR = Path("evm_cuda")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command, stream output to console."""
    print(f"$ {' '.join(cmd)}")
    kw.setdefault("check", True)
    return subprocess.run(cmd, **kw)


def detect_gpu() -> tuple[str, str]:
    """Return (gpu_name, cuda_arch) e.g. ('Tesla T4', '75')."""
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    parts = r.stdout.strip().split(", ")
    name = parts[0]
    cap = parts[1].replace(".", "")
    return name, cap


def clone_repo():
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(REPO_DIR)])


def build_extension(repo: Path, cuda_arch: str):
    """Build the pybind11 extension for the detected GPU architecture."""
    os.chdir(repo)

    # Install build deps (Kaggle has pip but not cmake/ninja)
    run([sys.executable, "-m", "pip", "install", "-q",
         "cmake", "ninja", "pybind11", "numpy", "scipy",
         "opencv-python", "requests"], check=True)

    # pybind11 CMake dir
    r = subprocess.run(
        [sys.executable, "-c", "import pybind11; print(pybind11.get_cmake_dir())"],
        capture_output=True, text=True, check=True,
    )
    os.environ["pybind11_DIR"] = r.stdout.strip()

    build_dir = repo / "cuda" / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    run([
        "cmake", "-S", "cuda", "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
        "-G", "Ninja",
    ], check=True)

    run(["cmake", "--build", str(build_dir), "--config", "Release", "-j"],
        check=True)
    print("Build complete.\n")


def download_samples(repo: Path):
    os.chdir(repo)
    run([sys.executable, "scripts/download_samples.py", "face", "baby"],
        check=True)


def run_profiler(repo: Path, name: str) -> dict | None:
    """Run a profiler, parse its output, return parsed timings or None on failure."""
    os.chdir(repo)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "cuda")

    script = f"scripts/profile_{name}.py"
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, env=env, check=True, timeout=180,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = getattr(e, "stderr", "") or ""
        print(f"\n{name} profiler failed: {e}\n{stderr[-500:]}")
        if "out of memory" in stderr.lower():
            print("GPU ran out of memory. Try a GPU with more VRAM.")
        return None

    print(result.stdout)
    return parse_profiler_output(result.stdout)


def parse_profiler_output(stdout: str) -> dict | None:
    """Extract the total time and per-stage medians from profiler output."""
    timings = {}
    total = None
    for line in stdout.splitlines():
        if "Pipeline total" in line:
            m = re.search(r"([\d.]+)s", line)
            if m:
                total = float(m.group(1))
        # Parse stage rows like "  color_cvt     0.0006s ..."
        m = re.match(r"\s*(.+?)\s+([\d.]+)s\s+([\d.]+)s\s+([\d.]+)s\s+([\d.]+)%", line)
        if m:
            stage = m.group(1).strip()
            median = float(m.group(2))
            timings[stage] = median
    if total is not None:
        timings["_total"] = total
    return timings if timings else None


def render_video(repo: Path, mode: str, inp: str, outp: str, **params):
    """Render one video end-to-end."""
    os.chdir(repo)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "cuda")

    if mode == "color":
        from evm_cuda.batched import magnify_color_gdown_ideal
        fn = magnify_color_gdown_ideal
    elif mode == "motion":
        from evm_cuda.batched import magnify_motion_lpyr_iir
        fn = magnify_motion_lpyr_iir
    else:
        raise ValueError(mode)

    os.makedirs("output", exist_ok=True)
    t0 = time.time()
    try:
        fn(inp, outp, **params)
        elapsed = time.time() - t0
        size = os.path.getsize(outp) / 1024 / 1024
        print(f"  {outp}: {size:.1f} MB, rendered in {elapsed:.2f}s")
        return elapsed
    except Exception as e:
        print(f"  {outp}: FAILED ({e})")
        return None


def compute_throughput(total_s: float, h: int, w: int, n_frames: int = 291) -> dict:
    """Compute throughput metrics from pipeline total time."""
    px_per_frame = h * w
    fps = n_frames / total_s
    gpx_per_s = fps * px_per_frame / 1e9
    fhd_1080p = gpx_per_s / (1920 * 1080) * 1e9
    return {
        "total_s": total_s,
        "fps": fps,
        "gpx_per_s": gpx_per_s,
        "1080p_fps": fhd_1080p,
    }


def main():
    print("=" * 60)
    print("EVM CUDA Benchmark")
    print("=" * 60)

    # Detect GPU
    gpu_name, cuda_arch = detect_gpu()
    print(f"GPU: {gpu_name}, compute capability: sm_{cuda_arch}\n")

    # Clone and build
    clone_repo()
    repo = REPO_DIR.resolve()
    build_extension(repo, cuda_arch)

    # Verify extension loads
    sys.path.insert(0, str(repo / "cuda"))
    os.chdir(repo)
    import evm_cuda
    assert evm_cuda.have_cuda, "CUDA not available"
    from evm_cuda import _evm_cuda
    print(f"Extension loaded. have_cuda=True\n")

    # Download samples
    download_samples(repo)

    # Run profilers
    print("\n" + "=" * 60)
    print("COLOR PIPELINE (face.mp4, 528x592)")
    print("=" * 60)
    color_t = run_profiler(repo, "color")

    print("\n" + "=" * 60)
    print("MOTION PIPELINE (baby.mp4, 960x544)")
    print("=" * 60)
    motion_t = run_profiler(repo, "motion")

    # Render videos
    print("\n" + "=" * 60)
    print("RENDER OUTPUT VIDEOS")
    print("=" * 60)
    render_video(repo, "color",
                 "data/face.mp4", "output/face_color.mp4",
                 alpha=50, level=4, fl=50/60, fh=60/60,
                 chrom_attenuation=1.0, sampling_rate=30.0)
    render_video(repo, "motion",
                 "data/baby.mp4", "output/baby_motion.mp4",
                 alpha=10, lambda_c=16, r1=0.4, r2=0.05,
                 chrom_attenuation=0.1)

    # Results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"GPU:    {gpu_name}")
    print(f"Arch:   sm_{cuda_arch}")

    results = {"gpu": gpu_name, "arch": cuda_arch}

    if color_t:
        ct = compute_throughput(color_t["_total"], 528, 592)
        print(f"\nColor pipeline (face.mp4):")
        print(f"  Total:    {ct['total_s']*1000:.1f} ms")
        print(f"  Throughput: {ct['gpx_per_s']:.2f} Gpx/s")
        print(f"  1080p:    {ct['1080p_fps']:.0f} fps")
        results["color"] = ct

    if motion_t:
        mt = compute_throughput(motion_t["_total"], 544, 960)
        print(f"\nMotion pipeline (baby.mp4):")
        print(f"  Total:    {mt['total_s']*1000:.1f} ms")
        print(f"  Throughput: {mt['gpx_per_s']:.2f} Gpx/s")
        print(f"  1080p:    {mt['1080p_fps']:.0f} fps")
        results["motion"] = mt

    # Reference comparison
    print(f"\nReference (NVIDIA H100):")
    print(f"  Color:  0.081s (1.12 Gpx/s, 542 fps @ 1080p)")
    print(f"  Motion: 0.181s (0.84 Gpx/s, 405 fps @ 1080p)")

    # Save results JSON
    results_path = repo / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
