#!/usr/bin/env python3
"""Profile FP16 vs FP32 motion pipeline on Kaggle GPU.

Builds the extension, then runs stage-by-stage profilers for both FP32
and FP16 motion pipelines, comparing per-stage timings.
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


def main():
    # Detect GPU
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, check=True)
    parts = r.stdout.strip().split(", ")
    gpu_name, cuda_arch = parts[0], parts[1].replace(".", "")
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

    # Profile FP32 motion pipeline (only if GPU has enough VRAM)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("cuda").resolve())

    # Check VRAM — FP32 motion needs ~24 GB
    mem_r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True)
    vram_mb = int(mem_r.stdout.strip().split()[0])

    fp32_output = ""
    if vram_mb >= 28000:  # 28 GB threshold
        print("=" * 60)
        print("FP32 MOTION PIPELINE PROFILE")
        print("=" * 60)
        try:
            result = subprocess.run(
                [sys.executable, "scripts/profile_motion.py"],
                capture_output=True, text=True, env=env, check=True, timeout=180)
            print(result.stdout)
            fp32_output = result.stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"FP32 profiler failed: {e}")
    else:
        print(f"GPU has {vram_mb} MB VRAM (< 28 GB needed for FP32 motion).")
        print("Skipping FP32 profile, running FP16 only.\n")

    # Profile FP16 motion pipeline (inline, since there's no separate profiler script)
    print("\n" + "=" * 60)
    print("FP16 MOTION PIPELINE PROFILE")
    print("=" * 60)

    sys.path.insert(0, str(Path("cuda").resolve()))
    import numpy as np
    from evm_cuda.batched import (DeviceBuffer, _read_frames, _d_binom5,
                                   _warmup_gpu_pool_motion, figure6_alpha_schedule)
    from evm_cuda import _evm_cuda

    def sync():
        _evm_cuda.device_synchronize()

    def median(xs):
        s = sorted(xs)
        n = len(s)
        return s[n//2] if n % 2 == 1 else (s[n//2-1] + s[n//2]) / 2

    VID = "data/baby.mp4"
    ALPHA = 10; LAMBDA_C = 16; R1 = 0.4; R2 = 0.05; CHROM_ATT = 0.1
    N_ITER = 5

    frames, fps = _read_frames(VID)
    n = len(frames)
    h, w = frames[0].shape[:2]
    print(f"clip: {n} frames, {h}x{w}, fps={fps:.1f}, iterations: {N_ITER}")

    levels = 1; hh, ww = h, w
    while hh >= 5 and ww >= 5:
        levels += 1; hh = (hh + 1) // 2; ww = (ww + 1) // 2
    alpha_sched = figure6_alpha_schedule(levels, ALPHA, LAMBDA_C, h, w)
    level_sizes = []
    ch, cw = h, w
    for _ in range(levels):
        level_sizes.append((ch, cw))
        ch = (ch + 1) // 2; cw = (cw + 1) // 2
    lvl_sizes = [s[0]*s[1] for s in level_sizes]
    total_band_floats = sum(s * (n*3) for s in lvl_sizes)
    clip_u8 = np.stack(frames, axis=0)
    ntsc_floats = n * h * w * 3

    _warmup_gpu_pool_motion(n, h, w, levels)
    sync()

    # Pre-allocate
    d_clip = DeviceBuffer.from_array(clip_u8)
    d_ntsc_f32 = DeviceBuffer(ntsc_floats * 4)
    d_ntsc_f16 = DeviceBuffer(ntsc_floats * 2)
    d_ntsc_planar = DeviceBuffer(n * 3 * h * w * 4)
    d_bands = DeviceBuffer(total_band_floats * 4)
    d_filtered = DeviceBuffer(total_band_floats * 4)
    d_delta_planar = DeviceBuffer(n * 3 * h * w * 4)
    d_out_u8 = DeviceBuffer(n * h * w * 3)
    max_sz = max(lvl_sizes)
    d_nt = DeviceBuffer(n * max_sz * 4)
    d_filt_nt = DeviceBuffer(n * max_sz * 4)
    level_offsets = []
    off = 0
    for sz in lvl_sizes:
        level_offsets.append(off)
        off += sz * n * 3

    def run_once():
        st = {}
        # Stage A: NTSC convert + f32->f16
        t0 = time.perf_counter()
        _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, d_ntsc_f32.ptr, n, h, w)
        _evm_cuda.f32_to_f16(d_ntsc_f32.ptr, d_ntsc_f16.ptr, ntsc_floats)
        sync()
        st["A) NTSC + f32->f16"] = time.perf_counter() - t0

        # Stage B: f16->f32 + planar + lpyr_build
        t0 = time.perf_counter()
        _evm_cuda.f16_to_f32(d_ntsc_f16.ptr, d_ntsc_f32.ptr, ntsc_floats)
        _evm_cuda.batched_to_planar_3ch(d_ntsc_f32.ptr, d_ntsc_planar.ptr, n, h, w)
        _evm_cuda.batched_lpyr_build(d_ntsc_planar.ptr, d_bands.ptr, n, h, w, levels, _d_binom5(), 5)
        sync()
        st["B) f16->f32 + lpyr_build"] = time.perf_counter() - t0

        # Stage C: temporal IIR
        t0 = time.perf_counter()
        for l in range(levels):
            sz = lvl_sizes[l]
            for c in range(3):
                sig_off = level_offsets[l] + c * n * sz
                _evm_cuda.batched_thwc_to_nt(d_bands.ptr_at(sig_off), d_nt.ptr, n, sz)
                _evm_cuda.batched_iir_bandpass(d_nt.ptr, d_filt_nt.ptr, n, sz, R1, R2)
                dst_off = level_offsets[l] + c * n * sz
                _evm_cuda.batched_nt_to_thwc_scaled(d_filt_nt.ptr, d_filtered.ptr_at(dst_off), n, sz, alpha_sched[l])
        sync()
        st["C) temporal IIR"] = time.perf_counter() - t0

        # Stage D1: lpyr_recon
        t0 = time.perf_counter()
        _evm_cuda.batched_lpyr_recon(d_filtered.ptr, d_delta_planar.ptr, n, h, w, levels, _d_binom5(), 5)
        sync()
        st["D1) lpyr_recon"] = time.perf_counter() - t0

        # Stage D2: f16->f32 + render
        t0 = time.perf_counter()
        _evm_cuda.f16_to_f32(d_ntsc_f16.ptr, d_ntsc_f32.ptr, ntsc_floats)
        _evm_cuda.batched_add_planar_quantize(d_ntsc_f32.ptr, d_delta_planar.ptr, d_out_u8.ptr, n, h, w, CHROM_ATT)
        sync()
        st["D2) f16->f32 + render"] = time.perf_counter() - t0

        return st

    # Warmup
    run_once()
    # Timed
    all_runs = [run_once() for _ in range(N_ITER)]

    stage_keys = list(all_runs[0].keys())
    print(f"\n{'Stage':<28s} {'median':>8s} {'min':>8s} {'max':>8s} {'%':>6s}")
    print("-" * 63)
    medians = {k: median([r[k] for r in all_runs]) for k in stage_keys}
    total_med = sum(medians.values())
    for k in stage_keys:
        vals = [r[k] for r in all_runs]
        pct = medians[k] / total_med * 100
        print(f"{k:<28s} {medians[k]:>7.4f}s {min(vals):>7.4f}s {max(vals):>7.4f}s {pct:>5.1f}%")
    print("-" * 63)
    print(f"{'Pipeline total (median)':.<28s} {total_med:>7.4f}s")


if __name__ == "__main__":
    main()
