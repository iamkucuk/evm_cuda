#!/usr/bin/env python3
"""Profile FP16 motion pipeline on Kaggle GPU.

Allocates/frees buffers per-iteration to match the real pipeline's memory
footprint (critical for fitting on 16 GB GPUs).
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
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True)
    parts = r.stdout.strip().split(", ")
    gpu_name, cuda_arch, vram = parts[0], parts[1].replace(".", ""), parts[2]
    print(f"GPU: {gpu_name}, sm_{cuda_arch}, VRAM: {vram}\n")

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

    run([sys.executable, "scripts/download_samples.py", "baby"])

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
    planar_floats = n * 3 * h * w

    _warmup_gpu_pool_motion(n, h, w, levels)
    sync()

    level_offsets = []
    off = 0
    for sz in lvl_sizes:
        level_offsets.append(off)
        off += sz * n * 3

    # Only d_clip and d_out_u8 persist across stages.
    d_clip = DeviceBuffer.from_array(clip_u8)
    d_out_u8 = DeviceBuffer(n * h * w * 3)

    def run_once():
        st = {}

        # Stage A: NTSC convert (FP32), store as FP16
        t0 = time.perf_counter()
        ntsc_f32 = DeviceBuffer(ntsc_floats * 4)
        ntsc_f16 = DeviceBuffer(ntsc_floats * 2)
        _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, ntsc_f32.ptr, n, h, w)
        _evm_cuda.f32_to_f16(ntsc_f32.ptr, ntsc_f16.ptr, ntsc_floats)
        sync()
        st["A) NTSC + f32->f16"] = time.perf_counter() - t0
        del ntsc_f32

        # Stage B: f16->f32 + planar + f32->f16 + lpyr_build_f16
        t0 = time.perf_counter()
        ntsc_f32 = DeviceBuffer(ntsc_floats * 4)
        _evm_cuda.f16_to_f32(ntsc_f16.ptr, ntsc_f32.ptr, ntsc_floats)
        planar = DeviceBuffer(planar_floats * 4)
        _evm_cuda.batched_to_planar_3ch(ntsc_f32.ptr, planar.ptr, n, h, w)
        del ntsc_f32
        planar_f16 = DeviceBuffer(planar_floats * 2)
        _evm_cuda.f32_to_f16(planar.ptr, planar_f16.ptr, planar_floats)
        del planar
        bands = DeviceBuffer(total_band_floats * 4)
        _evm_cuda.batched_lpyr_build_f16(
            planar_f16.ptr, bands.ptr, n, h, w, levels, _d_binom5(), 5)
        del planar_f16
        sync()
        st["B) lpyr_build (f16 scratch)"] = time.perf_counter() - t0

        # Stage C: temporal IIR (FP32)
        t0 = time.perf_counter()
        filtered = DeviceBuffer(total_band_floats * 4)
        max_sz = max(lvl_sizes)
        nt_buf = DeviceBuffer(n * max_sz * 4)
        filt_buf = DeviceBuffer(n * max_sz * 4)
        for l in range(levels):
            sz = lvl_sizes[l]
            for c in range(3):
                sig_off = level_offsets[l] + c * n * sz
                _evm_cuda.batched_thwc_to_nt(bands.ptr_at(sig_off), nt_buf.ptr, n, sz)
                _evm_cuda.batched_iir_bandpass(nt_buf.ptr, filt_buf.ptr, n, sz, R1, R2)
                dst_off = level_offsets[l] + c * n * sz
                _evm_cuda.batched_nt_to_thwc_scaled(filt_buf.ptr, filtered.ptr_at(dst_off), n, sz, alpha_sched[l])
        sync()
        st["C) temporal IIR"] = time.perf_counter() - t0
        del bands, nt_buf, filt_buf

        # Stage D1: lpyr_recon
        t0 = time.perf_counter()
        delta = DeviceBuffer(n * 3 * h * w * 4)
        _evm_cuda.batched_lpyr_recon(filtered.ptr, delta.ptr, n, h, w, levels, _d_binom5(), 5)
        sync()
        st["D1) lpyr_recon"] = time.perf_counter() - t0
        del filtered

        # Stage D2: f16->f32 + render
        t0 = time.perf_counter()
        ntsc_f32 = DeviceBuffer(ntsc_floats * 4)
        _evm_cuda.f16_to_f32(ntsc_f16.ptr, ntsc_f32.ptr, ntsc_floats)
        _evm_cuda.batched_add_planar_quantize(ntsc_f32.ptr, delta.ptr, d_out_u8.ptr, n, h, w, CHROM_ATT)
        sync()
        st["D2) f16->f32 + render"] = time.perf_counter() - t0
        del ntsc_f32, ntsc_f16, delta

        return st

    # Warmup
    run_once()
    # Timed
    all_runs = [run_once() for _ in range(N_ITER)]

    stage_keys = list(all_runs[0].keys())
    print(f"\n{'Stage':<30s} {'median':>8s} {'min':>8s} {'max':>8s} {'%':>6s}")
    print("-" * 65)
    medians = {k: median([r[k] for r in all_runs]) for k in stage_keys}
    total_med = sum(medians.values())
    for k in stage_keys:
        vals = [r[k] for r in all_runs]
        pct = medians[k] / total_med * 100
        print(f"{k:<30s} {medians[k]:>7.4f}s {min(vals):>7.4f}s {max(vals):>7.4f}s {pct:>5.1f}%")
    print("-" * 65)
    print(f"{'Pipeline total (median)':.<30s} {total_med:>7.4f}s")

    results = {"gpu": gpu_name, "vram": vram, "stages": medians, "total": total_med}
    with open("fp16_profile.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
