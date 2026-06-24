#!/usr/bin/env python3
"""GPU-only profiler: FP32 + FP16 for both pipelines on any GPU.

The CPU baseline is the same Python code regardless of GPU, so we use the
A100 measurements as reference. This script focuses on measuring how
the GPU pipelines scale across different hardware (P100, T4, A100, etc).

If a pipeline doesn't fit in VRAM (e.g. FP32 motion on 16 GB), it's skipped.

Push:
    kaggle kernels push -p kaggle/
Status:
    kaggle kernels status furkankucuk/evm-cuda-gpu-comparison
Pull:
    kaggle kernels output furkankucuk/evm-cuda-gpu-comparison -p ./results_gpu
"""
from __future__ import annotations
import gc, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

REPO_URL = "https://github.com/iamkucuk/eulerian-video-magnification-cuda.git"
BRANCH = "feature/kernel-optimization"
REPO_DIR = Path("evm_cuda")

# CPU reference (A100 run, 2026-06-24). These don't depend on the GPU.
CPU_REF = {
    "color": {
        "1) color_cvt": 1.418,
        "2) blur_dn": 6.373,
        "2b) D2H + reshape": 0.0,
        "3) ideal_bandpass": 0.061,
        "4) upsample + render": 2.510,
        "_total": 10.350,
    },
    "motion": {
        "A) NTSC convert": 2.159,
        "B) lpyr_build": 22.145,
        "C) temporal IIR": 4.711,
        "D1) lpyr_recon": 13.667,
        "D2) render": 3.596,
        "_total": 46.255,
    },
}


def run(cmd, **kw):
    print(f"$ {' '.join(cmd)}")
    kw.setdefault("check", True)
    return subprocess.run(cmd, **kw)


def sync():
    from evm_cuda import _evm_cuda
    _evm_cuda.device_synchronize()


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


def time_gpu_stages(setup_fn, n_iter=5):
    """Run a GPU stage profiler function n_iter times, return median per-stage."""
    setup_fn()  # warmup
    runs = [setup_fn() for _ in range(n_iter)]
    gc.collect()
    return {k: median([r[k] for r in runs]) for k in runs[0]}


def profile_color_fp32():
    from evm_cuda.batched import DeviceBuffer, _read_frames, _d_binom5_sum1, _warmup_gpu_pool
    from evm_cuda import _evm_cuda

    COLOR_VID = str(Path("data/face.mp4"))
    COLOR_LEVEL = 4
    COLOR_FL, COLOR_FH = 50/60, 60/60
    COLOR_CHROM = 1.0
    COLOR_SR = 30.0

    frames, fps = _read_frames(COLOR_VID)
    n = len(frames)
    h, w = frames[0].shape[:2]
    clip_u8 = np.stack(frames, axis=0)
    hl, wl = h, w
    for _ in range(COLOR_LEVEL):
        hl = (hl + 1) // 2; wl = (wl + 1) // 2

    _warmup_gpu_pool()
    sync()
    d_clip = DeviceBuffer.from_array(clip_u8)
    d_out = DeviceBuffer(n * h * w * 3)

    def run_once():
        st = {}
        t0 = time.perf_counter()
        ntsc = DeviceBuffer(n * h * w * 3 * 4)
        _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, ntsc.ptr, n, h, w)
        sync()
        st["1) color_cvt"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        planar = DeviceBuffer(n * 3 * h * w * 4)
        _evm_cuda.batched_to_planar_3ch(ntsc.ptr, planar.ptr, n, h, w)
        gdown = DeviceBuffer(n * 3 * hl * wl * 4)
        _evm_cuda.batched_blur_dn_color(planar.ptr, gdown.ptr, n * 3, h, w, COLOR_LEVEL, _d_binom5_sum1(), 5)
        sync()
        st["2) blur_dn"] = time.perf_counter() - t0
        del planar

        t0 = time.perf_counter()
        gd = gdown.download_f32(n * 3 * hl * wl).reshape(n, 3, hl, wl)
        gd = np.ascontiguousarray(gd.transpose(0, 2, 3, 1))
        del gdown
        st["2b) D2H + reshape"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        filt = np.empty_like(gd)
        for c in range(3):
            sig = np.ascontiguousarray(gd[..., c].reshape(n, hl * wl).T)
            d_sig = DeviceBuffer.from_array(sig)
            d_fo = DeviceBuffer(n * hl * wl * 4)
            _evm_cuda.batched_ideal_bandpass(d_sig.ptr, d_fo.ptr, n, hl * wl, COLOR_FL, COLOR_FH, COLOR_SR)
            filt[..., c] = d_fo.download_f32(n * hl * wl).reshape(hl * wl, n).T.reshape(n, hl, wl)
        sync()
        st["3) ideal_bandpass"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        gain = np.array([50, 50*COLOR_CHROM, 50*COLOR_CHROM], dtype=np.float32)
        filt = filt * gain
        d_filt = DeviceBuffer.from_array(np.ascontiguousarray(filt))
        _evm_cuda.batched_upsample_add_quantize(ntsc.ptr, d_filt.ptr, d_out.ptr, n, hl, wl, h, w, 1.0)
        sync()
        st["4) upsample + render"] = time.perf_counter() - t0
        del ntsc, d_filt
        st["_total"] = sum(v for k, v in st.items() if not k.startswith("_"))
        return st

    return time_gpu_stages(run_once)


def profile_color_fp16():
    from evm_cuda.batched import DeviceBuffer, _read_frames, _d_binom5_sum1, _warmup_gpu_pool
    from evm_cuda import _evm_cuda

    COLOR_VID = str(Path("data/face.mp4"))
    COLOR_LEVEL = 4
    COLOR_FL, COLOR_FH = 50/60, 60/60
    COLOR_CHROM = 1.0
    COLOR_SR = 30.0

    frames, fps = _read_frames(COLOR_VID)
    n = len(frames)
    h, w = frames[0].shape[:2]
    clip_u8 = np.stack(frames, axis=0)
    hl, wl = h, w
    for _ in range(COLOR_LEVEL):
        hl = (hl + 1) // 2; wl = (wl + 1) // 2

    ntsc_floats = n * h * w * 3
    _warmup_gpu_pool()
    sync()
    d_clip = DeviceBuffer.from_array(clip_u8)
    d_out = DeviceBuffer(n * h * w * 3)

    def run_once():
        st = {}
        t0 = time.perf_counter()
        ntsc_f32 = DeviceBuffer(ntsc_floats * 4)
        ntsc_f16 = DeviceBuffer(ntsc_floats * 2)
        _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, ntsc_f32.ptr, n, h, w)
        _evm_cuda.f32_to_f16(ntsc_f32.ptr, ntsc_f16.ptr, ntsc_floats)
        sync()
        st["1) color_cvt"] = time.perf_counter() - t0
        del ntsc_f32

        t0 = time.perf_counter()
        planar = DeviceBuffer(n * 3 * h * w * 2)
        _evm_cuda.batched_to_planar_3ch_f16(ntsc_f16.ptr, planar.ptr, n, h, w)
        gdown = DeviceBuffer(n * 3 * hl * wl * 4)
        _evm_cuda.batched_blur_dn_color_f16(planar.ptr, gdown.ptr, n * 3, h, w, COLOR_LEVEL, _d_binom5_sum1(), 5)
        sync()
        st["2) blur_dn"] = time.perf_counter() - t0
        del planar

        t0 = time.perf_counter()
        gd = gdown.download_f32(n * 3 * hl * wl).reshape(n, 3, hl, wl)
        gd = np.ascontiguousarray(gd.transpose(0, 2, 3, 1))
        del gdown
        st["2b) D2H + reshape"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        filt = np.empty_like(gd)
        for c in range(3):
            sig = np.ascontiguousarray(gd[..., c].reshape(n, hl * wl).T)
            d_sig = DeviceBuffer.from_array(sig)
            d_fo = DeviceBuffer(n * hl * wl * 4)
            _evm_cuda.batched_ideal_bandpass(d_sig.ptr, d_fo.ptr, n, hl * wl, COLOR_FL, COLOR_FH, COLOR_SR)
            filt[..., c] = d_fo.download_f32(n * hl * wl).reshape(hl * wl, n).T.reshape(n, hl, wl)
        sync()
        st["3) ideal_bandpass"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        gain = np.array([50, 50*COLOR_CHROM, 50*COLOR_CHROM], dtype=np.float32)
        filt = filt * gain
        d_filt = DeviceBuffer.from_array(np.ascontiguousarray(filt))
        _evm_cuda.batched_upsample_add_quantize_f16(ntsc_f16.ptr, d_filt.ptr, d_out.ptr, n, hl, wl, h, w, 1.0)
        sync()
        st["4) upsample + render"] = time.perf_counter() - t0
        del ntsc_f16, d_filt
        st["_total"] = sum(v for k, v in st.items() if not k.startswith("_"))
        return st

    return time_gpu_stages(run_once)


def profile_motion_fp32():
    from evm_cuda.batched import (DeviceBuffer, _read_frames, _d_binom5,
                                   _warmup_gpu_pool_motion, figure6_alpha_schedule)
    from evm_cuda import _evm_cuda

    MOTION_VID = str(Path("data/baby.mp4"))
    MOTION_ALPHA, MOTION_LAMBDA = 10, 16
    MOTION_R1, MOTION_R2 = 0.4, 0.05
    MOTION_CHROM = 0.1

    frames, fps = _read_frames(MOTION_VID)
    n = len(frames)
    h, w = frames[0].shape[:2]
    clip_u8 = np.stack(frames, axis=0)

    levels = 1; hh, ww = h, w
    while hh >= 5 and ww >= 5:
        levels += 1; hh = (hh + 1) // 2; ww = (ww + 1) // 2
    alpha_sched = figure6_alpha_schedule(levels, MOTION_ALPHA, MOTION_LAMBDA, h, w)
    level_sizes = []
    ch, cw = h, w
    for _ in range(levels):
        level_sizes.append((ch, cw)); ch = (ch + 1) // 2; cw = (cw + 1) // 2
    lvl_sizes = [s[0]*s[1] for s in level_sizes]
    total_band = sum(s * (n*3) for s in lvl_sizes)
    max_sz = max(lvl_sizes)

    level_offsets = []
    off = 0
    for sz in lvl_sizes:
        level_offsets.append(off); off += sz * n * 3

    _warmup_gpu_pool_motion(n, h, w, levels)
    sync()
    d_clip = DeviceBuffer.from_array(clip_u8)
    d_out = DeviceBuffer(n * h * w * 3)

    def run_once():
        st = {}
        t0 = time.perf_counter()
        ntsc = DeviceBuffer(n * h * w * 3 * 4)
        _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, ntsc.ptr, n, h, w)
        sync()
        st["A) NTSC convert"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        planar = DeviceBuffer(n * 3 * h * w * 4)
        _evm_cuda.batched_to_planar_3ch(ntsc.ptr, planar.ptr, n, h, w)
        bands = DeviceBuffer(total_band * 4)
        _evm_cuda.batched_lpyr_build(planar.ptr, bands.ptr, n, h, w, levels, _d_binom5(), 5)
        del planar
        sync()
        st["B) lpyr_build"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        filtered = DeviceBuffer(total_band * 4)
        nt_buf = DeviceBuffer(n * max_sz * 4)
        filt_buf = DeviceBuffer(n * max_sz * 4)
        for l in range(levels):
            sz = lvl_sizes[l]
            for c in range(3):
                sig_off = level_offsets[l] + c * n * sz
                _evm_cuda.batched_thwc_to_nt(bands.ptr_at(sig_off), nt_buf.ptr, n, sz)
                _evm_cuda.batched_iir_bandpass(nt_buf.ptr, filt_buf.ptr, n, sz, MOTION_R1, MOTION_R2)
                dst_off = level_offsets[l] + c * n * sz
                _evm_cuda.batched_nt_to_thwc_scaled(filt_buf.ptr, filtered.ptr_at(dst_off), n, sz, alpha_sched[l])
        sync()
        st["C) temporal IIR"] = time.perf_counter() - t0
        del bands, nt_buf, filt_buf

        t0 = time.perf_counter()
        delta = DeviceBuffer(n * 3 * h * w * 4)
        _evm_cuda.batched_lpyr_recon(filtered.ptr, delta.ptr, n, h, w, levels, _d_binom5(), 5)
        sync()
        st["D1) lpyr_recon"] = time.perf_counter() - t0
        del filtered

        t0 = time.perf_counter()
        _evm_cuda.batched_add_planar_quantize(ntsc.ptr, delta.ptr, d_out.ptr, n, h, w, MOTION_CHROM)
        sync()
        st["D2) render"] = time.perf_counter() - t0
        del ntsc, delta
        st["_total"] = sum(v for k, v in st.items() if not k.startswith("_"))
        return st

    return time_gpu_stages(run_once)


def profile_motion_fp16():
    from evm_cuda.batched import (DeviceBuffer, _read_frames, _d_binom5,
                                   _warmup_gpu_pool_motion, figure6_alpha_schedule)
    from evm_cuda import _evm_cuda

    MOTION_VID = str(Path("data/baby.mp4"))
    MOTION_ALPHA, MOTION_LAMBDA = 10, 16
    MOTION_R1, MOTION_R2 = 0.4, 0.05
    MOTION_CHROM = 0.1

    frames, fps = _read_frames(MOTION_VID)
    n = len(frames)
    h, w = frames[0].shape[:2]
    clip_u8 = np.stack(frames, axis=0)

    levels = 1; hh, ww = h, w
    while hh >= 5 and ww >= 5:
        levels += 1; hh = (hh + 1) // 2; ww = (ww + 1) // 2
    alpha_sched = figure6_alpha_schedule(levels, MOTION_ALPHA, MOTION_LAMBDA, h, w)
    level_sizes = []
    ch, cw = h, w
    for _ in range(levels):
        level_sizes.append((ch, cw)); ch = (ch + 1) // 2; cw = (cw + 1) // 2
    lvl_sizes = [s[0]*s[1] for s in level_sizes]
    total_band = sum(s * (n*3) for s in lvl_sizes)
    max_sz = max(lvl_sizes)
    ntsc_floats = n * h * w * 3
    planar_floats = n * 3 * h * w

    level_offsets = []
    off = 0
    for sz in lvl_sizes:
        level_offsets.append(off); off += sz * n * 3

    _warmup_gpu_pool_motion(n, h, w, levels)
    sync()
    d_clip = DeviceBuffer.from_array(clip_u8)
    d_out = DeviceBuffer(n * h * w * 3)

    def run_once():
        st = {}
        t0 = time.perf_counter()
        ntsc_f32 = DeviceBuffer(ntsc_floats * 4)
        ntsc_f16 = DeviceBuffer(ntsc_floats * 2)
        _evm_cuda.batched_bgr_u8_to_ntsc_f32(d_clip.ptr, ntsc_f32.ptr, n, h, w)
        _evm_cuda.f32_to_f16(ntsc_f32.ptr, ntsc_f16.ptr, ntsc_floats)
        sync()
        st["A) NTSC convert"] = time.perf_counter() - t0
        del ntsc_f32

        t0 = time.perf_counter()
        planar = DeviceBuffer(planar_floats * 2)
        _evm_cuda.batched_to_planar_3ch_f16(ntsc_f16.ptr, planar.ptr, n, h, w)
        bands_f32 = DeviceBuffer(total_band * 4)
        _evm_cuda.batched_lpyr_build_f16(planar.ptr, bands_f32.ptr, n, h, w, levels, _d_binom5(), 5)
        del planar
        bands = DeviceBuffer(total_band * 2)
        _evm_cuda.f32_to_f16(bands_f32.ptr, bands.ptr, total_band)
        del bands_f32
        sync()
        st["B) lpyr_build"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        filtered = DeviceBuffer(total_band * 2)
        nt_buf = DeviceBuffer(n * max_sz * 2)
        filt_buf = DeviceBuffer(n * max_sz * 2)
        for l in range(levels):
            sz = lvl_sizes[l]
            for c in range(3):
                sig_off = level_offsets[l] + c * n * sz
                _evm_cuda.batched_thwc_to_nt_f16(bands.ptr_at_half(sig_off), nt_buf.ptr, n, sz)
                _evm_cuda.batched_iir_bandpass_f16(nt_buf.ptr, filt_buf.ptr, n, sz, MOTION_R1, MOTION_R2)
                dst_off = level_offsets[l] + c * n * sz
                _evm_cuda.batched_nt_to_thwc_scaled_f16(filt_buf.ptr, filtered.ptr_at_half(dst_off), n, sz, alpha_sched[l])
        sync()
        st["C) temporal IIR"] = time.perf_counter() - t0
        del bands, nt_buf, filt_buf

        t0 = time.perf_counter()
        delta = DeviceBuffer(n * 3 * h * w * 2)
        _evm_cuda.batched_lpyr_recon_f16(filtered.ptr, delta.ptr, n, h, w, levels, _d_binom5(), 5)
        sync()
        st["D1) lpyr_recon"] = time.perf_counter() - t0
        del filtered

        t0 = time.perf_counter()
        _evm_cuda.batched_add_planar_quantize_f16(ntsc_f16.ptr, delta.ptr, d_out.ptr, n, h, w, MOTION_CHROM)
        sync()
        st["D2) render"] = time.perf_counter() - t0
        del ntsc_f16, delta
        st["_total"] = sum(v for k, v in st.items() if not k.startswith("_"))
        return st

    return time_gpu_stages(run_once)


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

    run([sys.executable, "scripts/download_samples.py", "face", "baby"])

    sys.path.insert(0, str(Path("cuda").resolve()))
    import numpy as np
    global np

    # --- Run GPU profilers ---
    results = {"gpu": gpu_name, "arch": cuda_arch, "vram": vram, "cpu_ref": CPU_REF}

    print("\n" + "=" * 60)
    print("COLOR PIPELINE — FP32")
    print("=" * 60)
    try:
        s = profile_color_fp32()
        results["color_fp32"] = s
        for k, v in s.items():
            tag = "" if k.startswith("_") else f"  {k:<24s} {v*1000:.1f}ms"
            if k == "_total": tag = f"  {'TOTAL':<24s} {v*1000:.1f}ms"
            print(tag)
    except Exception as e:
        print(f"  FAILED: {e}")
    gc.collect()

    print("\n" + "=" * 60)
    print("COLOR PIPELINE — FP16")
    print("=" * 60)
    try:
        s = profile_color_fp16()
        results["color_fp16"] = s
        for k, v in s.items():
            tag = "" if k.startswith("_") else f"  {k:<24s} {v*1000:.1f}ms"
            if k == "_total": tag = f"  {'TOTAL':<24s} {v*1000:.1f}ms"
            print(tag)
    except Exception as e:
        print(f"  FAILED: {e}")
    gc.collect()

    print("\n" + "=" * 60)
    print("MOTION PIPELINE — FP32")
    print("=" * 60)
    try:
        s = profile_motion_fp32()
        results["motion_fp32"] = s
        for k, v in s.items():
            tag = "" if k.startswith("_") else f"  {k:<24s} {v*1000:.1f}ms"
            if k == "_total": tag = f"  {'TOTAL':<24s} {v*1000:.1f}ms"
            print(tag)
    except Exception as e:
        print(f"  FAILED (likely OOM on <24 GB GPU): {e}")
    gc.collect()

    print("\n" + "=" * 60)
    print("MOTION PIPELINE — FP16")
    print("=" * 60)
    try:
        s = profile_motion_fp16()
        results["motion_fp16"] = s
        for k, v in s.items():
            tag = "" if k.startswith("_") else f"  {k:<24s} {v*1000:.1f}ms"
            if k == "_total": tag = f"  {'TOTAL':<24s} {v*1000:.1f}ms"
            print(tag)
    except Exception as e:
        print(f"  FAILED: {e}")
    gc.collect()

    # --- Render output videos (only those that fit) ---
    print("\n" + "=" * 60)
    print("RENDERING OUTPUT VIDEOS")
    print("=" * 60)
    os.makedirs("output", exist_ok=True)
    from evm_cuda.batched import magnify_color_gdown_ideal, magnify_color_gdown_ideal_fp16
    try:
        magnify_color_gdown_ideal("data/face.mp4", "output/face_fp32.mp4",
            alpha=50, level=4, fl=50/60, fh=60/60, chrom_attenuation=1.0, sampling_rate=30.0)
        print("  face_fp32.mp4 OK")
    except Exception as e:
        print(f"  face_fp32.mp4 FAILED: {e}")
    gc.collect()
    try:
        magnify_color_gdown_ideal_fp16("data/face.mp4", "output/face_fp16.mp4",
            alpha=50, level=4, fl=50/60, fh=60/60, chrom_attenuation=1.0, sampling_rate=30.0)
        print("  face_fp16.mp4 OK")
    except Exception as e:
        print(f"  face_fp16.mp4 FAILED: {e}")
    gc.collect()
    from evm_cuda.batched import magnify_motion_lpyr_iir
    try:
        magnify_motion_lpyr_iir("data/baby.mp4", "output/baby_fp32.mp4",
            alpha=10, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)
        print("  baby_fp32.mp4 OK")
    except Exception as e:
        print(f"  baby_fp32.mp4 FAILED (likely OOM): {e}")
    gc.collect()
    from evm_cuda.batched import magnify_motion_lpyr_iir_fp16
    try:
        magnify_motion_lpyr_iir_fp16("data/baby.mp4", "output/baby_fp16.mp4",
            alpha=10, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)
        print("  baby_fp16.mp4 OK")
    except Exception as e:
        print(f"  baby_fp16.mp4 FAILED: {e}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for pipeline in ("color", "motion"):
        cpu = CPU_REF[pipeline]
        fp32 = results.get(f"{pipeline}_fp32", {})
        fp16 = results.get(f"{pipeline}_fp16", {})
        if not fp32 and not fp16:
            continue
        print(f"\n{pipeline.upper()} — {'Stage':<24s} {'CPU':>9s} {'FP32':>9s} {'FP16':>9s}")
        print("-" * 55)
        for k in cpu:
            if k.startswith("_"): continue
            vc = cpu[k] * 1000
            v32 = fp32.get(k, 0) * 1000 if fp32 else 0
            v16 = fp16.get(k, 0) * 1000 if fp16 else 0
            print(f"  {k:<24s} {vc:>8.1f}ms {v32:>8.1f}ms {v16:>8.1f}ms")
        tc = cpu["_total"] * 1000
        t32 = fp32.get("_total", 0) * 1000 if fp32 else 0
        t16 = fp16.get("_total", 0) * 1000 if fp16 else 0
        print("-" * 55)
        print(f"  {'TOTAL':<24s} {tc:>8.1f}ms {t32:>8.1f}ms {t16:>8.1f}ms")
        if t32 > 0:
            print(f"  Speedup FP32: {tc/t32:.0f}x")
        if t16 > 0:
            print(f"  Speedup FP16: {tc/t16:.0f}x")

    print(f"\nGPU: {gpu_name} (sm_{cuda_arch}, {vram})")

    # List output videos
    outdir = Path("output")
    if outdir.exists():
        print("\nOutput videos:")
        for f in sorted(outdir.glob("*.mp4")):
            print(f"  {f.name}: {f.stat().st_size/1024/1024:.1f} MB")

    with open("gpu_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to gpu_comparison_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
