#!/usr/bin/env python3
"""Measure FP16 experiment ladder on motion baby (device-resident stages).

Reports stage times + accuracy vs CUDA FP32. Half-acc uses experimental
batched_lpyr_*_f16_halfacc bindings (not production motion path).
"""
from __future__ import annotations
import numpy as np

from evm.cuda.benchmark import run
from evm.cuda import batched, _evm_cuda
from evm.cuda.batched import DeviceBuffer, _d_binom5
from evm.cuda._common import figure6_alpha_schedule, read_frames as _read_frames

def accuracy_vs_fp32(fp16_out, fp32_out):
    d = np.abs(fp32_out.astype(np.float64) - fp16_out.astype(np.float64))
    a = np.clip(np.rint(fp32_out * 255), 0, 255).astype(np.int16)
    b = np.clip(np.rint(fp16_out * 255), 0, 255).astype(np.int16)
    du = np.abs(a - b)
    return {
        "rmse": float(np.sqrt((d**2).mean())),
        "max_abs": float(d.max()),
        "u8_max": int(du.max()),
        "u8_diff_frac": float((du > 0).mean()),
    }

def motion_fp16_halfacc(vid_path: str, **params):
    """Copy of magnify_motion_lpyr_iir_fp16 using half-acc build/recon."""
    frames, fps = _read_frames(vid_path)
    n = len(frames); h, w = frames[0].shape[:2]
    levels = 1
    hh, ww = h, w
    while hh >= 5 and ww >= 5:
        levels += 1; hh = (hh + 1) // 2; ww = (ww + 1) // 2
    alpha_sched = figure6_alpha_schedule(
        levels, params["alpha"], params["lambda_c"], h, w,
        exaggeration_factor=params.get("exaggeration_factor", _evm_cuda.exaggeration_factor))
    level_sizes = []
    ch, cw = h, w
    for _ in range(levels):
        level_sizes.append((ch, cw)); ch = (ch + 1) // 2; cw = (cw + 1) // 2
    clip_u8 = np.stack(frames, axis=0)
    ntsc_n = n * h * w * 3
    planar_n = n * 3 * h * w
    lvl_sizes = [s[0]*s[1] for s in level_sizes]
    total = sum(s * n * 3 for s in lvl_sizes)
    d_clip = DeviceBuffer.from_array(clip_u8)
    d_ntsc = DeviceBuffer(ntsc_n * 2)
    _evm_cuda.batched_bgr_u8_to_ntsc_f16(d_clip.ptr, d_ntsc.ptr, n, h, w)
    d_planar = DeviceBuffer(planar_n * 2)
    _evm_cuda.batched_to_planar_3ch_chan_outer_f16(d_ntsc.ptr, d_planar.ptr, n, h, w)
    d_bands = DeviceBuffer(total * 2)
    _evm_cuda.batched_lpyr_build_f16_halfacc(
        d_planar.ptr, d_bands.ptr, n, h, w, levels, _d_binom5(), 5)
    level_offsets = []
    off = 0
    for sz in lvl_sizes:
        level_offsets.append(off); off += sz * n * 3
    d_filt = DeviceBuffer(total * 2)
    for l in range(levels):
        sz = lvl_sizes[l]; a = float(alpha_sched[l])
        for c in range(3):
            sig = level_offsets[l] + c * n * sz
            _evm_cuda.batched_iir_bandpass_tn_f16(
                d_bands.ptr_at_half(sig), d_filt.ptr_at_half(sig),
                n, sz, params["r1"], params["r2"], a)
    d_delta = DeviceBuffer(n * 3 * h * w * 2)
    _evm_cuda.batched_lpyr_recon_f16_halfacc(
        d_filt.ptr, d_delta.ptr, n, h, w, levels, _d_binom5(), 5)
    d_out = DeviceBuffer(n * h * w * 3)
    _evm_cuda.batched_add_planar_quantize_f16(
        d_ntsc.ptr, d_delta.ptr, d_out.ptr, n, h, w, params.get("chrom_attenuation", 0.1))
    out = d_out.download_u8(n * h * w * 3).reshape(n, h, w, 3)
    return out.astype(np.float32) / 255.0

def main():
    params = dict(vid="data/baby.mp4", alpha=20.0, lambda_c=16.0, r1=0.4, r2=0.05, chrom_attenuation=0.1)
    n = 291
    print("GPU ladder: FP32 | FP16 (fused A + half bands + float-acc) | FP16+halfacc spatial")
    r32 = run("motion", "fp32", params, n_iter=5)
    r16 = run("motion", "fp16", params, n_iter=5)
    print("\n=== FP32 ==="); print(r32)
    print("\n=== FP16 production (fused Stage A, half bands, float-acc spatial) ==="); print(r16)

    fp32 = batched.magnify_motion_lpyr_iir("data/baby.mp4", "", **{k:v for k,v in params.items() if k!="vid"})
    fp16 = batched.magnify_motion_lpyr_iir_fp16("data/baby.mp4", "", **{k:v for k,v in params.items() if k!="vid"})
    acc16 = accuracy_vs_fp32(fp16, fp32)

    print("\n=== FP16 HALFACC spatial experiment (build+recon) ===")
    # time halfacc path roughly via manual stages not full benchmark harness
    import time
    # warm
    _ = motion_fp16_halfacc("data/baby.mp4", **{k:v for k,v in params.items() if k!="vid"})
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        out_ha = motion_fp16_halfacc("data/baby.mp4", **{k:v for k,v in params.items() if k!="vid"})
        # crude total wall including H2D of u8 inside function; not stage-split
        ts.append((time.perf_counter()-t0)*1e3)
    ts.sort()
    ha_ms = ts[len(ts)//2]
    acc_ha = accuracy_vs_fp32(out_ha, fp32)

    print(f"halfacc total wall median (includes decode path inside): {ha_ms:.1f} ms  [not comparable to stage compute]")
    print("Prefer build-level micro for halfacc; e2e accuracy:")
    print(acc_ha)

    print("\n=== SUMMARY (stage harness) ===")
    print(f"fp32  compute={r32.compute_ms:.1f} ms  fps={n/(r32.compute_ms/1000):.0f}")
    print(f"fp16  compute={r16.compute_ms:.1f} ms  fps={n/(r16.compute_ms/1000):.0f}  vs_fp32_compute={r16.compute_ms/r32.compute_ms:.3f}x")
    print(f"fp16 accuracy vs cuda fp32: rmse={acc16['rmse']:.4e} max={acc16['max_abs']:.4e} u8_max={acc16['u8_max']} diff_frac={acc16['u8_diff_frac']:.4f}")
    print(f"halfacc accuracy vs cuda fp32: rmse={acc_ha['rmse']:.4e} max={acc_ha['max_abs']:.4e} u8_max={acc_ha['u8_max']} diff_frac={acc_ha['u8_diff_frac']:.4f}")

    # microbench build only float-acc vs halfacc
    print("\n=== BUILD micro: float-acc f16 vs halfacc f16 ===")
    frames, _ = _read_frames("data/baby.mp4")
    n = len(frames); h, w = frames[0].shape[:2]
    imgs = np.random.default_rng(0).random((n*3, h, w)).astype(np.float16)
    imgs = np.ascontiguousarray(imgs)
    # actual: use real planar half from first frame batch size
    M = n * 3
    levels = 1
    hh, ww = h, w
    while hh >= 5 and ww >= 5:
        levels += 1; hh=(hh+1)//2; ww=(ww+1)//2
    total = 0
    ch, cw = h, w
    for _ in range(levels):
        total += ch*cw*M
        ch=(ch+1)//2; cw=(cw+1)//2
    d_in = DeviceBuffer.from_array(imgs)
    d_out = DeviceBuffer(total * 2)
    def sync():
        DeviceBuffer(4).download_u8(4)
    def time_k(fn, iters=7):
        for _ in range(2):
            fn(); sync()
        ts=[]
        for _ in range(iters):
            sync(); t0=time.perf_counter(); fn(); sync(); ts.append((time.perf_counter()-t0)*1e3)
        ts.sort(); return ts[len(ts)//2]
    t_fa = time_k(lambda: _evm_cuda.batched_lpyr_build_f16(d_in.ptr, d_out.ptr, n, h, w, levels, _d_binom5(), 5))
    t_ha = time_k(lambda: _evm_cuda.batched_lpyr_build_f16_halfacc(d_in.ptr, d_out.ptr, n, h, w, levels, _d_binom5(), 5))
    # accuracy of halfacc build vs float build
    d_ref = DeviceBuffer(total * 4)
    imgs_f32 = imgs.astype(np.float32)
    d_in32 = DeviceBuffer.from_array(imgs_f32)
    _evm_cuda.batched_lpyr_build(d_in32.ptr, d_ref.ptr, n, h, w, levels, _d_binom5(), 5)
    ref = d_ref.download_f32(total)
    d_tmp = DeviceBuffer(total * 4)
    _evm_cuda.batched_lpyr_build_f16(d_in.ptr, d_out.ptr, n, h, w, levels, _d_binom5(), 5)
    _evm_cuda.f16_to_f32(d_out.ptr, d_tmp.ptr, total)
    fa = d_tmp.download_f32(total)
    _evm_cuda.batched_lpyr_build_f16_halfacc(d_in.ptr, d_out.ptr, n, h, w, levels, _d_binom5(), 5)
    _evm_cuda.f16_to_f32(d_out.ptr, d_tmp.ptr, total)
    ha = d_tmp.download_f32(total)
    def stats(a,b):
        d=np.abs(a.astype(np.float64)-b.astype(np.float64))
        return float(np.sqrt((d**2).mean())), float(d.max())
    print(f"build_f16 float-acc: {t_fa:.2f} ms  vs_fp32_build rmse/max={stats(fa,ref)}")
    print(f"build_f16 half-acc:  {t_ha:.2f} ms  vs_fp32_build rmse/max={stats(ha,ref)}")

if __name__ == "__main__":
    main()
