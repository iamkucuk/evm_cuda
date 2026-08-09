#!/usr/bin/env python3
"""Prove (or disprove) FP16 cost on fused corr_dn and full lpyr_build.

Honest configs:
  f32            float storage + float MAC + float smem
  f16            half storage + float MAC + half smem  (production after fix)
  f16_halfacc    half storage + scalar half MAC        (NOT 2x Ampere path)
  f16_half2      half storage + packed __hfma2         (real 2x half datapath)

Optional: --ncu runs ncu on one kernel launch per mode (preferred when available).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Until the editable install lands (plan steps 1.10/1.12), point at src/.
sys.path.insert(0, str(ROOT / "src"))

from evm.cuda import _evm_cuda  # noqa: E402
from evm.cuda.batched import DeviceBuffer, _d_binom5  # noqa: E402


def sync():
    DeviceBuffer(4).download_u8(4)


def med_ms(fn, warmup=5, iters=21):
    for _ in range(warmup):
        fn()
        sync()
    ts = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=544)
    ap.add_argument("--W", type=int, default=960)
    ap.add_argument("--B", type=int, default=873)  # 291 frames * 3
    ap.add_argument("--ncu", action="store_true", help="profile one launch per mode with ncu")
    args = ap.parse_args()

    H, W, B = args.H, args.W, args.B
    Ho, Wo = (H + 1) // 2, (W + 1) // 2
    in_n = B * H * W
    out_n = B * Ho * Wo
    filt = _d_binom5()

    print("=== geometry ===")
    print(f"in  {B} x {H}x{W}  elems={in_n}  f32={in_n*4/1e6:.1f}MB  f16={in_n*2/1e6:.1f}MB")
    print(f"out {B} x {Ho}x{Wo} elems={out_n}")
    print("unique-ish R+W traffic lower bound for one corr_dn:")
    print(f"  f32: {(in_n*4 + out_n*4)/1e9:.3f} GB")
    print(f"  f16: {(in_n*2 + out_n*2)/1e9:.3f} GB")
    print("  (smem tile reuse means actual DRAM can be less than full R of input)")

    rng = np.random.default_rng(0)
    f32 = np.ascontiguousarray(rng.random((B, H, W), dtype=np.float32))
    f16 = np.ascontiguousarray(f32.astype(np.float16))

    d_f32_in = DeviceBuffer.from_array(f32)
    d_f32_out = DeviceBuffer(out_n * 4)
    d_f16_in = DeviceBuffer.from_array(f16)
    d_f16_out = DeviceBuffer(out_n * 2)

    modes = [
        ("f32", lambda: _evm_cuda.micro_corr_dn_fused(
            d_f32_in.ptr, d_f32_out.ptr, H, W, B, filt, 5, "f32")),
        ("f16", lambda: _evm_cuda.micro_corr_dn_fused(
            d_f16_in.ptr, d_f16_out.ptr, H, W, B, filt, 5, "f16")),
        ("f16_halfacc", lambda: _evm_cuda.micro_corr_dn_fused(
            d_f16_in.ptr, d_f16_out.ptr, H, W, B, filt, 5, "f16_halfacc")),
        ("f16_half2", lambda: _evm_cuda.micro_corr_dn_fused(
            d_f16_in.ptr, d_f16_out.ptr, H, W, B, filt, 5, "f16_half2")),
    ]

    print("\n=== single-level fused corr_dn wall time ===")
    results = {}
    for name, fn in modes:
        t = med_ms(fn)
        results[name] = t
        print(f"  {name:12s}  {t:7.2f} ms")

    base = results["f32"]
    print("\nratios vs f32 ( <1 = faster ):")
    for name, t in results.items():
        print(f"  {name:12s}  {t/base:.3f}x")

    # accuracy vs f32 host ref for half paths on one slice
    print("\n=== accuracy of half kernels vs f32 kernel (one slice) ===")
    _evm_cuda.micro_corr_dn_fused(d_f32_in.ptr, d_f32_out.ptr, H, W, B, filt, 5, "f32")
    ref = d_f32_out.download_f32(out_n).reshape(B, Ho, Wo)[0]
    for mode in ("f16", "f16_halfacc", "f16_half2"):
        _evm_cuda.micro_corr_dn_fused(d_f16_in.ptr, d_f16_out.ptr, H, W, B, filt, 5, mode)
        raw = d_f16_out.download_u8(out_n * 2)
        got = np.frombuffer(raw, dtype=np.float16).reshape(B, Ho, Wo)[0].astype(np.float32)
        d = np.abs(got - ref)
        print(f"  {mode:12s}  rmse={np.sqrt((d*d).mean()):.4e}  max={d.max():.4e}")

    # full pyramid build
    levels = 1
    hh, ww = H, W
    while hh >= 5 and ww >= 5:
        levels += 1
        hh = (hh + 1) // 2
        ww = (ww + 1) // 2
    n_frames = B // 3
    total = 0
    ch, cw = H, W
    for _ in range(levels):
        total += ch * cw * B
        ch = (ch + 1) // 2
        cw = (cw + 1) // 2
    d_bands_f32 = DeviceBuffer(total * 4)
    d_bands_f16 = DeviceBuffer(total * 2)

    print(f"\n=== full lpyr_build ({levels} levels, n_frames={n_frames}) ===")
    t_b32 = med_ms(lambda: _evm_cuda.batched_lpyr_build(
        d_f32_in.ptr, d_bands_f32.ptr, n_frames, H, W, levels, filt, 5))
    t_b16 = med_ms(lambda: _evm_cuda.batched_lpyr_build_f16(
        d_f16_in.ptr, d_bands_f16.ptr, n_frames, H, W, levels, filt, 5))
    t_bha = med_ms(lambda: _evm_cuda.batched_lpyr_build_f16_halfacc(
        d_f16_in.ptr, d_bands_f16.ptr, n_frames, H, W, levels, filt, 5))
    t_bh2 = med_ms(lambda: _evm_cuda.batched_lpyr_build_f16_half2(
        d_f16_in.ptr, d_bands_f16.ptr, n_frames, H, W, levels, filt, 5))
    print(f"  f32              {t_b32:7.2f} ms")
    print(f"  f16 float-acc    {t_b16:7.2f} ms  ratio={t_b16/t_b32:.3f}")
    print(f"  f16 scalar-half  {t_bha:7.2f} ms  ratio={t_bha/t_b32:.3f}")
    print(f"  f16 half2 dn     {t_bh2:7.2f} ms  ratio={t_bh2/t_b32:.3f}")

    print("\n=== interpretation keys ===")
    print("If f16 (half smem + float MAC) << f32: denser DRAM / smem was the win.")
    print("If f16_half2 << f16: packed half math is the missing lever.")
    print("If all half variants ~ f32: bottleneck is not storage bit-width")
    print("(access pattern / multi-pass / launch packaging dominate).")

    if args.ncu:
        print("\n=== NCU (one kernel, first mode that works) ===")
        # Write tiny C++? Better: use ncu --target-processes all python -c ...
        for mode in ("f32", "f16", "f16_half2"):
            cmd = [
                "ncu",
                "--set", "full",
                "--kernel-name-base", "demangled",
                "--launch-count", "1",
                "--target-processes", "all",
                sys.executable, "-c",
                (
                    "import sys; sys.path[:0]=%r; "
                    "from evm.cuda import _evm_cuda; from evm.cuda.batched import DeviceBuffer,_d_binom5; "
                    "import numpy as np; "
                    "H,W,B=%d,%d,%d; Ho,Wo=(H+1)//2,(W+1)//2; "
                    "rng=np.random.default_rng(0); "
                    "x=np.ascontiguousarray(rng.random((B,H,W),dtype=np.float32)); "
                    "filt=_d_binom5(); "
                    "if %r=='f32':\n"
                    " d_in=DeviceBuffer.from_array(x); d_out=DeviceBuffer(B*Ho*Wo*4); "
                    " _evm_cuda.micro_corr_dn_fused(d_in.ptr,d_out.ptr,H,W,B,filt,5,'f32')\n"
                    "else:\n"
                    " d_in=DeviceBuffer.from_array(x.astype(np.float16)); d_out=DeviceBuffer(B*Ho*Wo*2); "
                    " _evm_cuda.micro_corr_dn_fused(d_in.ptr,d_out.ptr,H,W,B,filt,5,%r)\n"
                ) % ([str(ROOT / "src")], H, W, B, mode, mode),
            ]
            print("\n$ " + " ".join(cmd[:6]) + f" ... mode={mode}")
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                text = (out.stdout or "") + (out.stderr or "")
                # keep the metrics lines that matter
                keys = (
                    "Duration",
                    "Memory Throughput",
                    "DRAM Throughput",
                    "Compute (SM) Throughput",
                    "L1/TEX",
                    "L2",
                    "Achieved Occupancy",
                    "Warp Cycles Per Issued Instruction",
                    "Stall",
                    "Sectors",
                    "corr_dn",
                )
                for line in text.splitlines():
                    if any(k in line for k in keys):
                        print(line)
                if out.returncode != 0 and "ERR_NVGPUCTRPERM" in text:
                    print("NCU counters blocked (ERR_NVGPUCTRPERM). Need admin / GPU toolkit perms.")
                    break
            except FileNotFoundError:
                print("ncu not on PATH")
                break
            except subprocess.TimeoutExpired:
                print("ncu timeout")
                break


if __name__ == "__main__":
    main()
