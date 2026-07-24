#!/usr/bin/env python3
"""Traffic-matched bound analysis without NCU counters.

Classifies stages as:
  - MEMORY-bound (low arithmetic intensity vs roofline ridge)
  - ACCESS-PATTERN / latency-bound (achieved GB/s << peak despite low AI)
  - COMPUTE-bound (high AI, high FLOP utilization) — not expected here
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "cuda")]

from evm_cuda import _evm_cuda  # noqa: E402
from evm_cuda.batched import DeviceBuffer, _d_binom5_sum1  # noqa: E402

# Force device completion via tiny D2H.
_SYNC = DeviceBuffer(4)


def device_sync() -> None:
    _SYNC.download_u8(4)


def time_ms(fn, warmup: int = 2, iters: int = 10) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
        device_sync()
    ts: list[float] = []
    for _ in range(iters):
        device_sync()
        t0 = time.perf_counter()
        fn()
        device_sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


def gbps(nbytes: float, ms: float) -> float:
    return nbytes / (ms / 1e3) / 1e9


def main() -> None:
    peak_bw = 3350.0  # H100 HBM3 GB/s (nominal)
    ridge = 20.0  # FLOP/byte @ 67 TF / 3.35 TB/s

    H, W, n = 960, 544, 291
    levels = 1
    hh, ww = H, W
    while hh >= 5 and ww >= 5:
        levels += 1
        hh = (hh + 1) // 2
        ww = (ww + 1) // 2
    level_hw: list[tuple[int, int]] = []
    ch, cw = H, W
    for _ in range(levels):
        level_hw.append((ch, cw))
        ch = (ch + 1) // 2
        cw = (cw + 1) // 2
    level_sizes = [h * w for h, w in level_hw]
    M = n * 3
    N_fine = H * W
    T = n

    print(f"GPU bindings: iir={hasattr(_evm_cuda, 'batched_iir_bandpass')}")
    print(f"Geometry: {H}x{W}, n={T}, levels={levels}, N_fine={N_fine}")
    print(f"H100 nominal peak BW={peak_bw} GB/s, ridge≈{ridge} FLOP/B")
    print()

    # --- Streaming reference: f32->f16->f32 (known traffic) ---
    print("=== Streaming reference (f32↔f16 round-trip) ===")
    nfloat = N_fine * T  # one channel of finest (N,T)
    d_a = DeviceBuffer(nfloat * 4)
    d_h = DeviceBuffer(nfloat * 2)
    d_b = DeviceBuffer(nfloat * 4)
    d_a.upload(np.random.randn(nfloat).astype(np.float32))
    # traffic: f32_to_f16 R4+W2, f16_to_f32 R2+W4 = 12 B/float = 3 * nfloat * 4
    ref_bytes = 3 * nfloat * 4

    def roundtrip():
        _evm_cuda.f32_to_f16(d_a.ptr, d_h.ptr, nfloat)
        _evm_cuda.f16_to_f32(d_h.ptr, d_b.ptr, nfloat)

    med, lo, hi = time_ms(roundtrip)
    print(
        f"  f32↔f16 roundtrip on {nfloat/1e6:.1f}M floats: "
        f"{med:.3f} ms  eff={gbps(ref_bytes, med):.0f} GB/s "
        f"({100 * gbps(ref_bytes, med) / peak_bw:.1f}% peak)"
    )
    print(f"  (range {lo:.3f}-{hi:.3f}); traffic={ref_bytes/1e9:.3f} GB")
    print()

    # --- Stage C pieces, finest level, 1 channel ---
    print("=== Stage C pieces (finest, 1 channel) ===")
    d_thwc = DeviceBuffer(N_fine * T * 4)
    d_nt = DeviceBuffer(N_fine * T * 4)
    d_out = DeviceBuffer(N_fine * T * 4)
    d_thwc.upload(np.random.randn(N_fine * T).astype(np.float32))

    med, _, _ = time_ms(
        lambda: _evm_cuda.batched_thwc_to_nt(d_thwc.ptr, d_nt.ptr, T, N_fine)
    )
    tr = 2 * N_fine * T * 4
    print(
        f"  thwc_to_nt:        {med:7.3f} ms  traffic={tr/1e9:.3f} GB  "
        f"eff={gbps(tr, med):6.1f} GB/s ({100 * gbps(tr, med) / peak_bw:5.2f}%)"
    )

    med, _, _ = time_ms(
        lambda: _evm_cuda.batched_iir_bandpass(
            d_nt.ptr, d_out.ptr, T, N_fine, 0.4, 0.05
        )
    )
    tr = 2 * N_fine * T * 4
    # ~5 FLOP/sample after t0
    flops = N_fine * (T - 1) * 5
    ai = flops / tr
    print(
        f"  iir_bandpass:      {med:7.3f} ms  traffic={tr/1e9:.3f} GB  "
        f"eff={gbps(tr, med):6.1f} GB/s ({100 * gbps(tr, med) / peak_bw:5.2f}%)  "
        f"AI≈{ai:.3f} FLOP/B"
    )

    med, _, _ = time_ms(
        lambda: _evm_cuda.batched_nt_to_thwc_scaled(
            d_out.ptr, d_thwc.ptr, T, N_fine, 1.0
        )
    )
    tr = 2 * N_fine * T * 4
    print(
        f"  nt_to_thwc_scaled: {med:7.3f} ms  traffic={tr/1e9:.3f} GB  "
        f"eff={gbps(tr, med):6.1f} GB/s ({100 * gbps(tr, med) / peak_bw:5.2f}%)"
    )

    def sandwich():
        _evm_cuda.batched_thwc_to_nt(d_thwc.ptr, d_nt.ptr, T, N_fine)
        _evm_cuda.batched_iir_bandpass(d_nt.ptr, d_out.ptr, T, N_fine, 0.4, 0.05)
        _evm_cuda.batched_nt_to_thwc_scaled(d_out.ptr, d_thwc.ptr, T, N_fine, 1.0)

    med, _, _ = time_ms(sandwich)
    tr = 6 * N_fine * T * 4
    print(
        f"  sandwich (1 ch):   {med:7.3f} ms  traffic={tr/1e9:.3f} GB  "
        f"eff={gbps(tr, med):6.1f} GB/s  => x3ch finest ≈ {med * 3:.2f} ms"
    )
    print()

    # --- Spatial proxy: blur_dn 1 level on M slices ---
    print("=== Spatial (batched blur_dn, 1 level, M=n*3) ===")
    Ho, Wo = (H + 1) // 2, (W + 1) // 2
    d_planar = DeviceBuffer(M * H * W * 4)
    d_g = DeviceBuffer(M * Ho * Wo * 4)
    d_planar.upload(np.random.randn(M * H * W).astype(np.float32))
    filt = _d_binom5_sum1()
    med, _, _ = time_ms(
        lambda: _evm_cuda.batched_blur_dn_color(
            d_planar.ptr, d_g.ptr, M, H, W, 1, filt, 5
        )
    )
    # cols: R MHW + W MH*Wo; rows: R MH*Wo + W M*Ho*Wo; plus D2D copies of input/out
    tr = M * (2 * H * W + 2 * H * Wo + 2 * Ho * Wo) * 4  # upper bound incl copies
    print(
        f"  blur_dn nlevs=1:   {med:7.3f} ms  rough_traffic≤{tr/1e9:.2f} GB  "
        f"eff≥{gbps(M * (H * W + H * Wo + Ho * Wo) * 4, med):.0f} GB/s (kernel-ish)"
    )
    print()

    # --- Full Stage C as in pipeline ---
    print("=== Full Stage C (all levels × 3 ch, pipeline loop) ===")
    total = sum(sz * n * 3 for sz in level_sizes)
    max_sz = max(level_sizes)
    d_bands = DeviceBuffer(total * 4)
    d_filtered = DeviceBuffer(total * 4)
    d_nt2 = DeviceBuffer(n * max_sz * 4)
    d_f2 = DeviceBuffer(n * max_sz * 4)
    d_bands.upload(np.random.randn(total).astype(np.float32))
    level_offsets: list[int] = []
    off = 0
    for sz in level_sizes:
        level_offsets.append(off)
        off += sz * n * 3

    def full_stage_c():
        for l, sz in enumerate(level_sizes):
            for c in range(3):
                sig_off = level_offsets[l] + c * n * sz
                _evm_cuda.batched_thwc_to_nt(
                    d_bands.ptr_at(sig_off), d_nt2.ptr, n, sz
                )
                _evm_cuda.batched_iir_bandpass(
                    d_nt2.ptr, d_f2.ptr, n, sz, 0.4, 0.05
                )
                _evm_cuda.batched_nt_to_thwc_scaled(
                    d_f2.ptr, d_filtered.ptr_at(sig_off), n, sz, 1.0
                )

    med, lo, hi = time_ms(full_stage_c, warmup=1, iters=5)
    tr = sum(3 * 3 * 2 * sz * n * 4 for sz in level_sizes)
    print(
        f"  full Stage C:      {med:7.2f} ms  (range {lo:.2f}-{hi:.2f})  "
        f"traffic={tr/1e9:.2f} GB  eff={gbps(tr, med):.1f} GB/s "
        f"({100 * gbps(tr, med) / peak_bw:.2f}% peak)"
    )

    def full_iir_only():
        for sz in level_sizes:
            for _c in range(3):
                _evm_cuda.batched_iir_bandpass(
                    d_nt2.ptr, d_f2.ptr, n, sz, 0.4, 0.05
                )

    med_i, _, _ = time_ms(full_iir_only, warmup=1, iters=5)
    tr_i = sum(3 * 2 * sz * n * 4 for sz in level_sizes)
    print(
        f"  IIR-only all lvls:  {med_i:7.2f} ms  traffic={tr_i/1e9:.2f} GB  "
        f"eff={gbps(tr_i, med_i):.1f} GB/s"
    )
    tax = med - med_i
    print(
        f"  layout tax ≈ StageC − IIR_only = {tax:.2f} ms "
        f"({100 * tax / med:.0f}% of Stage C wall)"
    )
    print()

    print("=== Bound classification ===")
    print(
        f"  IIR AI ≈ 0.2 FLOP/B << ridge {ridge} → theoretically MEMORY-bound"
    )
    print(
        "  If measured eff << peak (and << streaming ref): "
        "ACCESS-PATTERN / latency / serial-T bound, not HBM-saturated"
    )
    print(
        "  If layout tax large: retrieval/layout-bound around an already serial filter"
    )
    print("Done.")


if __name__ == "__main__":
    main()
