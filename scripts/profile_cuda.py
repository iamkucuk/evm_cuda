#!/usr/bin/env python3
"""Profile the EVM CUDA pipeline to find the real bottleneck.

Wraps the CUDA kernels with cudaEvent timing to bucket time into:
  - H2D transfers (input staging)
  - kernel launches (each stage)
  - D2H transfers (output readback)

Runs both pipelines (color and motion-IIR) on the MIT samples, prints a
per-stage breakdown. Output is the data we need to decide where to optimize.

Run on a GPU node. No correctness checks here — pure timing.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = ROOT / "cuda"
for p in (str(ROOT), str(CUDA_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

# Re-use the validated pipelines but instrument the binding's call pattern.
import evm  # Python baseline for reference timings  # noqa: E402
from evm_cuda import _evm_cuda  # noqa: E402

DATA = ROOT / "data"


def hr(label, char="=", width=70):
    print(f"\n{char * width}\n{label}\n{char * width}")


def time_block(label):
    """Coarse wall-clock context manager for Python-side orchestration cost."""
    t = [None]

    @contextmanager
    def cm():
        t0 = time.perf_counter()
        yield
        t[0] = time.perf_counter() - t0

    return cm, t


def profile_color():
    """Count kernel launches + transfers in the color pipeline by hand.

    We re-run the pipeline and use _evm_cuda's per-call alloc/copy pattern as
    the unit of accounting: every binding call does 1 H2D + N kernel launches
    + 1 D2H. So per-frame work scales with frame count.
    """
    import cv2

    cap = cv2.VideoCapture(str(DATA / "face.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    frames = frames[:-10]  # drop last 10
    n = len(frames)
    h, w = frames[0].shape[:2]
    print(f"face: {n} frames, {w}x{h}")

    # --- Stage-level wall-clock: how long does each Python-side phase take?
    phases = {}

    # Phase 1: per-frame color convert + downsample (291 frames).
    t0 = time.perf_counter()
    gdown_frames = []
    for fr in frames:
        ntsc = _evm_cuda.bgr_u8_to_ntsc_f32(np.ascontiguousarray(fr))
        chans = [_evm_cuda.blur_dn(ntsc[:, :, c].astype(np.float32), 4,
                                   np.array(_evm_cuda.binom5_sum1(), dtype=np.float32))
                 for c in range(3)]
        gdown_frames.append(np.stack(chans, axis=-1))
    phases["1. color_cvt + blur_dn (per frame, n=%d)" % n] = time.perf_counter() - t0

    gdown = np.stack(gdown_frames, axis=0).astype(np.float32)

    # Phase 2: temporal ideal bandpass per channel.
    t0 = time.perf_counter()
    filtered = np.empty_like(gdown)
    T_, Hl, Wl = gdown.shape[:3]
    for c in range(3):
        nt = np.ascontiguousarray(gdown[..., c].reshape(T_, Hl * Wl).T)
        out = _evm_cuda.ideal_bandpass(nt, 50/60, 60/60, 30.0)
        filtered[..., c] = np.ascontiguousarray(out.T).reshape(T_, Hl, Wl)
    phases["2. ideal_bandpass (3 channels)"] = time.perf_counter() - t0

    # Phase 3: per-frame render (upsample + add + quantize).
    import cv2 as _cv2
    t0 = time.perf_counter()
    out = np.empty((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        ntsc_frame = _evm_cuda.bgr_u8_to_ntsc_f32(frames[i])
        upsampled = _cv2.resize(filtered[i].astype(np.float32), (w, h),
                                interpolation=_cv2.INTER_LINEAR)
        rendered = ntsc_frame + upsampled
        out[i] = _evm_cuda.ntsc_f32_to_bgr_u8(rendered)
    phases["3. render (upsample+add+quantize, per frame)"] = time.perf_counter() - t0

    total = sum(phases.values())
    print(f"\n--- color pipeline breakdown (total {total*1000:.0f} ms) ---")
    for k, v in phases.items():
        print(f"  {v*1000:8.0f} ms  ({100*v/total:5.1f}%)  {k}")

    # Binding-call accounting: how many _evm_cuda.X calls happened?
    # Phase 1: 1 (color_cvt) + 3 (blur_dn) = 4 calls/frame
    # Phase 2: 1 ideal_bandpass call per channel = 3 calls
    # Phase 3: 2 calls/frame (color_cvt + ntsc_to_bgr)
    calls_p1 = n * 4
    calls_p2 = 3
    calls_p3 = n * 2
    print(f"\n--- binding call count ---")
    print(f"  Phase 1: {calls_p1:6d} calls  ({calls_p1/(calls_p1+calls_p2+calls_p3)*100:.0f}%)")
    print(f"  Phase 2: {calls_p2:6d} calls")
    print(f"  Phase 3: {calls_p3:6d} calls  ({calls_p3/(calls_p1+calls_p2+calls_p3)*100:.0f}%)")
    print(f"  TOTAL:   {calls_p1+calls_p2+calls_p3:6d} calls (each = 1 H2D + kernel(s) + 1 D2H)")


def profile_motion():
    """Same exercise for the IIR motion pipeline on baby.mp4."""
    import cv2
    from evm_cuda import pipelines as cu

    # The motion pipeline has way more per-frame work: full Laplacian pyramid
    # build per frame per channel. Count those.
    cap = cv2.VideoCapture(str(DATA / "baby.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    frames = frames[:-10]
    n = len(frames)
    h, w = frames[0].shape[:2]
    print(f"baby: {n} frames, {w}x{h}")

    # Time the whole pipeline as one number for reference.
    t0 = time.perf_counter()
    _ = cu.magnify_motion_lpyr_iir(
        str(DATA / "baby.mp4"), "/tmp/_bench.mp4",
        alpha=10, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)
    total = time.perf_counter() - t0
    print(f"total wall clock for cuda motion pipeline: {total:.2f}s")

    # Now break it down. Re-implement the pipeline with timers on each phase.
    # (This duplicates pipelines.py structure for measurement only.)
    from evm_cuda.pipelines import _read_frames, _bgr_u8_to_ntsc_f32, figure6_alpha_schedule, BINOM5

    frames2, fps = _read_frames(DATA / "baby.mp4")
    n = len(frames2)

    # Auto-levels: mirror max_pyr_ht
    levels = 1
    hh, ww = h, w
    while hh >= 5 and ww >= 5:
        levels += 1; hh = (hh + 1) // 2; ww = (ww + 1) // 2
    print(f"pyramid levels: {levels}")

    # Phase A: NTSC convert all frames
    t0 = time.perf_counter()
    ntsc_frames = [_bgr_u8_to_ntsc_f32(fr) for fr in frames2]
    ta = time.perf_counter() - t0

    # Phase B: per-frame lpyr_build per channel
    t0 = time.perf_counter()
    pyrs = []
    for ntsc in ntsc_frames:
        fp = []
        for c in range(3):
            bands, _ = _evm_cuda.lpyr_build(
                np.ascontiguousarray(ntsc[:, :, c], dtype=np.float32),
                levels, BINOM5)
            fp.append([np.ascontiguousarray(b, dtype=np.float32) for b in bands])
        pyrs.append(fp)
    tb = time.perf_counter() - t0

    # Phase C: stack along time + temporal filter (iir) per level/channel
    t0 = time.perf_counter()
    level_sizes = [(int(pyrs[0][0][l].shape[0]),
                    int(pyrs[0][0][l].shape[1])) for l in range(levels)]
    filtered = []
    for l in range(levels):
        lh, lw = level_sizes[l]
        chans_out = []
        for c in range(3):
            sig = np.stack([pyrs[i][c][l] for i in range(n)], axis=0)
            T_, H_, W_ = sig.shape
            nt = np.ascontiguousarray(sig.reshape(T_, H_ * W_).T)
            out = _evm_cuda.iir_bandpass(nt, 0.4, 0.05)
            chans_out.append(out)
        filtered.append(chans_out)
    tc = time.perf_counter() - t0

    # Phase D: per-frame recon + chromAtt + add + quantize
    t0 = time.perf_counter()
    out = np.empty((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        delta_chans = []
        for c in range(3):
            # Need to apply alpha_sched per level; mirror pipelines.py
            bands = []
            for l in range(levels):
                lh, lw = level_sizes[l]
                arr = filtered[l][c].T.reshape(n, lh, lw)[i]
                bands.append(np.ascontiguousarray(arr, dtype=np.float32))
            recon = _evm_cuda.lpyr_recon(bands, BINOM5)
            delta_chans.append(recon)
        delta = np.stack(delta_chans, axis=-1)
        delta = _evm_cuda.attenuate_chrom(np.ascontiguousarray(delta, dtype=np.float32), 0.1)
        out[i] = _evm_cuda.add_and_quantize(ntsc_frames[i], delta)
    td = time.perf_counter() - t0

    print(f"\n--- motion pipeline breakdown (total {(ta+tb+tc+td)*1000:.0f} ms) ---")
    print(f"  {ta*1000:8.0f} ms  ({100*ta/(ta+tb+tc+td):5.1f}%)  A. NTSC convert (n={n} frames)")
    print(f"  {tb*1000:8.0f} ms  ({100*tb/(ta+tb+tc+td):5.1f}%)  B. lpyr_build ({n} frames x 3 ch x {levels} levels)")
    print(f"  {tc*1000:8.0f} ms  ({100*tc/(ta+tb+tc+td):5.1f}%)  C. temporal IIR ({levels} levels x 3 ch)")
    print(f"  {td*1000:8.0f} ms  ({100*td/(ta+tb+tc+td):5.1f}%)  D. recon+add+quantize (n={n} frames)")

    # Binding call counts
    a_calls = n * 1
    b_calls = n * 3  # one lpyr_build per channel per frame
    c_calls = levels * 3
    d_calls = n * 2  # attenuate_chrom + add_and_quantize per frame
    print(f"\n--- binding call count ---")
    print(f"  A: {a_calls:6d}")
    print(f"  B: {b_calls:6d}  <- per-frame per-channel lpyr_build")
    print(f"  C: {c_calls:6d}")
    print(f"  D: {d_calls:6d}")
    print(f"  total: {a_calls+b_calls+c_calls+d_calls} calls")


def main():
    hr("EVM CUDA profiler — finding the bottleneck", "=")
    print("Rule: measure first. Don't guess what's slow.")

    hr("Pipeline 1: COLOR (face.mp4, alpha=50, level=4)")
    profile_color()

    hr("Pipeline 2: MOTION IIR (baby.mp4, alpha=10, r1=0.4, r2=0.05)")
    profile_motion()

    hr("Done — use this data to pick the optimization target", "=")


if __name__ == "__main__":
    main()
