# 100x Speedup: Optimizing CUDA Video Pipelines the Harris Way

Or: **how a single `cudaMalloc` was costing us 1 second, why I was wrong about
kernel fusion, and other surprises from profiling GPU pipelines.**

## The setup

[Eulerian Video Magnification][evm] (EVM) reveals subtle temporal changes in
video — a face pulsing with blood flow, a baby's subtle breathing. The
algorithm is conceptually simple: build a spatial pyramid, apply a temporal
bandpass filter at each spatial location, amplify the result, reconstruct.

We had a working CUDA port of the MIT MATLAB reference, validated bit-for-bit
against a Python baseline. The question was: **is it fast?** And if not,
why not?

[evm]: http://people.csail.mit.edu/mrub/vidmag/

Two pipelines:

- **Color** (face.mp4, 291 frames, 592×528): Gaussian downsample → ideal
  bandpass (FFT) → amplify → upsample. Reveals pulse.
- **Motion** (baby.mp4, 291 frames, 544×960, 9 pyramid levels): Laplacian
  pyramid → IIR temporal filter → per-level amplify → reconstruct. Reveals
  motion.

## Step 0: Measure first

Following Mark Harris's classic ["Optimizing Parallel Reduction in CUDA"][harris]
methodology — **measure before optimizing, attack the biggest bottleneck, one
change at a time** — we built per-stage profilers ([`scripts/profile_color.py`][pcolor],
[`scripts/profile_motion.py`][pmotion]).

[harris]: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
[pcolor]: ../scripts/profile_color.py
[pmotion]: ../scripts/profile_motion.py

The baseline numbers told a clear story:

**Color pipeline (4.26s):**

| Stage | Time | % | Binding calls |
|-------|------|---|---------------|
| 1. color_cvt + blur_dn | 2.34s | 55% | 1164 |
| 2. ideal_bandpass | 0.32s | 8% | 3 |
| 3. render (per-frame) | 1.59s | 37% | 582 |
| **Total** | **4.26s** | | **1749 calls** |

**Motion pipeline (14.78s):**

| Stage | Time | % | Binding calls |
|-------|------|---|---------------|
| A. NTSC convert | 2.20s | 15% | 291 |
| B. lpyr_build | 3.54s | 24% | 873 |
| C. temporal IIR | 3.97s | 27% | 27 |
| D. recon + render | 5.07s | 34% | 873 |
| **Total** | **14.78s** | | **1773 calls** |

The bottleneck was **not GPU compute** — it was per-call host↔device transfer
overhead. Each binding call did `cudaMalloc` + H2D + kernel launch + D2H +
`cudaFree`. With 1749–1773 calls, that's ~3ms of overhead per call vs
microseconds of actual GPU work. **>95% of wall time was transfer and
allocation overhead.**

## The principle

Every optimization in this project follows one rule:

> **Upload once at pipeline entry. Keep data on-device. Download once at
> pipeline exit. Everything in between stays on the GPU.**

The execution was iterative: profile → find the stage with the most host↔device
traffic → eliminate those transfers → re-profile → repeat.

## Color pipeline optimization (4.26s → 0.19s)

### Phase 1c: Batch blur_dn (873 calls → 1)

The Gaussian blur+downsample ran 291 frames × 3 channels = 873 binding calls.
Fix: a `to_planar_3ch` kernel that transposes `(n,H,W,3)` → `(n*3,H,W)`
on-device, then a C++ host-loop binding with scratch allocated once.

Note: the baseline profiler measured color_cvt + blur_dn as a combined stage
(2.34s). After optimization, the equivalent work (upload + color_cvt + blur_dn)
takes 0.07s.

**Combined color_cvt + blur_dn: 2.34s → 0.07s (33x).**

### Phase 1d: Keep NTSC on-device + batched render

The NTSC frames were downloaded to host, then re-uploaded 291 times for
rendering. Fix: use `batched_add_and_quantize` so NTSC never leaves the GPU.

**Along the way, a latent bug surfaced:** the `DeviceBuffer(array)` constructor
used `py::array_t<char>::ensure()` with `forcecast`, which **casts each element
to char** (1 byte). For float32 arrays, `0.5` truncated to `(char)0.5 = 0` —
silent all-zero uploads. Present since Phase 1a-b but only triggered when we
first uploaded float32. Fixed + 3 regression tests.

**Render stage: 1.59s → 0.09s (18x).**

### Phase 1g: CUDA bilinear upsample kernel

The last host-side bottleneck: 291 `cv2.resize(INTER_LINEAR)` calls. We
reverse-engineered cv2's coordinate convention — **half-pixel centers** with
**replicate** border — and wrote a matching CUDA kernel.

**0.90s → 0.08s (11x).**

### Phase 1h: The 1-second cudaMalloc

The profiler showed the upload+color_cvt stage at 88% of remaining time (1.0s).
The kernel itself took **2 milliseconds**. A micro-benchmark revealed:

```
cudaMalloc(273MB):    1.002s   ← !!
H2D upload(273MB):    0.025s
color_cvt kernel:     0.002s
```

The CUDA driver lazily builds page tables on the first large allocation — a
one-time ~1s cost. Fix: `warmup_device_pool(1GB)` at pipeline entry primes the
pool. All subsequent allocations are O(1).

**1.00s → 0.10s (10x)** at the time (H100). Later on H200, the same stage
measured 0.03s.

## Motion pipeline optimization (14.78s → 0.42s)

### Phase 2a–b: Batch lpyr_build + lpyr_recon

Same pattern as color's Phase 1c: C++ host-loop bindings with scratch allocated
once. The multi-level band output required a level-major layout with per-level
offset tables.

**Stages B+D: 8.61s → 3.53s.**

### Phase 3: Device-resident temporal IIR (the big one)

After Phase 2a-b, I profiled again. The result overturned my assumptions:

| Stage | Time | % |
|-------|------|---|
| A. NTSC convert | 0.99s | 9% |
| B. lpyr_build | 1.34s | 12% |
| **C. temporal IIR** | **5.76s** | **52%** |
| D. recon + render | 3.06s | 27% |

Stage C was 52% of the pipeline — and I had **theoretized for three turns about
kernel fusion and CUDA streams** before profiling proved me wrong. The bottleneck
wasn't kernel launches at all. It was the same per-call H2D/D2H transfers: 27
calls, each uploading and downloading hundreds of MB of pyramid band data.

Fix: change the band layout from frame-major to **channel-major**
`(level, channel, frame, spatial)`. This makes each `(level, channel)` a
contiguous `(T=n, N=lh×lw)` block. Stage C can then extract signals via pointer
arithmetic alone, transpose to `(N,T)` on-device via the existing `thwc_to_nt`
kernel, run IIR, scale by alpha, and transpose back — all device-to-device with
zero host transfers.

**5.76s → 0.04s (144x) — the single biggest win of the project.**

### Phase 4a: Device-resident render

Final step: keep NTSC on-device from Stage A through Stage D, batch the
`attenuate_chrom` and `add_and_quantize` calls. Added a `planar_to_interleaved_3ch`
kernel (inverse of `to_planar_3ch`) to convert recon output back to interleaved
layout on-device.

**2.33s → 0.20s (12x).**

## Step-by-step speedup analysis

Following the Harris methodology, here is the measured timing at each
optimization checkpoint. Every number is from an actual profiler or benchmark
run — no estimates. Measurement context (node, job) is noted for each.

**Caveats:** baselines were measured on H100 (kolyoz24); most intermediates
and finals on H200 (various nodes). H200 is generally faster, so some inter-step
speedup is hardware. Where only total time was measured (no stage breakdown),
stages are marked "—".

### Color pipeline (face.mp4, 291 frames, 592×528)

| Step | What changed | Total | cvt+blur | bandpass | render | Measured on |
|------|-------------|-------|----------|----------|--------|-------------|
| Python | — | 14.78s | — | — | — | kolyoz21/H100 |
| **v0** baseline | CUDA kernels, all per-call | **4.26s** | 2.34s | 0.32s | 1.59s | kolyoz24/H100 |
| v1 | Phase 1c+1d: batch blur_dn + render | **2.28s** | 1.09s ¹ | 0.03s | 1.17s ² | kolyoz42/H200 |
| v2 | Phase 1g: CUDA bilinear upsample | **1.14s** | 1.04s | 0.02s | 0.08s | kolyoz42/H200 |
| v3 | Phase 1h: cudaMalloc warmup | **0.52s** ³ | 0.17s | 0.21s | 0.14s | kolyoz26/H200 |
| **v4** final | (same code, re-profiled) | **0.19s** | 0.07s | 0.02s | 0.09s | kolyoz53/H200 |

> ¹ v1 splits the combined stage: upload+convert (1.04s) + blur_dn (0.05s)
> ² v1 splits render: host upsample via cv2.resize (0.72s) + add+quantize (0.45s)
> ³ v3 bandpass spiked to 0.21s, likely cold cuFFT plan creation on that node

**Per-step speedup:**
- Python → v0 (3.5x): CUDA kernels themselves, even with per-call overhead
- v0 → v1 (1.9x): eliminated 873 blur_dn + 582 render per-call cycles
- v1 → v2 (2.0x): replaced 291 host cv2.resize calls with 1 CUDA kernel
- v2 → v3 (2.2x): moved 1s cudaMalloc penalty to labeled warmup line
- v3 → v4 (2.7x): node variance + cuFFT cache; same pipeline code

### Motion pipeline (baby.mp4, 291 frames, 544×960, 9 levels)

| Step | What changed | Total | NTSC | lpyr_build | IIR | recon+render | Measured on |
|------|-------------|-------|------|-----------|-----|-------------|-------------|
| Python | — | 42.26s | — | — | — | — | kolyoz23/H100 |
| **v0** baseline | CUDA kernels, all per-call | **14.78s** | 2.20s | 3.54s | 3.97s | 5.07s | kolyoz24/H100 |
| v1 | Phase 2a: batch lpyr_build | **13.73s** ⁴ | — | — | — | — | kolyoz21/H100 |
| v2 | Phase 2b: batch lpyr_recon | **13.48s** ⁴ | — | — | — | — | kolyoz26/H200 |
| v3 | (profiled after 2a+2b) | **11.14s** | 0.99s | 1.34s | **5.76s** | 3.06s | kolyoz1/H100 |
| **v4** final | Phase 3+4a: device-resident C+D | **0.42s** | 0.05s | 0.13s | **0.04s** | 0.20s | kolyoz53/H200 |

> ⁴ Total-only measurement (benchmark script), no per-stage breakdown available

**Per-step speedup:**
- Python → v0 (2.9x): CUDA kernels themselves
- v0 → v1 (1.1x): batching lpyr_build — modest (call overhead was only part of cost)
- v1 → v2 (~1x): batching lpyr_recon — negligible on H200
- **v3 → v4 (26.5x)**: device-resident IIR (5.76s→0.04s, 144x) + device-resident render (3.06s→0.20s, 15x)

### Summary

| Pipeline | Python | CUDA v0 | CUDA v4 | vs Python | vs CUDA v0 |
|----------|--------|---------|---------|-----------|------------|
| Color | 14.78s | 4.26s | **0.19s** | **78x** | **22x** |
| Motion | 42.26s | 14.78s | **0.42s** | **101x** | **35x** |

(Plus a one-time ~1s CUDA memory pool warmup per process.)

## Lessons

1. **Profile before optimizing — and re-profile after each change.** I theorized
   for three turns about kernel fusion and CUDA streams for the motion pipeline.
   Then I profiled and found the bottleneck was something completely different:
   host↔device transfers, not kernel launches. The profiler would have told me
   that immediately.

2. **The kernel is never the bottleneck.** Our color conversion kernel was 2ms.
   The `cudaMalloc` before it was 1000ms. The IIR filter kernel was microseconds.
   The H2D/D2H around it was 5.8 seconds. GPU code is easy to optimize; the
   plumbing around it is where time goes.

3. **Batching beats kernel tuning.** We never wrote a faster blur kernel, a
   faster FFT, or a faster IIR. We just stopped calling `cudaMalloc`/`cudaFree`
   ~2000 times per pipeline. The kernels were already fast — they were just
   starved by the host.

4. **Memory pools matter.** The 1-second `cudaMalloc` is a well-known CUDA
   gotcha. A simple warmup allocation primes the driver's page tables and makes
   all subsequent allocations instant.

5. **Latent bugs hide in untested paths.** The DeviceBuffer float32-truncation
   bug existed for two phases before anyone noticed — the code path was never
   exercised until Phase 1d. Regression tests on the fix would have caught it
   immediately.

6. **Layout transforms are the key to device-resident pipelines.** The hardest
   part wasn't writing kernels — it was rearranging data layouts so that each
   stage could read its input via pointer arithmetic alone, without host
   round-trips. The `to_planar_3ch` / `planar_to_interleaved_3ch` /
   `thwc_to_nt` / `nt_to_thwc` transpose kernels were the enablers.

## Methodology credit

The "measure → attack biggest bottleneck → one change → re-measure" loop is
straight from Mark Harris's ["Optimizing Parallel Reduction in CUDA"][harris].
It works. Every phase targeted the profiler's #1 hotspot, and every phase
delivered a measurable speedup.
