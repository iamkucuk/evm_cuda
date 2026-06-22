# 10x Speedup: Optimizing a CUDA Video Pipeline the Harris Way

Or: **how a single `cudaMalloc` was costing us 1 second, and other surprises
from profiling a GPU pipeline.**

## The setup

[Eulerian Video Magnification][evm] (EVM) reveals subtle temporal changes in
video — a face pulsing with blood flow, a bridge swaying in the wind. The
algorithm is conceptually simple: build a spatial pyramid, apply a temporal
bandpass filter at each spatial location, amplify the result, reconstruct.

We had a working CUDA port of the MIT MATLAB reference, validated bit-for-bit
against a Python baseline (64/64 tests passing). The question was: **is it
fast?** And if not, why not?

[evm]: http://people.csail.mit.edu/mrub/vidmag/

## Step 0: Measure first

Following Mark Harris's classic ["Optimizing Parallel Reduction in CUDA"][harris]
methodology — **measure before optimizing, attack the biggest bottleneck, one
change at a time** — we built a profiler that breaks the color pipeline into
its four stages and counts binding calls.

[harris]: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

The baseline numbers (face.mp4, 291 frames, 592×548, H100):

| Stage | Time | % | Binding calls |
|-------|------|---|---------------|
| 1. color_cvt + blur_dn | 2.34s | 45% | 1164 |
| 2. ideal_bandpass | 0.32s | 6% | 3 |
| 3. render (per-frame) | 1.95s | 38% | 582 |
| 4. (other) | 0.59s | 11% | — |
| **Total** | **5.20s** | | **1749 calls** |

The bottleneck was **not GPU compute** — it was per-call host↔device transfer
overhead. Each binding call did `cudaMalloc` + H2D + kernel launch + D2H +
`cudaFree`. With 1749 calls, that's ~3ms of overhead per call vs microseconds
of actual GPU work. **>95% of wall time was transfer and allocation overhead.**

## The optimization phases

### Phase 1a–b: DeviceBuffer + batched stages

**Problem:** every kernel call crossed PCIe twice (H2D input, D2H output).

**Fix:** `DeviceBuffer` — a RAII wrapper around `cudaMalloc` that keeps data
on-device. Batched wrappers operate on raw device pointers, so the whole clip
stays on the GPU from upload to download.

**Result:** color convert and ideal bandpass stages batched. The remaining
stages still looped per-frame.

### Phase 1c: Batch blur_dn (the 873-call hotspot)

**Problem:** the Gaussian blur+downsample ran 291 frames × 3 channels = 873
binding calls, each doing 5 `cudaMalloc`s + H2D + kernel + D2H + 5 `cudaFree`s.

**Fix:** a `to_planar_3ch` kernel that transposes `(n,H,W,3)` → `(n*3,H,W)`
on-device, then a C++ host-loop binding that iterates over the contiguous
slices with scratch allocated once.

**Result:** 873 calls → 3 calls. Stage time: **2.34s → 0.05s (47x)**.

### Phase 1d: Keep NTSC on-device + batched render

**Problem:** the NTSC color-converted frames were downloaded to host, then
re-uploaded 291 times for the render stage's add-back + quantize.

**Fix:** use the existing `batched_add_and_quantize` kernel so the NTSC buffer
never leaves the GPU. The add-back happens on-device; only the final uint8
output crosses PCIe.

**Along the way, a latent bug surfaced:** the `DeviceBuffer(array)` constructor
used `py::array_t<char>::ensure()` with `forcecast`, which **casts each
element to char** (1 byte). For float32 arrays, every value like `0.5`
truncated to `(char)0.5 = 0` — silent all-zero uploads. The bug had been
present since Phase 1a-b but only manifested when we first uploaded a float32
array (uint8 was unaffected since `char == uint8`). Fixed with a raw-bytes
copy via the buffer protocol. Three regression tests now guard against it.

**Result:** render stage: **1.95s → 0.09s (22x)**.

### Phase 1g: CUDA bilinear upsample kernel

**Problem:** the last host-side bottleneck was 291 `cv2.resize(INTER_LINEAR)`
calls for upsampling the filtered signal back to full resolution.

**Fix:** a CUDA bilinear upsample kernel. First, we reverse-engineered cv2's
exact coordinate convention with a small test script: **half-pixel centers**
(`sx = (x+0.5) * in_W/out_W - 0.5`) with **replicate** (clamp-to-edge)
border handling — not reflect-101. Verified bit-exact on both 2× and odd-ratio
upsamples.

**Result:** 291 host calls → 1 kernel launch. Stage 4: **0.90s → 0.08s (11x)**.

### Phase 1h: The 1-second cudaMalloc

At this point the profiler showed S1 (upload + color convert) at **88% of
remaining time (1.0s)**. The color conversion kernel itself takes **2
milliseconds** — a trivial 3×3 matrix multiply per pixel. So where was the
other 998ms?

A micro-benchmark decomposed S1:

```
np.stack(frames):          0.063s
cudaMalloc(273MB clip):    1.002s   ← !!
H2D upload(273MB):         0.025s
color_cvt kernel:          0.002s
```

The **entire bottleneck was a single `cudaMalloc(273MB)`**. The CUDA driver
lazily builds page tables on the first large allocation — a one-time ~1s cost.
The second `cudaMalloc(1.09GB)` was instant because the driver's memory pool
was already warmed.

**Fix:** one line — call `warmup_device_pool(1GB)` at pipeline entry. This
allocates and immediately frees 1GB, priming the driver's page tables. The
warmup itself costs ~1ms (it's the same page-table setup, just moved out of
the hot path). All subsequent allocations are O(1).

**Result:** S1: **1.00s → 0.10s (10x)**. Pipeline total: **0.52s**.

## The final numbers

| Stage | Baseline | Optimized | Speedup |
|-------|---------|-----------|---------|
| S1 upload + color_cvt | 0.77s | 0.10s | 8x |
| S2 blur_dn (873 calls) | 2.34s | 0.08s | 31x |
| S3 ideal_bandpass | 0.32s | 0.21s | 1.5x |
| S4 render (582 calls) | 1.95s | 0.14s | 14x |
| **Pipeline total** | **5.20s** | **0.52s** | **10x** |

(Plus a one-time 1.0s warmup that primes the CUDA memory pool.)

The actual GPU compute across all stages is **~5ms**. The remaining ~0.5s is
Python orchestration overhead, host-side numpy operations, and the irreducible
PCIe transfers for input upload and output download.

## Lessons

1. **Profile before optimizing.** We expected the temporal FFT (cuFFT) to be
   the bottleneck. It was 0.02s. The real cost was per-call allocation
   overhead — invisible without measurement.

2. **The kernel is never the bottleneck (usually).** Our color conversion
   kernel was 2ms. The `cudaMalloc` before it was 1000ms. GPU code is easy to
   optimize; the plumbing around it is where time goes.

3. **Batching beats kernel tuning.** We never wrote a faster blur kernel or a
   faster FFT. We just stopped calling `cudaMalloc`/`cudaFree` 1749 times. The
   kernels were already fast — they were just starved by the host.

4. **Memory pools matter.** The 1-second `cudaMalloc` is a well-known CUDA
   gotcha. `cudaMallocAsync` (stream-ordered pool allocator) or a simple warmup
   alloc eliminates it.

5. **Latent bugs hide in untested paths.** The DeviceBuffer float32-truncation
   bug existed for two phases before anyone noticed — the code path was never
   exercised until Phase 1d. Regression tests on the fix would have caught it
   immediately.

## Methodology credit

The "measure → attack biggest bottleneck → one change → re-measure" loop is
straight from Mark Harris's ["Optimizing Parallel Reduction in CUDA"][harris].
It works. Every phase targeted the profiler's #1 hotspot, and every phase
delivered a measurable speedup.
