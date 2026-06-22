# Profiler Was Lying: Optimizing CUDA Video Pipelines the Harris Way

Or: **how the profiler itself became the biggest bug, why kernel fusion
didn't help, and what actually moves the needle after the transfers are
already gone.**

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

- **Color** (face.mp4, 291 frames, 528×592): Gaussian downsample → ideal
  bandpass (FFT) → amplify → upsample. Reveals pulse.
- **Motion** (baby.mp4, 291 frames, 960×544, 9 pyramid levels): Laplacian
  pyramid → IIR temporal filter → per-level amplify → reconstruct. Reveals
  motion.

This post covers two optimization phases:

- **Phase 1–4:** Eliminate per-call host↔device transfers (the classic
  "keep data on the GPU" story).
- **Phase 5:** Eliminate kernel launch overhead + cuFFT plan creation
  (the less obvious story of batched spatial kernels and profiler bugs).

## The principle

Every optimization in this project follows one rule:

> **Upload once at pipeline entry. Keep data on-device. Download once at
> pipeline exit. Everything in between stays on the GPU.**

The execution follows Mark Harris's classic ["Optimizing Parallel Reduction
in CUDA"][harris] methodology: **measure before optimizing, attack the
biggest bottleneck, one change at a time, re-profile.**

[harris]: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

---

## Part I: Kill the transfers (4.26s → 0.55s)

### Step 0: Measure first

We built per-stage profilers ([`scripts/profile_color.py`][pcolor],
[`scripts/profile_motion.py`][pmotion]).

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

### The optimizations

Each phase targeted the profiler's #1 hotspot:

1. **Batch blur_dn** (873 calls → 1): A `to_planar_3ch` kernel transposes
   `(n,H,W,3)` → `(n*3,H,W)` on-device, then a C++ host-loop binding with
   scratch allocated once. **2.34s → 0.07s.**

2. **Keep NTSC on-device + batched render**: The NTSC frames were downloaded
   to host, then re-uploaded 291 times for rendering. Fix: use
   `batched_add_and_quantize` so NTSC never leaves the GPU.
   **1.59s → 0.09s.**

3. **CUDA bilinear upsample kernel**: Replaced 291 `cv2.resize(INTER_LINEAR)`
   calls. Reverse-engineered cv2's coordinate convention — **half-pixel
   centers** with **replicate** border — and wrote a matching CUDA kernel.
   **0.90s → 0.08s.**

4. **The 1-second cudaMalloc**: The profiler showed the upload+color_cvt
   stage at 88% of remaining time (1.0s). The kernel itself took **2
   milliseconds**. The CUDA driver lazily builds page tables on the first
   large allocation — a one-time ~1s cost. Fix: `warmup_device_pool(1GB)`
   at pipeline entry. **1.00s → 0.10s.**

5. **Device-resident temporal IIR** (the biggest single win): Changed the
   band layout from frame-major to **channel-major**
   `(level, channel, frame, spatial)`. This makes each `(level, channel)`
   a contiguous `(T, N)` block, so Stage C can extract signals via pointer
   arithmetic alone, transpose on-device, run IIR, scale, and transpose
   back — all device-to-device with zero host transfers.
   **5.76s → 0.04s (144×).**

### Where that left us

After Phase 1–4, the profilers reported Color = 0.19s, Motion = 0.42s.
**Those numbers were wrong.**

---

## Part II: The profiler was lying (0.19s → 0.55s → 0.085s)

### The critical bug

The `batched_*` wrappers are all fire-and-forget: they queue work on stream 0
and return immediately after launching kernels. They contain **zero**
`cudaDeviceSynchronize` calls. Sync only happens implicitly when a *blocking*
`cudaMemcpy` (D2H) executes.

This broke the **motion profiler's per-stage breakdown** completely:

- Stage A (NTSC convert) ends with an async kernel → no sync
- Stage B (lpyr_build) ends with async kernels → no sync
- Stage C (temporal IIR) ends with async kernels → no sync
- Stage D's `download_u8` is the only blocking call → it blocks until ALL
  of A+B+C+D finish

So `perf_counter()` around Stages A, B, C captured **Python host overhead
only** (microseconds), and all the actual GPU compute piled up and got
attributed to Stage D's download. The reported "Stage D = 47%" was inflated;
Stages A/B/C were understated to near-zero.

**Fix:** Added a `device_synchronize()` binding and bracketed every stage
in both profilers with explicit sync.

The real numbers (cold-start, single run, but properly sync'd):

| | Reported (broken) | Real (sync'd) |
|---|---|---|
| Color total | 0.19s | **0.55s** |
| Motion total | 0.42s | **0.56s** |

The profiler had been underreporting by **~3×**. Every "optimization" in
Phase 1–4 was real, but the final numbers were wrong.

### Steady-state measurement

Once the sync was fixed, a second problem surfaced: **cold-start vs
steady-state**. The profiler ran each stage exactly once, so every number
included one-time CUDA driver costs (JIT compile, kernel binary load, context
setup). The `_warmup_gpu_pool` only primed the memory pool, not the kernels.

Fix: rewrite the profilers to run **N=5 timed iterations** with a **warmup
run** excluded from timing, pre-allocate all device buffers before timing,
and report **median + min/max**. Video decode and encode excluded entirely.

The steady-state numbers (median of 5, kolyoz21/H100):

| Stage | Cold (sync'd) | Steady-state | Ratio |
|---|---|---|---|
| **Color total** | 0.554s | **0.085s** | 6.5× |
| color_cvt | 0.053s | 0.0006s | 88× |
| blur_dn | 0.088s | 0.005s | 18× |
| ideal_bandpass | 0.287s | 0.005s | 57× |
| render | 0.120s | 0.070s | 1.7× |
| **Motion total** | 0.559s | **0.184s** | 3.0× |
| NTSC convert | 0.081s | 0.0009s | 90× |
| lpyr_build | 0.207s | 0.022s | 9.4× |
| temporal IIR | 0.061s | 0.041s | 1.5× |
| render | 0.113s | 0.104s | 1.1× |

The color_cvt and NTSC convert stages dropped by **88–90×** once warmed up
— they were almost entirely kernel JIT/binary load cost, not actual compute.
The real kernel time is sub-millisecond. The render stage barely moved
(1.1–1.7×), confirming it's genuine sustained work.

---

## Part III: Kill the launch overhead (Phase 5)

With honest numbers in hand, the per-stage breakdown revealed the next target.

### Phase 5a: cuFFT plan cache (ideal_bandpass 0.287s → 0.005s)

The `batched_ideal_bandpass` wrapper created and destroyed two `cufftPlanMany`
plans per call (3 channels × 2 plans = 6 plans per pipeline run). cuFFT plan
creation involves kernel autotuning — expensive.

Fix: a static cache keyed on `(T, N)` in `bindings.cpp`. Channels 2 and 3
hit the cache after channel 1 warms it.

**ideal_bandpass: 0.287s → 0.005s (57×).** The biggest single-stage win of
Phase 5.

### Phase 5b: Batched spatial kernels (lpyr_build 0.207s → 0.022s)

This was the real prize. The `batched_lpyr_build` wrapper had a host loop
calling `lpyr_build_device` **M=873 times** (291 frames × 3 channels), each
doing ~40 spatial kernel launches = **~35k total launches**. At ~5μs launch
overhead each, that's ~175ms of pure launch overhead — which matched the
measured 207ms almost exactly. **The stage was almost entirely launch
overhead, not kernel compute.**

Fix: added batched variants of `corr_dn_rows`, `corr_dn_cols`, `up_conv_rows`,
`up_conv_cols` that process B slices per launch via the grid z-dimension.
Each thread computes `(x, y, b)` where `b` indexes the batch slice. Per-thread
math is identical to the single-slice kernels — zero tolerance risk.

The channel-major band output layout (irregular per-slice offsets:
`slice_off(m) = (m%3)*n_frames + m/3`) required new scatter/gather kernels to
bridge frame-major scratch buffers and channel-major band storage:
`scatter_subtract`, `scatter`, `gather`, `gather_add`.

Rewrote `batched_lpyr_build` and `batched_lpyr_recon` to iterate levels in
the host (8 iterations) and batch all M slices per spatial kernel launch.
Total launches: **~35k → ~50** (700× reduction).

Scratch memory increased from 3×H×W (~6MB) to 4×M×H×W (~7.3GB for baby.mp4)
— fits comfortably on H200/H100.

| Stage | Before | After | Speedup |
|---|---|---|---|
| lpyr_build | 0.207s | 0.022s | **9.4×** |
| lpyr_recon | 0.096s | 0.016s | **6.0×** |

Same approach applied to `batched_blur_dn_color` (color Stage 2):
**0.088s → 0.005s (18×).**

### Phase 5c: Kernel fusions (correct, but negligible)

Flush with the batched-kernel success, I tried the obvious next step: fuse
adjacent kernels to eliminate intermediate buffers. Four attempts:

1. **Fold `attenuate_chrom` into `add_and_quantize`**: Added a `chrom_att`
   parameter. Eliminates one full-res kernel pass. **Result: 0.122s →
   0.122s. No measurable change.**

2. **Fold per-level alpha scaling into `nt_to_thwc` transpose**: Added a
   scaled transpose variant. Eliminates 27 `scale_inplace` launches.
   **Result: within noise.**

3. **Fuse bilinear upsample + add_and_quantize** (color render): One kernel
   reads filtered signal + NTSC frame, interpolates inline, writes uint8.
   Eliminates the 1.8GB intermediate buffer. **Result: 0.075s → 0.082s.
   Within noise.**

4. **Fuse planar→interleaved into add_and_quantize** (motion render): Reads
   delta directly from planar layout, folding the transpose inline.
   Eliminates the transpose pass + intermediate buffer. **Result: 0.122s →
   0.118s. Within noise.**

**All four fusions are correct (66/66 tests pass, bit-identical output), but
none moved the needle.** Why? The render stage is **memory-bandwidth bound**,
not launch-overhead bound. It reads the full NTSC frame (1.8GB) every call.
Eliminating a transpose or an intermediate buffer saves one read/write of the
*delta* (which is smaller), but the NTSC read dominates regardless.

This was the most important lesson of Phase 5: **once you've eliminated
launch overhead, kernel fusion stops helping.** The next bottleneck is raw
memory bandwidth, and the only way past it is to change the data format
(FP16), change the access pattern (texture cache), or eliminate the data
movement entirely.

### Phase 5d: Warp-level micro-optimizations (small wins, hard lessons)

With launch overhead gone and kernel fusion exhausted, the next tier of
Harris-style techniques is warp-level: register allocation, occupancy hints,
constant memory. We tried two.

**Register hoisting for filter taps** — the batched spatial kernels read the
5-tap binom5 filter from a global-memory pointer inside the convolution loop.
Loading all 5 taps into a local array at kernel entry (with `#pragma unroll`)
forces them into registers, eliminating 5 global reads per output pixel.

| Stage | Before | After | Speedup |
|---|---|---|---|
| blur_dn | 0.0052s | 0.0046s | **−12%** |
| lpyr_build | 0.0222s | 0.0196s | **−12%** |
| lpyr_recon | 0.0164s | 0.0148s | **−10%** |

Real, reproducible gains — consistent across all iterations and multiple runs.
These three stages use the 5-tap filter heaviest, so the effect is exactly
where expected.

**`__launch_bounds__`** — we added occupancy hints to every hot-path kernel,
then measured. The results were a textbook lesson in *why you must profile
every change*:

| Kernel | `__launch_bounds__` | Effect | Kept? |
|---|---|---|---|
| Batched spatial (1024 threads) | `(1024, 2)` | Neutral-to-positive | ✅ Kept |
| IIR (256 threads) | `(256, 8)` | **+21% regression** — register spills! | ❌ Removed |
| Render (256 threads) | `(256, 4)` | ±10% (within noise) | ❌ Removed |
| Transpose/color_cvt | `(256, 4)` / `(1024, 2)` | No measurable effect | ❌ Removed |

The IIR regression was the sharpest lesson. The IIR kernel uses 32 registers
in its sequential per-thread loop. `__launch_bounds__(256, 8)` demands
256 × 8 = 2048 threads per SM, needing 32 × 2048 = 65536 registers — exactly
the SM's maximum. The compiler achieved this by **spilling registers to local
memory** (slow global memory), which made the sequential loop slower. Higher
occupancy achieved via spills is *worse*, not better — a point Harris makes
explicitly in the presentation but is easy to forget when applying the hint
mechanically.

The render kernels showed ±10% run-to-run variance regardless of
`__launch_bounds__`, confirming they're purely memory-bandwidth bound.
Occupancy hints can't help a kernel that's waiting on memory, not compute.

**Net impact: small.** The spatial gains (10-12% on ~20% of GPU time) moved
the total motion pipeline by ~2%. The dominant render stage (57-83%) didn't
budge. This is the fundamental wall: Harris-style compute optimizations
address compute and launch overhead, not memory bandwidth.

---

## The full picture

### Component-by-component speedup vs the CUDA v0 baseline

**Color pipeline (face.mp4, 291 frames, 528×592):**

| Component | CUDA v0 | Now (steady) | Speedup |
|---|---:|---:|---:|
| color_cvt | ~1.0s | 0.0006s | ~1700× |
| blur_dn | ~1.3s | 0.0046s | ~280× |
| D2H + reshape | — | 0.004s | — |
| ideal_bandpass | 0.32s | 0.0047s | 68× |
| upsample + render | 1.59s | 0.067s | 24× |
| **Total** | **4.26s** | **0.081s** | **53×** |

**Motion pipeline (baby.mp4, 291 frames, 960×544, 9 levels):**

| Component | CUDA v0 | Now (steady) | Speedup |
|---|---:|---:|---:|
| NTSC convert | 2.20s | 0.0009s | 2444× |
| lpyr_build | 3.54s | 0.020s | 177× |
| temporal IIR | 3.97s | 0.041s | 97× |
| lpyr_recon | ~2.5s | 0.015s | ~167× |
| render | ~2.6s | 0.104s | ~25× |
| **Total** | **14.78s** | **0.181s** | **82×** |

(Steady-state = median of 5 iterations, warmup run excluded, device buffers
pre-allocated, decode/encode excluded. v0 numbers include decode/encode, so
the real end-to-end speedup is somewhat less.)

### Where the time goes now

The GPU pipeline is now extremely fast (0.081s color, 0.181s motion). But the
**full end-to-end pipeline call** — including video decode and encode — tells
a different story:

| Component | Color (face.mp4) | Motion (baby.mp4) |
|---|---|---|
| GPU pipeline (steady-state, profiled) | 0.081s | 0.181s |
| Full pipeline call (incl. decode + encode) | 2.65s | 2.85s |
| **Implied decode + encode overhead** | **~2.6s** | **~2.7s** |

The cv2.VideoWriter encode (mp4v codec, CPU-side) is roughly **10× slower
than the entire GPU pipeline**. We optimized the GPU to the point where the
video codec — which we didn't touch at all — is now the dominant cost by an
order of magnitude. NVENC (GPU hardware encode) is the clear next target for
end-to-end throughput.

Within the GPU pipeline itself, the render stage dominates (83% color, 58%
motion). It is **memory-bandwidth bound**: reading the full-res NTSC frame
(1.8GB for motion) + writing 455MB uint8 output. Every other stage has been
optimized to near-zero.

The remaining GPU optimization opportunities (documented in
[`HANDOFF.md`](../HANDOFF.md)) are all about reducing that bandwidth:
FP16 NTSC storage (halves the read), texture memory for cached reads, or
eliminating the NTSC intermediate entirely (BGR→NTSC inline in the render
kernel).

---

## Lessons

1. **Profile before optimizing — and verify the profiler itself.** The
   biggest bug in this project wasn't in any kernel — it was in the profiler
   itself. Missing `cudaDeviceSynchronize` made every number 3× too optimistic.
   The fix was one line of code, but it changed every conclusion about where
   to optimize next.

2. **Cold-start ≠ steady-state.** A single profiler run includes one-time
   kernel JIT/binary load costs that inflate every stage. The color_cvt
   kernel takes 0.6ms in steady state but 53ms on first invocation — an 88×
   difference. Always warm up and measure median of N.

3. **Launch overhead is real.** 35,000 kernel launches at 5μs each = 175ms of
   pure overhead. The batched spatial kernels collapsed that to ~50 launches.
   The per-thread math didn't change at all — the win was entirely in the
   dispatch.

4. **Kernel fusion stops helping once you're bandwidth-bound.** Four fusion
   attempts, all correct, all negligible. The render stage reads 1.8GB of
   NTSC data — eliminating a 0.5GB intermediate buffer doesn't register.
   Once launch overhead is gone, the only way forward is changing the data
   format or access pattern.

5. **The kernel is never the bottleneck (until it is).** In Phase 1–4, the
   kernel was never the bottleneck — it was always transfers and allocations.
   In Phase 5, the kernel launch overhead finally became the bottleneck —
   and batched spatial kernels fixed it. Now the kernel's memory bandwidth
   is the bottleneck. Each phase has its own enemy.

6. **Layout transforms are the key to device-resident pipelines.** The
   hardest part wasn't writing kernels — it was rearranging data layouts so
   that each stage could read its input via pointer arithmetic alone. The
   `to_planar_3ch` / `thwc_to_nt` / scatter/gather kernels were the enablers.

7. **The thing you didn't optimize becomes the bottleneck.** We spent two
   phases optimizing the GPU pipeline (0.081s color, 0.181s motion). Then we
   measured the full end-to-end call: 2.7s. The video codec — cv2.VideoWriter
   with mp4v on the CPU — is 10× slower than the entire GPU pipeline. We never
   touched it because it wasn't the bottleneck. Now it is. The lesson: always
   measure the full end-to-end path, not just the part you're optimizing.

8. **Higher occupancy via register spills is worse, not better.**
   `__launch_bounds__(256, 8)` on the IIR kernel demanded 65536 registers
   per SM — exactly the maximum — so the compiler spilled to local memory to
   fit. The result: a 21% *regression*. More concurrent threads that run
   slower is a net loss. The IIR kernel's sequential per-thread loop doesn't
   benefit from occupancy anyway — there's no memory latency to hide when
   the thread is doing pure arithmetic. Removed the hint, recovered the speed.

9. **Register hoisting helps compute-bound stages, not bandwidth-bound ones.**
   Loading the 5-tap filter into registers gave 10-12% gains on the spatial
   kernels (which do real convolution work). The same technique on the render
   kernel would be meaningless — it's waiting on memory, not re-reading
   filter taps. Match the optimization to the bottleneck type.

## Methodology credit

The "measure → attack biggest bottleneck → one change → re-measure" loop is
straight from Mark Harris's ["Optimizing Parallel Reduction in CUDA"][harris].
It works — but only if the profiler is trustworthy. Verify your measurement
tools before trusting their conclusions.
