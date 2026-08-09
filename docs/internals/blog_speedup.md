# Implementing Eulerian Video Magnification on CUDA

A study in porting Eulerian Video Magnification (EVM) from Python/NumPy to
CUDA, analyzing the bottlenecks at each level of the GPU memory hierarchy
and the optimization strategies that address them.

> **Note on module paths (added 2026-08-09).** This post names modules as they
> were when the measurements were taken. The 2026-08 library restructure moved
> the GPU package without changing its code, so the benchmark harness named
> `evm_cuda.benchmark` below is imported today as `evm.cuda.benchmark`. Nothing
> else here has been edited.

## The algorithm

[Eulerian Video Magnification][evm] (Wu et al., 2012) reveals subtle
temporal changes in video that are invisible to the naked eye. Blood flow
in a human face. The breathing of a sleeping infant. Unlike Lagrangian
approaches that track features through the frame (optical flow), EVM
treats each pixel as a time series, amplifies the frequencies of
interest, and reconstructs.

[evm]: http://people.csail.mit.edu/mrub/vidmag/

The reference implementation has two pipelines.

**Color magnification** reveals low-frequency color changes (pulse):

```
Input clip (n frames, H×W×3, uint8 BGR)
  → Convert to NTSC color space (YIQ float)
  → Build Gaussian pyramid (downsample ×L levels)
  → Temporal ideal bandpass filter (FFT, per spatial location)
  → Amplify by α (with chrominance attenuation)
  → Upsample back to full resolution
  → Add amplified signal to original NTSC frames
  → Convert back to BGR uint8
Output clip
```

**Motion magnification** reveals larger-scale spatial movement (breathing):

```
Input clip (n frames, H×W×3, uint8 BGR)
  → Convert to NTSC color space (YIQ float)
  → Build Laplacian pyramid (L levels)
  → Per-level temporal IIR bandpass filter
  → Amplify by Figure-6 α schedule (frequency-dependent)
  → Reconstruct pyramid
  → Add amplified delta to original NTSC frames
  → Convert back to BGR uint8
Output clip
```

Both share the same skeleton: spatial decomposition, temporal filtering,
amplification, spatial reconstruction, render. The difference is the
decomposition depth (4 levels for color, 9 for motion) and the temporal
filter type (FFT ideal vs. recursive IIR).

## CUDA implementation architecture

### The DeviceBuffer abstraction

The pipeline is device-resident. Data enters the GPU once as a uint8 input
clip, passes through all stages as on-device float32 buffers, and exits
once as a uint8 output clip. No intermediate host transfers happen within
the pipeline.

This is built on a `DeviceBuffer` class, a thin RAII wrapper around
`cudaMalloc` that exposes a raw device pointer (`uintptr_t`) to the
pybind11 bindings:

```cpp
class DeviceBuffer {
    ptr: *mut c_void  // cudaMalloc'd, auto-freed on drop
    nbytes: usize
}
```

The Python wrapper (`batched.py`) manages buffer lifetimes and passes
`ptr` to C++ kernels. Pointer arithmetic (`ptr_at(float_offset)`)
addresses sub-buffers within a single allocation. For example, extracting
channel `c` at pyramid level `l` from the channel-major band buffer is
just an offset calculation.

### Layout design for temporal filtering

The temporal filter operates on 1D time series. For each spatial location
`(y, x)`, it processes the sequence of values across `n` frames. To make
this efficient on the GPU, the band data is stored in channel-major layout:

```
(level, channel, frame, spatial)
```

This groups each `(level, channel)` pair as a contiguous `(T=n, N=H_l×W_l)`
block, so the temporal filter reads each time series with a simple
stride. Two transpose kernels bridge between the frame-major pipeline
layout and the channel-major band layout:

`thwc_to_nt` converts `(T, H, W, C)` to `(N, T)` by flattening spatial
and gathering along time. `nt_to_thwc_scaled` does the inverse, with an
optional scalar multiply folded in to apply per-level alpha amplification
without a separate kernel launch.

### Kernel inventory

The implementation has 32 CUDA kernels across 10 source files:

| File | Kernels | Purpose |
|------|---------|---------|
| `color_cvt.cu` | 2 | BGR↔NTSC conversion (3×3 matrix multiply per pixel) |
| `spatial.cu` | 8 | Separable 5-tap binomial filter: corr_dn/up_conv, single-slice + batched |
| `transpose.cu` | 4 | Layout transforms: planar↔interleaved, (T,H,W,C)↔(N,T) |
| `iir_bandpass.cu` | 1 | Recursive r1/r2 temporal filter (FP64 state per location) |
| `butter_bandpass.cu` | 1 | 1st-order Butterworth temporal filter |
| `ideal_bandpass.cu` | 3 | cuFFT C2C batched FFT + frequency mask + normalization |
| `lpyr.cu` | 8 | Pyramid build/recon (single-slice) + contiguous band ops |
| `blur_dn.cu` | 1 | Gaussian blur+downsample (calls corr_dn repeatedly) |
| `amplify_render.cu` | 7 | Gain, attenuation, add+quantize, fused upsample+add, fused planar+add |

## Bottleneck analysis

Performance was measured with the `evm_cuda.benchmark` harness
(`scripts/profile_full_comparison.py` invokes it), which brackets every
pipeline stage, including each H2D/D2H transfer, with
`cudaDeviceSynchronize` and reports the median of 5 iterations after one
warmup run. All device buffers are pre-allocated so `cudaMalloc` doesn't
contaminate kernel measurements.

The reference GPU for the measurements below is an **NVIDIA H100 80GB HBM3**
(sm_90). Prior A100 numbers are retained where relevant for comparison.
Numbers are split into **compute** (GPU kernels) and **transfer** (PCIe
H2D/D2H), since on the H100 transfers are a comparable cost to compute and
must be reported separately to be honest about the real pipeline cost.

### Steady-state timings (H100-80GB)

**Color pipeline (face.mp4, 291 frames, 528x592): 9 ms compute / 119 ms total**

| Stage | Time | Kind | What limits it |
|-------|------|------|-----------------|
| H2D: clip | 29.9 ms | transfer | PCIe upload of the input clip. |
| color_cvt | 1.2 ms | compute | Nothing. Trivial per-pixel 3x3 matrix multiply. |
| blur_dn | 5.6 ms | compute | Separable filter, batched across all slices. |
| D2H: gdown | 2.8 ms | transfer | Host round-trip for the per-channel FFT. |
| H2D: sig x3 | 3.9 ms | transfer | Per-channel bandpass input upload. |
| ideal_bandpass | 1.2 ms | compute | cuFFT compute. Plan is cached. |
| D2H: filt x3 | 3.0 ms | transfer | Per-channel bandpass output download. |
| H2D: filt | 4.7 ms | transfer | Gained-filter upload. |
| render | 0.6 ms | compute | See analysis below. |
| **D2H: output** | **65.6 ms** | **transfer** | **PCIe download of the output clip, 55% of total.** |

**Motion pipeline (baby.mp4, 291 frames, 960x544, 9 levels): 35.8 ms compute / 196 ms total (H100 remeasure)**

| Stage | Time | Kind | What limits it |
|-------|------|------|-----------------|
| H2D: clip | 49.9 ms | transfer | PCIe upload of the input clip. |
| NTSC convert | 2.0 ms | compute | Nothing. Trivial. |
| lpyr_build | 20.9 ms | compute | Separable filter, batched. |
| temporal IIR | 46.0 ms | compute | Algorithmic seriality. Each output depends on the previous two. |
| lpyr_recon | 14.2 ms | compute | Separable filter, batched. |
| render | 1.5 ms | compute | See analysis below. |
| **D2H: output** | **27.2 ms** | **transfer** | PCIe download of the output clip. |

**The headline change from earlier measurements:** the previous profiler
lumped the D2H download into the "render" stage, which inflated the H100
"render" to ~208 ms and made it look slower than A100. With transfers
separated, the render *kernel* is 0.6 to 1.5 ms and the dominant non-compute cost
is the output D2H (PCIe-bound). Transfers now dominate color total time (93%)
and are a substantial share of motion (48%).

### The render stage: now a minor cost

After the Level-4 optimization below (multiple-elements-per-thread), the
render *kernel* dropped to 0.6 ms (color) / 1.5 ms (motion) on the H100.
under 1% of compute time. The historical analysis that follows describes the
pre-optimization state (where render was 40-73% of compute and the prime
optimization target); it is retained because it explains the access-pattern
problem and the techniques that solved it.

Each output pixel reads ~12 bytes from the NTSC frame (3 floats), does about
35 FLOPs of math (bilinear interpolation plus NTSC-to-BGR matrix multiply),
and writes 3 bytes. The arithmetic intensity is 2.3 FLOPs per byte, which on
the A100 roofline puts this in the memory-bound regime.

But the kernel achieves only 0.4% of peak memory bandwidth and 0.003%
of peak FP32 throughput. The GPU has the bandwidth and the compute
capacity, but the kernel's data access patterns don't let it use either.

Each thread reads the NTSC frame from raw global memory through the
L2-to-DRAM path, with no software-managed caching. A single global
memory read takes 400 to 600 cycles to return. The thread's 35 FLOPs of
compute finish in about 35 cycles. Without enough independent memory
operations in flight, the SM sits idle for 90%+ of the time, waiting for
data that the memory subsystem could deliver much faster if asked the
right way.

Harris's reduction paper shows the same pattern across seven kernel
versions. His reduction starts at ~1.6 GB/s effective bandwidth (V1:
naive interleaved reads) and reaches ~17 GB/s (V7: unrolled, multiple
elements per thread). Both endpoints run on the same hardware. The 10x
difference comes entirely from how the kernel organizes data access:
coalescing, shared memory staging, warp-level coordination, and giving
each thread enough independent work to keep the memory pipeline full.

### The IIR filter: algorithmic seriality (30% of motion GPU time)

The recursive temporal filter is sequential along the time axis. Each
output sample depends on the previous two. The kernel assigns one thread
per spatial location, and each thread loops over all T frames. This
can't be parallelized across time without changing the algorithm
(block-parallel scan, cyclic reduction).

### The spatial filters: compute bound but fast (11 to 17%)

The separable 5-tap binomial filter operates on the pyramid levels. At
the finest level (960×544), each output pixel needs 5 multiply-adds per
axis times 2 axes = 10 FLOPs. The batched kernels process all n×3 slices
through the grid z-dimension. This stage is genuinely compute bound, but
it's small relative to render.

## Optimization analysis

### Level 1: Host-device transfer elimination

The initial CUDA port wrapped each kernel in a pybind11 binding that did
`cudaMalloc` + H2D + kernel + D2H + `cudaFree` per call. With 291 frames
processed per-frame, that was roughly 1,773 binding calls per pipeline
run, each incurring full transfer overhead. Profiling showed over 95% of
wall time was transfer and allocation, not GPU compute.

The DeviceBuffer pattern fixes this. Data enters the GPU once, stays
on-device through all stages including transposes and pyramid operations,
and exits once. The only remaining host round-trip is in the color
pipeline's Stage 2b, where the Gaussian pyramid gets downloaded to host
for a reshape before the per-channel FFT.

### Level 2: cuFFT plan caching

The ideal bandpass filter creates cuFFT plans (forward + inverse C2C) for
each of the 3 color channels. `cufftPlanMany` does internal autotuning on
each call: kernel selection, workspace sizing. That costs about 5 to 10
ms per plan on A100.

A static cache keyed on `(T, N)` fixes this. Channel 1 warms the cache,
channels 2 and 3 reuse it. The plan survives across pipeline invocations
within the same process, so repeated processing of same-dimension clips
skips plan creation entirely.

### Level 3: Batched spatial kernels

The Laplacian pyramid build processes n×3 = 873 independent image slices
(291 frames times 3 channels), each requiring about 40 kernel launches
across 8 pyramid levels. That's roughly 35,000 launches total. At about
5 μs launch overhead each, you get 175 ms of pure dispatch overhead. That
number matched the measured stage time almost exactly.

The batched spatial kernels add a batch dimension to the grid:

```cuda
// Grid: (ceil(W/32), ceil(Hout/32), B)   Block: (32, 32, 1)
// Each thread computes (x, y, slice). Per-thread math is identical
// to the single-slice kernel.
__global__ void corr_dn_rows_batched_kernel(
    const float* in, float* out,
    int H, int W, const float* filt,
    int stride_in, int stride_out, int B) { ... }
```

The grid z-dimension indexes the batch slice, so all B slices get
processed in a single kernel launch. This collapses roughly 35,000
launches down to about 50 (one per kernel per level), a 700x reduction.

Band storage is channel-major. Production motion now uses channel-outer
planar (`m' = c*n + f`) so band write/add is contiguous (`band_subtract` /
`band_add`) with no offset table. An earlier irregular scatter/gather path
handled frame-major planar; that was superseded in the further mid-pipeline
pass (see `blog_further_optimizations.md`).

### Level 4: Register and thread-level optimizations

Two techniques address the spatial and render kernels at the register and
thread level.

Filter tap register hoisting loads the 5-tap binomial filter into a
local array at kernel entry with `#pragma unroll`, forcing the
coefficients into registers instead of re-reading from global memory
inside the convolution loop:

```cuda
float f[5];
#pragma unroll
for (int k = 0; k < 5; ++k) f[k] = filt[k];
// convolution loop reads f[k] instead of filt[k]
```

This gave 10 to 12% gains on the spatial stages (blur_dn, lpyr_build,
lpyr_recon).

Multiple elements per thread addresses the render stage.
Each thread processes 4 adjacent pixels instead of 1, giving the
compiler 4 independent sets of memory reads to pipeline. The warp
scheduler fills the stall cycles on one read with the compute and memory
operations from the next:

```cuda
#pragma unroll
for (int e = 0; e < 4; ++e) {
    const int x = x0 + e * 32;
    // read NTSC[x], delta[x]: 4 independent read sequences pipelined
    // compute and write output[x]
}
```

This gave 22% gains on both render stages (measured on H100):

| Stage | Before | After |
|-------|--------|-------|
| Color render | 67.0 ms | 52.3 ms (-22%) |
| Motion render | 104.4 ms | 81.9 ms (-22%) |

The same technique was applied to the transpose kernels used in the IIR
stage, where each thread now handles 4 spatial locations.

### Level 5: FP16 storage

The render stage's memory traffic is dominated by reading the NTSC frame
in float32 (12 bytes per pixel). Storing intermediate buffers in half
precision (`__half`, 2 bytes per element instead of 4) halves the memory
traffic on every buffer read and write.

The implementation templates all batched spatial, transpose, IIR, and
render kernels on input/output type. When instantiated with `__half`,
each kernel reads via `__half2float`, computes in FP32, and writes via
`__float2half`. The IIR filter's accumulator stays FP64 regardless of
storage type.

**Motion FP16** stores NTSC, planar NTSC, bands, filtered bands, and
delta all in `__half`. The only FP32 buffers are the momentary NTSC
compute output (freed after one `f32_to_f16` conversion) and the band
output from `lpyr_build` (scatter kernels write float, converted to FP16
before the temporal filter).

| Stage | A100 FP32† | A100 FP16† | P100 FP32† | P100 FP16† | H100 FP32‡ | H100 FP16‡ |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| NTSC convert | 1.6 ms | 5.6 ms | 16.6 ms | 28.5 ms |  |  |
| lpyr_build | 35.3 ms | 37.0 ms | 402.8 ms | 183.4 ms |  |  |
| temporal IIR | 61.4 ms | 61.7 ms | 608.2 ms | 365.7 ms |  |  |
| lpyr_recon | 23.6 ms | 21.4 ms | 107.0 ms | 91.9 ms |  |  |
| render (kernel) | 82.3 ms¹ | 44.7 ms¹ | 8.6 ms | 5.6 ms |  |  |
| **compute total** | **~205 ms** | **~172 ms** | **1,143 ms** | **676 ms** | **35.8 ms** | **34.5 ms** |

† A100/P100 stage breakdown is **historical** (pre true half-band + dense smem
templates). Current P100 **color** is **26.3 / 21.8 ms** compute and current
P100 **motion FP16** is **139.7 ms** (`benches/bench_p100.json`, remeasured on
main); P100 **motion FP32** needs 16.3 GB and does not fit 16 GB.
‡ H100 compute totals postdate the true half-band + dense smem templates
(`benches/bench_h100.json`, per-stage detail in that JSON) but predate the
up_conv work, so like A100 they read a little pessimistic. Only the 3090 and
P100 records were taken after up_conv.

¹ A100/P100 historical render figures are from the pre-transfer-separation
profiler (which bundled the D2H download into "render").

**Color FP16** stores NTSC as `__half` (the dominant persistent buffer).
The Gaussian downsample output goes to FP32 (cuFFT bandpass needs float),
and the `filt` signal (FFT output) stays FP32. Only the NTSC buffer read
by the render kernel is halved.

The result is GPU-dependent. On the A100 (1935 GB/s), FP16 color is 17%
slower because the conversion overhead exceeds the bandwidth savings.
On the P100 (732 GB/s), FP16 color is 13% faster because every byte of
bandwidth matters. The reason is in the render kernel's memory access
pattern:

The color render kernel (`upsample_add_quantize`) reads 15 values per
output pixel: 3 NTSC values + 12 bilinear interpolation taps (4
neighbors x 3 channels) from `filt`. In FP32 that's 60 bytes. Halving
only NTSC saves 6 bytes (10%). The `filt` buffer stays FP32 and accounts
for 80% of the traffic. Compare to the motion render kernel
(`add_planar_quantize`), which reads 6 values per pixel (3 NTSC + 3
delta), so FP16 halves the traffic completely (24 to 12 bytes).

Precision: RMSE between FP32 and FP16 output is 0.0016 for motion, which
is 6.2x under the 0.01 end-to-end tolerance. The maximum per-pixel error
is 3/255 (3 uint8 quantization steps). For color FP16, the uint8 output
differs from FP32 by at most 2 LSB per channel. Motion FP16 peaks at 8.4 GB
on baby.mp4 and motion FP32 at 16.3 GB (measured on RTX 3090, 301 frames at
544x960), so FP16 fits a 16 GB card and FP32 does not. Those peaks used to be
misleading in a warm process: the device pool reuses a block only on an exact
byte-size match and held everything until process exit, so running several
differently sized configs in one process accumulated all of their footprints
and a later config failed even when it fit on its own. `benchmark.run` now
returns the pool's blocks after every config, so each one starts clean.

That residual also corrupted timings, not just capacity. Running all four
configs in one process on the 3090 left 0.03 GB free by the last one, and
motion FP16 then measured **302 ms** of compute instead of ~59 ms: lpyr_build
167 ms instead of 19, IIR 97 instead of 21, recon 29 instead of 14. The likely
cause is the driver placing allocations in host memory once the card is full.
The published multi-GPU numbers were never affected, because they were taken
with a fresh process per config; the point is that the in-process harness
disagreed with that method by 5x on the last config it ran, and now reproduces
it. Median of three in-process runs on 2026-08-02 against the stored
`benches/bench_rtx3090.json`: motion FP32 74.89 vs 75.36 ms (-0.6%), motion
FP16 58.95 vs 60.39 ms (-2.4%), color face FP32 9.54 vs 9.74 ms (-2.1%). Small
and all in one direction, but nothing in the compute path changed, so read it
as agreement rather than a speedup. Color face FP16 is excluded because it is
bimodal on that host: three standalone repeats gave 7.33 / 11.82 / 11.70 ms
against a stored 7.63 ms, and picking one sample would decide the answer.

## Throughput and theoretical limits

### Speedup vs CPU: three scenarios (H100-80GB, current production)

Re-measured on **current** production (H100, 1 warmup + median of 7).
Python CPU baselines unchanged (color 11,194 ms / motion 44,190 ms).

| Scenario | What it measures | Color FP32 | Color FP16 | Motion FP32 | Motion FP16 |
|----------|------------------|-----------:|-----------:|------------:|------------:|
| **CPU baseline** | Python/NumPy compute | 11,194 ms | 11,194 ms | 44,190 ms | 44,190 ms |
| **① Compute only** | GPU kernels (no transfers) | **4.9 ms** | **4.4 ms** | **35.8 ms** | **34.5 ms** |
|  *speedup* | | **~2,290×** | **~2,540×** | **~1,230×** | **~1,280×** |
| **② Compute + H2D** | + input upload from host | **34.6 ms** | **34.2 ms** | **82.1 ms** | **81.0 ms** |
|  *speedup* | | **~320×** | **~330×** | **~540×** | **~550×** |
| **③ Compute + H2D + D2H** | + output download to host | **103.6 ms** | **102.8 ms** | **196.0 ms** | **189.3 ms** |
|  *speedup* | | **~110×** | **~110×** | **~225×** | **~230×** |

Color: `face.mp4`. Motion: `baby.mp4`. Source: `benches/bench_h100.json`.

**① / ② / ③** mean the same as before: compute-only, inference (+H2D), full
file-to-array (+D2H). On this H100 run D2H alone is ~70 to 110 ms and dominates
wall time.

**vs older H100 doc numbers (~85 / 79 ms motion, ~9 ms color):** current is
~2.3 to 2.4x faster on motion and ~1.8 to 2x on color. That gap is **mostly the
mid-pipeline series** (TN IIR, sticky scratch, pool, smem downsample) that was
proven on 3090 and only remeasured on H100 here, not “half-band alone
gave 2× on H100.” FP32 moved almost as much as FP16; half-band / dense smem
adds a smaller FP16 edge (H100 motion 0.96×, color 0.90×).

### Multi-GPU comparison (compute-only, current production)

| GPU | BW | Color FP32 | Color FP16 | Motion FP32 | Motion FP16 |
|-----|-----|-----------|-----------|------------|------------|
| **T4** (16GB, sm_75) | 320 GB/s | **48.9 ms** | **39.7 ms** | does not fit* | **228.8 ms** |
| **P100** (16GB, sm_60) | 732 GB/s | **26.3 ms** | **21.8 ms** | does not fit* | **139.7 ms** |
| **RTX 3090** (24GB, sm_86) | ~936 GB/s | **10.1 ms** | **7.8 ms** | **90.4 ms** | **75.1 ms** |
| **A100** (80GB, sm_80) | 1,935 GB/s | **8.8 ms** | **8.2 ms** | **54.4 ms** | **48.2 ms** |
| **H100** (80GB, sm_90) | 3,350 GB/s | **4.9 ms** | **4.4 ms** | **35.8 ms** | **34.5 ms** |

Sources: `benches/bench_p100.json`, `bench_rtx3090.json`,
`bench_a100.json`, `bench_h100.json` (1 warmup + median of 7). The T4 row is a
single run of `colab/evm_cuda_benchmark.ipynb` on Colab's shared hardware, with
no stored JSON; treat it as indicative, not as a careful measurement.
Motion FP16 peaks at 8.4 GB and fits 16 GB; *motion FP32 peaks at 16.3 GB and
does not. P100 motion FP16 used to fail here as well, from pool residual left
by earlier configs in the same process; `benchmark.run` now releases between
configs. Prefer ≥24 GB for whole-clip motion in FP32.

**vs older multi-GPU rows (same table, pre-remeasure):** A100 motion ~209 →
**54 ms** (~3.8×), color ~72 → **8.8 ms** (~8×); P100 color ~138 → **26.3 ms**
(~5.2×). Those old rows predated the mid-pipeline series **and** true half-band;
again, do not attribute the whole jump to FP16 density alone.

### Measured throughput (current production)

Pixel rates from remeasured times and clip geometry (face 291×592×528;
baby 291×960×544). **1080p@30 ≈ 62.2 Mpx/s.** Stream counts are pixel-scaled
capacity, not a measured multi-stream harness.

At the realistic *inference* tier (②, input upload included, output kept on
the GPU):

| Pipeline (FP16) | GPU | ② time | Gpx/s (②) | ≈ 1080p@30 streams |
|-----------------|-----|-------:|----------:|-------------------:|
| Color face | RTX 3090 | 26.9 ms | ~3.4 | ~55 |
| Motion baby | RTX 3090 | 107.1 ms | ~1.4 | ~23 |

The compute-only ceiling (①, data already on the GPU) is higher:

| Pipeline (FP16) | GPU | ① time | Gpx/s (①) | ≈ 1080p@30 streams |
|-----------------|-----|-------:|----------:|-------------------:|
| Color face | RTX 3090 | 7.8 ms | ~12 | ~190 |
| Color face | A100 | 8.2 ms | ~11 | ~180 |
| Color face | H100 | 4.4 ms | ~21 | ~330 |
| Color face | P100 | 21.8 ms | ~4.3 | ~60 |
| Color face | T4 | 39.7 ms | ~2.3 | ~30 |
| Motion baby | RTX 3090 | 75.1 ms | ~2.0 | ~30 |
| Motion baby | A100 | 48.2 ms | ~3.2 | ~50 |
| Motion baby | H100 | 34.5 ms | ~4.4 | ~70 |
| Motion baby | P100 | 139.7 ms | ~1.1 | ~18 |
| Motion baby | T4 | 228.8 ms | ~0.69 | ~11 |

The full path (③) drops further and is usually PCIe-bound on the download.
Do not read ① stream counts as file-to-file real-time capacity.

Relative to 3090 motion FP16 compute: A100 ~1.6×, H100 ~2.2×.

### Realtime projection (pixel-scale)

| Resolution | 3090 motion ② | 3090 motion ① | 3090 color ② |
|-----------|--------------:|--------------:|-------------:|
| 1080p@30 streams | ~23 | ~30 | ~55 |
| 4K@30 streams | ~6 | ~8 | ~14 |

4K stream counts = 1080p / 4 (pixel-scale).

### Resource utilization: both underutilized

Pre-optimization, the render stage took 40 to 73% of GPU compute yet achieved
only 0.4% of peak memory bandwidth and 0.003% of peak FP32 throughput. Neither
resource was saturated. The Level-4 multiple-elements-per-thread fix (below)
addressed the worst of this, but the underlying access-pattern problem
remains the limiting factor on the bandwidth-bound stages.

Harris's reduction paper shows the same pattern. His seven kernel
versions all run on the same hardware, process the same data, and produce
the same result. The 10x performance difference (1.6 GB/s to 17 GB/s
effective bandwidth) comes from how the kernel organizes its interaction
with the memory subsystem:

- Coalesced access patterns so each warp transaction moves a full cache line
- Shared memory staging so data loaded once serves multiple threads
- Enough independent work per thread to keep the memory pipeline full while
  the warp scheduler overlaps computation with other warps' memory access
- Avoiding bank conflicts in shared memory so all banks serve simultaneously

The current render kernel gets the first item right (reads are coalesced)
but doesn't do the rest. The NTSC frame is read from raw global memory
through the L2-to-DRAM path with no software-managed staging, and each
thread does too little independent work to keep the memory pipeline busy.

## Open optimization surfaces

On the H100 the compute kernel cost is small (~5-36 ms on the current path); the remaining
headroom is split between (a) the transfer cost: the output D2H dominates
color total time and is a large share of motion, and (b) the still-memory-bound
render/IIR access patterns at the architectural level.

**Output D2H / host round-trips.** ~~The color pipeline does four host
round-trips (gdown D2H, per-channel sig/filt H2D/D2H, output D2H) totaling
~110 ms, 93% of its wall clock. The motion pipeline is cleaner (one input
H2D, one output D2H). Eliminating the color bandpass host round-trip (running
cuFFT on-device end-to-end) and keeping the result device-resident would cut
most of this.~~ **DONE (feature/kernel-optimization-A):** the color bandpass
is now fully device-resident. Stage 2b's D2H+reshape plus the surrounding
per-channel H2D/D2H round-trips are replaced by a single unified cuFFT call
over `(N=hl*wl*3, T=n)`. cuFFT filters each row independently, so the
unified batch is numerically identical to 3 per-channel calls. The color
pipeline now has exactly two end-to-end transfers (input H2D, output D2H);
only the unavoidable output D2H remains as a transfer-side cost.

**FP16 `filt` in color render.** The color render kernel reads 12 values
per pixel from `filt` (4 bilinear taps x 3 channels), which stays FP32
because it comes from the FFT output. Converting `filt` to FP16 before
render would halve that traffic. The filt buffer is small (~1 MB for
face.mp4), so the conversion is sub-millisecond. This would address the 80%
of render traffic that FP16 NTSC storage doesn't touch. *Reverted (final)
after measurement on two architectures:* on Tesla P100 (sm_60) the FP16-filt
render regressed from 4.8 ms to 10.8 ms (+125%); on H100 (sm_90) it was
neutral (0.6 to 0.7 ms, within noise, no measurable benefit). Root cause:
the 12 per-pixel `__half→float` conversions in the bilinear tap loop are
scalar and strided, so they cannot vectorize into `__half2` SIMD loads.
On sm_60, scalar `__half` operations run at ½ FP32 rate (only packed
`__half2` gets the 2× advantage), so the conversion overhead dominates the
bandwidth saving on the small (~5 MB) filt buffer. On sm_90, the
conversions are full-rate native instructions, so they're nearly free, but
then there's no win either, because the buffer is too small for the
bandwidth saving to register at 3350 GB/s. The blog's "P100 FP16 = 13%
faster" prediction held for FP16 *NTSC* storage (large contiguous buffer,
SIMD-friendly) but did not extrapolate to filt (small buffer, strided
access). The templated `FILT_T` infrastructure is preserved in the kernel
(defaults to `float`); the FP16 color pipeline keeps reading FP32 filt.

Texture hardware (Harris texture path). `cudaTextureObject_t` with
`cudaReadModeElementType` provides hardware-managed L1 texture cache with
spatial prefetch. The texture unit automatically fetches neighboring cache
lines, so adjacent threads benefit from each other's reads without
explicit shared memory management. For the color pipeline's bilinear
upsample, `tex2D` with linear filtering replaces the manual 4-tap
interpolation entirely: the hardware does it in one instruction with its
own cache.

NVENC/NVDEC. Video encode (~1.6s) is 10x slower than the GPU pipeline on
both color and motion. Hardware video codecs would address this end-to-end
bottleneck.

CUDA streams. All kernels use stream 0. Independent stages (e.g., IIR
filter across levels/channels) could overlap on multiple streams.
*Investigated and deferred:* the per-level IIR at the finest pyramid level
(sz ≈ H·W ≈ 500k threads) already saturates the H100's ~270k resident
threads, so multi-streaming yields ~0% gain where most of the 46 ms IIR cost
lives. Only the coarser (smaller) levels have spare SM capacity, and the
absolute time saved there is small, estimated 1 to 4% end-to-end motion
speedup. Not worth the binding/API churn (4 new bindings + 6 signature
changes + per-stream scratch) for that payoff.

For the IIR stage, an isolation-probe study (2026-07) showed the dominant cost
was **not** FP64 math or peak HBM saturation but **warp-uncoalesced `(N,T)`
access plus a redundant transpose sandwich** around a filter whose bands were
already `(T,N)`. Production now runs coalesced `iir_bandpass_tn` in place
(no `thwc_to_nt` / `nt_to_thwc` in Stage C). Further mid-pipeline work.
sticky lpyr scratch, free-list `DeviceMemPool`, channel-outer bands, smem
fused *downsample* (fused *upsample* regressed and was reverted), took
RTX 3090 motion FP32 compute from **~934 ms → ~100 ms** (~9×). Full
measurement-driven writeup:
[`docs/blog_further_optimizations.md`](blog_further_optimizations.md).
Full writeup of that arc:
[`docs/blog_further_optimizations.md`](blog_further_optimizations.md).

## Methodology

The optimization approach follows Harris's ["Optimizing Parallel Reduction
in CUDA"][harris]. The presentation's contribution is a specific
progression of memory-access optimizations: interleaved to sequential
addressing, loading data into shared memory, unrolling the final warp,
having each thread process multiple elements. Each version makes memory
access faster, not the math cheaper.

An underutilized GPU is usually suffering from poor data access patterns,
not from a lack of hardware capability. The reduction kernel starts at
1.6 GB/s and reaches 17 GB/s through seven versions that each improve how
the kernel interacts with the memory subsystem. The arithmetic doesn't
change. The data access does.

This project applied the same principle at three levels. At the pipeline
level: eliminating host-device transfers, batching kernel launches,
caching cuFFT plans. At the kernel level: register hoisting for filter
taps, multiple-elements-per-thread for the render stage (22%
improvement), and FP16 storage to halve memory traffic. At the system
level: profiling across two GPU generations (P100 and A100) to find that
the optimal precision choice depends on the hardware's memory bandwidth.
The render kernel's 0.4% bandwidth utilization indicates the same class
of problem Harris describes: the hardware can deliver far more data and
compute far more FLOPs than the kernel currently asks of it. The FP16
`filt` conversion, texture hardware, and NVENC optimizations in the open
surfaces section are the next steps along that roadmap.

The profilers run 5 timed iterations with a warmup run (to exclude kernel
JIT and binary load costs), pre-allocate all device buffers (to exclude
`cudaMalloc` from kernel measurements), and report median plus min/max per
stage. Video decode and encode are excluded to isolate GPU pipeline
performance. CPU stages use the same boundary definitions, profiling
`evm.magnify_*` with `perf_counter` around each algorithmic stage.

Measurements were taken on:
- **A100-SXM4-80GB** (sm_80): 1,935 GB/s bandwidth
- **Tesla P100-PCIE-16GB** (sm_60, Kaggle): 732 GB/s bandwidth

[harris]: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

92 unit and integration tests validate correctness against the Python
baseline (RMSE < 0.01 for end-to-end pipelines, per-kernel tolerances
from 10^-6 to 10^-4 depending on the operation). The full test suite and
profiler scripts are in the [repository][repo].

[repo]: https://github.com/iamkucuk/eulerian-video-magnification-cuda
