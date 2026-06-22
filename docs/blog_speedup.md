# Implementing Eulerian Video Magnification on CUDA

An investigation into porting the Eulerian Video Magnification (EVM)
algorithm from Python/NumPy to CUDA, analyzing the bottlenecks at each
level of the GPU memory hierarchy — from PCIe transfers down to register
allocation — and the optimization strategies that address them.

## The algorithm

[Eulerian Video Magnification][evm] (Wu et al., 2012) reveals subtle
temporal variations in video that are invisible to the naked eye — blood
flow in a human face, the breathing of a sleeping infant. Unlike
Lagrangian approaches (optical flow tracking), EVM operates in the
Eulerian frame: it treats each pixel as a time series, amplifies the
frequencies of interest, and reconstructs.

[evm]: http://people.csail.mit.edu/mrub/vidmag/

Two pipelines exist in the reference implementation:

### Color magnification

Reveals low-frequency color changes (blood flow pulse):

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

### Motion magnification

Reveals larger-scale spatial movement (breathing):

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

Both pipelines share the same structure: **spatial decomposition →
temporal filtering → amplification → spatial reconstruction → render.**
The key difference is the spatial decomposition depth (4 levels for
color, 9 for motion) and the temporal filter type (FFT ideal vs. IIR).

## CUDA implementation architecture

### The DeviceBuffer abstraction

The fundamental design decision is a **device-resident pipeline**: data
enters the GPU once as a uint8 input clip, passes through all stages as
on-device float32 buffers, and exits once as a uint8 output clip. No
intermediate host↔device transfers occur within the pipeline.

This is implemented via a `DeviceBuffer` class — a thin RAII wrapper
around `cudaMalloc` that exposes a raw device pointer (`uintptr_t`) to
the pybind11 bindings:

```cpp
class DeviceBuffer {
    ptr: *mut c_void  // cudaMalloc'd, auto-freed on drop
    nbytes: usize
}
```

The Python wrapper (`batched.py`) manages buffer lifetimes and passes
`ptr` to C++ kernels. Pointer arithmetic (`ptr_at(float_offset)`)
addresses sub-buffers within a single allocation — e.g., extracting
channel `c` at pyramid level `l` from the channel-major band buffer.

### Layout design for temporal filtering

The temporal filter operates on 1D time series — for each spatial
location `(y, x)`, the sequence of values across `n` frames. To make
this efficient on the GPU, the band data is stored in **channel-major
layout**:

```
(level, channel, frame, spatial)
```

This groups each `(level, channel)` pair as a contiguous `(T=n, N=H_l×W_l)`
block, enabling the temporal filter to read each time series via a
simple stride. Two transpose kernels bridge between the frame-major
pipeline layout and the channel-major band layout:

- `thwc_to_nt`: `(T, H, W, C)` → `(N, T)` — flatten spatial, gather time
- `nt_to_thwc_scaled`: `(N, T)` → `(T, H, W, C)` — scatter time, with
  optional per-call scalar multiply (folds alpha amplification into the
  transpose)

### Kernel inventory

The implementation spans 32 CUDA kernels across 9 source files:

| File | Kernels | Purpose |
|------|---------|---------|
| `color_cvt.cu` | 2 | BGR↔NTSC conversion (3×3 matrix multiply per pixel) |
| `spatial.cu` | 8 | Separable 5-tap binomial filter: corr_dn/up_conv, single-slice + batched |
| `transpose.cu` | 4 | Layout transforms: planar↔interleaved, (T,H,W,C)↔(N,T) |
| `iir_bandpass.cu` | 1 | Recursive r1/r2 temporal filter (FP64 state per location) |
| `butter_bandpass.cu` | 1 | 1st-order Butterworth temporal filter |
| `ideal_bandpass.cu` | 3 | cuFFT C2C batched FFT + frequency mask + normalization |
| `lpyr.cu` | 8 | Pyramid build/recon (single-slice) + scatter/gather (batched) |
| `blur_dn.cu` | 1 | Gaussian blur+downsample (calls corr_dn repeatedly) |
| `amplify_render.cu` | 7 | Gain, attenuation, add+quantize, fused upsample+add, fused planar+add |

## Bottleneck analysis

Performance was measured using stage-by-stage profilers
(`scripts/profile_color.py`, `scripts/profile_motion.py`) that bracket
each pipeline stage with `cudaDeviceSynchronize` and report median of 5
iterations (with a warmup run, all device buffers pre-allocated).
Measurements were taken on NVIDIA H100 (kolyoz21, TRUBA HPC).

### Steady-state timings

**Color pipeline (face.mp4, 291 frames, 528×592) — 0.081s:**

| Stage | Time | Share | Bottleneck type |
|-------|------|-------|-----------------|
| color_cvt | 0.6 ms | 0.7% | Compute (trivial — per-pixel 3×3 matrix) |
| blur_dn | 4.6 ms | 5.7% | Compute (separable filter, batched) |
| D2H + reshape | 4.0 ms | 5.0% | PCIe bandwidth + host transform |
| ideal_bandpass | 4.7 ms | 5.8% | cuFFT compute (plan cached) |
| **upsample + render** | **67.0 ms** | **82.8%** | **GPU memory bandwidth** |

**Motion pipeline (baby.mp4, 291 frames, 960×544, 9 levels) — 0.181s:**

| Stage | Time | Share | Bottleneck type |
|-------|------|-------|-----------------|
| NTSC convert | 0.9 ms | 0.5% | Compute (trivial) |
| lpyr_build | 19.6 ms | 10.8% | Compute (separable filter, batched) |
| temporal IIR | 40.8 ms | 22.6% | Algorithmic seriality |
| lpyr_recon | 14.8 ms | 8.2% | Compute (separable filter, batched) |
| **render** | **104.4 ms** | **57.9%** | **GPU memory bandwidth** |

### Three bottleneck regimes

The profiler reveals three distinct performance-limiting mechanisms:

**1. GPU memory bandwidth (render stage, 58–83%)**

The render kernels read the full-resolution NTSC frame (float32,
n×H×W×3 = 1.8 GB for motion) and write the uint8 output. At H100's
~3 TB/s memory bandwidth, reading 1.8 GB takes ~0.6 ms in theory — but
the kernel also performs bilinear interpolation and NTSC→BGR conversion
per pixel, and the actual measured time is 104 ms. The kernel is
saturated by memory traffic, not compute.

Kernel fusion experiments confirm this: merging the bilinear upsample +
add + quantize into a single kernel (eliminating a 1.8 GB intermediate
buffer and one kernel launch) produced no measurable improvement — the
dominant NTSC frame read overwhelms any savings on the smaller delta
buffer.

**2. Algorithmic seriality (IIR filter, 23%)**

The IIR temporal filter is inherently sequential along the time axis —
each output sample depends on the previous two. The kernel assigns one
thread per spatial location, and each thread loops over all T frames.
This cannot be parallelized across time without changing the algorithm
(e.g., block-parallel scan). It's the only stage where the bottleneck
is the algorithm itself, not the hardware.

**3. Compute (spatial filters, 11–19%)**

The separable 5-tap binomial filter (corr_dn, up_conv) operates on the
pyramid levels. At the finest level (960×544), each output pixel
requires 5 multiply-adds per axis × 2 axes = 10 FLOPs. The batched
kernels process all n×3 slices via the grid z-dimension, achieving high
throughput. This stage is compute-bound but fast relative to the render
stage.

## Optimization analysis

### Level 1: Host-device transfer elimination

The initial CUDA port wrapped each kernel in a pybind11 binding that
performed `cudaMalloc` + H2D + kernel + D2H + `cudaFree` per call. With
291 frames processed per-frame, this resulted in ~1,773 binding calls per
pipeline run — each incurring full transfer overhead. Profiling showed
**>95% of wall time was transfer and allocation**, not GPU compute.

The DeviceBuffer pattern eliminates this entirely: data enters the GPU
once, stays on-device through all stages (including transposes and
pyramid operations), and exits once. The only remaining host round-trip
is the color pipeline's Stage 2b, where the Gaussian pyramid is
downloaded to host for a reshape before the per-channel FFT — a target
for future device-resident FFT optimization.

### Level 2: cuFFT plan caching

The ideal bandpass filter creates cuFFT plans (forward + inverse C2C) for
each of the 3 color channels. `cufftPlanMany` performs internal
autotuning (kernel selection, workspace sizing) on each call — measured
at ~5–10 ms per plan on H100.

A static cache keyed on `(T, N)` eliminates redundant plan creation:
channel 1 warms the cache, channels 2 and 3 reuse it. The plan survives
across pipeline invocations within the same process, making repeated
processing of same-dimension clips effectively free of plan overhead.

### Level 3: Batched spatial kernels

The Laplacian pyramid build processes n×3 = 873 independent image slices
(291 frames × 3 channels), each requiring ~40 kernel launches across 8
pyramid levels — a total of ~35,000 launches. At ~5 μs launch overhead
each, this accounts for ~175 ms of pure dispatch overhead, which matched
the measured stage time almost exactly.

The batched spatial kernels add a batch dimension to the grid:

```cuda
// Grid: (ceil(W/32), ceil(Hout/32), B)   Block: (32, 32, 1)
// Each thread computes (x, y, slice) — per-thread math is identical
// to the single-slice kernel.
__global__ void corr_dn_rows_batched_kernel(
    const float* in, float* out,
    int H, int W, const float* filt,
    int stride_in, int stride_out, int B) { ... }
```

The grid z-dimension indexes the batch slice, allowing all B slices to
be processed in a single kernel launch. This collapses ~35,000 launches
to ~50 (one per kernel per level), a 700× reduction.

The channel-major band output layout uses irregular per-slice offsets
(`offset = chan × n_frames + frame`), which prevents simple stride-based
batching for band writes. Four scatter/gather kernels bridge this:

- `scatter_subtract`: band[l] = current − hi2 (pyramid build)
- `scatter`: coarsest residual band write
- `gather`: coarsest band read (recon)
- `gather_add`: output = band[l] + residual (recon)

### Level 4: Register-level optimizations

At the finest grain, the 5-tap separable filter can benefit from
register-level tuning. Two techniques were evaluated:

**Filter tap register hoisting.** The batched spatial kernels originally
read filter coefficients from a global-memory pointer inside the
convolution loop. Loading all 5 taps into a local array at kernel entry
(with `#pragma unroll`) forces them into registers:

```cuda
float f[5];
#pragma unroll
for (int k = 0; k < 5; ++k) f[k] = filt[k];
// ... convolution loop reads f[k] instead of filt[k]
```

This produced consistent 10–12% gains on the three spatial stages:

| Stage | Before | After |
|-------|--------|-------|
| blur_dn | 5.2 ms | 4.6 ms (−12%) |
| lpyr_build | 22.2 ms | 19.6 ms (−12%) |
| lpyr_recon | 16.4 ms | 14.8 ms (−10%) |

**`__launch_bounds__` occupancy hints.** The effect was kernel-dependent:

| Kernel | Result | Reason |
|--------|--------|--------|
| Batched spatial (1024 threads) | Retained — consistent with register pressure | Genuine register pressure from filter loop |
| IIR (256 threads) | Removed — 21% regression | Forced register spills to local memory |
| Render (256 threads) | Removed — no effect | Bandwidth-bound; occupancy irrelevant |

The IIR regression illustrates the occupancy trade-off: demanding
`minBlocksPerSM=8` with 256 threads and 32 registers per thread requires
exactly 65,536 registers (the SM maximum). The compiler achieved this by
spilling to local memory, making each thread slower. Since the IIR
kernel is sequential (one thread loops over all T frames), higher
occupancy provides no latency-hiding benefit — there's no memory latency
to hide when the thread is doing pure arithmetic.

## Results

### Component-level speedup vs. baseline

**Color pipeline (4.26s → 0.081s, 53×):**

| Component | Baseline | Optimized | Speedup |
|-----------|----------|-----------|---------|
| color_cvt | ~1.0s | 0.6 ms | ~1,700× |
| blur_dn | ~1.3s | 4.6 ms | ~280× |
| ideal_bandpass | 0.32s | 4.7 ms | 68× |
| upsample + render | 1.59s | 67.0 ms | 24× |

**Motion pipeline (14.78s → 0.181s, 82×):**

| Component | Baseline | Optimized | Speedup |
|-----------|----------|-----------|---------|
| NTSC convert | 2.20s | 0.9 ms | 2,444× |
| lpyr_build | 3.54s | 19.6 ms | 181× |
| temporal IIR | 3.97s | 40.8 ms | 97× |
| lpyr_recon | ~2.5s | 14.8 ms | ~169× |
| render | ~2.6s | 104.4 ms | ~25× |

### End-to-end perspective

The GPU pipeline is now fast enough that the dominant end-to-end cost has
shifted outside the GPU entirely. Video encoding (cv2.VideoWriter with
mp4v codec, CPU-side) takes ~2.6–2.7s — roughly 15× longer than the
entire GPU pipeline:

| Component | Color | Motion |
|-----------|-------|--------|
| GPU pipeline | 0.081s | 0.181s |
| Full pipeline (incl. decode + encode) | ~2.65s | ~2.85s |

The video codec is the clear next target: NVDEC (hardware decode) and
NVENC (hardware encode) would address both the encoding latency and the
input upload transfer.

## Open optimization surfaces

The render stage (58–83% of GPU time) is memory-bandwidth bound. The
three viable strategies, in order of estimated impact:

1. **FP16 NTSC storage** — storing the NTSC frame in half-precision halves
   the read bandwidth. The NTSC values are in [0, 1] with ~10⁻⁶ precision
   requirements; FP16's 11-bit mantissa is likely sufficient but requires
   tolerance validation against the Python baseline.

2. **Texture memory for NTSC reads** — `cudaTextureObject_t` with
   `cudaReadModeElementType` provides cached reads with hardware spatial
   locality. The render kernel reads each NTSC pixel exactly once, but
   texture cache could improve effective bandwidth via the L2-backed
   texture path.

3. **Eliminate the NTSC intermediate** — fuse the BGR→NTSC conversion
   into the render kernel, reading BGR uint8 directly and converting
   inline. This eliminates the 1.8 GB NTSC buffer entirely but requires
   reordering the pipeline (NTSC is currently computed in an early stage
   and reused).

For the IIR stage, the algorithmic seriality can only be addressed by
replacing the recursive filter with a block-parallel formulation (e.g.,
cyclic reduction or scan-based IIR), which would change the numerical
characteristics and require re-validation.

## Methodology

All measurements follow Harris's ["Optimizing Parallel Reduction in
CUDA"][harris] framework: measure first, attack the largest bottleneck,
make one change, re-profile. The profilers run 5 timed iterations with a
warmup run (to exclude kernel JIT/binary load costs), pre-allocate all
device buffers (to exclude `cudaMalloc` from kernel measurements), and
report median + min/max per stage. Video decode and encode are excluded
to isolate GPU pipeline performance.

[harris]: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

74 unit and integration tests validate correctness against the Python
baseline (RMSE < 0.01 for end-to-end pipelines, per-kernel tolerances
from 10⁻⁶ to 10⁻⁴ depending on the operation). The full test suite and
profiler scripts are in the [repository][repo].

[repo]: https://github.com/iamkucuk/evm_cuda
