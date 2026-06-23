# Implementing Eulerian Video Magnification on CUDA

A study in porting Eulerian Video Magnification (EVM) from Python/NumPy to
CUDA, analyzing the bottlenecks at each level of the GPU memory hierarchy
and the optimization strategies that address them.

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

The implementation has 32 CUDA kernels across 9 source files:

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

Performance was measured with stage-by-stage profilers
(`scripts/profile_color.py`, `scripts/profile_motion.py`) that bracket
each pipeline stage with `cudaDeviceSynchronize` and report median of 5
iterations with a warmup run. All device buffers are pre-allocated so
`cudaMalloc` doesn't contaminate kernel measurements. Measurements were
taken on NVIDIA H100 (kolyoz21, TRUBA HPC).

### Steady-state timings

**Color pipeline (face.mp4, 291 frames, 528×592): 0.081s total**

| Stage | Time | Share | What limits it |
|-------|------|-------|-----------------|
| color_cvt | 0.6 ms | 0.7% | Nothing. Trivial per-pixel 3×3 matrix multiply. |
| blur_dn | 4.6 ms | 5.7% | Compute. Separable filter, batched across all slices. |
| D2H + reshape | 4.0 ms | 5.0% | PCIe bandwidth plus a host-side numpy transpose. |
| ideal_bandpass | 4.7 ms | 5.8% | cuFFT compute. Plan is cached. |
| **upsample + render** | **52.3 ms** | **64.6%** | See analysis below. |

**Motion pipeline (baby.mp4, 291 frames, 960×544, 9 levels): 0.181s total**

| Stage | Time | Share | What limits it |
|-------|------|-------|-----------------|
| NTSC convert | 0.9 ms | 0.5% | Nothing. Trivial. |
| lpyr_build | 19.6 ms | 10.8% | Compute. Separable filter, batched. |
| temporal IIR | 40.8 ms | 22.6% | Algorithmic seriality. Each output depends on the previous two. |
| lpyr_recon | 14.8 ms | 8.2% | Compute. Separable filter, batched. |
| **render** | **81.9 ms** | **45.2%** | See analysis below. |

### The render stage: underutilized resources

The render stage takes 45 to 65% of GPU time. Each output pixel reads
~12 bytes from the NTSC frame (3 floats), does about 35 FLOPs of math
(bilinear interpolation plus NTSC-to-BGR matrix multiply), and writes 3
bytes. The arithmetic intensity is 2.3 FLOPs per byte, which on the H100
roofline puts this in the memory-bound regime.

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

### The IIR filter: algorithmic seriality (23% of motion GPU time)

The recursive temporal filter is sequential along the time axis. Each
output sample depends on the previous two. The kernel assigns one thread
per spatial location, and each thread loops over all T frames. This
can't be parallelized across time without changing the algorithm
(block-parallel scan, cyclic reduction).

### The spatial filters: compute bound but fast (11 to 19%)

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
ms per plan on H100.

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

The channel-major band output layout uses irregular per-slice offsets
(`offset = chan × n_frames + frame`), which prevents simple stride-based
batching for band writes. Four scatter/gather kernels handle this:
`scatter_subtract` for band writes during build, `scatter` for the
coarsest residual, `gather` for coarsest band reads during recon, and
`gather_add` for combining bands with the residual during recon.

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

Multiple elements per thread (Harris V6) addresses the render stage.
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

This gave 22% gains on both render stages:

| Stage | Before | After |
|-------|--------|-------|
| Color render | 67.0 ms | 52.3 ms (-22%) |
| Motion render | 104.4 ms | 81.9 ms (-22%) |

The same technique was applied to the transpose kernels used in the IIR
stage, where each thread now handles 4 spatial locations.

## Throughput and theoretical limits

### Measured throughput

The GPU pipeline processes pixels at the following rates (whole pipeline,
not just render):

| Pipeline | Resolution | Time (291 frames) | Throughput |
|----------|-----------|-------------------|------------|
| Color | 528×592 | 0.081s | **1.12 Gpx/s** (0.89 ns/px) |
| Motion | 960×544 | 0.181s | **0.84 Gpx/s** (1.19 ns/px) |

Motion is slower per pixel because the Laplacian pyramid does 9 levels of
decomposition and reconstruction, plus the IIR filter is sequential per
location.

### Realtime performance projection

Scaling linearly by pixel count (the bottleneck stages scale with pixels):

| Resolution | Color | Motion | Realtime (30 fps)? |
|-----------|-------|--------|---------------------|
| 1080p (1920×1080) | 542 fps | 405 fps | **18× and 13× headroom** |
| 4K (3840×2160) | 135 fps | 101 fps | **4.5× and 3.4× headroom** |
| Max @ 30 fps | 8156×4588 | 7052×3966 | Beyond 8K |

At 1080p, a single H100 can run the full color pipeline at 542 fps and
motion at 405 fps. Even at 4K, both pipelines exceed 30 fps with 3× to 4×
margin. The maximum resolution for 30 fps realtime exceeds 8K for color
and approaches 8K for motion.

These are GPU-only numbers. The end-to-end pipeline (including video
decode and encode) is currently bottlenecked by the CPU codec at about
2.7s per clip, which limits realtime throughput to roughly 0.1 fps
regardless of GPU speed.

### Resource utilization: both underutilized

The render stage takes 45 to 65% of GPU time, yet it achieves only 0.4%
of the H100's peak memory bandwidth and 0.003% of peak FP32 throughput.
Neither resource is saturated. The GPU has the bandwidth and the compute
capacity, but the kernel's data access patterns don't let it use either.

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

The render stage (45 to 65% of GPU time) underutilizes both memory
bandwidth and compute. The remaining headroom is in changing how the
kernel accesses data, not how much data it processes. Three approaches
correspond to later stages of Harris's progression.

Shared memory tiling (Harris V3). Load a tile of the NTSC frame into
shared memory cooperatively, then have each thread read from shared memory
(~20 cycle access) instead of global memory (~400 cycle access). Adjacent
threads share loaded cache lines, and the `__syncthreads()` barrier
ensures the tile is fully loaded before any thread reads from it. The
shared memory layout must be padded to avoid bank conflicts (32 banks, 4
bytes each; a stride that's a multiple of 32 causes all threads to hit the
same bank).

Texture hardware (Harris texture path). `cudaTextureObject_t` with
`cudaReadModeElementType` provides hardware-managed L1 texture cache with
spatial prefetch. The texture unit automatically fetches neighboring cache
lines, so adjacent threads benefit from each other's reads without
explicit shared memory management. For the color pipeline's bilinear
upsample, `tex2D` with linear filtering replaces the manual 4-tap
interpolation entirely: the hardware does it in one instruction with its
own cache.

Multiple elements per thread, extended (Harris V6+). The current V6
implementation processes 4 pixels per thread. Increasing to 8 or 16 would
give the compiler more independent memory operations to pipeline, further
filling the gaps where the SM waits for data. The tradeoff is register
pressure: each pixel needs registers for its intermediate values, and
spilling to local memory would negate the benefit.

FP16 NTSC storage. Storing the NTSC frame in half-precision halves the
bytes per read, which doubles the effective cache capacity and improves
coalescing density. The NTSC values are in [0, 1] with roughly 10⁻⁶
precision requirements. FP16's 11-bit mantissa is likely sufficient but
needs tolerance validation.

For the IIR stage, the algorithmic seriality can only be addressed by
replacing the recursive filter with a block-parallel formulation (cyclic
reduction or scan-based IIR). That changes the numerical characteristics
and would require re-validation.

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

This project applied the same principle at two levels. At the pipeline
level: eliminating host-device transfers, batching kernel launches,
caching cuFFT plans. At the kernel level: register hoisting for filter
taps, and V6 multiple-elements-per-thread for the render stage (22%
improvement). The render kernel's 0.4% bandwidth utilization indicates
the same class of problem Harris describes: the hardware can deliver far
more data and compute far more FLOPs than the kernel currently asks of it.
The shared-memory tiling, texture hardware, and extended V6 optimizations
in the open surfaces section are the next steps along that roadmap.

The profilers run 5 timed iterations with a warmup run (to exclude kernel
JIT and binary load costs), pre-allocate all device buffers (to exclude
`cudaMalloc` from kernel measurements), and report median plus min/max per
stage. Video decode and encode are excluded to isolate GPU pipeline
performance.

[harris]: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

74 unit and integration tests validate correctness against the Python
baseline (RMSE < 0.01 for end-to-end pipelines, per-kernel tolerances
from 10⁻⁶ to 10⁻⁴ depending on the operation). The full test suite and
profiler scripts are in the [repository][repo].

[repo]: https://github.com/iamkucuk/evm_cuda
