# Implementing Eulerian Video Magnification on CUDA

A look at porting Eulerian Video Magnification (EVM) from Python/NumPy to
CUDA, and where the time actually goes once you do. The interesting part
isn't the speedup itself. It's that the bottleneck keeps moving as you
optimize, and each level of the GPU memory hierarchy has its own physics.

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

The core design choice is a device-resident pipeline. Data enters the
GPU once as a uint8 input clip, passes through all stages as on-device
float32 buffers, and exits once as a uint8 output clip. No intermediate
host transfers happen within the pipeline.

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
| **upsample + render** | **67.0 ms** | **82.8%** | **Memory latency** (see throughput analysis below). |

**Motion pipeline (baby.mp4, 291 frames, 960×544, 9 levels): 0.181s total**

| Stage | Time | Share | What limits it |
|-------|------|-------|-----------------|
| NTSC convert | 0.9 ms | 0.5% | Nothing. Trivial. |
| lpyr_build | 19.6 ms | 10.8% | Compute. Separable filter, batched. |
| temporal IIR | 40.8 ms | 22.6% | Algorithmic seriality. Each output depends on the previous two. |
| lpyr_recon | 14.8 ms | 8.2% | Compute. Separable filter, batched. |
| **render** | **104.4 ms** | **57.9%** | **Memory latency** (see throughput analysis below). |

### Three bottleneck regimes

The profiler data falls into three categories, each governed by different
physics.

**The render stage is memory-latency bound (58 to 83% of GPU time).**

The render kernels read the full-resolution NTSC frame in float32
(n×H×W×3 = 1.8 GB for motion) and write the uint8 output. The arithmetic
intensity is about 2.3 FLOPs per byte, which puts it nominally in the
memory-bound regime on H100's roofline. But the kernel achieves only 0.4%
of peak bandwidth. The problem isn't that the kernel moves too much data
for the memory controller to handle. The problem is that each thread
issues a global memory read, then stalls for 400+ cycles waiting for it,
with no software caching to hide the latency.

Kernel fusion confirmed that the bottleneck isn't raw data volume.
Merging the bilinear upsample, add, and quantize into a single kernel
eliminated a 1.8 GB intermediate buffer and one kernel launch. It
produced no measurable improvement. Eliminating reads doesn't help when
the remaining reads still stall on latency. The fix is faster reads
(texture units, shared memory tiling), not fewer reads.

**The IIR filter is algorithmically serial (23% of motion GPU time).**

The recursive temporal filter is sequential along the time axis. Each
output sample depends on the previous two. The kernel assigns one thread
per spatial location, and each thread loops over all T frames. You can't
parallelize across time without changing the algorithm (block-parallel
scan, cyclic reduction). This is the one stage where the bottleneck is
the math itself, not the hardware moving data around.

**The spatial filters are compute bound but fast (11 to 19%).**

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
for a reshape before the per-channel FFT. That's a candidate for a
device-resident FFT in a future pass.

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

### Level 4: Register-level tuning

At the finest grain, the 5-tap separable filter can benefit from
register-level tuning. Two techniques were evaluated.

**Filter tap register hoisting.** The batched spatial kernels originally
read filter coefficients from a global-memory pointer inside the
convolution loop. Loading all 5 taps into a local array at kernel entry
(with `#pragma unroll`) forces them into registers:

```cuda
float f[5];
#pragma unroll
for (int k = 0; k < 5; ++k) f[k] = filt[k];
// convolution loop reads f[k] instead of filt[k]
```

This gave consistent 10 to 12% gains on the three spatial stages:

| Stage | Before | After |
|-------|--------|-------|
| blur_dn | 5.2 ms | 4.6 ms (-12%) |
| lpyr_build | 22.2 ms | 19.6 ms (-12%) |
| lpyr_recon | 16.4 ms | 14.8 ms (-10%) |

**`__launch_bounds__` occupancy hints.** The effect was kernel-dependent.
On the batched spatial kernels, it was retained because the filter loop
creates genuine register pressure. On the IIR kernel, it caused a 21%
regression: demanding `minBlocksPerSM=8` with 256 threads and 32 registers
per thread requires exactly 65,536 registers, the SM maximum. The compiler
hit that target by spilling registers to local memory, which made each
thread slower. Since the IIR kernel is sequential (one thread loops over
all T frames), higher occupancy provides no latency-hiding benefit. There
is no memory latency to hide when the thread is doing pure arithmetic.

On the render kernels, `__launch_bounds__` had no measurable effect. The
render kernel is waiting on memory, not compute, so occupancy hints are
irrelevant.

## Results

### Component-level speedup vs. baseline

**Color pipeline: 4.26s → 0.081s (53x)**

| Component | Baseline | Optimized | Speedup |
|-----------|----------|-----------|---------|
| color_cvt | ~1.0s | 0.6 ms | ~1,700x |
| blur_dn | ~1.3s | 4.6 ms | ~280x |
| ideal_bandpass | 0.32s | 4.7 ms | 68x |
| upsample + render | 1.59s | 67.0 ms | 24x |

**Motion pipeline: 14.78s → 0.181s (82x)**

| Component | Baseline | Optimized | Speedup |
|-----------|----------|-----------|---------|
| NTSC convert | 2.20s | 0.9 ms | 2,444x |
| lpyr_build | 3.54s | 19.6 ms | 181x |
| temporal IIR | 3.97s | 40.8 ms | 97x |
| lpyr_recon | ~2.5s | 14.8 ms | ~169x |
| render | ~2.6s | 104.4 ms | ~25x |

### End-to-end perspective

The GPU pipeline is now fast enough that the dominant end-to-end cost has
shifted outside the GPU. Video encoding (cv2.VideoWriter with mp4v codec,
running on the CPU) takes about 2.6 to 2.7s, roughly 15x longer than the
entire GPU pipeline:

| Component | Color | Motion |
|-----------|-------|--------|
| GPU pipeline | 0.081s | 0.181s |
| Full pipeline (incl. decode + encode) | ~2.65s | ~2.85s |

The video codec is the clear next target. NVDEC (hardware decode) and
NVENC (hardware encode) would address both the encoding latency and the
input upload transfer.

## Open optimization surfaces

The render stage (58 to 83% of GPU time) is memory-latency bound, achieving
under 1% of peak bandwidth. The kernel stalls on global memory read latency
because it uses no software caching. Three strategies address this directly.

**Texture memory for NTSC reads.** `cudaTextureObject_t` provides hardware-
managed L1 texture cache. The render kernel currently reads the NTSC frame
from raw global memory (L2 to DRAM path). Texture units would cache the
reads in L1, turning 400+ cycle DRAM latency into ~30 cycle L1 hits on
spatially local accesses. Since adjacent threads read adjacent pixels, the
texture cache hit rate should be high. This is the most direct fix for a
latency-bound kernel and requires no algorithmic change.

**Shared memory tiling.** Load a block of the NTSC frame into shared memory
cooperatively, then have each thread read from shared memory. This is the
classic solution to the latency problem that Harris describes in the
reduction paper. It requires a tiled kernel rewrite but gives full control
over the caching behavior.

**FP16 NTSC storage.** Storing the NTSC frame in half-precision halves the
bytes per read, which doubles the effective bandwidth for the same latency.
The NTSC values are in [0, 1] with roughly 10⁻⁶ precision requirements.
FP16's 11-bit mantissa is likely sufficient but needs tolerance validation
against the Python baseline.

**Eliminate the NTSC intermediate entirely.** Fuse the BGR-to-NTSC conversion
into the render kernel, reading BGR uint8 (1 byte per channel) directly
instead of NTSC float32 (4 bytes per channel). This cuts the read volume by
4× and eliminates the 1.8 GB NTSC buffer, but requires reordering the
pipeline since NTSC is currently computed early and reused by both the
filter and render stages.

For the IIR stage, the algorithmic seriality can only be addressed by
replacing the recursive filter with a block-parallel formulation (cyclic
reduction or scan-based IIR). That changes the numerical characteristics
and would require re-validation.

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

At 1080p, a single H100 could run the full color pipeline at 542 fps and
motion at 405 fps. Even at 4K, both pipelines exceed 30 fps with 3× to 4×
margin. The maximum resolution for 30 fps realtime exceeds 8K for color
and approaches 8K for motion.

These are GPU-only numbers. The end-to-end pipeline (including video
decode and encode) is currently bottlenecked by the CPU codec at about
2.7s per clip, which limits realtime throughput to roughly 0.1 fps
regardless of GPU speed.

### Resource utilization: why the render stage is slow

The render stage dominates GPU time (58 to 83%), yet it achieves only
**0.4% of the H100's peak memory bandwidth** and **0.003% of peak FP32
throughput**. Neither resource is saturated.

The arithmetic intensity is about 2.3 FLOPs per byte. On the roofline
model, the crossover between memory-bound and compute-bound on H100 is
at ~3.4 FLOPs/byte, so the kernel is nominally in the memory-bound
regime. But achieving only 0.4% of peak bandwidth means it isn't actually
saturating memory. The real limiter is **memory latency**, not bandwidth.

Each thread reads ~12 bytes from the NTSC frame (3 floats), does about 35
FLOPs of math (bilinear interpolation plus NTSC-to-BGR matrix multiply),
and writes 3 bytes. The reads go through L2 cache to DRAM with no
software caching (no shared memory, no texture units). A global memory
read takes 400 to 600 cycles. The thread's 35 FLOPs finish in about 35
cycles. Without enough concurrent threads in flight to overlap, the SM
stalls on the read latency for 90%+ of the time.

This explains why kernel fusion didn't help. Eliminating an intermediate
buffer read doesn't matter when the threads are already stalled on the
remaining reads. The fix is not fewer reads but faster reads: texture
units (hardware-managed L1 cache), shared memory tiling (load a block to
shared memory once, compute from it), or processing multiple output
pixels per thread to amortize the read latency.

### Theoretical ceiling

If the render kernel could achieve 50% of peak memory bandwidth (a
typical achievable fraction for well-optimized kernels), the render stage
would drop from 67 ms to under 1 ms for color, and from 104 ms to under
2 ms for motion. The full pipelines would drop to roughly 15 ms (color)
and 40 ms (motion), processing 1080p at over 2,000 fps.

That is the headroom. The current implementation uses less than 1% of it.

## Methodology

All measurements follow Harris's ["Optimizing Parallel Reduction in
CUDA"][harris] framework: measure first, attack the largest bottleneck,
make one change, re-profile. The profilers run 5 timed iterations with a
warmup run (to exclude kernel JIT and binary load costs), pre-allocate
all device buffers (to exclude `cudaMalloc` from kernel measurements),
and report median plus min/max per stage. Video decode and encode are
excluded to isolate GPU pipeline performance.

[harris]: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

74 unit and integration tests validate correctness against the Python
baseline (RMSE < 0.01 for end-to-end pipelines, per-kernel tolerances
from 10⁻⁶ to 10⁻⁴ depending on the operation). The full test suite and
profiler scripts are in the [repository][repo].

[repo]: https://github.com/iamkucuk/evm_cuda
