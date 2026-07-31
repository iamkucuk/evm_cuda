# Further optimizations: layout, allocation, and what the measurements said

A follow-up to [Implementing Eulerian Video Magnification on CUDA](blog_speedup.md).

The first writeup covered the stack that made a correct CUDA port fast:
device-resident pipelines, batched spatial launches, cuFFT plan caching,
multi-element render, and FP16 storage. After that work, motion on an H100
was already in the tens of milliseconds of compute. On a consumer RTX 3090
the same motion path was still near a second of compute, and the question
changed.

Are we hitting a theoretical limit (bandwidth or compute), or failing to
because of access patterns, host overhead, or thrashing device memory?

This post walks that question step by step, with a stage table after every
change that stayed. The EVM math did not change: same Laplacian levels, same
r1/r2 IIR, same Figure-6 alpha. What changed is how memory is addressed, how
buffers live across calls, and which half of the pyramid is worth fusing.

Headline numbers for the mid-pipeline series (RTX 3090, `data/baby.mp4`, 291
frames at 960x544, 9 levels, FP32 motion *compute only*):

| Arc | Compute |
|---|---:|
| Before this series | ~934 ms |
| After coalesced TN IIR + sticky lpyr scratch | ~377-407 ms |
| After DeviceBuffer free-list + smem fused downsample | ~96-104 ms |
| Total (series) | about 9x mid-pipeline compute |
| **Current production (remeasured)** | **3090 90.4/75.1; A100 54.4/48.2; H100 35.8/34.5 (FP32/FP16 motion)** |

H2D/D2H and encode are reported, but they were not the target. After this
series, transfer often exceeds mid-pipeline compute on file-to-file runs, so
product latency is a different problem (NVDEC/NVENC, async H2D).

**Cross-GPU note:** Older H100 (~85 ms motion) and A100 (~209 ms motion) doc
numbers predate this mid-pipeline series. Remeasuring them *after* TN IIR /
sticky / pool / smem-down (and later true half-band) shows ~2.4× (H100) and
~3.8× (A100) on motion FP32. That is the series landing on those GPUs, not
half-band alone. Half-band / dense smem is the extra FP16 edge (3090 motion
0.83×; H100 only 0.96× because the path is already tiny and transfer-bound).

Stage tables for each keep are in this post. Pre/post CUDA FP32 A/B on
baby.mp4 (same params, tree before this series vs production): not
bit-identical; float max abs 1/255, float RMSE ~1.5e-6, 66 of ~456M uint8
pixels differ by 1 LSB. Drift is at quantize boundaries, not a layout bug.

---

## Provenance and methodology

| Item | Value |
|---|---|
| GPU (production timings) | RTX 3090 24 GB (sm_86, WSL2) |
| Isolation also on | H100 80 GB for TN vs NT IIR diagnosis |
| Clip | `data/baby.mp4`, 291 frames, 960x544, 9 Laplacian levels |
| Harness | `evm_cuda.benchmark`: median after untimed warmup; `cudaDeviceSynchronize` per stage |
| Scope | Mid-pipeline stages A-D2 (NTSC, build, IIR, recon, render) |
| Correctness | Batched + pipeline CUDA tests green after each keep |

Nsight Compute GPU counters are not available under WSL2 on the development
host used for most timings. Saturation claims use
traffic-accounted isolation probes: same data volume, different access or
math, wall time and effective GB/s against a friendly streaming baseline
(~850-860 GB/s flat copy on the 3090, about 90% of ~936 GB/s GDDR6X peak).

Absolute milliseconds move with GPU load. Ratios and which stage collapses
are the useful part. Two baseline numbers for the same post-TN+sticky code
generation appear in the record (~377 ms end-of-session median vs ~407 ms
later median-of-7 re-baseline). That gap is harness noise, not a missing fix.
Pool-and-later steps use the ~407 ms Step 0 so one continuous table stays
consistent. The layout session's 934 to 377 is the earlier arc.

Working rule after every real fix:

1. Stage table first: who owns the milliseconds?
2. Calibrate the roof: a friendly stream must approach peak BW.
3. Isolate with same-traffic probes: copy vs math, NT vs TN, FP32 vs FP64.
4. Validate the fix in the production pipeline, not only a microbench.
5. Re-stage the table. The next bottleneck is allowed to change.
6. Keep the numerical contract green.

---

## The arc in one table

Motion FP32 compute on 3090 (`baby.mp4`):

| Milestone | Compute | What changed |
|---|---:|---|
| Start of this series | ~934 ms | Batched path, but NT IIR + multi-GB scratch thrash |
| Level 1: coalesced `(T,N)` IIR | ~718 ms | Drop NT sandwich; filter where bands live |
| Level 2: sticky lpyr scratch | ~377-407 ms | Grow-only scratch for build/recon/blur |
| Level 3: DeviceBuffer free-list pool | ~113-133 ms | No hot-path `cudaFree` for stage buffers |
| Level 4: channel-outer + contiguous bands | ~106-110 ms | Layout foundation; wall change in noise |
| Level 5: smem fused downsample | ~98-102 ms | Small real Stage B win |
| Level 6: smem fused upsample | 127-189 ms | Regressed, then reverted |
| Production | ~96-104 ms | about 9x vs series start |
| Level 7: up_conv tile/reflect/taps | not yet landed | 1.70x rows, 1.46x cols in isolation (3090); 1.27x total GPU kernel time (A100 trace) |

Full stage collapse (pre-pool ~407 ms era to production ~100 ms era):

| Stage | Before pool | After pool + smem down |
|---|---:|---:|
| A) NTSC | 67 ms | ~3 ms |
| B) lpyr_build | 163 ms | ~36-39 ms |
| C) IIR | 87 ms | ~27-30 ms |
| D1) recon | 72 ms | ~27-30 ms |
| D2) render | 16 ms | ~5 ms |
| Compute | ~407 ms | ~100 ms |

---

## Level 1: Stage C was not math-bound; it was uncoalesced

### Calibration

On the 3090, a flat float copy and an f32/f16 round-trip both reach about
850-860 GB/s. The device can saturate bandwidth when the access pattern is
friendly.

### Isolation probes (finest level, one channel, N=522240, T=291)

| Probe | What it does | 3090 time | Eff. BW |
|---|---|---:|---:|
| `flat_copy` | Coalesced `out[i]=in[i]` | ~1.4 ms | ~860 GB/s |
| `strided_NT_copy` | Same indexing as production IIR, no math: thread `n` loops `t`, `addr=n*T+t` | ~48-55 ms | ~25 GB/s |
| `iir_fp64` (old production) | Real bandpass, FP64 state, `(N,T)` | ~39 ms | ~31 GB/s |
| `iir_fp32` | Same bandpass, FP32 state | ~39 ms | ~31 GB/s |
| `iir_single_pole` | One pole only | ~42 ms | ~29 GB/s |
| `coalesced_TN_copy` | `addr=t*N+n` | ~1.5-1.7 ms | ~730-810 GB/s |
| `iir_fp32_TN` | Full bandpass on `(T,N)` | ~1.5 ms | ~760-820 GB/s |

H100 told the same story with cleaner absolutes: NT IIR ~5.0 ms to TN IIR
0.47 ms (about 12x). TN IIR matched TN copy and the flat stream (~77% of
3350 GB/s).

### What the probes ruled out

| Hypothesis | Verdict |
|---|---|
| IIR is just slow math / dependency chain as the primary cost | Too strong. Pure memory walk with IIR indexing is already ~3% of peak BW. |
| FP64 fidelity is the first wall on 3090 | No. FP32 state matches FP64 time. |
| Second pole / subtraction is expensive | No. Single-pole matches full bandpass. |
| We need a different filter (Riesz) to go faster | Orthogonal. Same r1/r2 IIR is fine once the layout is coalesced. |

### What they ruled in

Production Stage C was:

```text
band (T,N)  ->  thwc_to_nt  ->  IIR on (N,T)  ->  nt_to_thwc_scaled  ->  band (T,N)
```

Laplacian band storage was already `(T, N)` per channel. The pipeline paid two
full transposes to put the filter on `(N, T)`, where time is contiguous per
thread, but warp lanes at fixed `t` are spaced by `T` floats (~1164 B). That
is uncoalesced across the warp.

Coalesced form keeps the same recurrence (one location per thread, serial `t`
in registers) but addresses `t*N + n`, so a warp at fixed `t` is contiguous in
`n`.

### The production change

Replace the three-call sandwich with one kernel:

```cpp
// Production: bands already (T,N); scale folds per-level alpha.
iir_bandpass_tn_kernel<<<...>>>(in, out, T, N, r1, r2, alpha);
// addr = t * N + n
```

Python Stage C becomes a nested loop over levels and channels calling
`batched_iir_bandpass_tn` (and the f16 twin). No intermediate `(N,T)` scratch.
Legacy `(N,T)` IIR remains for unit tests and probes.

### Measured impact (3090 motion FP32)

| | Stage C | Compute total |
|---|---:|---:|
| Before | 323 ms | 934 ms |
| After TN IIR | 84 ms (-74%) | 718 ms (-23%) |

The win is larger than "IIR kernel alone." We removed both layout transforms
that owned a large fraction of Stage C bytes (probe sandwich traffic was about
two-thirds transpose).

---

## Level 2: After IIR, spatial stages were inflated by multi-GB alloc thrash

With Stage C no longer dominant, the table reweighted:

| Stage | After TN IIR | Share of compute |
|---|---:|---:|
| B) lpyr_build | ~390 ms | ~54% |
| D1) recon | ~182 ms | ~25% |
| C) IIR | ~84 ms | ~12% |

Spatial probes on the same 3090:

| Work | Time | Note |
|---|---:|---|
| Finest 4-kernel chain (cols+rows+up x2, M slices) | ~23 ms | Sum of parts matches chain (no launch tax) |
| Full `batched_lpyr_build` (9 levels) | ~280-350 ms | about 15x the finest chain |
| `corr_dn_*` unique-byte model | often 50%+ of peak BW | Nearer the roof than IIR ever was |

The 5-tap kernels are not a 3%-of-peak coalescing disaster. Full build cost was
inflated by something else: allocating four full-resolution scratch buffers
(`M x H x W` floats each, multi-GB) on every `batched_lpyr_build` call, then
freeing them. Recon did the same with two buffers; blur did it too.

### The change: sticky grow-only scratch

```cpp
struct StickyScratch {
    void* ptr = nullptr;
    size_t capacity = 0;
    void* ensure(size_t nbytes);  // cudaMalloc only when capacity grows
};
// Carve slots: sticky_f32_slots(M*H*W, 4) for build, 2 for recon, etc.
```

FP32 and FP16 build/recon/blur use sticky slots. Only small per-call tables
still allocated and freed at this point.

### Measured impact (3090 motion FP32)

| Stage | After TN only | After sticky | Delta |
|---|---:|---:|---:|
| B) build | 390 ms | 158 ms | -59% |
| D1) recon | 182 ms | 70 ms | -61% |
| C) IIR | 84 ms | 82 ms | ~flat |
| Compute | 718 ms | 377 ms | -47% |

Combined with the layout fix: 934 to 377 ms compute (-60%) on that session.
FP16 motion compute moved 709 to 452 ms (-36%) on the same arc.

### What the numbers said about theoretical limits

```
Friendly stream / TN IIR path     ####################  near BW peak
Production (N,T) IIR (old)        #                     ~3% BW
Spatial 5-tap (unique bytes)      ############          often half peak+
Multi-GB cudaMalloc every call    (host/driver wall, not a FLOP roof)
```

1. Missing both bandwidth and compute utilization usually means the access
   schedule or host work is wrong, not that the algorithm must change.
2. Roofline AI alone misled on IIR. Low FLOP/byte looked memory-bound; a pure
   strided copy with the same indexing proved the walk never reached the
   memory roof.
3. Same recurrence, different layout put IIR on the bandwidth roof (TN IIR
   matches TN copy on H100 and on clean 3090 runs).
4. Alloc thrash can dominate kernel time even when the kernels themselves are
   respectable. Measure build wall against a finest-level chain; a huge gap
   points at setup, not taps.

---

## Level 3: The next slow stages were still allocator walls

After TN + sticky, a fresh median-of-7 re-baseline on the same node landed at
~407 ms compute (same code generation as the ~377 ms session end; see
provenance). Isolation said something uncomfortable:

| Work | Isolated kernel | Pipeline stage |
|---|---:|---:|
| NTSC | ~2.6 ms at ~93% peak BW | ~55-67 ms |
| lpyr_build | ~40-45 ms | ~160 ms |

That gap is not color math and not 5-tap math. Sticky scratch had fixed lpyr
internals. The Python pipeline stage buffers (NTSC, planar, bands, filtered,
delta, output) still RAII'd `cudaMalloc` / `cudaFree` every call.

### The change: free-list DeviceMemPool

```text
DeviceMemPool: free-list by exact size
  alloc(n)  -> reuse bin[n] if any, else cudaMalloc
  release   -> push back to bin[n]  (no cudaFree until process exit)
```

Same idea as sticky scratch: known geometry, reuse pages, don't return multi-GB
regions to the OS between stages or between timed iterations. Stage buffers
allocate sequentially (same peak VRAM as before). The benchmark's VRAM
pre-check was softened so pool-held memory doesn't look like an OOM skip after
a warm run.

### Measured (Step 1 of the progressive log)

| Stage | Before | After (best of pair) | Delta |
|---|---:|---:|---:|
| A) NTSC | 67.4 | 2.7 | -96% |
| B) build | 163.1 | 39.4 | -76% |
| C) IIR | 87.4 | 29.8 | -66% |
| D1) recon | 72.4 | 36.4 | ~-50% |
| D2) render | 16.2 | 4.8 | -70% |
| Compute | 406.6 | 113.2 | -72% |

Stage A now matches the isolated kernel. Stage B matches isolated build. That
is the definition of "the wall was alloc/first-touch."

### What this ruled out

| Hypothesis | Verdict |
|---|---|
| NTSC is a fat color kernel | No. Kernel was already on the BW roof. |
| Build is 160 ms of 5-tap work | No. ~40 ms of work + ~120 ms of thrash. |
| We need a different algorithm to leave ~400 ms | No. Same EVM math; stop throwing device memory away. |

This is the single largest win of the whole series after the first blog.
Everything after is foundation or a few milliseconds.

---

## Level 4: Layout foundation without a wall-time win

After the pool, Stage B still paid a scatter/gather bridge between frame-major
spatial scratch and channel-major bands for TN IIR.

### Channel-outer planar

Planar packing was frame-major (`m = f*3 + c`) while bands wanted
channel-outer (`m' = c*n + f`). That forced a permutation in the offset table.

Change: `batched_to_planar_3ch_chan_outer` so planar and bands share
`m' = c*n + f`. Offsets become affine: `level_off + m*N`. Render reads
channel-outer delta directly.

Result: correctness/layout foundation. Compute ~113 to ~110 ms, inside noise.

### Contiguous band write/add

With affine layout, scatter is no longer a scatter. It is a dense subtract/add
over `M*N` floats. Drop the per-call offset table H2D; use `band_subtract` /
`band_add` into `out_base + level_off`.

Result: cleaner code. Compute ~110 to ~106 ms, still noise-level.

### Honest read

Both steps were worth doing for the design, not for a blog-scale speedup. They
remove a class of irregular layout tax so the next kernel work is not fighting
a permutation. They did not beat run-to-run variance the way the pool did.

---

## Level 5: Fuse the right half of the pyramid

### Bound after the pool (compute only)

Clean sequential probes (unique-byte model, ~936 GB/s peak):

| Op | % peak BW (unique) | Role |
|---|---:|---|
| NTSC / planar | ~90-94% | Done |
| corr_dn rows/cols | ~63-67% | Near roof |
| up_conv rows/cols | ~28-32% | Weak access |
| Stage B as a whole | ~45% | Mix of the above + band write |
| Stage C (packaged) | ~18% | Multi-launch; 1ch IIR ~1.3-1.6x TN copy when clean |

Arithmetic intensity is still well below the FP32/HBM ridge. Nothing here needs
more FLOPs.

### Smem fused downsample (kept)

Production order is still cols then rows (matlabPyrTools). Unlike dense 5x5
global fusion (already a wash or loss earlier), this path:

1. Cooperatively loads a 2D input tile into shared memory
2. Horizontal 5-tap, then vertical 5-tap from smem
3. Writes the half-res output once

That removes the intermediate `(H, W/2)` global write/read between the two 1D
passes, the same pattern as OpenCV's `cuda::pyrDown`, with L2 binom5 +
reflect1 so the numerical contract stays.

Wired into FP32/FP16 `batched_lpyr_build` and `batched_blur_dn_color`. The
dense `corr_dn_2d` variant was probe-only and is since removed as dead code.

| Stage | Pre-smem | After smem down |
|---|---:|---:|
| B) build | ~39-45 ms | ~36-39 ms |
| Compute | ~106-110 ms | ~98-102 ms |

Small, real, and consistent with the bound: corr_dn was already BW-leaning;
fusion only buys intermediate traffic.

### Smem fused upsample (reverted)

Symmetric idea for recon/build upsample (rows-then-cols product from a coarse
tile in smem). Structurally close to the dense `up_conv_2d` that had already
regressed recon earlier.

| Stage | Pre | Fused up |
|---|---:|---:|
| B) build | ~36-39 | 50-104 |
| D1) recon | ~27 | ~41-42 |
| Compute | ~100 | 127-189 |

Reverted and removed. Separable `up_conv` stays in production; the fused-up
variant no longer lives in the tree.

Fusion is not a virtue by itself. Down fusion helped. Up fusion, same family as
the old dense product, hurt twice. Do not re-attempt up_conv fusion without a
new isolation A/B that beats separable on recon and build.

---

## Level 7: up_conv was slow for three reasons, none of them fusion

Level 5 measured up_conv at 28-32% of peak, tried to fix it by fusing, lost
twice, and closed with "do not re-attempt up_conv fusion." That ruled out one
approach. It never diagnosed why the separable kernels were weak in the first
place. The bound table carried "weak access" as an unexplained entry from then
on.

This level opens the kernels instead of the launch structure.

Status: measured and validated, not yet landed in the tree. Numbers below come
from a patched scratch copy; the production tables above are still pre-Level-7.

### Where the clock actually is

Nsight Systems traces the real motion pipeline without needing counters, so it
works where Nsight Compute does not (see below). A100, `baby.mp4`, FP32 motion,
per-kernel totals across the whole run:

| Kernel | Instances | Total | Share |
|---|---:|---:|---:|
| `up_conv_cols_batched` | 16 | 19.85 ms | 35.7% |
| `up_conv_rows_batched` | 16 | 11.65 ms | 21.0% |
| `corr_dn_fused_smem_batched` | 8 | 5.17 ms | 9.3% |
| `band_subtract` | 8 | 4.35 ms | 7.8% |
| `band_add` | 8 | 4.31 ms | 7.8% |
| `iir_bandpass_tn` | 27 | 4.05 ms | 7.3% |
| `add_planar_quantize` | 1 | 2.56 ms | 4.6% |
| `to_planar_3ch_chan_outer` | 1 | 2.06 ms | 3.7% |
| `bgr_u8_to_ntsc_f32` | 1 | 1.54 ms | 2.8% |
| Total GPU kernel time | | 55.55 ms | |

The two up_conv kernels are 56.7% of all GPU time. Instance counts match the
host loops in `bindings.cpp` (8 build levels, 16 up_conv across build and
recon, 27 IIR launches for 9 levels x 3 channels), which is a cheap check that
the trace lines up with the code.

### Calibration

The in-tree `probe_flat_copy_f32` was the roof reference for earlier levels. A
vectorized `float4` grid-stride copy beats it:

| Reference | RTX 3090 | A100 80GB |
|---|---:|---:|
| Theoretical peak | 936 GB/s | 2039 GB/s |
| `float4` copy (read+write) | 845 GB/s (90.3%) | 1695 GB/s (83.1%) |
| `float4` read only | 910 GB/s (97.2%) | 1768 GB/s (86.7%) |
| `float4` write only | 882 GB/s (94.2%) | 1770 GB/s (86.8%) |
| in-tree `probe_flat_copy_f32` | 838 GB/s | 1624 GB/s |

The in-tree probe understates the roof by about 4% on A100. Saturation figures
computed against it, including the 28-32% in Level 5, are optimistic by that
margin. Everything below uses the `float4` copy number.

### Occupancy, ruled out without a profiler

Nsight Compute refuses to collect counters on both development hosts
(`ERR_NVGPUCTRPERM`). On the cluster there is no route to elevation. On the
WSL2 box the GPU is driven by the Windows driver through `/dev/dxg`, so there
is no `nvidia` kernel module to pass `NVreg_RestrictProfilingToAdminUsers=0`
to, and `sudo ncu` fails the same way (tested). The permission lives on the
Windows side.

Occupancy is still answerable without counters. `cuobjdump -res-usage` reports
registers and shared memory per kernel from the binary; `cudaGetDeviceProperties`
reports the SM limits from the device. That is enough to compute theoretical
occupancy:

| Kernel | Regs | Smem | 3090 occupancy | A100 occupancy |
|---|---:|---:|---:|---:|
| `up_conv_rows<f,f>` | 36 | 1792 B | 100% (threads) | 75% (registers) |
| `up_conv_cols<f,f>` | 34 | 768 B | 100% (threads) | 75% (registers) |
| `corr_dn_fused_smem<f,f>` | 40 | 5440 B | 100% (threads) | 75% (registers) |
| `iir_bandpass_tn<f,f>` | 40 | 0 | 100% (threads) | 75% (registers) |
| `band_subtract` | 12 | 0 | 100% (threads) | 100% (threads) |

On the 3090 every kernel reaches full occupancy and up_conv still runs at a
third of the roof. Occupancy is not the explanation. On the A100 the pyramid
kernels cap at 75% on the register file (34-40 registers x 2048 threads
overruns 65536), which is worth at most 1.33x and does not account for a
kernel sitting at 15-22%.

`band_subtract` is the control: 12 registers, full occupancy, plainly
coalesced, and it runs at 97-103% of the copy roof on the same hardware. The
memory system is fine. Whatever slows up_conv belongs to up_conv.

### Four hypotheses, from reading the kernels

| Tag | Hypothesis | Where |
|---|---|---|
| E1 | Tile over-fetch | `UY = SP_BY + 2*SP_HALO + 2 = 14` |
| E2 | Warp misalignment | `UXW = SP_BX/2 + 2*SP_HALO + 4 = 24` |
| E3 | Integer modulo | `reflect1`, 5 calls per output element |
| E4 | Dead taps | `for k in 0..4 { if ((r & 1) == 0) ... }` |

**E1.** `up_conv_rows` loads a tile of 14 input rows to produce 8 output rows.
The bound was sized in output space. Because the input is half resolution, the
8 output rows of a block map to `src` in `[yo0/2 - 1, yo0/2 + 4]`, seven rows.
The source comment already admits the slack: "generous for pad=2."

**E2.** `up_conv_cols` tiles `[8][24]`. The load loop indexes `ly = i / 24`,
`lx = i % 24`, and 24 is not a multiple of the warp size. Lanes 0-23 read row
`y0`, lanes 24-31 read row `y0+1`, one `in_W` away. Every warp splits across
two rows.

**E3.** `reflect1` folds with `i % period`. GPUs have no hardware integer
divide, so each call is roughly 20-30 instructions. up_conv calls it five times
per output element in the tap loop, plus once per tile element on load.

**E4.** The tap loop runs all five taps and keeps the ones landing on an even
upsampled index. `reflect1`'s period is `2*(n-1)`, always even, so reflection
preserves parity: `(r & 1)` equals `((yo + k) & 1)`, which is known before the
loop starts. The predicate can be hoisted into the loop bounds. Three taps
survive when `yo` is even, two when it is odd.

### Method

One kernel body per kernel, parameterized by template argument, so a baseline
and a variant differ only in the knob under test. The repo kernel runs in the
same binary as a control; if its time does not match the copied baseline, the
copy is unfaithful and the comparison is void. It matched to within 0.2%. Every
variant's output is compared against the repo kernel's on device. All results
below are bit-exact, max absolute difference 0.0.

### Iterative measurement, RTX 3090

Finest level (L0), `M = 873` slices, medians of 11:

| Step | up_conv_rows | speedup | % roof | up_conv_cols | speedup | % roof |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 4.770 ms | 1.00x | 33.4% | 8.631 ms | 1.00x | 36.9% |
| a = E1 (tile) | 4.071 ms | 1.16x | 39.1% | 8.460 ms | 1.02x | 37.6% |
| b = E3 (reflect) | 3.584 ms | 1.31x | 44.4% | 7.713 ms | 1.12x | 41.2% |
| c = E4 (taps) | 4.438 ms | 1.07x | 35.9% | 7.402 ms | 1.17x | 43.0% |
| a + b | 3.177 ms | 1.48x | 50.5% | 7.041 ms | 1.23x | 45.7% |
| a + b + c | 2.807 ms | **1.70x** | 56.7% | 5.898 ms | **1.46x** | 54.0% |

### Iterative measurement, A100

| Step | up_conv_rows | speedup | % roof | up_conv_cols | speedup | % roof |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 4.990 ms | 1.00x | 15.5% | 7.125 ms | 1.00x | 21.7% |
| a = E1 (tile) | 4.225 ms | 1.18x | 18.3% | 6.974 ms | 1.02x | 22.2% |
| b = E3 (reflect) | 3.041 ms | 1.64x | 25.5% | 5.464 ms | 1.30x | 28.4% |
| c = E4 (taps) | 4.473 ms | 1.13x | 17.4% | 6.303 ms | 1.13x | 24.7% |
| a + b | 2.48-3.01 ms | 1.66-2.01x | 25.8% | 5.371 ms | 1.33x | 29.0% |
| a + b + c | 2.278 ms | **2.22x** | 34.1% | 4.833 ms | **1.48x** | 32.2% |

The `a + b` row on A100 measured 2.482 ms in one binary and 3.011 ms in
another that also carried the E4 code path. Same source for that
configuration, so the gap is compilation or scheduling, not a code difference.
The 3090 pair agreed (3.177 vs 3.153). The end-to-end trace below is the
number to trust.

E3 is the single largest knob on both GPUs, and it is larger on the A100
(1.64x vs 1.31x on rows). That fits: the A100 has proportionally more memory
bandwidth per integer ALU, so index math binds sooner.

### E2 was wrong

| Variant | 3090 | A100 |
|---|---:|---:|
| `UXW = 24` (baseline) | 1.00x | 1.00x |
| `UXW = 32` (warp-aligned) | 0.96x | 0.94x |
| `UXW = 20` (minimal) | 1.02x | 1.02x |

Padding the tile to a warp multiple makes it slower on both architectures. A
`[8][32]` tile loads 33% more elements than `[8][24]`, and the alignment saved
does not cover the extra fetch. Tile size matters more than tile alignment
here. E1 says the same thing from the other side: both wins came from fetching
fewer elements, not from fetching them more neatly.

It is worth recording as a failure, because warp-aligning the tile is the
obvious first thing to try.

### The production change

Three files: the two kernels, the shared helper, and the DESIGN.md entry that
records the numerical contract. A divide-free reflect that falls through to
the modulo version for anything more than a period out of range:

```cpp
__device__ __forceinline__ int reflect1_fast(int i, int n) {
    if (n == 1) return 0;
    if (i < 0) i = -i;                           // mirror about 0
    const int period = 2 * (n - 1);
    if (i >= n) i = period - i;                  // mirror about n-1
    if (i < 0 || i >= n) return reflect1(i, n);  // rare: >1 period out
    return i;
}
```

And in the two production up_conv kernels: `reflect1` to `reflect1_fast`, the
tap loop hoisted to even taps only, and both tile bounds rederived from the
block constants the way the rest of the file does it, so they still track
`SP_BX` / `SP_BY` if those ever move:

```cpp
constexpr int UY  = SP_BY / 2 + SP_HALO + 2;  // 8, was SP_BY + 2*SP_HALO + 2 = 14
constexpr int UXW = SP_BX / 2 + SP_HALO + 2;  // 20, was SP_BX/2 + 2*SP_HALO + 4 = 24
```

The bound halves because BY output rows span only BY/2 input rows when the
input is half resolution. The old bound counted in output space and fetched
about twice what it used.

```cpp
// (r & 1) == ((yo + k) & 1); parity is known before the loop
#pragma unroll
for (int k = (yo & 1); k < 5; k += 2) {
    int r = reflect1_fast(yo + (k - pad), up_H);
    int src = r >> 1;
    ...
}
```

The `*_halfacc` variants are left alone. They are experiment-only and off the
production path.

### Measured in the pipeline

Full motion pipeline, A100, before and after in the same job on the same node:

| Kernel | Before | After | Change |
|---|---:|---:|---|
| `up_conv_cols_batched` | 19.85 ms | 13.31 ms | **1.49x** |
| `up_conv_rows_batched` | 11.65 ms | 6.50 ms | **1.79x** |
| `corr_dn_fused_smem` | 5.174 ms | 5.173 ms | 1.00x |
| `band_subtract` | 4.347 ms | 4.346 ms | 1.00x |
| `band_add` | 4.313 ms | 4.316 ms | 1.00x |
| `iir_bandpass_tn` | 4.050 ms | 4.040 ms | 1.00x |
| `add_planar_quantize` | 2.564 ms | 2.551 ms | 1.00x |
| **Total GPU kernel time** | **55.55 ms** | **43.80 ms** | **1.27x** |

The untouched kernels stayed within 0.1%. That is the control: the difference
comes from the patch and not from the state of the node.

Test suite: 95 of 95 green on the patched build.

### Honest read

Even with all three changes, up_conv sits at 32-34% of the roof on the A100
and 54-57% on the 3090. E1, E3 and E4 together explain roughly half the
original gap. The rest is unexplained, and naming it needs the NCU counters
neither host will collect.

Two limits on the numbers. The 3090 was measured at kernel level, not end to
end, so the production tables in this post are unchanged. And absolute
milliseconds moved between sessions (an earlier trace of the same unpatched
build totalled 63.69 ms against 55.55 ms here), so the before/after ratio is
the measurement and the absolutes are not comparable across runs.

### FP16 goes through the same body

Both precisions instantiate one template: `<float, float>` and
`<__half, __half>` of the same `up_conv_*_batched_kernel`. There is no
type-specific route in production, and no `if constexpr` was added. The
`_halfacc` kernels are separate non-template functions, experiment-only, and
deliberately untouched.

Measured on A100, same binary pair, orig against patched:

| Instantiation | Before | After | Speedup |
|---|---:|---:|---:|
| `up_conv_rows<float, float>` | 4.992 ms | 2.318 ms | 2.15x |
| `up_conv_rows<__half, __half>` | 5.056 ms | 2.363 ms | 2.14x |
| `up_conv_cols<float, float>` | 8.726 ms | 4.961 ms | 1.76x |
| `up_conv_cols<__half, __half>` | 7.323 ms | 4.970 ms | 1.47x |

Shared memory tracks the storage type as it did before, since the tile is
declared `__shared__ In tile[UY][SP_BX]`: rows 1792 to 1024 bytes in FP32 and
896 to 512 in FP16, cols 768 to 640 and 384 to 320. Registers are unchanged
(31-32), and `STACK` and `LOCAL` are 0 in all four instantiations, so the
even-tap loop still unrolls and the 5-tap filter array stays in registers
despite the runtime-dependent loop start.

FP16 moves half the bytes and takes the same time as FP32, before the patch
and after it. A DRAM-bound kernel would have halved. up_conv never was one,
which is why the fixes that worked were index math and fetch count rather than
anything about precision, and why both instantiations gain the same multiple
from one shared body.

---

## What production is after this series

| Piece | State |
|---|---|
| Stage buffers | Free-list `DeviceMemPool` (no hot-path `cudaFree`) |
| lpyr scratch | Sticky grow-only |
| Temporal IIR | Coalesced `(T,N)` + alpha scale; FP64 state |
| Planar / bands | Channel-outer; contiguous band write/add |
| Downsample | Smem fused cols then rows |
| Upsample | Separable rows then cols (fused tried and failed) |
| FP16 spatial | Same templates as FP32 (`__shared__ In tile`); half bands end-to-end |
| Color blur | First level reads caller input (no full-frame D2D into sticky) |
| File I/O | Still transfer/encode dominated |

### Cumulative compute (3090 baby)

| Milestone | Compute ms | vs series start |
|---|---:|---:|
| Start | ~934 | 1.0x |
| TN IIR | ~718 | ~1.3x |
| Sticky scratch | ~377-407 | ~2.3-2.5x |
| DeviceMemPool | ~113-133 | ~7-8x |
| Layout foundation | ~106-110 | ~8.5x |
| Smem fused down | ~96-104 | ~9x |
| True half bands + dense smem (remeasured) | **3090 90.4/75.1; A100 54.4/48.2; H100 35.8/34.5** | **~10× / ~12× on 3090** |

### Current production stage table (3090, baby motion)

Fresh process per config; 1 warmup + median of 7 (`benches/bench_rtx3090.json`).

| Stage | FP32 (ms) | FP16 (ms) | ratio |
|---|---:|---:|---:|
| A) NTSC | 2.9 | 1.7 | 0.59 |
| B) lpyr_build | 32.5 | 26.8 | 0.82 |
| C) IIR | 24.7 | 23.4 | 0.95 |
| D1) recon | 25.3 | 20.8 | 0.82 |
| D2) render | 5.0 | 2.4 | 0.48 |
| **Compute** | **90.4** | **75.1** | **0.83** |
| H2D+D2H | 113.1 | 104.4 | 0.92 |
| **TOTAL** | **203.5** | **179.5** | **0.88** |

Same-day color (fresh process each): face compute **10.1 → 7.8 ms** (0.77×);
baby color compute **15.8 → 12.2 ms** (0.77×). Accuracy: motion FP16 vs CUDA
FP32 RMSE **0.00232** / max **5** LSB; color face RMSE **0.00071** / max **1** LSB.

Cross-GPU remeasure (same code): A100 motion **54.4 → 48.2 ms**, color face
**8.8 → 8.2 ms** (`benches/bench_a100.json`); H100 motion **35.8 → 34.5 ms**,
color face **4.9 → 4.4 ms** (`benches/bench_h100.json`).

3090 throughput at the inference tier (②, upload included): motion FP16
~1.4 Gpx/s (~23 concurrent 1080p@30 streams), color face FP16 ~3.4 Gpx/s
(~55 streams). The compute-only ceiling (①) is higher: ~2.0 / ~30 motion
and ~12 / ~190 color. Not file-to-file (PCIe/codec bound). A100/H100 motion
FP16 ① ≈ 50 / 70 streams at 1080p@30 (same scaling).

A rough 4K compute-only motion stream count is ~1080p/4 (~8 on 3090 FP16 ①).
VRAM may force tiling on full-res long clips; whole-clip resident 4K motion
exceeds 24 GB without tiling.

---

## What we learned

1. Re-stage the table after every real fix. After sticky + TN, the next slow
   stages were still inflated by DeviceBuffer thrash, not 5-tap math.
2. Isolation vs pipeline is a first-class tool. Kernel at 93% BW and stage at
   60 ms means measure the allocator, not the matvec.
3. Same filter, fix layout beat inventing a new temporal algorithm for Stage C.
   TN IIR matches copy on clean runs.
4. Layout cleanups can be correct and still free. Channel-outer + contiguous
   band were good engineering; they were not another 3x.
5. Fuse the near-roof pass that still pays an intermediate buffer. Do not fuse
   the weak-access product that already lost once. Down yes; up no.
6. After ~100 ms compute, transfer owns the wall. Further mid-pipeline work is
   for purity and paper numbers, not for real-time file-to-file without
   NVDEC/NVENC / async H2D.
7. A failed fix can close a question it never answered. Fused up_conv lost
   twice, and up_conv stayed "weak access" in the bound table for the rest of
   the series. The separable kernels were still carrying a 14-row tile for 8
   rows of output, an integer modulo five times per output element, and a
   five-iteration loop that used two or three of those iterations. Fusion was
   never required to fix any of it.
8. Missing counters slowed the diagnosis but did not block it. Registers and
   shared memory come out of `cuobjdump -res-usage`, SM limits out of
   `cudaGetDeviceProperties`, kernel timings out of Nsight Systems, and one
   template argument per hypothesis covers the rest. NCU would have named the
   residual gap faster.

Harris's reduction paper still applies: performance is often about how the
kernel talks to the memory system. Here that meant warp-coalesced temporal
addresses, not throwing multi-GB scratch away every pyramid call, and not
malloc/freeing stage buffers every stage.

---

## Open surfaces

| Surface | Status |
|---|---|
| TN-coalesced IIR | Done |
| Sticky lpyr scratch | Done |
| DeviceBuffer free-list pool | Done |
| Channel-outer / contiguous band | Done (foundation; little wall win) |
| Smem fused downsample | Done (small win) |
| Smem / dense fused upsample | Ruled out (regression, twice) |
| up_conv tile bound / divide-free reflect / even-tap loop | Measured and validated; patch not yet landed |
| Warp-aligning the up_conv tile (`UXW = 32`) | Ruled out (slower on sm_80 and sm_86) |
| Remaining up_conv gap (still ~2/3 off roof on A100) | Open: needs NCU counters, blocked on both hosts |
| True half-band motion + dense smem templates | Done (production; ~0.84× FP32 on 3090) |
| Packed `__half2` / scalar half-acc spatial | Experiment-only; not production |
| Stage C multi-series IIR / CUDA Graphs | Open: kernel can sit near BW; stage is still ~9x3 launches |
| PCIe + encode | Open: product path, not mid-pipeline |
| Scan / blocked IIR | Still low ROI while TN IIR matches copy on clean runs |

---

## One-line summary

After the first CUDA port was correct and fast, motion on a 3090 was still
about 0.9 s of compute because Stage C walked memory uncoalesced and pyramid
calls threw multi-GB scratch away. Coalesced TN IIR and sticky scratch cut that
to about 0.4 s. A free-list pool then collapsed the remaining fake stage walls
to real kernel times (about another 4x). Channel-outer layout cleaned the band
bridge without buying much wall time. Smem fused downsample shaved a few more
milliseconds. Fused upsample regressed again, so production keeps separable up.
True half-band storage with dense half smem (same templates as FP32) then lands
near **90 ms FP32 / 76 ms FP16** mid-pipeline compute for baby.mp4 (~10-12x from
the start of this series). File-to-file is still dominated by transfer and encode.

[repo]: https://github.com/iamkucuk/eulerian-video-magnification-cuda
