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
| **Current production (remeasured)** | **3090: 90.4 / 75.1 ms; H100: 35.8 / 34.5 ms (FP32/FP16)** |

H2D/D2H and encode are reported, but they were not the target. After this
series, transfer often exceeds mid-pipeline compute on file-to-file runs, so
product latency is a different problem (NVDEC/NVENC, async H2D).

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
| True half bands + dense smem (remeasured) | **3090 90.4/75.1; H100 35.8/34.5** | **~10× / ~12× on 3090** |

### Current production stage table (3090, baby motion)

Fresh process per config; 1 warmup + median of 7 (`output/bench_osiris_3090.json`).

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

H100 remeasure (same code, `output/bench_truba_h100.json`): motion compute
**35.8 → 34.5 ms**; color face **4.9 → 4.4 ms**.

A rough 4K compute-only FPS estimate (pixel-scale from ~90 ms / 291 frames on
3090 FP32) lands around 190-200 FPS of pure mid-pipeline. VRAM may force tiling
on full-res long clips; whole-clip resident 4K motion exceeds 24 GB without tiling.

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
