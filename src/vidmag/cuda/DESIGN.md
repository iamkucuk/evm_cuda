# CUDA Port Design

This document records the kernel-by-kernel mapping from the Python baseline
(`vidmag.cpu`, at `src/vidmag/cpu/`) to the CUDA port (`cuda/` kernels behind the
`src/vidmag/cuda/` wrapper), the grid/block rationale, and the
precision choices behind the per-stage tolerances. It is the authoritative
reference for the numerical contract.

## Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Binding | Raw CUDA (`.cu`/`nvcc`) + pybind11 | Maximum control, no heavyweight framework dep |
| Deploy | Any machine with NVIDIA GPU + CUDA toolkit | Standard CMake + nvcc build (`make build`) |
| Mode | Batch — whole clip in device memory | Simplest, supports the FFT-based ideal filter natively |
| GPU | Portable: `sm_60 sm_70 sm_80 sm_89 sm_90` | One `.so` covers P100 through H100 |
| Precision | FP32 hot path (including the IIR state) + FP64 Butterworth accumulators + optional FP16 storage | FP32 matches Python tolerances; FP16 halves VRAM for memory-constrained GPUs |

Further mid-pipeline writeup:
[how it was made fast](../../../docs/internals/making-it-fast.md).

## Repository layout

```
cuda/
├── include/
│   ├── evm_common.cuh     # __constant__ arrays: BINOM5, BINOM5_SUM1,
│   │                      # NTSC matrices; reflect1() device helper; kDropLast,
│   │                      # kExaggerationFactor
│   └── evm_check.cuh      # CUDA_CHECK / CUFFT_CHECK macros (abort on error)
├── kernels/
│   ├── color_cvt.cu       # bgr_u8 <-> ntsc_f32 (per-pixel matvec + quantize)
│   ├── spatial.cu         # corr_dn / up_conv (single-slice + batched variants)
│   ├── transpose.cu       # (T,H,W,C) <-> (N,T), planar<->interleaved, scaled transpose
│   ├── iir_bandpass.cu    # per-pixel FP32-state r1/r2 recursion
│   ├── butter_bandpass.cu # 1st-order Butter via scipy coeffs (host)
│   ├── ideal_bandpass.cu  # cuFFT C2C batched + mask kernel + 1/T normalize
│   ├── lpyr.cu            # build/recon (single-slice) + contiguous band ops
│   ├── blur_dn.cu         # blur_dn (single-slice; batched variant is in bindings.cpp)
│   └── amplify_render.cu  # add+quantize, fused upsample+add, fused planar+add
├── bindings.cpp           # pybind11 module: per-kernel + batched_* wrappers,
│                          # cuFFT plan cache, batched lpyr/blur orchestration
└── CMakeLists.txt         # CUDA-optional; CUDAToolkit, pybind11. Driven by
                           # scikit-build-core from the root pyproject.toml
                           # (`pip install .`); setup.py is gone.

src/vidmag/cuda/              # Python wrapper package (it used to live in cuda/)
├── __init__.py            # lazy surface for the 4 magnify_* pipelines
├── runtime.py             # have_cuda probe, butter coeffs
├── pipelines.py           # non-batched magnify_* (ideal/butter motion pipelines)
├── batched.py             # optimized device-resident magnify_* (color/iir)
└── _vidmag_cuda*.so          # the compiled extension, written here by CMake
```

## Kernel-by-kernel mapping

**Single-slice kernels** (used by `pipelines.py` and unit tests):

| Python baseline | CUDA kernel | Grid / Block | Tolerance |
|---|---|---|---|
| `vidmag.rgb_to_yiq`, `_rgb_frame_to_ntsc` | `color_cvt.cu:bgr_u8_to_ntsc_f32_kernel` | `(⌈W/32⌉,⌈H/32⌉) / (32,32,1)` | `<1e-6` |
| `vidmag.yiq_to_rgb`, `_ntsc_to_bgr_uint8` | `color_cvt.cu:ntsc_f32_to_bgr_u8_kernel` | same | `<1e-6` (≤1 LSB on u8) |
| `vidmag.corr_dn_axis` (axis=0) | `spatial.cu:corr_dn_rows_kernel` | `(⌈W/32⌉,⌈Ho/32⌉) / (32,32,1)` | `<1e-5` |
| `vidmag.corr_dn_axis` (axis=1) | `spatial.cu:corr_dn_cols_kernel` | `(⌈Wo/32⌉,⌈H/32⌉) / (32,32,1)` | `<1e-5` |
| `vidmag.up_conv_axis` (axis=0) | `spatial.cu:up_conv_rows_kernel` | `(⌈W/32⌉,⌈outH/32⌉) / (32,32,1)` | `<1e-5` |
| `vidmag.up_conv_axis` (axis=1) | `spatial.cu:up_conv_cols_kernel` | `(⌈outW/32⌉,⌈H/32⌉) / (32,32,1)` | `<1e-5` |
| `vidmag.build_lpyr` | `lpyr.cu:lpyr_build_device` (host loop) | per level → spatial kernels | `<1e-5` per band |
| `vidmag.recon_lpyr` | `lpyr.cu:lpyr_recon_device` (host loop) | per level → spatial kernels | `<1e-5` |
| `vidmag.blur_dn` | `blur_dn.cu:blur_dn_device` (host loop) | per level → corr_dn | `<1e-5` |
| `vidmag.iir_bandpass` | `iir_bandpass.cu:iir_bandpass_kernel` | `(⌈N/256⌉) / (256,1,1)` | `<1e-5` |
| `vidmag.butter_bandpass` | `butter_bandpass.cu:butter_bandpass_kernel` | same | `<1e-5` |
| `vidmag.ideal_bandpass` | `ideal_bandpass.cu` (3 sub-kernels + cuFFT) | mask: `(⌈TN/256⌉) / (256,1,1)` | `<1e-4` |
| `vidmag.figure6_alpha_schedule` | host-side, in `batched.py`/`pipelines.py` | n/a (small host array) | n/a |
| `vidmag._amplify_lpyr_stack` add+quantize | `amplify_render.cu:add_and_quantize_kernel` | `(⌈W/32⌉,⌈H/32⌉) / (32,32,1)` | `<1e-6` (≤1 LSB on u8) |

**Batched kernels** (used by `batched.py` — the optimized production path):

| Operation | CUDA kernel | Grid / Block | Notes |
|---|---|---|---|
| Smem fused corr_dn (down) | `spatial.cu:corr_dn_fused_smem_batched` | `(⌈Wo/32⌉,⌈Ho/8⌉,B) / (32,8,1)` | Production build/blur; cols→rows in smem |
| Separable up_conv | `spatial.cu:up_conv_{rows,cols}_batched` | `(⌈W/128⌉,⌈H/8⌉,B) / (32,8,1)` | Production recon/build up; fused up **not** wired. **No shared memory** — see below. Four outputs per thread, spaced one warp apart. `up_conv_cols_batched_combine` folds the band add/subtract into its store |
| Batched lpyr_build | `bindings.cpp:batched_lpyr_build` | host loop over levels | smem fused down + sep up (no smem), the subtract **fused into** the up_conv store |
| Batched lpyr_recon | `bindings.cpp:batched_lpyr_recon` | host loop over levels | sep up, the add **fused into** the up_conv store |
| Batched blur_dn | `bindings.cpp:batched_blur_dn_color` | host loop over nlevs | smem fused down; frame-major (color) or as called |
| Contiguous band ops | `lpyr.cu:band_subtract/band_add` | `(⌈n/256⌉) / (256,1,1)` | channel-outer affine; no offset table. Still used by the per-frame and half-accumulate paths; the batched FP32/FP16 pipelines now fuse this into up_conv |
| TN IIR (+scale) | `iir_bandpass.cu:iir_bandpass_tn_kernel` | `(⌈N/256⌉) / (256,1,1)` | motion Stage C; no transpose sandwich |
| Scaled transpose | `transpose.cu:nt_to_thwc_kernel` (+scale) | `(⌈N/256⌉) / (256,1,1)` | color / legacy layout helpers |
| Fused upsample+add+quant | `amplify_render.cu:upsample_add_quantize_kernel` | `(⌈MHW/256⌉) / (256,1,1)` | color pipeline render |
| Fused planar+add+quant | `amplify_render.cu:add_planar_quantize_kernel` | `(⌈W/32⌉,⌈H/32⌉,n) / (32,32,1)` | motion render; **channel-outer** delta |
| Planar 3ch (frame / chan-outer) | `transpose.cu:to_planar_3ch*` | `(⌈n*HW/256⌉) / (256,1,1)` | color: frame-major; motion: `*_chan_outer` |
| DeviceMemPool | `bindings.cpp:DeviceMemPool` | n/a | free-list by size for pipeline DeviceBuffers |
| Sticky lpyr scratch | `bindings.cpp:sticky_f*_slots` | n/a | grow-only scratch for build/recon/blur |
| cuFFT plan cache | `bindings.cpp:g_fft_cache` | n/a | keyed on (T,N) |
| FP16 conversion | `fp16_cvt.cu:f32_to_f16 / f16_to_f32` | `(⌈n/256⌉) / (256,1,1)` | residual casts (e.g. color gdown→FFT); motion bands stay half end-to-end |

## FP16 storage rationale

Both pipelines support an FP16 storage path (`magnify_color_gdown_ideal_fp16`
and `magnify_motion_lpyr_iir_fp16` in `batched.py`). Batched spatial kernels are
templated on storage type `In`/`Out`. Where a kernel does stage data — the
downsample kernels; the enlargement kernels no longer do, see below — the tile is
declared `__shared__ In tile`, so half storage stays dense in it; MAC is float
via `cvt_in`/`cvt_out` at the arithmetic edge. Instantiating `<__half,__half>` is the production FP16 path —
same code as FP32, not a forked algorithm.

**Motion FP16:** NTSC, planar, bands, filtered bands, and delta are `__half`
end-to-end (no full float band stack). Stage A is fused `u8→YIQ→half`. Peak
VRAM on baby.mp4 (301 frames, 544x960) measures 8.4 GB, against 16.3 GB for
FP32, so FP16 motion fits a 16 GB card and FP32 motion does not. Fresh remeasure (1 warmup + median of 7):

| GPU | Motion FP32 | Motion FP16 | ratio |
|---|---:|---:|---:|
| RTX 3090 | **90.4 ms** | **75.1 ms** | **0.83×** |
| A100 80GB | **54.4 ms** | **48.2 ms** | **0.89×** |
| H100 80GB | **35.8 ms** | **34.5 ms** | **0.96×** |

Accuracy vs CUDA FP32 (baby): RMSE **0.00140**, max **2** LSB (re-measured 2026-08-18).

The three timing rows above predate three later rounds of work on the motion path — the
up_conv retune, then the FP32 IIR state with the band combine folded into the up_conv
store, then shared memory taken out of the two enlargement kernels. They are kept as the
record of that measurement. Current figures, both re-measured on the branch and stored
with the commit they were taken at: RTX 3090 **40.3 / 26.8 ms**
(`benches/bench_rtx3090.json`, 2026-08-18) and P100 **does not fit / 82.8 ms**
(`benches/bench_p100.json`, 2026-08-22). The A100 and H100 have not been re-run.

**Color FP16:** NTSC + planar blur scratch are `__half`. Final Gaussian gdown
converts to FP32 for cuFFT; `filt` stays FP32. First blur level reads the
caller's planar input (no full-frame D2D copy into sticky scratch). Fresh
remeasure:

| GPU / clip | Color FP32 | Color FP16 | ratio |
|---|---:|---:|---:|
| P100 / face | **26.3 ms** | **21.8 ms** | **0.83×** |
| 3090 / face | **10.1 ms** | **7.8 ms** | **0.77×** |
| A100 / face | **8.8 ms** | **8.2 ms** | **0.93×** |
| H100 / face | **4.9 ms** | **4.4 ms** | **0.90×** |
| 3090 / baby | **15.8 ms** | **12.2 ms** | **0.77×** |

Accuracy vs CUDA FP32 (face): RMSE **0.00071**, max **1** LSB.
Source: `benches/bench_rtx3090.json`, `benches/bench_a100.json`,
`benches/bench_h100.json`, `benches/bench_p100.json`.
Re-measured on the P100 on 2026-08-22, on current code: color **26.4 / 21.9 ms**,
against **26.3 / 21.8 ms** in the row above. Colour is unchanged, as it must be —
it builds no Laplacian pyramid, so none of the three motion-path changes reaches
it, which is what makes the pre- and post-change rows in this table comparable.
Motion FP16 on the same run is **82.8 ms**, down from 139.7 ms; motion FP32 needs
16.3 GB and still does not fit a 16 GB card, and the harness reports the skip.

The device pool holds released blocks until process exit and reuses one only on
an exact byte-size match, so several differently sized configs in one process
used to accumulate every footprint. `benchmark.run` now calls
`_vidmag_cuda.free_device_pool()` from a `finally`, which also covers the OOM
paths. Two things came out of that: configs that used to die with a real
`cudaMalloc` failure now run, and the timings stopped being distorted. With the
3090 down to 0.03 GB free, the last config measured 302 ms of compute against
~59 ms on a clean card.

The multi-config harness now reproduces the fresh-process numbers in
`benches/bench_rtx3090.json`. Median of three in-process runs, 2026-08-02,
against the stored record:

| Config | Stored (fresh process) | In-process, remeasured | Delta |
|---|---:|---:|---:|
| motion FP32 | 75.36 ms | 74.89 ms | -0.6% |
| motion FP16 | 60.39 ms | 58.95 ms | -2.4% |
| color face FP32 | 9.74 ms | 9.54 ms | -2.1% |

The stored numbers stand; this is a confirmation, not a remeasure to publish.
Color face FP16 is left out on purpose: on this host it is bimodal, three
standalone repeats gave 7.33 / 11.82 / 11.70 ms against a stored 7.63 ms, so a
single sample proves nothing. Anything measured here needs several repeats.

## Why the enlargement kernels use no shared memory

The two enlargement kernels (`up_conv_rows_batched`, `up_conv_cols_batched`)
staged their input in `__shared__` tiles, as the downsample kernel beside them
still does. That was measured and reversed: for these two kernels the staging
cost more than it saved.

The reason is the access pattern. An enlargement output reads only two or three
inputs — the 5-tap filter is halved by parity, since only taps landing on an
even upsampled index survive — and neighbouring threads read overlapping
inputs. The cache already serves that overlap. Staging it in shared memory buys
nothing, and costs a barrier plus a load loop in which the tile shape rather
than the warp decides the access: the column kernel's 20-wide tile left 96 of
256 threads idle during the load and split each warp's reads across two short
unaligned segments.

Measured on an RTX 3090 at the largest pyramid level of baby.mp4
(291 frames, 544x960, 3 channels), against a measured read+write ceiling of
863 GB/s. All variants produce bit-identical output.

| Variant | Enlarge rows | Enlarge columns |
|---|---:|---:|
| Shared tile, one output per thread (the old form) | 474 GB/s (55%) | 471 GB/s (55%) |
| No shared memory, one output per thread | 720 GB/s (83%) | 671 GB/s (78%) |
| Shared tile, four outputs per thread | — | 683 GB/s (79%) |
| No shared memory, four outputs adjacent, 16-byte stores | 862 GB/s (100%) | 864 GB/s (100%) |
| **No shared memory, four outputs one warp apart (shipped)** | **825 GB/s (96%)** | **797 GB/s (92%)** |

The adjacent-output form with 16-byte stores is the fastest, and is not what
ships. It requires the width to be divisible by four, which two pyramid levels
of a 544x960 clip are not (widths 15 and 30), so it would need a second kernel
and a scalar fallback for those levels, plus separate treatment for `__half`.
The shipped form spaces each thread's four outputs one warp apart, so every
store from a warp is still 32 consecutive values, no alignment is required, and
one kernel serves every level and both storage types. That trades roughly six
percentage points for not having three kernels where one will do.

The downsample kernel keeps its shared tile: it reads a 2x2 neighbourhood plus
halo per output and genuinely reuses staged data, and it measures 94% of the
ceiling as it stands.

## Precision rationale

The Python baseline uses FP64 in `pyramids.py` and `filters.py` (its
round-trip is `<1e-9`) but FP32 in `video.py` and the color pipeline. The
CUDA port uses FP32 throughout the hot path with two specific FP64
exceptions:

1. **Butterworth accumulators** (`yh_prev`/`yl_prev` in
   `butter_bandpass_kernel`) are FP64 in registers. Rationale: a length-300
   temporal recursion accumulates floating-point error proportional to
   `sqrt(T) · eps`; FP32 `eps ≈ 1.2e-7` predicts ~2e-6 worst-case, which would
   eat most of the `<1e-5` budget. Arrays stay FP32 — only the running state
   is FP64.

   **The IIR kernels are the measured exception to that reasoning.** `y1` and
   `y2` in `iir_bandpass_kernel` and `iir_bandpass_tn_kernel` were FP64 on the
   same argument, and the argument turned out to be too pessimistic for this
   filter: the r1/r2 form is a pair of leaky running averages, and a leaky
   average forgets its own error rather than accumulating it, so the drift
   never approaches the `sqrt(T)` bound. Measured over the dominant pyramid
   level of baby.mp4 (T=291, 960x544), the largest difference between an FP32
   state and an FP64 one is **4.023e-07**, against a `<1e-5` budget.

   The reason to care is that FP64 is not free on the hardware this targets: a
   GeForce card runs double-precision arithmetic at a sixty-fourth of its
   single-precision rate, so the kernel was arithmetic-bound rather than
   memory-bound. Same kernel, same access pattern, only the state type
   changed:

   | State | Time | Effective bandwidth |
   |---|---:|---:|
   | FP64 | 4.95 ms | 246 GB/s |
   | FP32 | 1.50 ms | 809 GB/s |

   Peak on that card is 936 GB/s, so the FP32 form is memory-bound, which is
   where a filter this simple should be. In the full motion pipeline the stage
   went from 24.65 ms to 6.36 ms. Butterworth keeps its FP64 state: it is a
   true recursion with feedback on its own output, the argument above does
   apply to it, and it is not on the hot path.

2. **Ideal bandpass** (`cufftComplex` = `float2`). cuFFT's FP32 plan vs
   numpy's FP64 FFT is the reason this stage has the looser `<1e-4`
   tolerance. If a tighter tolerance is ever required, switch to a
   `CUFFT_Z2Z` double-precision plan (drop-in via `runtime.py`).

## reflect1 helper

The single most tolerance-critical piece. `vidmag::reflect1(i, n)` in
`evm_common.cuh` reproduces numpy's `mode='reflect'` (== MATLAB `reflect1`):
half-sample symmetric reflection without duplicating the edge sample.

```cpp
__device__ int reflect1(int i, int n) {
    if (n == 1) return 0;
    const int period = 2 * (n - 1);
    if (i < 0) i = -i;             // reflection is symmetric about 0
    if (i >= period) i %= period;  // skipped on the hot path
    if (i >= n) i = period - i;
    return i;
}
```

This form is there for speed, not clarity. The GPU has no hardware
integer divide, so `i % period` costs ~20-30 instructions, and `up_conv`
called this five times per output element. Spatial callers never reach more
than one period past an edge, so the modulo is skipped for them; it stays
in the general path so the function's contract is unchanged for any input.
Measured 1.3-1.6x on `up_conv` alone.

The even period also means reflection preserves parity, which `up_conv`
uses to hoist its tap predicate: `(r & 1)` equals `((yo + k) & 1)` and is
known before the loop runs.

Any mistake here propagates into every pyramid band and the Laplacian
round-trip. Verified against numpy's behaviour in `tests/cuda/test_spatial.py`
indirectly (the per-band `<1e-5` assertions fail immediately if reflection
is off-by-one).

## Layout choice: motion bands vs color FFT

**Motion (production IIR):** Laplacian bands are stored **channel-outer**
`(level, channel, frame, spatial)` so each `(level, channel)` block is
contiguous **`(T, N)`**. Stage C runs `batched_iir_bandpass_tn` **in place**
(addr `t*N+n`) with alpha folded into the write scale — **no**
`thwc_to_nt` / `nt_to_thwc` sandwich.

Spatial scratch during build/recon is also channel-outer after
`batched_to_planar_3ch_chan_outer` (`m' = c*n + f`), so band writes are
contiguous, not irregular scatter. That contiguity is what lets the batched
pipelines fold the band subtract/add into the `up_conv_cols` store instead of
running it as a second pass: the band a level combines with has exactly the
index and stride the up_conv is already writing.

**Color (cuFFT ideal bandpass):** still uses `(N,T)` via
`thwc_to_nt` / plan-many with `istride=1, idist=T` — fastest batched 1-D
cuFFT layout. That path is separate from motion Stage C.

Alternative rejected for motion IIR: keep frame-major bands and walk
strided `n*T+t` — uncoalesced; measured ~12× worse than TN on probes.

## Pipeline composition

Two pipeline implementations exist:

- **`pipelines.py`** — the non-batched reference path (per-frame H2D/D2H per
  binding call). Used for `magnify_motion_lpyr_ideal` and
  `magnify_motion_lpyr_butter` (which `batched.py` doesn't implement).
  Matches `src/vidmag/cpu/magnify.py` line-for-line.
- **`batched.py`** — the optimized device-resident path for
  `magnify_color_gdown_ideal` and `magnify_motion_lpyr_iir`. Upload once,
  keep data on-device through all stages (batched spatial kernels, on-device
  transpose+IIR, fused render), download only the final uint8 output.

What's on-device vs on-host (batched.py):

| Step | Where | Why |
|---|---|---|
| Frame read, drop-last-10, fps | Host | I/O-bound, OpenCV VideoCapture |
| NTSC convert | Device (batched) | Per-pixel matvec, all frames at once |
| Pyramid build/recon, blur_dn | Device (batched spatial) | smem fused down + sep up; sticky scratch |
| Temporal filter (motion) | Device | TN IIR in place on channel-outer bands; alpha as scale |
| Temporal filter (color) | Device | planar + cuFFT ideal on `(N,T)` |
| Figure-6 schedule | Host | Small `n_levels`-length float array |
| Fused render | Device | planar add+quant (motion) / upsample+add+quant (color) |
| Video encode | Host | PyAV (libx264) |

The color bandpass (Stage 2b) is fully device-resident: the previous host
round-trip (downsampled clip D2H + reshape for the per-channel
ideal_bandpass) is replaced by a single unified cuFFT call over
`(N=hl*wl*3, T=n)` plus device-resident transpose/gain kernels. Both
pipelines now run device-resident through all intermediate stages; the only
remaining host round-trips are the input clip H2D at entry and the uint8
output D2H at exit.

## Known divergences from MATLAB (intentional)

These are documented per the "CUDA matches Python, not MATLAB"
rule. The Python baseline is the oracle.

1. **Color pipeline upsample** uses `cv2.INTER_LINEAR` (half-pixel-centered
   bilinear), same as the Python baseline's choice at
   `src/vidmag/cpu/magnify.py:191`.
   MATLAB's `imresize` uses a different grid; this is a Python-baseline
   choice we inherit, not a CUDA choice.

2. **uint8 rounding** uses CUDA's `rintf` (round-half-to-even by default),
   matching `numpy.round`. Verified in `tests/cuda/test_color_cvt.py` to
   within ≤1 LSB.

3. **cuFFT plan lifecycle**: plans are cached by `(T, N)` in `bindings.cpp`'s
   `g_fft_cache`. The first call creates the plan; subsequent calls (same
   clip dimensions) reuse it. This eliminates the ~5-10ms autotuning cost
   per plan that the per-call lifecycle incurred.

## Validation strategy

1. **Build succeeds.** `make build` produces
   `src/vidmag/cuda/_vidmag_cuda*.so` (and installs it at `vidmag/cuda/` in the wheel).
2. **Each kernel matches the Python baseline within its tolerance.**
   `tests/cuda/test_*.py` (36 tests across 9 test files).
3. **End-to-end pipelines match the Python baseline within `<0.01` RMSE** on
   synthetic clips and on `face.mp4` / `baby.mp4` (`test_pipelines.py`).
4. **Python baseline still matches MIT.** The existing
   `tests/test_against_mit_reference.py` (unchanged) confirms the oracle
   itself hasn't drifted.
