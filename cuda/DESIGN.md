# CUDA Port Design

This document records the kernel-by-kernel mapping from the Python baseline
(`evm/`) to the CUDA port (`cuda/`), the grid/block rationale, and the
precision choices behind the per-stage tolerances. It is the written record
AGENTS.md §2 requires for any numerical-contract decision.

## Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Binding | Raw CUDA (`.cu`/`nvcc`) + pybind11 | Maximum control, no heavyweight framework dep |
| Deploy | TRUBA ARF-ACC (`ssh truba`, GPU queues) | Has internet on login nodes, dedicated GPU partitions |
| Mode | Batch — whole clip in device memory | Simplest, supports the FFT-based ideal filter natively |
| GPU | Portable: `sm_60 sm_70 sm_80 sm_89 sm_90` | One `.so` covers TRUBA's whole fleet (P100→H100) |
| Precision | FP32 hot path + FP64 IIR accumulators + optional FP16 storage | FP32 matches Python tolerances; FP16 halves VRAM for memory-constrained GPUs |

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
│   ├── iir_bandpass.cu    # per-pixel FP64-state r1/r2 recursion
│   ├── butter_bandpass.cu # 1st-order Butter via scipy coeffs (host)
│   ├── ideal_bandpass.cu  # cuFFT C2C batched + mask kernel + 1/T normalize
│   ├── lpyr.cu            # build/recon (single-slice) + scatter/gather (batched)
│   ├── blur_dn.cu         # blur_dn (single-slice; batched variant is in bindings.cpp)
│   └── amplify_render.cu  # add+quantize, fused upsample+add, fused planar+add
├── bindings.cpp           # pybind11 module: per-kernel + batched_* wrappers,
│                          # cuFFT plan cache, batched lpyr/blur orchestration
├── evm_cuda/              # Python wrapper package
│   ├── __init__.py        # lazy surface for the 4 magnify_* pipelines
│   ├── runtime.py         # have_cuda probe, butter coeffs
│   ├── pipelines.py       # non-batched magnify_* (ideal/butter motion pipelines)
│   └── batched.py         # optimized device-resident magnify_* (color/iir)
├── CMakeLists.txt         # enable_language(CUDA), CUDAToolkit, pybind11
└── setup.py               # pip-installable; shells out to CMake
```

## Kernel-by-kernel mapping

**Single-slice kernels** (used by `pipelines.py` and unit tests):

| Python baseline | CUDA kernel | Grid / Block | Tolerance |
|---|---|---|---|
| `evm.rgb_to_yiq`, `_rgb_frame_to_ntsc` | `color_cvt.cu:bgr_u8_to_ntsc_f32_kernel` | `(⌈W/32⌉,⌈H/32⌉) / (32,32,1)` | `<1e-6` |
| `evm.yiq_to_rgb`, `_ntsc_to_bgr_uint8` | `color_cvt.cu:ntsc_f32_to_bgr_u8_kernel` | same | `<1e-6` (≤1 LSB on u8) |
| `evm.corr_dn_axis` (axis=0) | `spatial.cu:corr_dn_rows_kernel` | `(⌈W/32⌉,⌈Ho/32⌉) / (32,32,1)` | `<1e-5` |
| `evm.corr_dn_axis` (axis=1) | `spatial.cu:corr_dn_cols_kernel` | `(⌈Wo/32⌉,⌈H/32⌉) / (32,32,1)` | `<1e-5` |
| `evm.up_conv_axis` (axis=0) | `spatial.cu:up_conv_rows_kernel` | `(⌈W/32⌉,⌈outH/32⌉) / (32,32,1)` | `<1e-5` |
| `evm.up_conv_axis` (axis=1) | `spatial.cu:up_conv_cols_kernel` | `(⌈outW/32⌉,⌈H/32⌉) / (32,32,1)` | `<1e-5` |
| `evm.build_lpyr` | `lpyr.cu:lpyr_build_device` (host loop) | per level → spatial kernels | `<1e-5` per band |
| `evm.recon_lpyr` | `lpyr.cu:lpyr_recon_device` (host loop) | per level → spatial kernels | `<1e-5` |
| `evm.blur_dn` | `blur_dn.cu:blur_dn_device` (host loop) | per level → corr_dn | `<1e-5` |
| `evm.iir_bandpass` | `iir_bandpass.cu:iir_bandpass_kernel` | `(⌈N/256⌉) / (256,1,1)` | `<1e-5` |
| `evm.butter_bandpass` | `butter_bandpass.cu:butter_bandpass_kernel` | same | `<1e-5` |
| `evm.ideal_bandpass` | `ideal_bandpass.cu` (3 sub-kernels + cuFFT) | mask: `(⌈TN/256⌉) / (256,1,1)` | `<1e-4` |
| `evm.figure6_alpha_schedule` | host-side, in `batched.py`/`pipelines.py` | n/a (small host array) | n/a |
| `evm._amplify_lpyr_stack` add+quantize | `amplify_render.cu:add_and_quantize_kernel` | `(⌈W/32⌉,⌈H/32⌉) / (32,32,1)` | `<1e-6` (≤1 LSB on u8) |

**Batched kernels** (used by `batched.py` — the optimized production path):

| Operation | CUDA kernel | Grid / Block | Notes |
|---|---|---|---|
| Batched corr_dn/up_conv | `spatial.cu:*_batched_kernel` | `(⌈W/32⌉,⌈Ho/32⌉,B) / (32,32,1)` | B=M slices via grid.z; identical per-thread math |
| Batched lpyr_build | `bindings.cpp:batched_lpyr_build` | host loop over levels | scatter_subtract for channel-major band writes |
| Batched lpyr_recon | `bindings.cpp:batched_lpyr_recon` | host loop over levels | gather/gather_add for channel-major band reads |
| Batched blur_dn | `bindings.cpp:batched_blur_dn_color` | host loop over nlevs | frame-major output, no scatter needed |
| Scatter/gather | `lpyr.cu:scatter_subtract/gather/gather_add/scatter` | `(⌈n/256⌉,B) / (256,1,1)` | bridges frame-major scratch ↔ channel-major bands |
| Scaled transpose | `transpose.cu:nt_to_thwc_kernel` (+scale param) | `(⌈N/256⌉) / (256,1,1)` | folds alpha amplification into transpose |
| Fused upsample+add+quant | `amplify_render.cu:upsample_add_quantize_kernel<NTSC_T>` | `(⌈MHW/256⌉) / (256,1,1)` | color pipeline render; templated on NTSC type (float/__half); filt stays float* (FFT output) |
| Fused planar+add+quant | `amplify_render.cu:add_planar_quantize_kernel<NTSC_T>` | `(⌈W/32⌉,⌈H/32⌉,n) / (32,32,1)` | motion pipeline render; templated on NTSC type |
| cuFFT plan cache | `bindings.cpp:g_fft_cache` | n/a | keyed on (T,N); eliminates per-call plan creation |
| V6 multiple elements/thread | render + transpose kernels | 4 px/thread via `#pragma unroll` | pipelines independent reads for latency hiding (22% render) |
| FP16 storage (both pipelines) | All batched kernels templated on In/Out type | `cvt_in`/`cvt_out` in evm_common.cuh | __half storage, FP32 compute, FP64 IIR accumulator unchanged |
| FP16 blur_dn_color | `bindings.cpp:batched_blur_dn_color_f16` | host loop over nlevs | reads __half NTSC planar, downsamples in FP16 scratch, converts to FP32 for FFT |
| FP16 conversion | `fp16_cvt.cu:f32_to_f16 / f16_to_f32` | `(⌈n/256⌉) / (256,1,1)` | one-time conversion at NTSC creation boundary |

## FP16 storage rationale

Both pipelines support an FP16 storage path (`magnify_color_gdown_ideal_fp16`
and `magnify_motion_lpyr_iir_fp16` in `batched.py`). All batched spatial,
transpose, IIR, and render kernels are templated on input/output type.
When instantiated with `__half`, reads convert via `__half2float` and
writes via `__float2half`. Compute stays FP32 throughout.

**Motion FP16:** Stores NTSC, planar, bands, filtered bands, and delta all
in `__half`. Halves VRAM (23 GB to 12 GB for baby.mp4) and halves the
render stage's memory traffic (82 ms to 45 ms on A100, 8.6 ms to 5.6 ms on
P100). The IIR accumulator stays FP64 regardless of storage type.

**Color FP16:** Stores NTSC as `__half` (the dominant persistent buffer
read by render). The Gaussian downsample output goes to FP32 (cuFFT
bandpass needs float). The `filt` signal (FFT output) stays FP32. Only the
NTSC buffer is halved.

Precision: RMSE between FP32 and FP16 output is 0.0016 for motion, which
is 6.2x under the 0.01 end-to-end tolerance. The maximum per-pixel error
is 3/255 (3 uint8 quantization steps). For color FP16, the uint8 output
differs from FP32 by at most 2 LSB per channel.

**FP16 color is GPU-dependent.** The color render kernel reads 15 values
per output pixel (3 NTSC + 12 bilinear filt taps). FP16 NTSC reduces total
traffic by only 10% (filt stays FP32). On the A100 (1935 GB/s bandwidth),
this is invisible and conversion overhead makes FP16 slower. On the P100
(732 GB/s), the 10% traffic reduction is measurable and FP16 is 13% faster.

## Precision rationale

The Python baseline uses FP64 in `pyramids.py` and `filters.py` (its
round-trip is `<1e-9`) but FP32 in `video.py` and the color pipeline. The
CUDA port uses FP32 throughout the hot path with two specific FP64
exceptions:

1. **IIR/Butter accumulators** (`y1`, `y2` in `iir_bandpass_kernel`,
   `yh_prev`/`yl_prev` in `butter_bandpass_kernel`) are FP64 in registers.
   Rationale: a length-300 temporal recursion accumulates floating-point
   error proportional to `sqrt(T) · eps`; FP32 `eps ≈ 1.2e-7` would yield
   ~2e-6 worst-case, eating most of the `<1e-5` budget. FP64 keeps the
   accumulator drift well under `1e-7`, leaving headroom for the per-step
   rounding. Arrays stay FP32 — only the running state is FP64.

2. **Ideal bandpass** (`cufftComplex` = `float2`). cuFFT's FP32 plan vs
   numpy's FP64 FFT is the reason this stage has the looser `<1e-4`
   tolerance. If a tighter tolerance is ever required, switch to a
   `CUFFT_Z2Z` double-precision plan (drop-in via `runtime.py`).

## reflect1 helper

The single most tolerance-critical piece. `evm::reflect1(i, n)` in
`evm_common.cuh` reproduces numpy's `mode='reflect'` (== MATLAB `reflect1`):
half-sample symmetric reflection without duplicating the edge sample.

```cpp
__device__ int reflect1(int i, int n) {
    if (n == 1) return 0;
    const int period = 2 * (n - 1);
    i = i % period;  if (i < 0) i += period;
    if (i >= n) i = period - i;
    return i;
}
```

Any mistake here propagates into every pyramid band and the Laplacian
round-trip. Verified against numpy's behaviour in `tests/cuda/test_spatial.py`
indirectly (the per-band `<1e-5` assertions fail immediately if reflection
is off-by-one).

## Layout choice: (T,H,W,C) ↔ (N,T)

The video arrives as `(T,H,W,C)` row-major (T-stride = `H*W*C`). The
temporal filters want each spatial location's length-T series contiguous, so
we transpose to `(N,T)` with `N = H*W*C` before the filter kernel, and back
after. (`transpose.cu`).

Alternative considered: leave `(T,H,W,C)` and let the IIR kernel do strided
T-access. Rejected — strided reads along T are uncoalesced and the
bandwidth hit dwarfs the transpose cost.

For cuFFT, the `(N,T)` layout maps directly onto `cufftPlanMany` with
`istride=1, idist=T` — exactly the fastest cuFFT configuration for batched
1-D transforms.

## Pipeline composition

Two pipeline implementations exist:

- **`pipelines.py`** — the non-batched reference path (per-frame H2D/D2H per
  binding call). Used for `magnify_motion_lpyr_ideal` and
  `magnify_motion_lpyr_butter` (which `batched.py` doesn't implement).
  Matches `evm/magnify.py` line-for-line.
- **`batched.py`** — the optimized device-resident path for
  `magnify_color_gdown_ideal` and `magnify_motion_lpyr_iir`. Upload once,
  keep data on-device through all stages (batched spatial kernels, on-device
  transpose+IIR, fused render), download only the final uint8 output.

What's on-device vs on-host (batched.py):

| Step | Where | Why |
|---|---|---|
| Frame read, drop-last-10, fps | Host | I/O-bound, OpenCV VideoCapture |
| NTSC convert | Device (batched) | Per-pixel matvec, all frames at once |
| Pyramid build/recon, blur_dn | Device (batched spatial kernels) | grid.z = M slices per launch |
| Temporal filter | Device | On-device transpose + IIR, alpha folded into transpose |
| Figure-6 schedule | Host | Small `n_levels`-length float array |
| Fused render (upsample/planar + add + quant) | Device | Eliminates intermediate buffers |
| Video encode | Host | OpenCV VideoWriter |

The only remaining host round-trip in the color pipeline is Stage 2b
(downsampled clip D2H + reshape for the per-channel ideal_bandpass). The
motion pipeline is fully device-resident through Stages A–D.

## Known divergences from MATLAB (intentional)

These are documented per AGENTS.md §1's "CUDA matches Python, not MATLAB"
rule. The Python baseline is the oracle.

1. **Color pipeline upsample** uses `cv2.INTER_LINEAR` (half-pixel-centered
   bilinear), same as the Python baseline's choice at `evm/magnify.py:191`.
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

1. **Build succeeds on TRUBA.** `bash deploy/build.sh` produces
   `cuda/evm_cuda/_evm_cuda.so`.
2. **Each kernel matches the Python baseline within its tolerance.**
   `tests/cuda/test_*.py` (48 tests across 8 test files).
3. **End-to-end pipelines match the Python baseline within `<0.01` RMSE** on
   synthetic clips and on `face.mp4` / `baby.mp4` (`test_pipelines.py`).
4. **Python baseline still matches MIT.** The existing
   `tests/test_against_mit_reference.py` (unchanged) confirms the oracle
   itself hasn't drifted.
