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
| Precision | FP32 hot path + FP64 IIR accumulators | Matches Python baseline's per-stage tolerances |

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
│   ├── spatial.cu         # corr_dn / up_conv (separable, reflect1, 5-tap)
│   ├── transpose.cu       # (T,H,W,C) <-> (N,T) for temporal coalescing
│   ├── iir_bandpass.cu    # per-pixel FP64-state r1/r2 recursion
│   ├── butter_bandpass.cu # 1st-order Butter via scipy coeffs (host)
│   ├── ideal_bandpass.cu  # cuFFT C2C batched + mask kernel + 1/T normalize
│   ├── lpyr.cu            # build_lpyr / recon_lpyr (host loop over levels)
│   ├── blur_dn.cu         # blur_dn (host loop of corr_dn)
│   └── amplify_render.cu  # apply_channel_gain, attenuate_chrom, add+quantize
├── bindings.cpp           # pybind11 module: thin per-kernel wrappers
├── evm_cuda/              # Python wrapper package
│   ├── __init__.py        # lazy surface for the 4 magnify_* pipelines
│   ├── runtime.py         # have_cuda probe, butter coeffs, to_contiguous_f32
│   └── pipelines.py       # the 4 magnify_* orchestrators
├── CMakeLists.txt         # enable_language(CUDA), CUDAToolkit, pybind11
└── setup.py               # pip-installable; shells out to CMake
```

## Kernel-by-kernel mapping

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
| `evm.figure6_alpha_schedule` | host-side, in `pipelines.py` | n/a (small host array) | n/a |
| `evm._amplify_lpyr_stack` add+quantize | `amplify_render.cu:add_and_quantize_kernel` | `(⌈W/32⌉,⌈H/32⌉) / (32,32,1)` | `<1e-6` (≤1 LSB on u8) |

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

The four magnify pipelines (`magnify_color_gdown_ideal`,
`magnify_motion_lpyr_ideal`/`_butter`/`_iir`) live in
`cuda/evm_cuda/pipelines.py`. They mirror the structure of
`evm/magnify.py` line-for-line: frame read → drop-last-10 → NTSC convert →
pyramid build → temporal filter → amplify → recon → add → quantize.

What's on-device vs on-host:

| Step | Where | Why |
|---|---|---|
| Frame read, drop-last-10, fps | Host | I/O-bound, OpenCV VideoCapture |
| NTSC convert | Device | Per-pixel matvec |
| Pyramid build/recon, blur_dn | Device (host-orchestrated) | Each level = spatial kernel launch |
| Temporal filter | Device | The hot per-pixel loop |
| Figure-6 schedule | Host | Small `n_levels`-length float array |
| Add + quantize + clip | Device | Per-pixel |
| Video encode | Host | OpenCV VideoWriter |

The host orchestration does mean per-frame device↔host round-trips for the
pyramid bands (each frame's pyramid is copied back, stacked, then re-uploaded
for the temporal filter). A follow-up optimization can stage whole-clip
pyramid stacks on-device and avoid the round-trip; this is a perf
optimization, not a correctness concern, and is left for after the
validation lands.

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

3. **cuFFT plan lifecycle** is per-call (created and destroyed inside
   `_evm_cuda.ideal_bandpass`). This is acceptable for the accuracy
   comparison; a production-realtime path should cache plans across calls.

## Validation strategy

1. **Build succeeds on TRUBA.** `bash deploy/build.sh` produces
   `cuda/evm_cuda/_evm_cuda.so`.
2. **Each kernel matches the Python baseline within its tolerance.**
   `tests/cuda/test_*.py` (30 tests).
3. **End-to-end pipelines match the Python baseline within `<0.01` RMSE** on
   synthetic clips and on `face.mp4` / `baby.mp4` (`test_pipelines.py`).
4. **Python baseline still matches MIT.** The existing
   `tests/test_against_mit_reference.py` (unchanged) confirms the oracle
   itself hasn't drifted.
