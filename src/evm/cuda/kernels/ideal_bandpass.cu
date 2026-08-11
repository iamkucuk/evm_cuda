// ideal_bandpass.cu — FFT brick-wall bandpass via cuFFT.
//
// Port of evm/filters.py:ideal_bandpass, which mirrors MATLAB
// ideal_bandpassing.m. The filter has three steps:
//
//   1. Forward FFT along the time axis, per spatial location.
//   2. Zero every bin whose one-sided frequency is OUTSIDE (wl, wh).
//   3. Inverse FFT, take the real part.
//
// The mask uses frequencies `freqs = (0..n-1)/n * sampling_rate` with STRICT
// inequalities `freqs > wl & freqs < wh`. NOTE: this is NOT fftfreq — the
// mask is built on the un-folded 0..n-1 axis, so the upper-half FFT bins
// (which physically correspond to negative frequencies) get masked as if
// they were very-high positive frequencies and end up zeroed for typical
// (wl, wh) inside [0, Nyquist]. This matches the Python reference exactly;
// do not "fix" it with fftfreq — you'd change the in-band gain.
//
// Layout: (N, T) row-major post-transpose. We use cufftPlanMany with
// idist=odist=T, istride=ostride=1 — i.e. each row is one length-T signal —
// batched over N locations. Plans are cached on the host (runtime.py).
//
// Numerical contract (< 1e-4 vs Python): cuFFT's float32 plan vs numpy.fft's
// float64. Tolerance budget accommodates the FFT backend difference.
//
// NOTE on precision: we deliberately use C2C float32 plans to match the rest
// of the FP32 pipeline. If a tighter tolerance is required, switch to R2C/C2R
// or a double-precision C2C plan; both are drop-in via runtime.py.

#include <cufft.h>
#include "../include/evm_common.cuh"
#include "../include/evm_check.cuh"

namespace evm {

// Mask multiplier: zero out complex bins whose frequency is outside (wl, wh).
// Frequencies follow the MATLAB convention freqs[k] = k / T * sampling_rate.
__global__ void apply_ideal_mask_kernel(
    cufftComplex* __restrict__ spec,  // (N, T) complex
    int T, int N, float wl, float wh, float sampling_rate)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = T * N;
    if (idx >= total) return;

    const int t = idx % T;  // which frequency bin
    const float freq = static_cast<float>(t) / static_cast<float>(T) * sampling_rate;
    const bool keep = (freq > wl) && (freq < wh);
    if (!keep) {
        spec[idx].x = 0.0f;
        spec[idx].y = 0.0f;
    }
}

// Copy a real float array into the real part of a complex array (pre-FFT).
__global__ void real_to_complex_kernel(
    const float* __restrict__ real,
    cufftComplex* __restrict__ cplx,
    int total)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    cplx[idx].x = real[idx];
    cplx[idx].y = 0.0f;
}

// Take the real part of a complex array, normalizing by 1/T to match
// numpy.ifft (cuFFT's inverse does not normalize on its own).
__global__ void real_part_kernel_unnormalized(
    const cufftComplex* __restrict__ cplx,
    float* __restrict__ real,
    int total, int T)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    real[idx] = cplx[idx].x * (1.0f / static_cast<float>(T));
}

// --- host orchestration ----------------------------------------------------
//
// The caller owns the cuFFT plan (created/cached in runtime.py). We do
// real->cplx forward, mask, cplx->real inverse, return the real result.
// Working buffer `tmp` is N*T cufftComplex elements, allocated by caller.

void launch_ideal_bandpass(
    const float* in, float* out, cufftComplex* tmp,
    int T, int N, float wl, float wh, float sampling_rate,
    cufftHandle plan_fwd, cufftHandle plan_inv,
    cudaStream_t stream)
{
    const int total = T * N;
    int block = 256;
    int grid = div_up(total, block);

    real_to_complex_kernel<<<grid, block, 0, stream>>>(in, tmp, total);

    // cufftExecC2C is asynchronous w.r.t. the host but ordered w.r.t. the
    // caller's stream when the plan was created with that stream.
    CUFFT_CHECK(cufftExecC2C(plan_fwd, tmp, tmp, CUFFT_FORWARD));

    int mgrid = div_up(total, block);
    apply_ideal_mask_kernel<<<mgrid, block, 0, stream>>>(
        tmp, T, N, wl, wh, sampling_rate);

    CUFFT_CHECK(cufftExecC2C(plan_inv, tmp, tmp, CUFFT_INVERSE));

    real_part_kernel_unnormalized<<<grid, block, 0, stream>>>(tmp, out, total, T);
}

}  // namespace evm
