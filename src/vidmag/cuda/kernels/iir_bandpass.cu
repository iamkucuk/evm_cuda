// iir_bandpass.cu — direct r1/r2 IIR bandpass filter, per spatial location.
//
// Port of vidmag/cpu/filters.py:iir_bandpass, which mirrors MATLAB
// amplify_spatial_lpyr_temporal_iir.m:
//
//   y1[n] = (1 - r1) * y1[n-1] + r1 * x[n]
//   y2[n] = (1 - r2) * y2[n-1] + r2 * x[n]
//   out[n] = y1[n] - y2[n]            (require r1 > r2)
//
// Initial state: y1[0] = y2[0] = x[0]  (NOT zero). out[0] = 0 by construction.
//
// Layouts:
//   (N,T) row-major — legacy; one thread per n, contiguous T, but warp lanes
//     at fixed t are spaced by T (uncoalesced). Kept for unit tests / probes.
//   (T,N) row-major — production; one thread per n, addr = t*N + n so a warp
//     at fixed t is contiguous in n (coalesced). Band buffers from lpyr are
//     already (T,N) per channel.
//
// Grid: (ceil(N/256))  Block: (256, 1, 1).
//
// Numerical contract (< 1e-5 vs Python), enforced by tests/cuda/conftest.py
// TOL["iir"] and by the end-to-end comparison against the NumPy reference.
//
// The recursion state is FP32, not FP64. It was FP64, and the change is worth
// recording because it is not obvious. On a GeForce card double-precision
// arithmetic runs at a sixty-fourth of single-precision rate, and this kernel
// is four multiply-adds per element with a serial dependency, so it was
// arithmetic-bound rather than memory-bound. Measured on an RTX 3090, on the
// dominant pyramid level of the motion pipeline (291 frames of 960x544):
//
//     FP64 state    4.95 ms    246 GB/s
//     FP32 state    1.50 ms    809 GB/s   (peak is 936 GB/s)
//
// So the same work becomes memory-bound, which is as good as it can be, and
// the whole stage runs about three times faster. The cost is accuracy: the two
// differ by about 4e-7 over a 291-step recursion, which is twenty-five times
// inside the 1e-5 the contract allows, and the suite checks the real figure on
// real data rather than trusting that estimate.
//
// Vectorised loads were measured too and are not used: four elements per
// thread gave 1.44 ms against 1.50 ms, which does not justify the extra code.

#include "../include/evm_common.cuh"

namespace evm {

// --- (N,T) layout: dst[n*T + t] (legacy / tests) ---------------------------

template <typename In, typename Out>
__global__ void iir_bandpass_kernel(
    const In* __restrict__ in,   // (N, T) row-major
    Out* __restrict__ out,       // (N, T) row-major
    int T, int N, double r1, double r2)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const In* x = in  + n * T;
    Out* o = out + n * T;

    float y1 = cvt_in<In>(x[0]);
    float y2 = y1;
    const float one_minus_r1 = 1.0f - static_cast<float>(r1);
    const float one_minus_r2 = 1.0f - static_cast<float>(r2);
    const float f_r1 = static_cast<float>(r1);
    const float f_r2 = static_cast<float>(r2);

    o[0] = cvt_out<Out>(0.0f);
    for (int t = 1; t < T; ++t) {
        const float xt = cvt_in<In>(x[t]);
        y1 = fmaf(one_minus_r1, y1, f_r1 * xt);
        y2 = fmaf(one_minus_r2, y2, f_r2 * xt);
        o[t] = cvt_out<Out>(y1 - y2);
    }
}

// --- (T,N) layout: addr = t*N + n (production, coalesced) ------------------
// Optional scale folds per-level alpha into the write (was nt_to_thwc_scaled).

template <typename In, typename Out>
__global__ void iir_bandpass_tn_kernel(
    const In* __restrict__ in,   // (T, N) row-major
    Out* __restrict__ out,       // (T, N) row-major
    int T, int N, double r1, double r2, float scale)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float y1 = cvt_in<In>(in[n]);   // t = 0
    float y2 = y1;
    const float one_minus_r1 = 1.0f - static_cast<float>(r1);
    const float one_minus_r2 = 1.0f - static_cast<float>(r2);
    const float f_r1 = static_cast<float>(r1);
    const float f_r2 = static_cast<float>(r2);

    out[n] = cvt_out<Out>(0.0f);
    for (int t = 1; t < T; ++t) {
        const int idx = t * N + n;
        const float xt = cvt_in<In>(in[idx]);
        y1 = fmaf(one_minus_r1, y1, f_r1 * xt);
        y2 = fmaf(one_minus_r2, y2, f_r2 * xt);
        out[idx] = cvt_out<Out>((y1 - y2) * scale);
    }
}

void launch_iir_bandpass(const float* in, float* out, int T, int N,
                         double r1, double r2, cudaStream_t stream) {
    int block = 256;
    int grid = div_up(N, block);
    iir_bandpass_kernel<float, float><<<grid, block, 0, stream>>>(in, out, T, N, r1, r2);
}

void launch_iir_bandpass_f16(const __half* in, __half* out, int T, int N,
                             double r1, double r2, cudaStream_t stream) {
    int block = 256;
    int grid = div_up(N, block);
    iir_bandpass_kernel<__half, __half><<<grid, block, 0, stream>>>(in, out, T, N, r1, r2);
}

void launch_iir_bandpass_tn(const float* in, float* out, int T, int N,
                            double r1, double r2, float scale,
                            cudaStream_t stream) {
    int block = 256;
    int grid = div_up(N, block);
    iir_bandpass_tn_kernel<float, float><<<grid, block, 0, stream>>>(
        in, out, T, N, r1, r2, scale);
}

void launch_iir_bandpass_tn_f16(const __half* in, __half* out, int T, int N,
                                double r1, double r2, float scale,
                                cudaStream_t stream) {
    int block = 256;
    int grid = div_up(N, block);
    iir_bandpass_tn_kernel<__half, __half><<<grid, block, 0, stream>>>(
        in, out, T, N, r1, r2, scale);
}

// ---------------------------------------------------------------------------
// Diagnostic probes (not used by the production pipeline).
//
// Isolate Stage-C bounds without Nsight counters:
//   1) flat_copy     — coalesced element-wise copy of N*T floats
//   2) strided_t_copy — same (N,T) access as IIR (one thread per n, loop t)
//   3) iir_fp32_state — same recurrence as production, FP32 accumulators
//
// Interpretation (same N,T, same timing harness):
//   flat_copy ~ stream peak, strided_t_copy << flat  → access pattern / coalescing
//   strided_t_copy ~ iir_fp64                         → memory pattern dominates IIR
//   iir_fp32 << iir_fp64 time, near strided           → FP64 cost dominates
//   iir_fp64 >> strided_t_copy                        → dep/compute on top of access
// ---------------------------------------------------------------------------

__global__ void flat_copy_f32_kernel(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     int n_elem)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elem) return;
    out[i] = in[i];
}

// Same indexing as iir_bandpass_kernel: thread n owns row n, loops t.
__global__ void strided_t_copy_f32_kernel(const float* __restrict__ in,
                                          float* __restrict__ out,
                                          int T, int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* x = in + n * T;
    float* o = out + n * T;
    #pragma unroll 1
    for (int t = 0; t < T; ++t) {
        o[t] = x[t];
    }
}

// Production IIR math with FP32 state (breaks <1e-5 contract — probe only).
__global__ void iir_bandpass_fp32_state_kernel(const float* __restrict__ in,
                                               float* __restrict__ out,
                                               int T, int N,
                                               float r1, float r2)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* x = in + n * T;
    float* o = out + n * T;
    float y1 = x[0];
    float y2 = x[0];
    const float one_minus_r1 = 1.0f - r1;
    const float one_minus_r2 = 1.0f - r2;
    o[0] = 0.0f;
    for (int t = 1; t < T; ++t) {
        float xt = x[t];
        y1 = one_minus_r1 * y1 + r1 * xt;
        y2 = one_minus_r2 * y2 + r2 * xt;
        o[t] = y1 - y2;
    }
}

void launch_flat_copy_f32(const float* in, float* out, int n_elem,
                          cudaStream_t stream) {
    int block = 256;
    int grid = div_up(n_elem, block);
    flat_copy_f32_kernel<<<grid, block, 0, stream>>>(in, out, n_elem);
}

void launch_strided_t_copy_f32(const float* in, float* out, int T, int N,
                               cudaStream_t stream) {
    int block = 256;
    int grid = div_up(N, block);
    strided_t_copy_f32_kernel<<<grid, block, 0, stream>>>(in, out, T, N);
}

// Single-pole IIR (one accumulator) — same indexing, less arithmetic than bandpass.
__global__ void iir_single_pole_fp32_kernel(const float* __restrict__ in,
                                            float* __restrict__ out,
                                            int T, int N, float r)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* x = in + n * T;
    float* o = out + n * T;
    float y = x[0];
    const float one_minus_r = 1.0f - r;
    o[0] = 0.0f;
    for (int t = 1; t < T; ++t) {
        y = one_minus_r * y + r * x[t];
        o[t] = y;
    }
}

void launch_iir_single_pole_fp32(const float* in, float* out, int T, int N,
                                 float r, cudaStream_t stream) {
    int block = 256;
    int grid = div_up(N, block);
    iir_single_pole_fp32_kernel<<<grid, block, 0, stream>>>(in, out, T, N, r);
}

// Coalesced (T,N) walk: thread n loops t, address = t*N + n (warp-contiguous at fixed t).
__global__ void coalesced_tn_copy_f32_kernel(const float* __restrict__ in,
                                             float* __restrict__ out,
                                             int T, int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    for (int t = 0; t < T; ++t) {
        const int idx = t * N + n;
        out[idx] = in[idx];
    }
}

// Same recurrence as FP32 bandpass, but (T,N) layout for coalesced loads/stores.
__global__ void iir_bandpass_fp32_tn_kernel(const float* __restrict__ in,
                                            float* __restrict__ out,
                                            int T, int N,
                                            float r1, float r2)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float y1 = in[n];  // t=0
    float y2 = y1;
    const float one_minus_r1 = 1.0f - r1;
    const float one_minus_r2 = 1.0f - r2;
    out[n] = 0.0f;
    for (int t = 1; t < T; ++t) {
        const int idx = t * N + n;
        float xt = in[idx];
        y1 = one_minus_r1 * y1 + r1 * xt;
        y2 = one_minus_r2 * y2 + r2 * xt;
        out[idx] = y1 - y2;
    }
}

void launch_coalesced_tn_copy_f32(const float* in, float* out, int T, int N,
                                  cudaStream_t stream) {
    int block = 256;
    int grid = div_up(N, block);
    coalesced_tn_copy_f32_kernel<<<grid, block, 0, stream>>>(in, out, T, N);
}

void launch_iir_bandpass_fp32_tn(const float* in, float* out, int T, int N,
                                 float r1, float r2, cudaStream_t stream) {
    int block = 256;
    int grid = div_up(N, block);
    iir_bandpass_fp32_tn_kernel<<<grid, block, 0, stream>>>(
        in, out, T, N, r1, r2);
}

void launch_iir_bandpass_fp32_state(const float* in, float* out, int T, int N,
                                    float r1, float r2, cudaStream_t stream) {
    int block = 256;
    int grid = div_up(N, block);
    iir_bandpass_fp32_state_kernel<<<grid, block, 0, stream>>>(
        in, out, T, N, r1, r2);
}

}  // namespace evm
