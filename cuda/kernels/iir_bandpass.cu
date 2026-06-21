// iir_bandpass.cu — direct r1/r2 IIR bandpass filter, per spatial location.
//
// Port of evm/filters.py:iir_bandpass, which mirrors MATLAB
// amplify_spatial_lpyr_temporal_iir.m:
//
//   y1[n] = (1 - r1) * y1[n-1] + r1 * x[n]
//   y2[n] = (1 - r2) * y2[n-1] + r2 * x[n]
//   out[n] = y1[n] - y2[n]            (require r1 > r2)
//
// Initial state: y1[0] = y2[0] = x[0]  (NOT zero). out[0] = 0 by construction.
//
// Input layout: (N, T) row-major (N = H*W*C, contiguous T per location).
// This is the post-transpose layout from transpose.cu; the host wrapper
// handles the (T,H,W,C) -> (N,T) round-trip.
//
// Grid: (ceil(N/256))  Block: (256, 1, 1). One thread per location, sequential
// loop over T. State y1/y2 live in registers (FP64) so accumulated error
// stays well under the 1e-5 tolerance budget.
//
// Numerical contract (< 1e-5 vs Python): the Python baseline also accumulates
// in float64; matching it requires FP64 state here. The I/O arrays stay FP32
// (matching the rest of the pipeline).

#include "../include/evm_common.cuh"

namespace evm {

__global__ void iir_bandpass_kernel(
    const float* __restrict__ in,   // (N, T) row-major
    float* __restrict__ out,        // (N, T) row-major
    int T, int N, double r1, double r2)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float* x = in  + n * T;
    float* o = out + n * T;

    // Initial state = first sample (evm/filters.py:114-116).
    double y1 = static_cast<double>(x[0]);
    double y2 = static_cast<double>(x[0]);
    const double one_minus_r1 = 1.0 - r1;
    const double one_minus_r2 = 1.0 - r2;

    o[0] = 0.0f;  // y1 - y2 == 0
    for (int t = 1; t < T; ++t) {
        double xt = static_cast<double>(x[t]);
        y1 = one_minus_r1 * y1 + r1 * xt;
        y2 = one_minus_r2 * y2 + r2 * xt;
        o[t] = static_cast<float>(y1 - y2);
    }
}

void launch_iir_bandpass(const float* in, float* out, int T, int N,
                         double r1, double r2, cudaStream_t stream) {
    int block = 256;
    int grid = div_up(N, block);
    iir_bandpass_kernel<<<grid, block, 0, stream>>>(in, out, T, N, r1, r2);
}

}  // namespace evm
