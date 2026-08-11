// butter_bandpass.cu — first-order Butterworth bandpass = difference of two
// 1st-order lowpass IIR filters.
//
// Port of evm/filters.py:butter_bandpass, which mirrors MATLAB
// amplify_spatial_lpyr_temporal_butter.m:
//
//   (b_high, a_high) = butter(order, fh/nyq, 'low')   # faster lowpass
//   (b_low,  a_low)  = butter(order, fl/nyq, 'low')   # slower lowpass
//   out = lfilter(b_high, a_high, x) - lfilter(b_low, a_low, x)
//
// The (b, a) coefficients are produced by scipy.signal.butter on the host
// (see evm_cuda/runtime.py); the kernel just runs the resulting 1st-order
// direct-form-II transposed recursion per location.
//
// For order=1: b=[b0,b1], a=[1,a1]. lfilter zero-inits (y[-1]=0, x[-1]=0).
// Recurrence:  y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
//
// Input layout: (N, T) row-major (post-transpose). One thread per location,
// FP64 accumulators (y_prev, x_prev) in registers.
//
// Numerical contract (< 1e-5 vs Python): Python uses scipy.lfilter with
// float64 internally when the input is float64. Our accumulators are FP64 so
// the per-step error matches lfilter's reference implementation.

#include "../include/evm_common.cuh"

namespace evm {

__global__ void butter_bandpass_kernel(
    const float* __restrict__ in,   // (N, T)
    float* __restrict__ out,        // (N, T)
    int T, int N,
    double b0_high, double b1_high, double a1_high,
    double b0_low,  double b1_low,  double a1_low)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float* x = in + n * T;
    float* o = out + n * T;

    // Zero initial state: x_prev=0, y_prev=0 for BOTH filters (scipy.lfilter
    // default zi=None).
    double yh_prev = 0.0, xh_prev = 0.0;
    double yl_prev = 0.0, xl_prev = 0.0;

    for (int t = 0; t < T; ++t) {
        double xt = static_cast<double>(x[t]);
        double yh = b0_high * xt + b1_high * xh_prev - a1_high * yh_prev;
        double yl = b0_low  * xt + b1_low  * xl_prev - a1_low  * yl_prev;
        xh_prev = xt; yh_prev = yh;
        xl_prev = xt; yl_prev = yl;
        o[t] = static_cast<float>(yh - yl);
    }
}

void launch_butter_bandpass(const float* in, float* out, int T, int N,
                            double b0_high, double b1_high, double a1_high,
                            double b0_low,  double b1_low,  double a1_low,
                            cudaStream_t stream) {
    int block = 256;
    int grid = div_up(N, block);
    butter_bandpass_kernel<<<grid, block, 0, stream>>>(
        in, out, T, N, b0_high, b1_high, a1_high, b0_low, b1_low, a1_low);
}

}  // namespace evm
