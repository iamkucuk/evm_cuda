// transpose.cu — layout transpose so the temporal filters see contiguous time.
//
// The video lives in (T,H,W,C) row-major: walking along T strides by H*W*C.
// The per-pixel temporal filters (iir / butter / ideal) want each spatial
// location's time series contiguous. We transpose (T,H,W,C) <-> (N,T) where
// N = H*W*C, so a length-T 1-D filter over a location reads N consecutive
// floats.
//
// Grid: (ceil(N/256))  Block: (256, 1, 1). Each thread copies one location's
// full length-T series (a strided gather from src, contiguous scatter to dst).
//
// This is a layout transform only — bit-exact, no tolerance implications.

#include "../include/evm_common.cuh"

namespace evm {

// (T,H,W,C) row-major  ->  (N,T) row-major, where N = H*W*C.
// src[n*T_strided + t] -> dst[t*N + n]   with T_strided = H*W*C.
__global__ void thwc_to_nt_kernel(
    const float* __restrict__ src,  // (T, H*W*C) row-major, stride N between frames
    float* __restrict__ dst,        // (N, T)     row-major
    int T, int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* s = src + n;
    float* d = dst + n * T;
    for (int t = 0; t < T; ++t) {
        d[t] = s[t * N];
    }
}

// (N,T) row-major -> (T,H,W,C) row-major (inverse of the above).
__global__ void nt_to_thwc_kernel(
    const float* __restrict__ src,  // (N, T)
    float* __restrict__ dst,        // (T, N) with leading stride N
    int T, int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* s = src + n * T;
    float* d = dst + n;
    for (int t = 0; t < T; ++t) {
        d[t * N] = s[t];
    }
}

void launch_thwc_to_nt(const float* src, float* dst, int T, int N,
                       cudaStream_t stream) {
    int block = 256;
    int grid = div_up(N, block);
    thwc_to_nt_kernel<<<grid, block, 0, stream>>>(src, dst, T, N);
}

void launch_nt_to_thwc(const float* src, float* dst, int T, int N,
                       cudaStream_t stream) {
    int block = 256;
    int grid = div_up(N, block);
    nt_to_thwc_kernel<<<grid, block, 0, stream>>>(src, dst, T, N);
}

}  // namespace evm
