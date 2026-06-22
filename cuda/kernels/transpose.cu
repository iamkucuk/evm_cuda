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

// (n,H,W,3) interleaved  ->  (n*3,H,W) planar (frame-major, then channel).
// Used by the batched color pipeline so each frame-channel is a contiguous
// (H,W) block that blur_dn_device can consume directly via pointer offset.
// Bit-exact layout transform — no FP, no tolerance implications.
//
// Each thread handles one pixel (n,y,x): reads 3 contiguous floats from the
// interleaved source, scatters them to 3 separate planes.
__global__ void to_planar_3ch_kernel(
    const float* __restrict__ src,  // (n,H,W,3) row-major
    float* __restrict__ dst,        // (n*3,H,W) row-major
    int n, int H, int W)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * H * W;
    if (idx >= total) return;
    const int x = idx % W;
    const int yw = idx / W;
    const int y = yw % H;
    const int f = yw / H;
    const float* s = src + (yw * W + x) * 3;
    const int plane_off = (f * 3) * H * W + y * W + x;
    dst[plane_off + 0 * H * W] = s[0];
    dst[plane_off + 1 * H * W] = s[1];
    dst[plane_off + 2 * H * W] = s[2];
}

void launch_to_planar_3ch(const float* src, float* dst, int n, int H, int W,
                          cudaStream_t stream) {
    int block = 256;
    int grid = div_up(n * H * W, block);
    to_planar_3ch_kernel<<<grid, block, 0, stream>>>(src, dst, n, H, W);
}

// (n*3,H,W) planar  ->  (n,H,W,3) interleaved (inverse of to_planar_3ch).
// Used by the motion pipeline's Stage D to convert recon output back to
// interleaved for attenuate_chrom + add_and_quantize.
// Bit-exact layout transform — no FP, no tolerance implications.
__global__ void planar_to_interleaved_3ch_kernel(
    const float* __restrict__ src,  // (n*3,H,W) row-major
    float* __restrict__ dst,        // (n,H,W,3) row-major
    int n, int H, int W)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * H * W;
    if (idx >= total) return;
    const int x = idx % W;
    const int yw = idx / W;
    const int y = yw % H;
    const int f = yw / H;
    const int plane_off = (f * 3) * H * W + y * W + x;
    float* d = dst + (yw * W + x) * 3;
    d[0] = src[plane_off + 0 * H * W];
    d[1] = src[plane_off + 1 * H * W];
    d[2] = src[plane_off + 2 * H * W];
}

void launch_planar_to_interleaved_3ch(const float* src, float* dst,
                                      int n, int H, int W, cudaStream_t stream) {
    int block = 256;
    int grid = div_up(n * H * W, block);
    planar_to_interleaved_3ch_kernel<<<grid, block, 0, stream>>>(src, dst, n, H, W);
}

}  // namespace evm
