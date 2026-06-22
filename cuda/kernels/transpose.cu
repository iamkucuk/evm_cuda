// transpose.cu — layout transpose so the temporal filters see contiguous time.
//
// The video lives in (T,H,W,C) row-major: walking along T strides by H*W*C.
// The per-pixel temporal filters (iir / butter / ideal) want each spatial
// location's time series contiguous. We transpose (T,H,W,C) <-> (N,T) where
// N = H*W*C, so a length-T 1-D filter over a location reads N consecutive
// floats.
//
// Access pattern analysis for (N,T) <-> (T,N) transpose:
//   thwc_to_nt: reads src[t*N + n] (coalesced), writes dst[n*T + t] (stride-T)
//   nt_to_thwc: reads src[n*T + t] (coalesced), writes dst[t*N + n] (coalesced)
//
// The classic shared-memory tiled matrix transpose doesn't apply here because
// T is small (~291) and N is large (~500K): a [TILE_T][TILE_N] 2D tile would
// need 291*256*4 = 288 KB of shared memory, exceeding the H100's 228 KB per SM.
//
// Instead, each thread processes ELEMS_PER_THREAD spatial locations (Harris V6:
// "multiple elements per thread"). This amortizes the per-thread loop overhead
// and gives the compiler more independent memory operations to pipeline,
// improving instruction-level parallelism and hiding latency through the
// warp scheduler rather than through occupancy.
//
// This is a layout transform only — bit-exact, no tolerance implications.

#include "../include/evm_common.cuh"

namespace evm {

constexpr int ELEMS_PER_THREAD = 4;

// (T,H,W,C) -> (N,T): each thread handles ELEMS_PER_THREAD locations.
__global__ void thwc_to_nt_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int T, int N)
{
    const int base_n = blockIdx.x * blockDim.x * ELEMS_PER_THREAD;
    const int tid = threadIdx.x;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        const int n = base_n + tid + e * blockDim.x;
        if (n >= N) return;
        const float* s = src + n;
        float* d = dst + n * T;
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            d[t] = s[t * N];
        }
    }
}

// (N,T) -> (T,N) with optional scale (folds alpha amplification into the
// transpose for the motion pipeline's Stage C).
__global__ void nt_to_thwc_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int T, int N, float scale)
{
    const int base_n = blockIdx.x * blockDim.x * ELEMS_PER_THREAD;
    const int tid = threadIdx.x;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        const int n = base_n + tid + e * blockDim.x;
        if (n >= N) return;
        const float* s = src + n * T;
        float* d = dst + n;
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            d[t * N] = s[t] * scale;
        }
    }
}

void launch_thwc_to_nt(const float* src, float* dst, int T, int N,
                       cudaStream_t stream) {
    int block = 256;
    int grid = div_up(N, block * ELEMS_PER_THREAD);
    thwc_to_nt_kernel<<<grid, block, 0, stream>>>(src, dst, T, N);
}

void launch_nt_to_thwc(const float* src, float* dst, int T, int N,
                       cudaStream_t stream) {
    int block = 256;
    int grid = div_up(N, block * ELEMS_PER_THREAD);
    nt_to_thwc_kernel<<<grid, block, 0, stream>>>(src, dst, T, N, 1.0f);
}

void launch_nt_to_thwc_scaled(const float* src, float* dst, int T, int N,
                              float scale, cudaStream_t stream) {
    int block = 256;
    int grid = div_up(N, block * ELEMS_PER_THREAD);
    nt_to_thwc_kernel<<<grid, block, 0, stream>>>(src, dst, T, N, scale);
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
