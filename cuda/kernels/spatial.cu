// spatial.cu — separable correlate+downsample (corr_dn) and its transpose
// upsample+convolve (up_conv). Direct port of evm/pyramids.py:corr_dn_axis /
// up_conv_axis, which in turn mirror matlabPyrTools corrDn / upConv.
//
// Both operate on a single axis of a 2-D single-channel image. The
// multi-level pyramid and multi-channel wrappers call them in sequence.
//
// Grid:  (ceil(Wout/32), ceil(Hout/32))   Block: (32, 32, 1)
// Each thread computes one output element by gathering 5 input samples
// under reflect1 padding and dot-producting with the (flipped) binom5 kernel.
//
// Numerical contract (tolerance < 1e-5 vs Python, see DESIGN.md):
//   - Filter is applied as correlation (kernel flipped inside the math
//     below; matches the Python `filt[::-1]` convention).
//   - reflect1 padding via evm::reflect1(i, n) device helper.
//   - Downsample keeps source indices 0,2,4,...; upsample stuffs data at
//     even dest indices and zeros at odd (MATLAB start=[1,1]).

#include "../include/evm_common.cuh"

namespace evm {

// corr_dn along axis=0 (rows / y). Output rows = (H + 1) / 2.
__global__ void corr_dn_rows_kernel(
    const float* __restrict__ in,   // (H*W) row-major
    float* __restrict__ out,        // (((H+1)/2)*W)
    int H, int W, const float* filt, int filt_len)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int yo = blockIdx.y * blockDim.y + threadIdx.y;
    const int Ho = (H + 1) / 2;
    if (x >= W || yo >= Ho) return;

    // Output index yo corresponds to source row 2*yo (start=[1,1] -> idx 0).
    const int src_center = 2 * yo;
    const int pad = filt_len / 2;
    float acc = 0.0f;
    for (int k = 0; k < filt_len; ++k) {
        // Correlation: sample at src_center + (k - pad); weight by filt[k]
        // (matches np.convolve with reversed kernel).
        int src = reflect1(src_center + (k - pad), H);
        acc += filt[k] * in[src * W + x];
    }
    out[yo * W + x] = acc;
}

// corr_dn along axis=1 (cols / x). Output cols = (W + 1) / 2.
__global__ void corr_dn_cols_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int H, int W, const float* filt, int filt_len)
{
    const int xo = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int Wo = (W + 1) / 2;
    if (xo >= Wo || y >= H) return;

    const int src_center = 2 * xo;
    const int pad = filt_len / 2;
    float acc = 0.0f;
    for (int k = 0; k < filt_len; ++k) {
        int src = reflect1(src_center + (k - pad), W);
        acc += filt[k] * in[y * W + src];
    }
    out[y * Wo + xo] = acc;
}

// up_conv along axis=0 (rows / y). Output has out_H rows (= next-finer size).
// Input has in_H rows; data is "stuffed" at even output positions then convolved.
//
// The reference (evm/pyramids.py:up_conv_axis) builds a length-2*in_H upsampled
// array u with u[2*i] = img[i] and u[odd] = 0, reflect1-pads it by `pad=2` on
// each side, then convolves 'valid' with the reversed kernel. The convolution
// taps samples of u that may lie OUTSIDE [0, 2*in_H) at the boundaries; those
// are brought back via reflect1 over the 2*in_H axis. We replicate that
// analytically here so we don't have to materialise the (sparse) upsampled
// buffer:
//
//   out[yo] = sum_{k=0..4} filt[k] * u[ reflect1(yo + k - pad, 2*in_H) ]
//
// where u[m] = img[m/2] if m is even, else 0. reflect1 is the SAME helper
// used by corr_dn; it correctly maps e.g. u[-2] -> u[2] (which holds img[1]).
__global__ void up_conv_rows_kernel(
    const float* __restrict__ in,   // (in_H * W) row-major
    float* __restrict__ out,        // (out_H * W)
    int in_H, int out_H, int W,
    const float* filt, int filt_len)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int yo = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || yo >= out_H) return;

    const int pad = filt_len / 2;
    const int up_H = 2 * in_H;  // period of the upsampled array
    float acc = 0.0f;
    for (int k = 0; k < filt_len; ++k) {
        int u_idx = yo + (k - pad);
        int r = reflect1(u_idx, up_H);   // reflected index into [0, 2*in_H)
        if ((r & 1) == 0) {              // u[r] is nonzero only at even r
            int src = r / 2;
            acc += filt[k] * in[src * W + x];
        }
    }
    out[yo * W + x] = acc;
}

// up_conv along axis=1 (cols / x).
__global__ void up_conv_cols_kernel(
    const float* __restrict__ in,   // (H * in_W)
    float* __restrict__ out,        // (H * out_W)
    int H, int in_W, int out_W,
    const float* filt, int filt_len)
{
    const int xo = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (xo >= out_W || y >= H) return;

    const int pad = filt_len / 2;
    const int up_W = 2 * in_W;
    float acc = 0.0f;
    for (int k = 0; k < filt_len; ++k) {
        int u_idx = xo + (k - pad);
        int r = reflect1(u_idx, up_W);
        if ((r & 1) == 0) {
            int src = r / 2;
            acc += filt[k] * in[y * in_W + src];
        }
    }
    out[y * out_W + xo] = acc;
}

// --- host launchers --------------------------------------------------------

void launch_corr_dn_rows(const float* in, float* out, int H, int W,
                         const float* filt, int filt_len, cudaStream_t stream) {
    int Ho = (H + 1) / 2;
    dim3 block(32, 32, 1);
    dim3 grid(div_up(W, 32), div_up(Ho, 32), 1);
    corr_dn_rows_kernel<<<grid, block, 0, stream>>>(
        in, out, H, W, filt, filt_len);
}

void launch_corr_dn_cols(const float* in, float* out, int H, int W,
                         const float* filt, int filt_len, cudaStream_t stream) {
    int Wo = (W + 1) / 2;
    dim3 block(32, 32, 1);
    dim3 grid(div_up(Wo, 32), div_up(H, 32), 1);
    corr_dn_cols_kernel<<<grid, block, 0, stream>>>(
        in, out, H, W, filt, filt_len);
}

void launch_up_conv_rows(const float* in, float* out,
                         int in_H, int out_H, int W,
                         const float* filt, int filt_len, cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(div_up(W, 32), div_up(out_H, 32), 1);
    up_conv_rows_kernel<<<grid, block, 0, stream>>>(
        in, out, in_H, out_H, W, filt, filt_len);
}

void launch_up_conv_cols(const float* in, float* out,
                         int H, int in_W, int out_W,
                         const float* filt, int filt_len, cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(div_up(out_W, 32), div_up(H, 32), 1);
    up_conv_cols_kernel<<<grid, block, 0, stream>>>(
        in, out, H, in_W, out_W, filt, filt_len);
}

// ===========================================================================
// Batched variants — process B independent (H,W) slices in a single launch.
// Used by the batched Laplacian pyramid build/reconstruct to collapse the
// M-slice host loop (~35k launches) into ~40 launches (one per kernel per
// level). Each slice occupies a contiguous block of `slice_stride` elements;
// the grid z-dimension indexes the batch.
//
// Templated on In/Out types to support both FP32 and FP16 storage. The
// convolution arithmetic is always FP32 (acc is float). When In=__half,
// reads convert via __half2float; when Out=__half, writes convert via
// __float2half. This lets the pipeline store buffers in FP16 to halve VRAM
// without changing the numerical results of the compute.
// ===========================================================================
// Batched kernels with shared-memory tiles (5-tap binom).
// Block is (32, 8): enough threads to hide latency, SM footprint small.
// Halo = 2 (pad for filt_len=5). Reflect applied when loading into SM so the
// compute loop is a plain 5-tap without per-tap reflect1 in the hot path for
// interior; borders still correct via reflected loads.
// ===========================================================================

constexpr int SP_BX = 32;
constexpr int SP_BY = 8;
constexpr int SP_HALO = 2;
// Horizontal tile width in input samples for corr_dn_cols / up_conv_cols:
// output xo tile maps to centers 2*xo0 .. 2*(xo0+BX-1), plus ±HALO.
constexpr int SP_X_IN = 2 * SP_BX + 2 * SP_HALO;  // 68
// Vertical tile height in input samples for corr_dn_rows:
constexpr int SP_Y_IN = 2 * SP_BY + 2 * SP_HALO;  // 20

template <typename In, typename Out>
__global__ void corr_dn_cols_batched_kernel(
    const In* __restrict__ in,
    Out* __restrict__ out,
    int H, int W, const float* filt, int filt_len,
    int slice_stride_in, int slice_stride_out, int B)
{
    const int xo0 = blockIdx.x * SP_BX;
    const int y0  = blockIdx.y * SP_BY;
    const int b   = blockIdx.z;
    const int Wo  = (W + 1) / 2;
    if (b >= B) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    float f[5];
    #pragma unroll
    for (int k = 0; k < 5; ++k) f[k] = filt[k];

    // Shared tile: BY rows × (2*BX + 2*HALO) input columns, float compute type.
    __shared__ float tile[SP_BY][SP_X_IN];

    const In* sin = in + b * slice_stride_in;
    // First input column index for this block's tile (may be negative → reflect).
    const int x_base = 2 * xo0 - SP_HALO;

    // Cooperative load: each thread loads multiple elements of the tile.
    const int tile_elems = SP_BY * SP_X_IN;
    const int tid = ty * SP_BX + tx;
    const int nthreads = SP_BX * SP_BY;
    for (int i = tid; i < tile_elems; i += nthreads) {
        const int ly = i / SP_X_IN;
        const int lx = i - ly * SP_X_IN;
        const int gy = y0 + ly;
        const int gx = reflect1(x_base + lx, W);
        float v = 0.0f;
        if (gy >= 0 && gy < H) {
            v = cvt_in<In>(sin[gy * W + gx]);
        }
        tile[ly][lx] = v;
    }
    __syncthreads();

    const int xo = xo0 + tx;
    const int y  = y0 + ty;
    if (xo >= Wo || y >= H) return;

    // Center of the 5-tap in tile coordinates: global center 2*xo maps to
    // tile_x = 2*xo - x_base = 2*xo - (2*xo0 - HALO) = 2*(xo-xo0) + HALO.
    const int tc = 2 * tx + SP_HALO;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 5; ++k) {
        acc += f[k] * tile[ty][tc + (k - SP_HALO)];
    }
    out[b * slice_stride_out + y * Wo + xo] = cvt_out<Out>(acc);
}

template <typename In, typename Out>
__global__ void corr_dn_rows_batched_kernel(
    const In* __restrict__ in,
    Out* __restrict__ out,
    int H, int W, const float* filt, int filt_len,
    int slice_stride_in, int slice_stride_out, int B)
{
    const int x0  = blockIdx.x * SP_BX;
    const int yo0 = blockIdx.y * SP_BY;
    const int b   = blockIdx.z;
    const int Ho  = (H + 1) / 2;
    if (b >= B) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    float f[5];
    #pragma unroll
    for (int k = 0; k < 5; ++k) f[k] = filt[k];

    // Shared: (2*BY + 2*HALO) rows × BX cols
    __shared__ float tile[SP_Y_IN][SP_BX];

    const In* sin = in + b * slice_stride_in;
    const int y_base = 2 * yo0 - SP_HALO;

    const int tile_elems = SP_Y_IN * SP_BX;
    const int tid = ty * SP_BX + tx;
    const int nthreads = SP_BX * SP_BY;
    for (int i = tid; i < tile_elems; i += nthreads) {
        const int ly = i / SP_BX;
        const int lx = i - ly * SP_BX;
        const int gx = x0 + lx;
        const int gy = reflect1(y_base + ly, H);
        float v = 0.0f;
        if (gx >= 0 && gx < W) {
            v = cvt_in<In>(sin[gy * W + gx]);
        }
        tile[ly][lx] = v;
    }
    __syncthreads();

    const int x  = x0 + tx;
    const int yo = yo0 + ty;
    if (x >= W || yo >= Ho) return;

    const int tc = 2 * ty + SP_HALO;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 5; ++k) {
        acc += f[k] * tile[tc + (k - SP_HALO)][tx];
    }
    out[b * slice_stride_out + yo * W + x] = cvt_out<Out>(acc);
}

// up_conv: output yo maps to upsampled index; only even reflected samples hit.
// Shared-mem loads the *input* coarse grid neighborhood.

template <typename In, typename Out>
__global__ void up_conv_rows_batched_kernel(
    const In* __restrict__ in,
    Out* __restrict__ out,
    int in_H, int out_H, int W,
    const float* filt, int filt_len,
    int slice_stride_in, int slice_stride_out, int B)
{
    const int x0  = blockIdx.x * SP_BX;
    const int yo0 = blockIdx.y * SP_BY;
    const int b   = blockIdx.z;
    if (b >= B) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    float f[5];
    #pragma unroll
    for (int k = 0; k < 5; ++k) f[k] = filt[k];

    // Input rows needed: for yo in [yo0, yo0+BY), u = yo+k-2, r=reflect1(u,2*in_H),
    // if even src=r/2. Worst-case src spans roughly yo0/2 ± a few.
    // Load in_H neighborhood: from (yo0 - HALO)/2 to (yo0+BY-1+HALO)/2 with margin.
    // Simpler: load a vertical strip of input rows covering [yo0-HALO, yo0+BY+HALO]
    // mapped through even samples — load ceil range of in_H with halo.
    // Use tile of (BY + 2*HALO + 2) input rows × BX cols — generous for pad=2.
    constexpr int UY = SP_BY + 2 * SP_HALO + 2;  // 14
    __shared__ float tile[UY][SP_BX];

    const In* sin = in + b * slice_stride_in;
    // First input row we might need: floor((yo0 - HALO) / 2) - 1, clamp via reflect later
    const int y_in0 = (yo0 - SP_HALO) / 2 - 1;

    const int tile_elems = UY * SP_BX;
    const int tid = ty * SP_BX + tx;
    const int nthreads = SP_BX * SP_BY;
    for (int i = tid; i < tile_elems; i += nthreads) {
        const int ly = i / SP_BX;
        const int lx = i - ly * SP_BX;
        const int gx = x0 + lx;
        const int gy = reflect1(y_in0 + ly, in_H);
        float v = 0.0f;
        if (gx >= 0 && gx < W) {
            v = cvt_in<In>(sin[gy * W + gx]);
        }
        tile[ly][lx] = v;
    }
    __syncthreads();

    const int x  = x0 + tx;
    const int yo = yo0 + ty;
    if (x >= W || yo >= out_H) return;

    const int pad = SP_HALO;
    const int up_H = 2 * in_H;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 5; ++k) {
        int u_idx = yo + (k - pad);
        int r = reflect1(u_idx, up_H);
        if ((r & 1) == 0) {
            int src = r / 2;
            // Map src to tile row: src - y_in0
            int ly = src - y_in0;
            if (ly >= 0 && ly < UY) {
                acc += f[k] * tile[ly][tx];
            } else {
                // Fallback (should be rare if UY generous)
                acc += f[k] * cvt_in<In>(sin[src * W + x]);
            }
        }
    }
    out[b * slice_stride_out + yo * W + x] = cvt_out<Out>(acc);
}

template <typename In, typename Out>
__global__ void up_conv_cols_batched_kernel(
    const In* __restrict__ in,
    Out* __restrict__ out,
    int H, int in_W, int out_W,
    const float* filt, int filt_len,
    int slice_stride_in, int slice_stride_out, int B)
{
    const int xo0 = blockIdx.x * SP_BX;
    const int y0  = blockIdx.y * SP_BY;
    const int b   = blockIdx.z;
    if (b >= B) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    float f[5];
    #pragma unroll
    for (int k = 0; k < 5; ++k) f[k] = filt[k];

    constexpr int UX = SP_BX + 2 * SP_HALO + 2;  // 14 input cols in tile? need more
    // For horizontal up_conv, input x needed ~ xo/2 ± halo. Tile width ~ BX/2 + halo + margin
    // Use UX = SP_BX + 4 (generous for BX=32 outputs spanning ~16 input centers + pad)
    // Actually outputs xo0..xo0+31 need src up to ~xo0/2+16+2. Width ~ BX/2+HALO+2 ~ 20.
    // Use 24.
    constexpr int UXW = SP_BX / 2 + 2 * SP_HALO + 4;  // 24
    __shared__ float tile[SP_BY][UXW];

    const In* sin = in + b * slice_stride_in;
    const int x_in0 = (xo0 - SP_HALO) / 2 - 1;

    const int tile_elems = SP_BY * UXW;
    const int tid = ty * SP_BX + tx;
    const int nthreads = SP_BX * SP_BY;
    for (int i = tid; i < tile_elems; i += nthreads) {
        const int ly = i / UXW;
        const int lx = i - ly * UXW;
        const int gy = y0 + ly;
        const int gx = reflect1(x_in0 + lx, in_W);
        float v = 0.0f;
        if (gy >= 0 && gy < H) {
            v = cvt_in<In>(sin[gy * in_W + gx]);
        }
        tile[ly][lx] = v;
    }
    __syncthreads();

    const int xo = xo0 + tx;
    const int y  = y0 + ty;
    if (xo >= out_W || y >= H) return;

    const int pad = SP_HALO;
    const int up_W = 2 * in_W;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 5; ++k) {
        int u_idx = xo + (k - pad);
        int r = reflect1(u_idx, up_W);
        if ((r & 1) == 0) {
            int src = r / 2;
            int lx = src - x_in0;
            if (lx >= 0 && lx < UXW) {
                acc += f[k] * tile[ty][lx];
            } else {
                acc += f[k] * cvt_in<In>(sin[y * in_W + src]);
            }
        }
    }
    out[b * slice_stride_out + y * out_W + xo] = cvt_out<Out>(acc);
}

// --- batched launchers (FP32 storage) --------------------------------------

void launch_corr_dn_rows_batched(const float* in, float* out,
                                 int H, int W, const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream) {
    int Ho = (H + 1) / 2;
    dim3 block(SP_BX, SP_BY, 1);
    dim3 grid(div_up(W, SP_BX), div_up(Ho, SP_BY), B);
    corr_dn_rows_batched_kernel<float, float><<<grid, block, 0, stream>>>(
        in, out, H, W, filt, filt_len, stride_in, stride_out, B);
}

void launch_corr_dn_cols_batched(const float* in, float* out,
                                 int H, int W, const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream) {
    int Wo = (W + 1) / 2;
    dim3 block(SP_BX, SP_BY, 1);
    dim3 grid(div_up(Wo, SP_BX), div_up(H, SP_BY), B);
    corr_dn_cols_batched_kernel<float, float><<<grid, block, 0, stream>>>(
        in, out, H, W, filt, filt_len, stride_in, stride_out, B);
}

void launch_up_conv_rows_batched(const float* in, float* out,
                                 int in_H, int out_H, int W,
                                 const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream) {
    dim3 block(SP_BX, SP_BY, 1);
    dim3 grid(div_up(W, SP_BX), div_up(out_H, SP_BY), B);
    up_conv_rows_batched_kernel<float, float><<<grid, block, 0, stream>>>(
        in, out, in_H, out_H, W, filt, filt_len, stride_in, stride_out, B);
}

void launch_up_conv_cols_batched(const float* in, float* out,
                                 int H, int in_W, int out_W,
                                 const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream) {
    dim3 block(SP_BX, SP_BY, 1);
    dim3 grid(div_up(out_W, SP_BX), div_up(H, SP_BY), B);
    up_conv_cols_batched_kernel<float, float><<<grid, block, 0, stream>>>(
        in, out, H, in_W, out_W, filt, filt_len, stride_in, stride_out, B);
}

// --- batched launchers (FP16 storage, FP32 compute) ------------------------

void launch_corr_dn_rows_batched_f16(const __half* in, __half* out,
                                 int H, int W, const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream) {
    int Ho = (H + 1) / 2;
    dim3 block(SP_BX, SP_BY, 1);
    dim3 grid(div_up(W, SP_BX), div_up(Ho, SP_BY), B);
    corr_dn_rows_batched_kernel<__half, __half><<<grid, block, 0, stream>>>(
        in, out, H, W, filt, filt_len, stride_in, stride_out, B);
}

void launch_corr_dn_cols_batched_f16(const __half* in, __half* out,
                                 int H, int W, const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream) {
    int Wo = (W + 1) / 2;
    dim3 block(SP_BX, SP_BY, 1);
    dim3 grid(div_up(Wo, SP_BX), div_up(H, SP_BY), B);
    corr_dn_cols_batched_kernel<__half, __half><<<grid, block, 0, stream>>>(
        in, out, H, W, filt, filt_len, stride_in, stride_out, B);
}


// EXPERIMENT: half-accumulate separable up_conv (f16 launchers only).
__global__ void up_conv_rows_batched_halfacc_kernel(
    const __half* __restrict__ in,
    __half* __restrict__ out,
    int in_H, int out_H, int W, const float* filt, int filt_len,
    int slice_stride_in, int slice_stride_out, int B)
{
    const int x  = blockIdx.x * SP_BX + threadIdx.x;
    const int yo0 = blockIdx.y * SP_BY;
    const int b  = blockIdx.z;
    if (b >= B) return;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __half f[5];
#pragma unroll
    for (int k = 0; k < 5; ++k) f[k] = __float2half(filt[k]);

    // Input rows needed: for output yo in [yo0, yo0+BY), upsampled index yo,
    // even reflected samples only. Load a generous float-height tile of source rows.
    constexpr int UY = SP_BY + 2 * SP_HALO + 4;
    __shared__ __half tile[UY][SP_BX];

    const __half* sin = in + b * slice_stride_in;
    // Map roughly: source y ~ yo/2
    const int y_src0 = (yo0 - SP_HALO) / 2 - 1;

    for (int i = ty * SP_BX + tx; i < UY * SP_BX; i += SP_BX * SP_BY) {
        const int ly = i / SP_BX;
        const int lx = i - ly * SP_BX;
        const int gx = blockIdx.x * SP_BX + lx;
        const int gy = reflect1(y_src0 + ly, in_H);
        __half v = __float2half(0.0f);
        if (gx >= 0 && gx < W) v = sin[gy * W + gx];
        tile[ly][lx] = v;
    }
    __syncthreads();

    const int yo = yo0 + ty;
    if (x >= W || yo >= out_H) return;

    const int pad = SP_HALO;
    const int up_H = 2 * in_H;
    __half acc = __float2half(0.0f);
#pragma unroll
    for (int k = 0; k < 5; ++k) {
        const int ry = reflect1(yo + (k - pad), up_H);
        if (ry & 1) continue;
        const int sy = ry / 2;
        const int ly = sy - y_src0;
        if (ly >= 0 && ly < UY) {
            acc = __hadd(acc, __hmul(f[k], tile[ly][tx]));
        } else {
            acc = __hadd(acc, __hmul(f[k], sin[sy * W + x]));
        }
    }
    out[b * slice_stride_out + yo * W + x] = acc;
}

__global__ void up_conv_cols_batched_halfacc_kernel(
    const __half* __restrict__ in,
    __half* __restrict__ out,
    int H, int in_W, int out_W, const float* filt, int filt_len,
    int slice_stride_in, int slice_stride_out, int B)
{
    const int xo0 = blockIdx.x * SP_BX;
    const int y  = blockIdx.y * SP_BY + threadIdx.y;
    const int b  = blockIdx.z;
    if (b >= B) return;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __half f[5];
#pragma unroll
    for (int k = 0; k < 5; ++k) f[k] = __float2half(filt[k]);

    constexpr int UX = SP_BX + 2 * SP_HALO + 4;
    __shared__ __half tile[SP_BY][UX];

    const __half* sin = in + b * slice_stride_in;
    const int x_src0 = (xo0 - SP_HALO) / 2 - 1;

    for (int i = ty * SP_BX + tx; i < SP_BY * UX; i += SP_BX * SP_BY) {
        const int ly = i / UX;
        const int lx = i - ly * UX;
        const int gy = blockIdx.y * SP_BY + ly;
        const int gx = reflect1(x_src0 + lx, in_W);
        __half v = __float2half(0.0f);
        if (gy >= 0 && gy < H) v = sin[gy * in_W + gx];
        tile[ly][lx] = v;
    }
    __syncthreads();

    const int xo = xo0 + tx;
    if (xo >= out_W || y >= H) return;

    const int pad = SP_HALO;
    const int up_W = 2 * in_W;
    __half acc = __float2half(0.0f);
#pragma unroll
    for (int k = 0; k < 5; ++k) {
        const int rx = reflect1(xo + (k - pad), up_W);
        if (rx & 1) continue;
        const int sx = rx / 2;
        const int lx = sx - x_src0;
        if (lx >= 0 && lx < UX) {
            acc = __hadd(acc, __hmul(f[k], tile[ty][lx]));
        } else {
            acc = __hadd(acc, __hmul(f[k], sin[y * in_W + sx]));
        }
    }
    out[b * slice_stride_out + y * out_W + xo] = acc;
}

void launch_up_conv_rows_batched_f16(const __half* in, __half* out,
                                 int in_H, int out_H, int W,
                                 const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream) {
    dim3 block(SP_BX, SP_BY, 1);
    dim3 grid(div_up(W, SP_BX), div_up(out_H, SP_BY), B);
    up_conv_rows_batched_kernel<__half, __half><<<grid, block, 0, stream>>>(
        in, out, in_H, out_H, W, filt, filt_len, stride_in, stride_out, B);
}

void launch_up_conv_rows_batched_f16_halfacc(const __half* in, __half* out,
                                 int in_H, int out_H, int W,
                                 const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream) {
    dim3 block(SP_BX, SP_BY, 1);
    dim3 grid(div_up(W, SP_BX), div_up(out_H, SP_BY), B);
    up_conv_rows_batched_halfacc_kernel<<<grid, block, 0, stream>>>(
        in, out, in_H, out_H, W, filt, filt_len, stride_in, stride_out, B);
}

void launch_up_conv_cols_batched_f16(const __half* in, __half* out,
                                 int H, int in_W, int out_W,
                                 const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream) {
    dim3 block(SP_BX, SP_BY, 1);
    dim3 grid(div_up(out_W, SP_BX), div_up(H, SP_BY), B);
    up_conv_cols_batched_kernel<__half, __half><<<grid, block, 0, stream>>>(
        in, out, H, in_W, out_W, filt, filt_len, stride_in, stride_out, B);
}

void launch_up_conv_cols_batched_f16_halfacc(const __half* in, __half* out,
                                 int H, int in_W, int out_W,
                                 const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream) {
    dim3 block(SP_BX, SP_BY, 1);
    dim3 grid(div_up(out_W, SP_BX), div_up(H, SP_BY), B);
    up_conv_cols_batched_halfacc_kernel<<<grid, block, 0, stream>>>(
        in, out, H, in_W, out_W, filt, filt_len, stride_in, stride_out, B);
}

// ===========================================================================
// OpenCV-style fused corr_dn: cols-then-rows via shared-memory tile.
// PRODUCTION downsample path for batched_lpyr_build / batched_blur_dn_color.
//
// Production order is still cols then rows (matches matlabPyrTools). Unlike
// the earlier dense 5x5 global corr_dn_2d (probe-only, later removed), this
// cooperatively loads a 2D input tile once, then applies horizontal 5-tap then
// vertical 5-tap from smem. Removes the intermediate (H, W/2) global write/read
// between the two passes.
// ===========================================================================

// Output tile (BX, BY); input tile covers 2× in each axis + ±HALO.
constexpr int PD_BX = 32;
constexpr int PD_BY = 8;
constexpr int PD_HALO = 2;
constexpr int PD_X_IN = 2 * PD_BX + 2 * PD_HALO;  // 68
constexpr int PD_Y_IN = 2 * PD_BY + 2 * PD_HALO;  // 20

template <typename In, typename Out>
__global__ void corr_dn_fused_smem_batched_kernel(
    const In* __restrict__ in,
    Out* __restrict__ out,
    int H, int W, const float* filt, int filt_len,
    int slice_stride_in, int slice_stride_out, int B)
{
    const int xo0 = blockIdx.x * PD_BX;
    const int yo0 = blockIdx.y * PD_BY;
    const int b   = blockIdx.z;
    const int Ho  = (H + 1) / 2;
    const int Wo  = (W + 1) / 2;
    if (b >= B) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    float f[5];
#pragma unroll
    for (int k = 0; k < 5; ++k) f[k] = filt[k];

    // Input tile: [2*yo0-HALO, ...) × [2*xo0-HALO, ...)
    __shared__ float tile[PD_Y_IN][PD_X_IN];

    const In* sin = in + b * slice_stride_in;
    const int y_base = 2 * yo0 - PD_HALO;
    const int x_base = 2 * xo0 - PD_HALO;

    const int tile_elems = PD_Y_IN * PD_X_IN;
    const int tid = ty * PD_BX + tx;
    const int nthreads = PD_BX * PD_BY;
    for (int i = tid; i < tile_elems; i += nthreads) {
        const int ly = i / PD_X_IN;
        const int lx = i - ly * PD_X_IN;
        const int gy = reflect1(y_base + ly, H);
        const int gx = reflect1(x_base + lx, W);
        tile[ly][lx] = cvt_in<In>(sin[gy * W + gx]);
    }
    __syncthreads();

    const int xo = xo0 + tx;
    const int yo = yo0 + ty;
    if (xo >= Wo || yo >= Ho) return;

    // Tile coords of centers: global (2*yo, 2*xo) -> tile (2*ty+HALO, 2*tx+HALO)
    const int tcy = 2 * ty + PD_HALO;
    const int tcx = 2 * tx + PD_HALO;

    // cols-then-rows: for each of 5 source rows, horizontal 5-tap, then vertical.
    float mid[5];
#pragma unroll
    for (int ky = 0; ky < 5; ++ky) {
        float acc_h = 0.0f;
        const int ty_src = tcy + (ky - PD_HALO);
#pragma unroll
        for (int kx = 0; kx < 5; ++kx) {
            acc_h += f[kx] * tile[ty_src][tcx + (kx - PD_HALO)];
        }
        mid[ky] = acc_h;
    }
    float acc = 0.0f;
#pragma unroll
    for (int ky = 0; ky < 5; ++ky) {
        acc += f[ky] * mid[ky];
    }
    out[b * slice_stride_out + yo * Wo + xo] = cvt_out<Out>(acc);
}


// ===========================================================================
// EXPERIMENT: half-accumulate fused corr_dn (f16 path only).
// Shared tile + MACs in __half; FP32 production path untouched.
// Numerics differ from float-acc; measure vs float-acc f16.
// ===========================================================================
__global__ void corr_dn_fused_smem_batched_halfacc_kernel(
    const __half* __restrict__ in,
    __half* __restrict__ out,
    int H, int W, const float* filt, int filt_len,
    int slice_stride_in, int slice_stride_out, int B)
{
    const int xo0 = blockIdx.x * PD_BX;
    const int yo0 = blockIdx.y * PD_BY;
    const int b   = blockIdx.z;
    const int Ho  = (H + 1) / 2;
    const int Wo  = (W + 1) / 2;
    if (b >= B) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __half f[5];
#pragma unroll
    for (int k = 0; k < 5; ++k) f[k] = __float2half(filt[k]);

    __shared__ __half tile[PD_Y_IN][PD_X_IN];

    const __half* sin = in + b * slice_stride_in;
    const int y_base = 2 * yo0 - PD_HALO;
    const int x_base = 2 * xo0 - PD_HALO;

    const int tile_elems = PD_Y_IN * PD_X_IN;
    const int tid = ty * PD_BX + tx;
    const int nthreads = PD_BX * PD_BY;
    for (int i = tid; i < tile_elems; i += nthreads) {
        const int ly = i / PD_X_IN;
        const int lx = i - ly * PD_X_IN;
        const int gy = reflect1(y_base + ly, H);
        const int gx = reflect1(x_base + lx, W);
        tile[ly][lx] = sin[gy * W + gx];
    }
    __syncthreads();

    const int xo = xo0 + tx;
    const int yo = yo0 + ty;
    if (xo >= Wo || yo >= Ho) return;

    const int tcy = 2 * ty + PD_HALO;
    const int tcx = 2 * tx + PD_HALO;

    __half mid[5];
#pragma unroll
    for (int ky = 0; ky < 5; ++ky) {
        __half acc_h = __float2half(0.0f);
        const int ty_src = tcy + (ky - PD_HALO);
#pragma unroll
        for (int kx = 0; kx < 5; ++kx) {
            acc_h = __hadd(acc_h, __hmul(f[kx], tile[ty_src][tcx + (kx - PD_HALO)]));
        }
        mid[ky] = acc_h;
    }
    __half acc = __float2half(0.0f);
#pragma unroll
    for (int ky = 0; ky < 5; ++ky) {
        acc = __hadd(acc, __hmul(f[ky], mid[ky]));
    }
    out[b * slice_stride_out + yo * Wo + xo] = acc;
}

void launch_corr_dn_fused_smem_batched(const float* in, float* out,
                                       int H, int W, const float* filt, int filt_len,
                                       int stride_in, int stride_out, int B,
                                       cudaStream_t stream) {
    const int Ho = (H + 1) / 2;
    const int Wo = (W + 1) / 2;
    dim3 block(PD_BX, PD_BY, 1);
    dim3 grid(div_up(Wo, PD_BX), div_up(Ho, PD_BY), B);
    corr_dn_fused_smem_batched_kernel<float, float><<<grid, block, 0, stream>>>(
        in, out, H, W, filt, filt_len, stride_in, stride_out, B);
}

void launch_corr_dn_fused_smem_batched_f16(const __half* in, __half* out,
                                       int H, int W, const float* filt, int filt_len,
                                       int stride_in, int stride_out, int B,
                                       cudaStream_t stream) {
    const int Ho = (H + 1) / 2;
    const int Wo = (W + 1) / 2;
    dim3 block(PD_BX, PD_BY, 1);
    dim3 grid(div_up(Wo, PD_BX), div_up(Ho, PD_BY), B);
    // Production: float accumulate (cvt_in/out). Half-acc: *_f16_halfacc.
    corr_dn_fused_smem_batched_kernel<__half, __half><<<grid, block, 0, stream>>>(
        in, out, H, W, filt, filt_len, stride_in, stride_out, B);
}

void launch_corr_dn_fused_smem_batched_f16_halfacc(const __half* in, __half* out,
                                       int H, int W, const float* filt, int filt_len,
                                       int stride_in, int stride_out, int B,
                                       cudaStream_t stream) {
    const int Ho = (H + 1) / 2;
    const int Wo = (W + 1) / 2;
    dim3 block(PD_BX, PD_BY, 1);
    dim3 grid(div_up(Wo, PD_BX), div_up(Ho, PD_BY), B);
    corr_dn_fused_smem_batched_halfacc_kernel<<<grid, block, 0, stream>>>(
        in, out, H, W, filt, filt_len, stride_in, stride_out, B);
}


}  // namespace evm
