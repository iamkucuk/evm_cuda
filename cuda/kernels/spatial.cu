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
// Numerical contract (tolerance < 1e-5 vs Python, AGENTS.md §2):
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

template <typename In, typename Out>
__launch_bounds__(1024, 2)
__global__ void corr_dn_rows_batched_kernel(
    const In* __restrict__ in,
    Out* __restrict__ out,
    int H, int W, const float* filt, int filt_len,
    int slice_stride_in, int slice_stride_out, int B)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int yo = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    const int Ho = (H + 1) / 2;
    if (x >= W || yo >= Ho || b >= B) return;

    float f[5];
    #pragma unroll
    for (int k = 0; k < 5; ++k) f[k] = filt[k];
    const int src_center = 2 * yo;
    const int pad = filt_len / 2;
    const In* sin = in + b * slice_stride_in;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 5; ++k) {
        int src = reflect1(src_center + (k - pad), H);
        acc += f[k] * cvt_in<In>(sin[src * W + x]);
    }
    out[b * slice_stride_out + yo * W + x] = cvt_out<Out>(acc);
}

template <typename In, typename Out>
__launch_bounds__(1024, 2)
__global__ void corr_dn_cols_batched_kernel(
    const In* __restrict__ in,
    Out* __restrict__ out,
    int H, int W, const float* filt, int filt_len,
    int slice_stride_in, int slice_stride_out, int B)
{
    const int xo = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    const int Wo = (W + 1) / 2;
    if (xo >= Wo || y >= H || b >= B) return;

    float f[5];
    #pragma unroll
    for (int k = 0; k < 5; ++k) f[k] = filt[k];
    const int src_center = 2 * xo;
    const int pad = filt_len / 2;
    const In* sin = in + b * slice_stride_in;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 5; ++k) {
        int src = reflect1(src_center + (k - pad), W);
        acc += f[k] * cvt_in<In>(sin[y * W + src]);
    }
    out[b * slice_stride_out + y * Wo + xo] = cvt_out<Out>(acc);
}

template <typename In, typename Out>
__launch_bounds__(1024, 2)
__global__ void up_conv_rows_batched_kernel(
    const In* __restrict__ in,
    Out* __restrict__ out,
    int in_H, int out_H, int W,
    const float* filt, int filt_len,
    int slice_stride_in, int slice_stride_out, int B)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int yo = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (x >= W || yo >= out_H || b >= B) return;

    float f[5];
    #pragma unroll
    for (int k = 0; k < 5; ++k) f[k] = filt[k];
    const int pad = filt_len / 2;
    const int up_H = 2 * in_H;
    const In* sin = in + b * slice_stride_in;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 5; ++k) {
        int u_idx = yo + (k - pad);
        int r = reflect1(u_idx, up_H);
        if ((r & 1) == 0) {
            int src = r / 2;
            acc += f[k] * cvt_in<In>(sin[src * W + x]);
        }
    }
    out[b * slice_stride_out + yo * W + x] = cvt_out<Out>(acc);
}

template <typename In, typename Out>
__launch_bounds__(1024, 2)
__global__ void up_conv_cols_batched_kernel(
    const In* __restrict__ in,
    Out* __restrict__ out,
    int H, int in_W, int out_W,
    const float* filt, int filt_len,
    int slice_stride_in, int slice_stride_out, int B)
{
    const int xo = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (xo >= out_W || y >= H || b >= B) return;

    float f[5];
    #pragma unroll
    for (int k = 0; k < 5; ++k) f[k] = filt[k];
    const int pad = filt_len / 2;
    const int up_W = 2 * in_W;
    const In* sin = in + b * slice_stride_in;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 5; ++k) {
        int u_idx = xo + (k - pad);
        int r = reflect1(u_idx, up_W);
        if ((r & 1) == 0) {
            int src = r / 2;
            acc += f[k] * cvt_in<In>(sin[y * in_W + src]);
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
    dim3 block(32, 32, 1);
    dim3 grid(div_up(W, 32), div_up(Ho, 32), B);
    corr_dn_rows_batched_kernel<float, float><<<grid, block, 0, stream>>>(
        in, out, H, W, filt, filt_len, stride_in, stride_out, B);
}

void launch_corr_dn_cols_batched(const float* in, float* out,
                                 int H, int W, const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream) {
    int Wo = (W + 1) / 2;
    dim3 block(32, 32, 1);
    dim3 grid(div_up(Wo, 32), div_up(H, 32), B);
    corr_dn_cols_batched_kernel<float, float><<<grid, block, 0, stream>>>(
        in, out, H, W, filt, filt_len, stride_in, stride_out, B);
}

void launch_up_conv_rows_batched(const float* in, float* out,
                                 int in_H, int out_H, int W,
                                 const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(div_up(W, 32), div_up(out_H, 32), B);
    up_conv_rows_batched_kernel<float, float><<<grid, block, 0, stream>>>(
        in, out, in_H, out_H, W, filt, filt_len, stride_in, stride_out, B);
}

void launch_up_conv_cols_batched(const float* in, float* out,
                                 int H, int in_W, int out_W,
                                 const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(div_up(out_W, 32), div_up(H, 32), B);
    up_conv_cols_batched_kernel<float, float><<<grid, block, 0, stream>>>(
        in, out, H, in_W, out_W, filt, filt_len, stride_in, stride_out, B);
}

// --- batched launchers (FP16 storage, FP32 compute) ------------------------

void launch_corr_dn_rows_batched_f16(const __half* in, __half* out,
                                 int H, int W, const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream) {
    int Ho = (H + 1) / 2;
    dim3 block(32, 32, 1);
    dim3 grid(div_up(W, 32), div_up(Ho, 32), B);
    corr_dn_rows_batched_kernel<__half, __half><<<grid, block, 0, stream>>>(
        in, out, H, W, filt, filt_len, stride_in, stride_out, B);
}

void launch_corr_dn_cols_batched_f16(const __half* in, __half* out,
                                 int H, int W, const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream) {
    int Wo = (W + 1) / 2;
    dim3 block(32, 32, 1);
    dim3 grid(div_up(Wo, 32), div_up(H, 32), B);
    corr_dn_cols_batched_kernel<__half, __half><<<grid, block, 0, stream>>>(
        in, out, H, W, filt, filt_len, stride_in, stride_out, B);
}

void launch_up_conv_rows_batched_f16(const __half* in, __half* out,
                                 int in_H, int out_H, int W,
                                 const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(div_up(W, 32), div_up(out_H, 32), B);
    up_conv_rows_batched_kernel<__half, __half><<<grid, block, 0, stream>>>(
        in, out, in_H, out_H, W, filt, filt_len, stride_in, stride_out, B);
}

void launch_up_conv_cols_batched_f16(const __half* in, __half* out,
                                 int H, int in_W, int out_W,
                                 const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(div_up(out_W, 32), div_up(H, 32), B);
    up_conv_cols_batched_kernel<__half, __half><<<grid, block, 0, stream>>>(
        in, out, H, in_W, out_W, filt, filt_len, stride_in, stride_out, B);
}

}  // namespace evm
