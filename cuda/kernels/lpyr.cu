// lpyr.cu — Laplacian pyramid build/reconstruct, host-orchestrated.
//
// Port of evm/pyramids.py:build_lpyr / recon_lpyr, which mirror
// matlabPyrTools buildLpyr / reconLpyr. The host owns the per-level size
// bookkeeping (pind) and the level loop; each level calls into spatial.cu's
// corr_dn / up_conv kernels. Device memory for the level bands is managed
// by the caller (see evm_cuda/pipelines.py).
//
// Layout: each level band is a flat (H_l * W_l) float32 buffer on the device.
// The host keeps an array of device pointers + the per-level (H_l, W_l).
//
// Numerical contract (< 1e-5 vs Python):
//   - Uses BINOM5 (L2-normalized, NOT renormalized to sum=1) — see the
//     warning in evm/pyramids.py:30-32 and evm_common.cuh.
//   - build_lpyr downsamples columns-then-rows; the band at each level is
//     (current image) - upsample(downsample(current image)). The coarsest
//     band IS the residual lowpass (not a difference).
//   - recon_lpyr is the exact inverse: band + upsample(recon of sub-pyramid).
//
// Note on the round-trip tolerance: evm/'s FP64 round-trip is < 1e-9; the
// CUDA port's FP32 round-trip is held to < 1e-5 (AGENTS.md table).

#include <vector>
#include "../include/evm_common.cuh"
#include "../include/evm_check.cuh"
#include <cuda_fp16.h>

namespace evm {

// Forward decls from spatial.cu (same namespace, separate translation unit).
void launch_corr_dn_rows(const float* in, float* out, int H, int W,
                         const float* filt, int filt_len, cudaStream_t stream);
void launch_corr_dn_cols(const float* in, float* out, int H, int W,
                         const float* filt, int filt_len, cudaStream_t stream);
void launch_up_conv_rows(const float* in, float* out,
                         int in_H, int out_H, int W,
                         const float* filt, int filt_len, cudaStream_t stream);
void launch_up_conv_cols(const float* in, float* out,
                         int H, int in_W, int out_W,
                         const float* filt, int filt_len, cudaStream_t stream);
// Batched variants (from spatial.cu) — process B slices per launch.
void launch_corr_dn_rows_batched(const float* in, float* out,
                                 int H, int W, const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream);
void launch_corr_dn_cols_batched(const float* in, float* out,
                                 int H, int W, const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream);
void launch_up_conv_rows_batched(const float* in, float* out,
                                 int in_H, int out_H, int W,
                                 const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream);
void launch_up_conv_cols_batched(const float* in, float* out,
                                 int H, int in_W, int out_W,
                                 const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream);

// --- elementwise helpers (file-local) --------------------------------------

__global__ void subtract_inplace_kernel(
    const float* a, float* b, int n)  // b := a - b
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    b[idx] = a[idx] - b[idx];
}

__global__ void add_inplace_kernel(
    const float* a, float* b, int n)  // b := a + b
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    b[idx] = a[idx] + b[idx];
}

// --- host orchestration ----------------------------------------------------

// Compute the per-level (H, W) sizes, equivalent to matlabPyrTools maxPyrHt.
// Pure host-side bookkeeping; no device work.
std::vector<std::pair<int, int>> lpyr_level_sizes(int H, int W, int levels) {
    std::vector<std::pair<int, int>> sizes;
    sizes.reserve(levels);
    int h = H, w = W;
    for (int l = 0; l < levels; ++l) {
        sizes.emplace_back(h, w);
        h = (h + 1) / 2;
        w = (w + 1) / 2;
    }
    return sizes;
}

// Build a Laplacian pyramid from `img` (H*W float32 on device).
// Caller pre-allocates each band buffer (band[l]) according to lpyr_level_sizes.
// Uses BINOM5 (passed in via filt for testability; production passes kBinom5).
//
// Recursion unrolled iteratively: at each level, compute lo2 = downsample(img),
// hi2 = upsample(lo2) back to current size, band[l] = img - hi2, descend.
// The coarsest band[levels-1] gets the residual lowpass (lo2 of the deepest).
void lpyr_build_device(
    const float* img, int H, int W,
    float** band_ptrs,                 // [levels] device pointers (caller-alloc)
    const std::pair<int, int>* sizes,  // [levels]
    int levels,
    const float* filt, int filt_len,
    float* scratch_a, float* scratch_b, float* scratch_c,  // device scratch
    cudaStream_t stream)
{
    // We need a working buffer chain. scratch_a holds current image (size of
    // current level), scratch_b holds lo (downsampled in x), scratch_c holds
    // lo2 (downsampled in x then y). At each level we also need a buffer to
    // hold hi (= upsample of lo2 in y) and hi2 (= upsample of hi in x) at the
    // CURRENT image size; we reuse band[l] as that scratch and overwrite it
    // with (img - hi2) at the end.
    //
    // Copy img -> scratch_a (current).
    CUDA_CHECK(cudaMemcpyAsync(scratch_a, img, H * W * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));

    for (int l = 0; l < levels; ++l) {
        const auto [h, w] = sizes[l];
        int hn = (h + 1) / 2;
        int wn = (w + 1) / 2;

        if (l == levels - 1) {
            // Coarsest level: residual lowpass = the current image.
            CUDA_CHECK(cudaMemcpyAsync(band_ptrs[l], scratch_a,
                                       h * w * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
            break;
        }

        // lo  = corr_dn(scratch_a, axis=1) -> (h, wn)
        launch_corr_dn_cols(scratch_a, scratch_b, h, w, filt, filt_len, stream);
        // lo2 = corr_dn(lo,     axis=0) -> (hn, wn)
        launch_corr_dn_rows(scratch_b, scratch_c, h, wn, filt, filt_len, stream);

        // hi  = up_conv(lo2, axis=0, out_size=h) -> (h, wn)
        launch_up_conv_rows(scratch_c, scratch_b, hn, h, wn,
                            filt, filt_len, stream);
        // hi2 = up_conv(hi,  axis=1, out_size=w) -> (h, w)
        launch_up_conv_cols(scratch_b, band_ptrs[l], h, wn, w,
                            filt, filt_len, stream);

        // band[l] = scratch_a - hi2  (per element)
        subtract_inplace_kernel<<<div_up(h*w, 256), 256, 0, stream>>>(
            scratch_a, band_ptrs[l], h * w);

        // Descend: scratch_a := lo2 (next current image).
        // We need scratch_c's contents to survive into the next iteration as
        // scratch_a. Ping-pong by swapping the two device pointers via a copy.
        CUDA_CHECK(cudaMemcpyAsync(scratch_a, scratch_c,
                                   hn * wn * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

// Reconstruct an image from its Laplacian pyramid. Inverse of lpyr_build_device.
// Caller passes band_ptrs + sizes; output written to `out` (H*W float32).
void lpyr_recon_device(
    const float* const* band_ptrs,
    const std::pair<int, int>* sizes,
    int levels,
    const float* filt, int filt_len,
    float* out,
    float* scratch_lo,    // device, size of coarsest band
    float* scratch_hi,    // device, size of intermediate level
    cudaStream_t stream)
{
    // Walk from coarsest to finest. Start with the residual (coarsest band).
    const auto [ch, cw] = sizes[levels - 1];
    CUDA_CHECK(cudaMemcpyAsync(scratch_lo, band_ptrs[levels - 1],
                               ch * cw * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));

    for (int l = levels - 2; l >= 0; --l) {
        const auto [h, w] = sizes[l];
        const auto [ph, pw] = sizes[l + 1];
        // hi  = up_conv(scratch_lo, axis=0, out_size=h) -> (h, pw)
        launch_up_conv_rows(scratch_lo, scratch_hi, ph, h, pw,
                            filt, filt_len, stream);
        // res = up_conv(hi, axis=1, out_size=w) -> (h, w)
        launch_up_conv_cols(scratch_hi, out, h, pw, w,
                            filt, filt_len, stream);
        // out = band[l] + res
        add_inplace_kernel<<<div_up(h*w, 256), 256, 0, stream>>>(
            band_ptrs[l], out, h * w);

        // scratch_lo := out  (becomes the input for the next finer level)
        if (l > 0) {
            CUDA_CHECK(cudaMemcpyAsync(scratch_lo, out, h * w * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
        }
    }
}

// ===========================================================================
// Scatter/gather kernels for the channel-major band layout.
//
// The band buffer is laid out (level, channel, frame, spatial) so that Stage C's
// temporal filter sees contiguous (T, N) blocks. This makes per-slice offsets
// irregular: slice_off(m) = (m%3)*n_frames + m/3. These kernels bridge the
// frame-major scratch buffers (regular strides) and the channel-major band
// storage (scattered) via a pre-computed offset table.
//
// Grid: (ceil(n_per_slice/256), B). blockIdx.y = slice index m.
// ===========================================================================

// Scatter-subtract: dst[offsets[m] + px] = a[m*stride + px] - b[m*stride + px].
// Used to write band[l] = current - hi2 into channel-major band storage.
__global__ void scatter_subtract_kernel(
    const float* __restrict__ a,       // frame-major (B, n_per_slice)
    const float* __restrict__ b,       // frame-major (B, n_per_slice)
    float* __restrict__ dst,           // scattered (channel-major)
    const int* __restrict__ offsets,   // [B] offset in floats per slice
    int n_per_slice, int B)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (px >= n_per_slice || m >= B) return;
    int ai = m * n_per_slice + px;
    dst[offsets[m] + px] = a[ai] - b[ai];
}

// Scatter-copy: dst[offsets[m] + px] = src[m*stride + px].
// Used for the coarsest band (residual lowpass).
__global__ void scatter_kernel(
    const float* __restrict__ src,     // frame-major (B, n_per_slice)
    float* __restrict__ dst,           // scattered (channel-major)
    const int* __restrict__ offsets,   // [B] offset in floats per slice
    int n_per_slice, int B)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (px >= n_per_slice || m >= B) return;
    dst[offsets[m] + px] = src[m * n_per_slice + px];
}

// Gather-add: dst[m*stride + px] = src[offsets[m] + px] + b[m*stride + px].
// Used in recon: out = band[l] (scattered) + res (frame-major).
__global__ void gather_add_kernel(
    const float* __restrict__ src,     // scattered band data (channel-major)
    const float* __restrict__ b,       // frame-major res (B, n_per_slice)
    float* __restrict__ dst,           // frame-major output (B, n_per_slice)
    const int* __restrict__ offsets,   // [B] offset in floats per slice
    int n_per_slice, int B)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (px >= n_per_slice || m >= B) return;
    int di = m * n_per_slice + px;
    dst[di] = src[offsets[m] + px] + b[di];
}

// Gather: dst[m*stride + px] = src[offsets[m] + px].
// Used in recon to read the coarsest band (scattered) into frame-major scratch.
__global__ void gather_kernel(
    const float* __restrict__ src,     // scattered band data (channel-major)
    float* __restrict__ dst,           // frame-major (B, n_per_slice)
    const int* __restrict__ offsets,   // [B] offset in floats per slice
    int n_per_slice, int B)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (px >= n_per_slice || m >= B) return;
    dst[m * n_per_slice + px] = src[offsets[m] + px];
}

void launch_scatter_subtract(const float* a, const float* b, float* dst,
                             const int* offsets, int n_per_slice, int B,
                             cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid(div_up(n_per_slice, 256), B, 1);
    scatter_subtract_kernel<<<grid, block, 0, stream>>>(
        a, b, dst, offsets, n_per_slice, B);
}

void launch_scatter(const float* src, float* dst,
                    const int* offsets, int n_per_slice, int B,
                    cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid(div_up(n_per_slice, 256), B, 1);
    scatter_kernel<<<grid, block, 0, stream>>>(
        src, dst, offsets, n_per_slice, B);
}

void launch_gather_add(const float* src, const float* b, float* dst,
                       const int* offsets, int n_per_slice, int B,
                       cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid(div_up(n_per_slice, 256), B, 1);
    gather_add_kernel<<<grid, block, 0, stream>>>(
        src, b, dst, offsets, n_per_slice, B);
}

void launch_gather(const float* src, float* dst,
                   const int* offsets, int n_per_slice, int B,
                   cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid(div_up(n_per_slice, 256), B, 1);
    gather_kernel<<<grid, block, 0, stream>>>(
        src, dst, offsets, n_per_slice, B);
}

// ===========================================================================
// FP16 variants: read __half scratch, convert to float, write float bands.
// Used when lpyr_build's scratch buffers are stored as __half to save VRAM.
// ===========================================================================

__global__ void scatter_subtract_f16_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ dst,
    const int* __restrict__ offsets,
    int n_per_slice, int B)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (px >= n_per_slice || m >= B) return;
    int ai = m * n_per_slice + px;
    dst[offsets[m] + px] = __half2float(a[ai]) - __half2float(b[ai]);
}

__global__ void scatter_f16_kernel(
    const __half* __restrict__ src,
    float* __restrict__ dst,
    const int* __restrict__ offsets,
    int n_per_slice, int B)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (px >= n_per_slice || m >= B) return;
    dst[offsets[m] + px] = __half2float(src[m * n_per_slice + px]);
}

void launch_scatter_subtract_f16(const __half* a, const __half* b, float* dst,
                                 const int* offsets, int n_per_slice, int B,
                                 cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid(div_up(n_per_slice, 256), B, 1);
    scatter_subtract_f16_kernel<<<grid, block, 0, stream>>>(
        a, b, dst, offsets, n_per_slice, B);
}

void launch_scatter_f16(const __half* src, float* dst,
                    const int* offsets, int n_per_slice, int B,
                    cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid(div_up(n_per_slice, 256), B, 1);
    scatter_f16_kernel<<<grid, block, 0, stream>>>(
        src, dst, offsets, n_per_slice, B);
}

}  // namespace evm
