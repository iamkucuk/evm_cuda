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

}  // namespace evm
