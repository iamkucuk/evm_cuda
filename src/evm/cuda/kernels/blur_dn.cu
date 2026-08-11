// blur_dn.cu — Gaussian blur+downsample (color pipeline), host-orchestrated.
//
// Port of evm/pyramids.py:blur_dn / blur_dn_clr, which mirror matlabPyrTools
// blurDn / blurDnClr. Repeatedly applies corr_dn separably (rows then cols),
// `nlevs` times, per color channel. Each call halves H and W.
//
// Uses BINOM5_SUM1 (sum-normalized to 1.0); blurDn re-normalizes filt/sum(filt)
// at every call (evm/pyramids.py:129), which is a no-op for BINOM5_SUM1 but
// part of the contract.
//
// The host loop runs the 2*nlevs corr_dn launches per channel. We expose a
// single-channel variant; the wrapper layer iterates over channels.
//
// Numerical contract (< 1e-5 vs Python): same spatial primitives as the
// Laplacian pyramid, so the same FP32 precision budget applies.

#include <vector>
#include "../include/evm_common.cuh"
#include "../include/evm_check.cuh"

namespace evm {

void launch_corr_dn_rows(const float* in, float* out, int H, int W,
                         const float* filt, int filt_len, cudaStream_t stream);
void launch_corr_dn_cols(const float* in, float* out, int H, int W,
                         const float* filt, int filt_len, cudaStream_t stream);

// Blur+downsample a single 2-D channel `nlevs` times, in place via ping-pong
// scratch buffers. Caller allocates scratch_a (size = max level size) and
// scratch_b (size = next-coarser level size). Output ends up in `out` (size
// of the final coarsest level); smaller buffers are returned by the caller
// via the level-size table.
//
// For simplicity we let the caller pass a pair of equal-sized scratch buffers
// sized to the LARGEST level (H*W) — the extra capacity is unused at coarse
// levels. This trades a little device memory for simpler bookkeeping.
void blur_dn_device(
    const float* in, int H, int W,
    float* out,
    int nlevs,
    const float* filt, int filt_len,
    float* scratch_a, float* scratch_b,   // each >= H*W floats
    cudaStream_t stream)
{
    if (nlevs <= 0) {
        CUDA_CHECK(cudaMemcpyAsync(out, in, H * W * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
        return;
    }
    // current := in
    CUDA_CHECK(cudaMemcpyAsync(scratch_a, in, H * W * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));
    float* cur = scratch_a;
    float* nxt = scratch_b;
    int ch = H, cw = W;
    for (int l = 0; l < nlevs; ++l) {
        int wn = (cw + 1) / 2;   // after cols downsample
        int hn = (ch + 1) / 2;   // after rows downsample (cols already done)
        // cols downsample: cur (ch x cw) -> nxt (ch x wn)
        launch_corr_dn_cols(cur, nxt, ch, cw, filt, filt_len, stream);
        // rows downsample: nxt (ch x wn) -> cur (hn x wn)
        launch_corr_dn_rows(nxt, cur, ch, wn, filt, filt_len, stream);
        ch = hn; cw = wn;
        // cur now holds the downsampled image; loop continues.
    }
    CUDA_CHECK(cudaMemcpyAsync(out, cur, ch * cw * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));
}

}  // namespace evm
