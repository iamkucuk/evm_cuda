// amplify_render.cu — the final stage of every EVM pipeline.
//
// Combines, per pixel and per channel:
//   1. (optional) per-channel gain (color pipeline: [alpha, alpha*chromAtt x2])
//   2. add-back to the original NTSC frame
//   3. (motion pipelines) chromAtt on the I,Q channels post-reconstruction
//   4. NTSC -> RGB, clip [0,1], *255, banker's-round, cast to uint8 BGR
//
// Two specialized kernels are exposed because the two pipelines attenuate
// chrominance at different points (evm/magnify.py):
//
//   - magnify_color_gdown_ideal:   gain applied to the FILTERED signal
//                                  before upsample+add (Python does this in
//                                  numpy before calling cv2.resize). So the
//                                  kernel here only does ntsc->bgr.
//   - magnify_motion_lpyr_*:       the per-level gain is applied during the
//                                  pyramid reconstruction; here we add the
//                                  reconstructed (chromAtt-scaled) delta to
//                                  the frame and quantize.
//
// For both pipelines the FINAL per-pixel step (ntsc -> bgr u8 with clip and
// banker's rounding) is identical — that's `ntsc_to_bgr_u8`, which is the
// same math as ntsc_f32_to_bgr_u8_kernel in color_cvt.cu (we re-export that
// launcher rather than duplicating it).

#include "../include/evm_common.cuh"

namespace evm {

// Forward decl from color_cvt.cu.
void launch_ntsc_f32_to_bgr_u8(const float* yiq, unsigned char* bgr,
                                int H, int W, cudaStream_t stream);

// Apply per-channel gain to a filtered signal (color pipeline).
// in-place: gain[0]*Y, gain[1]*I, gain[2]*Q.
__global__ void apply_channel_gain_kernel(
    float* __restrict__ sig, int H, int W,
    float g0, float g1, float g2)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int px = (y * W + x) * 3;
    sig[px + 0] *= g0;
    sig[px + 1] *= g1;
    sig[px + 2] *= g2;
}

// Add the (already-gained and chromAtt-attenuated) filtered delta to the
// NTSC frame, then convert to BGR u8. Single fused kernel saves a global
// write/read pair vs. add-then-convert.
__global__ void add_and_quantize_kernel(
    const float* __restrict__ ntsc_frame,  // (H,W,3)
    const float* __restrict__ delta,       // (H,W,3) — reconstruction
    unsigned char* __restrict__ bgr_out,   // (H,W,3)
    int H, int W)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int px = (y * W + x) * 3;

    float y_ = ntsc_frame[px + 0] + delta[px + 0];
    float i_ = ntsc_frame[px + 1] + delta[px + 1];
    float q_ = ntsc_frame[px + 2] + delta[px + 2];

    float r = kYiqToRgb[0][0]*y_ + kYiqToRgb[0][1]*i_ + kYiqToRgb[0][2]*q_;
    float g = kYiqToRgb[1][0]*y_ + kYiqToRgb[1][1]*i_ + kYiqToRgb[1][2]*q_;
    float b = kYiqToRgb[2][0]*y_ + kYiqToRgb[2][1]*i_ + kYiqToRgb[2][2]*q_;
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);
    bgr_out[px + 0] = static_cast<unsigned char>(rintf(b * 255.0f));
    bgr_out[px + 1] = static_cast<unsigned char>(rintf(g * 255.0f));
    bgr_out[px + 2] = static_cast<unsigned char>(rintf(r * 255.0f));
}

// Scale the I,Q channels of a delta buffer by chromAtt (motion pipelines).
// Y is left untouched. evm/magnify.py:_amplify_lpyr_stack.
__global__ void attenuate_chrom_kernel(
    float* __restrict__ delta, int H, int W, float chrom_att)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int px = (y * W + x) * 3;
    delta[px + 1] *= chrom_att;
    delta[px + 2] *= chrom_att;
}

void launch_apply_channel_gain(float* sig, int H, int W,
                               float g0, float g1, float g2,
                               cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(div_up(W, 32), div_up(H, 32), 1);
    apply_channel_gain_kernel<<<grid, block, 0, stream>>>(sig, H, W, g0, g1, g2);
}

void launch_attenuate_chrom(float* delta, int H, int W, float chrom_att,
                            cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(div_up(W, 32), div_up(H, 32), 1);
    attenuate_chrom_kernel<<<grid, block, 0, stream>>>(delta, H, W, chrom_att);
}

// Bilinear upsample of a stack of M frames, each (in_H, in_W, 3), to
// (out_H, out_W, 3). Replaces host-side cv2.resize(INTER_LINEAR) in the
// color pipeline's render stage.
//
// Coordinate convention (reverse-engineered from cv2, bit-exact match):
//   sx = (x + 0.5) * (in_W / out_W) - 0.5   (half-pixel centers)
//   border: replicate (clamp to edge), NOT reflect-101.
//
// Each thread handles one output pixel (all 3 channels in registers).
__global__ void bilinear_upsample_3ch_kernel(
    const float* __restrict__ src,   // (M, in_H, in_W, 3)
    float* __restrict__ dst,         // (M, out_H, out_W, 3)
    int M, int in_H, int in_W, int out_H, int out_W)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * out_H * out_W;
    if (idx >= total) return;

    const int x = idx % out_W;
    const int tmp = idx / out_W;
    const int y = tmp % out_H;
    const int m = tmp / out_H;

    const float scale_x = static_cast<float>(in_W) / out_W;
    const float scale_y = static_cast<float>(in_H) / out_H;
    float sx = (x + 0.5f) * scale_x - 0.5f;
    float sy = (y + 0.5f) * scale_y - 0.5f;

    int x0 = static_cast<int>(floorf(sx));
    int y0 = static_cast<int>(floorf(sy));
    const float fx = sx - x0;
    const float fy = sy - y0;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    // Replicate border (clamp to edge).
    x0 = max(0, min(x0, in_W - 1));
    x1 = max(0, min(x1, in_W - 1));
    y0 = max(0, min(y0, in_H - 1));
    y1 = max(0, min(y1, in_H - 1));

    const float* s = src + static_cast<size_t>(m) * in_H * in_W * 3;
    float* d = dst + static_cast<size_t>(m) * out_H * out_W * 3;

    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w01 = fx * (1.0f - fy);
    const float w10 = (1.0f - fx) * fy;
    const float w11 = fx * fy;

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float v00 = s[(y0 * in_W + x0) * 3 + c];
        const float v01 = s[(y0 * in_W + x1) * 3 + c];
        const float v10 = s[(y1 * in_W + x0) * 3 + c];
        const float v11 = s[(y1 * in_W + x1) * 3 + c];
        d[(y * out_W + x) * 3 + c] = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
    }
}

void launch_bilinear_upsample_3ch(const float* src, float* dst,
                                  int M, int in_H, int in_W,
                                  int out_H, int out_W, cudaStream_t stream) {
    const int block = 256;
    const int grid = div_up(M * out_H * out_W, block);
    bilinear_upsample_3ch_kernel<<<grid, block, 0, stream>>>(
        src, dst, M, in_H, in_W, out_H, out_W);
}

void launch_add_and_quantize(const float* ntsc_frame, const float* delta,
                             unsigned char* bgr_out, int H, int W,
                             cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(div_up(W, 32), div_up(H, 32), 1);
    add_and_quantize_kernel<<<grid, block, 0, stream>>>(
        ntsc_frame, delta, bgr_out, H, W);
}

// Plain ntsc->bgr u8 conversion for the color pipeline (where the add-back
// has already happened in NTSC space, possibly after a CPU/GPU upsample).
// Re-uses the canonical implementation in color_cvt.cu.
void launch_ntsc_to_bgr_u8(const float* ntsc, unsigned char* bgr,
                           int H, int W, cudaStream_t stream) {
    launch_ntsc_f32_to_bgr_u8(ntsc, bgr, H, W, stream);
}

}  // namespace evm
