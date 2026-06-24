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
// Grid: (ceil(W/32), ceil(H/32))  Block: (32, 32, 1)
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

// Add the (already-gained and optionally chromAtt-attenuated) filtered delta
// to the NTSC frame, then convert to BGR u8. Single fused kernel saves a
// global write/read pair vs. add-then-convert.
//
// chrom_att scales the I,Q channels of delta BEFORE the add. When chrom_att=1
// (color pipeline) this is a no-op and the kernel is identical to the old
// version. For the motion pipeline, passing chrom_attenuation here lets us
// skip the separate attenuate_chrom kernel launch entirely (one fewer
// full-res pass over the delta buffer).
//
// Grid: (ceil(W/32), ceil(H/32))  Block: (32, 32, 1)
__global__ void add_and_quantize_kernel(
    const float* __restrict__ ntsc_frame,  // (H,W,3)
    const float* __restrict__ delta,       // (H,W,3) — reconstruction
    unsigned char* __restrict__ bgr_out,   // (H,W,3)
    int H, int W, float chrom_att)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int px = (y * W + x) * 3;

    // Fold chromAtt into the delta read (motion pipeline). chrom_att=1.0
    // (color pipeline) makes this a plain copy.
    float y_ = ntsc_frame[px + 0] + delta[px + 0];
    float i_ = ntsc_frame[px + 1] + delta[px + 1] * chrom_att;
    float q_ = ntsc_frame[px + 2] + delta[px + 2] * chrom_att;

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

// Fused add + NTSC->BGR quantize reading delta from PLANAR layout.
//
// The motion pipeline's lpyr_recon outputs delta in planar (n*3, H, W) layout
// (frame-major, then channel). The existing add_and_quantize expects interleaved
// (n, H, W, 3) delta — requiring a separate planar_to_interleaved_3ch transpose
// pass first. This variant reads the 3 delta channels directly from planar
// layout, folding the transpose inline and eliminating the intermediate buffer
// + one full-res kernel pass.
//
// Fused add + NTSC->BGR quantize reading delta from PLANAR layout.
//
// The motion pipeline's lpyr_recon outputs delta in planar (n*3, H, W) layout
// (frame-major, then channel). The existing add_and_quantize expects interleaved
// (n, H, W, 3) delta, requiring a separate planar_to_interleaved_3ch transpose
// pass first. This variant reads the 3 delta channels directly from planar
// layout, folding the transpose inline and eliminating the intermediate buffer
// + one full-res kernel pass.
//
// Planar delta layout: delta[(f*3 + c) * H * W + y * W + x] for frame f, chan c.
//
// Grid: (ceil(W/(32*ELEMS)), ceil(H/32), n)  Block: (32, 32, 1)
// Each thread processes ELEMS adjacent pixels along x (multiple elements per
// thread). This amortizes the per-pixel overhead and gives the
// compiler independent memory operations to pipeline for latency hiding.
constexpr int ADD_PLANAR_ELEMS = 4;

template <typename NTSC_T>
__global__ void add_planar_quantize_kernel(
    const NTSC_T* __restrict__ ntsc,
    const NTSC_T* __restrict__ delta_planar,
    unsigned char* __restrict__ bgr_out,
    int n, int H, int W, float chrom_att)
{
    const int x0 = blockIdx.x * blockDim.x * ADD_PLANAR_ELEMS + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int f = blockIdx.z;
    if (y >= H || f >= n) return;

    const int spatial_base = y * W + x0;
    const int px_base = (f * H * W + spatial_base) * 3;
    const NTSC_T* dplane = delta_planar + static_cast<size_t>(f) * 3 * H * W;
    const NTSC_T* dy_plane = dplane;
    const NTSC_T* di_plane = dplane + H * W;
    const NTSC_T* dq_plane = dplane + 2 * H * W;

    #pragma unroll
    for (int e = 0; e < ADD_PLANAR_ELEMS; ++e) {
        const int x = x0 + e * 32;
        if (x >= W) break;
        const int spatial = spatial_base + e * 32;
        const int px = px_base + e * 32 * 3;

        float dy = cvt_in<NTSC_T>(dy_plane[spatial]);
        float di = cvt_in<NTSC_T>(di_plane[spatial]) * chrom_att;
        float dq = cvt_in<NTSC_T>(dq_plane[spatial]) * chrom_att;

        float y_ = cvt_in<NTSC_T>(ntsc[px + 0]) + dy;
        float i_ = cvt_in<NTSC_T>(ntsc[px + 1]) + di;
        float q_ = cvt_in<NTSC_T>(ntsc[px + 2]) + dq;

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
}

// Scale the I,Q channels of a delta buffer by chromAtt (motion pipelines).
// Y is left untouched. evm/magnify.py:_amplify_lpyr_stack.
// Grid: (ceil(W/32), ceil(H/32))  Block: (32, 32, 1)
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

// Scale a flat float32 array by a scalar. Used to apply per-level alpha
// amplification to IIR-filtered pyramid bands on-device.
// Grid: (ceil(n/256))  Block: (256, 1, 1)
__global__ void scale_inplace_kernel(float* __restrict__ data, int n, float scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] *= scale;
}

void launch_scale_inplace(float* data, int n, float scale, cudaStream_t stream) {
    const int block = 256;
    const int grid = div_up(n, block);
    scale_inplace_kernel<<<grid, block, 0, stream>>>(data, n, scale);
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
// Grid: (ceil(M*out_H*out_W/256))  Block: (256, 1, 1)
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
                             float chrom_att, cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(div_up(W, 32), div_up(H, 32), 1);
    add_and_quantize_kernel<<<grid, block, 0, stream>>>(
        ntsc_frame, delta, bgr_out, H, W, chrom_att);
}

void launch_add_planar_quantize(const float* ntsc, const float* delta_planar,
                                unsigned char* bgr_out,
                                int n, int H, int W, float chrom_att,
                                cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(div_up(W, 32 * ADD_PLANAR_ELEMS), div_up(H, 32), n);
    add_planar_quantize_kernel<float><<<grid, block, 0, stream>>>(
        ntsc, delta_planar, bgr_out, n, H, W, chrom_att);
}

void launch_add_planar_quantize_f16(const __half* ntsc, const __half* delta_planar,
                                    unsigned char* bgr_out,
                                    int n, int H, int W, float chrom_att,
                                    cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(div_up(W, 32 * ADD_PLANAR_ELEMS), div_up(H, 32), n);
    add_planar_quantize_kernel<__half><<<grid, block, 0, stream>>>(
        ntsc, delta_planar, bgr_out, n, H, W, chrom_att);
}

// Fused bilinear-upsample + add + NTSC->BGR quantize (color pipeline render).
//
// Combines what was previously two kernels with a full-res intermediate
// (M, out_H, out_W, 3) float32 buffer between them:
//   1. bilinear_upsample_3ch_kernel: filt (M,in_H,in_W,3) -> upsampled float
//   2. add_and_quantize_kernel:      ntsc + upsampled -> bgr u8
//
// Each output pixel now reads 4 source taps from `filt` (the small filtered
// signal at the pyramid level), interpolates inline, reads ntsc[px], adds,
// and writes directly to the uint8 output. No intermediate buffer, one launch.
//
// Each thread processes UPSAMPLE_ELEMS adjacent pixels along x,
// amortizing the per-pixel overhead and giving the compiler independent
// memory operations to pipeline for latency hiding.
//
// Coordinate convention: same as bilinear_upsample_3ch_kernel (half-pixel
// centers + replicate border — bit-exact match to cv2 INTER_LINEAR).
// Grid: (ceil(out_W/(32*UPSAMPLE_ELEMS)), ceil(out_H/32), M)
// Block: (32, 32, 1)
constexpr int UPSAMPLE_ELEMS = 4;

// Templated on NTSC_T: the NTSC buffer may be stored as float (FP32 pipeline)
// or __half (FP16 pipeline). The filt buffer stays float — it comes from the
// FFT output, which is always FP32 regardless of pipeline precision.
template <typename NTSC_T>
__global__ void upsample_add_quantize_kernel(
    const NTSC_T* __restrict__ ntsc,  // (M, out_H, out_W, 3)
    const float* __restrict__ filt,   // (M, in_H, in_W, 3)
    unsigned char* __restrict__ bgr_out,  // (M, out_H, out_W, 3)
    int M, int in_H, int in_W, int out_H, int out_W, float chrom_att)
{
    const int x0 = blockIdx.x * blockDim.x * UPSAMPLE_ELEMS + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int m = blockIdx.z;
    if (y >= out_H || m >= M) return;

    const float scale_x = static_cast<float>(in_W) / out_W;
    const float scale_y = static_cast<float>(in_H) / out_H;
    const float* f = filt + static_cast<size_t>(m) * in_H * in_W * 3;
    const NTSC_T* n = ntsc + static_cast<size_t>(m) * out_H * out_W * 3;
    unsigned char* o = bgr_out + static_cast<size_t>(m) * out_H * out_W * 3;
    const int px_row = (y * out_W + x0) * 3;

    #pragma unroll
    for (int e = 0; e < UPSAMPLE_ELEMS; ++e) {
        const int x = x0 + e * 32;
        if (x >= out_W) break;
        const int px = px_row + e * 32 * 3;

        float sx = (x + 0.5f) * scale_x - 0.5f;
        float sy = (y + 0.5f) * scale_y - 0.5f;
        int sx0 = static_cast<int>(floorf(sx));
        int sy0 = static_cast<int>(floorf(sy));
        const float fx = sx - sx0;
        const float fy = sy - sy0;
        int sx1 = sx0 + 1;
        int sy1 = sy0 + 1;
        sx0 = max(0, min(sx0, in_W - 1));
        sx1 = max(0, min(sx1, in_W - 1));
        sy0 = max(0, min(sy0, in_H - 1));
        sy1 = max(0, min(sy1, in_H - 1));

        const float w00 = (1.0f - fx) * (1.0f - fy);
        const float w01 = fx * (1.0f - fy);
        const float w10 = (1.0f - fx) * fy;
        const float w11 = fx * fy;

        float y_ = cvt_in<NTSC_T>(n[px + 0]);
        float i_ = 0.0f;
        float q_ = 0.0f;
        for (int c = 0; c < 3; ++c) {
            const float v00 = f[(sy0 * in_W + sx0) * 3 + c];
            const float v01 = f[(sy0 * in_W + sx1) * 3 + c];
            const float v10 = f[(sy1 * in_W + sx0) * 3 + c];
            const float v11 = f[(sy1 * in_W + sx1) * 3 + c];
            float delta = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
            if (c == 0) y_ += delta;
            else if (c == 1) i_ = cvt_in<NTSC_T>(n[px + 1]) + delta * chrom_att;
            else             q_ = cvt_in<NTSC_T>(n[px + 2]) + delta * chrom_att;
        }

        float r = kYiqToRgb[0][0]*y_ + kYiqToRgb[0][1]*i_ + kYiqToRgb[0][2]*q_;
        float g = kYiqToRgb[1][0]*y_ + kYiqToRgb[1][1]*i_ + kYiqToRgb[1][2]*q_;
        float b = kYiqToRgb[2][0]*y_ + kYiqToRgb[2][1]*i_ + kYiqToRgb[2][2]*q_;
        r = fminf(fmaxf(r, 0.0f), 1.0f);
        g = fminf(fmaxf(g, 0.0f), 1.0f);
        b = fminf(fmaxf(b, 0.0f), 1.0f);
        o[px + 0] = static_cast<unsigned char>(rintf(b * 255.0f));
        o[px + 1] = static_cast<unsigned char>(rintf(g * 255.0f));
        o[px + 2] = static_cast<unsigned char>(rintf(r * 255.0f));
    }
}

void launch_upsample_add_quantize(const float* ntsc, const float* filt,
                                  unsigned char* bgr_out,
                                  int M, int in_H, int in_W,
                                  int out_H, int out_W, float chrom_att,
                                  cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(div_up(out_W, 32 * UPSAMPLE_ELEMS), div_up(out_H, 32), M);
    upsample_add_quantize_kernel<float><<<grid, block, 0, stream>>>(
        ntsc, filt, bgr_out, M, in_H, in_W, out_H, out_W, chrom_att);
}

void launch_upsample_add_quantize_f16(const __half* ntsc, const float* filt,
                                      unsigned char* bgr_out,
                                      int M, int in_H, int in_W,
                                      int out_H, int out_W, float chrom_att,
                                      cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(div_up(out_W, 32 * UPSAMPLE_ELEMS), div_up(out_H, 32), M);
    upsample_add_quantize_kernel<__half><<<grid, block, 0, stream>>>(
        ntsc, filt, bgr_out, M, in_H, in_W, out_H, out_W, chrom_att);
}

}  // namespace evm
