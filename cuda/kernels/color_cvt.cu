// color_cvt.cu — BGR uint8 <-> NTSC YIQ float32 color conversion.
//
// Port of evm/video.py:rgb_to_yiq / yiq_to_rgb and evm/magnify.py:
// _rgb_frame_to_ntsc / _ntsc_to_bgr_uint8. Two kernels are exposed:
//
//   bgr_u8_to_ntsc_f32  : (H,W,3) uint8 BGR  ->  (H,W,3) float32 YIQ in [..]
//   ntsc_f32_to_bgr_u8  : (H,W,3) float32 YIQ ->  (H,W,3) uint8 BGR (clipped,
//                                                   banker-rounded, *255)
//
// Grid:  (ceil(W/32), ceil(H/32))   Block: (32, 32, 1)
// Each thread does one pixel; the 3 color channels live in registers.
//
// Numerical contract (tolerance < 1e-6 vs Python, see DESIGN.md):
//   - Matrices kRgbToYiq / kYiqToRgb verbatim from evm_common.cuh.
//   - /255.0 on the u8->float path; clip[0,1] + rintf(*255) on float->u8.
//   - rintf uses round-half-to-even by default, matching numpy.round.
//
// Channel order note: input is BGR (OpenCV native). rgb = bgr reversed, so
//   yiq[0] (Y) =  0.299*B + 0.587*G + 0.114*R    etc.
// We collapse the channel swap into the matrix by indexing b,r,g explicitly.

#include "../include/evm_common.cuh"

namespace evm {

__global__ void bgr_u8_to_ntsc_f32_kernel(
    const unsigned char* __restrict__ bgr,  // (H*W*3) uint8, BGR order
    float* __restrict__ yiq,                // (H*W*3) float32, YIQ order
    int H, int W)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int px = (y * W + x) * 3;
    // BGR u8 -> RGB float in [0,1]
    const float b = bgr[px + 0] * (1.0f / 255.0f);
    const float g = bgr[px + 1] * (1.0f / 255.0f);
    const float r = bgr[px + 2] * (1.0f / 255.0f);

    // yiq = M_rgb_to_yiq . [R;G;B]   (M stored row-major as [Y][I][Q])
    // Note: evm/video.py stores the matrix with rows (Y,I,Q) and computes
    // yiq = rgb @ M.T, so the columns of M.T are the matrix rows below.
    yiq[px + 0] = kRgbToYiq[0][0] * r + kRgbToYiq[0][1] * g + kRgbToYiq[0][2] * b;
    yiq[px + 1] = kRgbToYiq[1][0] * r + kRgbToYiq[1][1] * g + kRgbToYiq[1][2] * b;
    yiq[px + 2] = kRgbToYiq[2][0] * r + kRgbToYiq[2][1] * g + kRgbToYiq[2][2] * b;
}

__global__ void ntsc_f32_to_bgr_u8_kernel(
    const float* __restrict__ yiq,          // (H*W*3) float32, YIQ order
    unsigned char* __restrict__ bgr,        // (H*W*3) uint8, BGR order
    int H, int W)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int px = (y * W + x) * 3;
    const float y_ = yiq[px + 0];
    const float i_ = yiq[px + 1];
    const float q_ = yiq[px + 2];

    // rgb = M_yiq_to_rgb . [Y;I;Q]
    float r = kYiqToRgb[0][0] * y_ + kYiqToRgb[0][1] * i_ + kYiqToRgb[0][2] * q_;
    float g = kYiqToRgb[1][0] * y_ + kYiqToRgb[1][1] * i_ + kYiqToRgb[1][2] * q_;
    float b = kYiqToRgb[2][0] * y_ + kYiqToRgb[2][1] * i_ + kYiqToRgb[2][2] * q_;

    // clip [0,1], *255, banker's round (rintf default mode), cast u8
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);
    bgr[px + 0] = static_cast<unsigned char>(rintf(b * 255.0f));
    bgr[px + 1] = static_cast<unsigned char>(rintf(g * 255.0f));
    bgr[px + 2] = static_cast<unsigned char>(rintf(r * 255.0f));
}

// --- host launchers (called from bindings.cpp) -----------------------------

void launch_bgr_u8_to_ntsc_f32(const unsigned char* bgr, float* yiq,
                                int H, int W, cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(div_up(W, 32), div_up(H, 32), 1);
    bgr_u8_to_ntsc_f32_kernel<<<grid, block, 0, stream>>>(bgr, yiq, H, W);
}

void launch_ntsc_f32_to_bgr_u8(const float* yiq, unsigned char* bgr,
                                int H, int W, cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(div_up(W, 32), div_up(H, 32), 1);
    ntsc_f32_to_bgr_u8_kernel<<<grid, block, 0, stream>>>(yiq, bgr, H, W);
}

}  // namespace evm
