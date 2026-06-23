// fp16_cvt.cu — FP16 ↔ FP32 batch conversion kernels.
//
// Used by the motion pipeline's FP16 storage path. The pipeline's compute
// kernels stay FP32; only the intermediate storage buffers are __half to
// halve VRAM usage (23 GB → 12 GB for baby.mp4).
//
// Each kernel converts one direction at a buffer boundary:
//   f32_to_f16: read float32, write __half
//   f16_to_f32: read __half, write float32
//
// Grid: (ceil(n/256))  Block: (256, 1, 1)

#include "../include/evm_common.cuh"
#include <cuda_fp16.h>

namespace evm {

__global__ void f32_to_f16_kernel(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __float2half(src[idx]);
}

__global__ void f16_to_f32_kernel(
    const __half* __restrict__ src,
    float* __restrict__ dst,
    int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __half2float(src[idx]);
}

void launch_f32_to_f16(const float* src, __half* dst, int n,
                       cudaStream_t stream) {
    int block = 256;
    int grid = div_up(n, block);
    f32_to_f16_kernel<<<grid, block, 0, stream>>>(src, dst, n);
}

void launch_f16_to_f32(const __half* src, float* dst, int n,
                       cudaStream_t stream) {
    int block = 256;
    int grid = div_up(n, block);
    f16_to_f32_kernel<<<grid, block, 0, stream>>>(src, dst, n);
}

}  // namespace evm
