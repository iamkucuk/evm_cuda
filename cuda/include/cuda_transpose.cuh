#ifndef CUDA_TRANSPOSE_CUH
#define CUDA_TRANSPOSE_CUH

#include <cuda_runtime.h>

extern "C" cudaError_t launch_transpose_frame_to_pixel(
    const float* d_frame_major, float* d_pixel_major,
    int width, int height, int channels, int num_frames,
    dim3 gridSize, dim3 blockSize);

extern "C" cudaError_t launch_transpose_pixel_to_frame(
    const float* d_pixel_major, float* d_frame_major,
    int width, int height, int channels, int num_frames,
    dim3 gridSize, dim3 blockSize);

#endif // CUDA_TRANSPOSE_CUH