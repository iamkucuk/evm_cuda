#pragma once

#include <cuda_runtime.h>
#include <vector>

// CUDA spatial Gaussian filter for EVM
// Input/output are float32, 3-channel images (height x width x 3)
// Kernel is a square, odd-sized float32 array (host pointer)
// All pointers are host pointers unless otherwise specified

// Returns true on success, false on error
bool cudaSpatiallyFilterGaussian(
    const float* input, // H x W x 3, row-major
    float* output,      // H x W x 3, row-major
    int height,
    int width,
    int channels,       // should be 3
    const float* kernel,
    int kernel_size
);

// CUDA pyrDown: Gaussian blur + subsample
// Output: (height/2) x (width/2) x 3
bool cudaPyrDown(
    const float* input, // H x W x 3
    float* output,      // (H/2) x (W/2) x 3
    int height,
    int width,
    int channels,
    const float* kernel,
    int kernel_size
);

// CUDA pyrUp: zero-insert + Gaussian blur (kernel * 4)
// Output: (height*2) x (width*2) x 3
bool cudaPyrUp(
    const float* input, // H x W x 3
    float* output,      // (H*2) x (W*2) x 3
    int height,
    int width,
    int channels,
    const float* kernel,
    int kernel_size
);
