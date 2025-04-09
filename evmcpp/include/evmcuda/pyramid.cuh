#ifndef EVMCUDA_PYRAMID_CUH
#define EVMCUDA_PYRAMID_CUH

#include <cuda_runtime.h>
#include <cstddef> // For size_t
#include <opencv2/core/types.hpp> // For cv::Size

namespace evmcuda {

/**
 * @brief Performs Gaussian filtering and downsampling by 2 (custom pyrDown) on the GPU.
 * Assumes input is 3-channel float.
 *
 * @param d_input Pointer to input image data (float, 3 channels) on the GPU.
 * @param d_output Pointer to output image data (float, 3 channels) on the GPU. Must be pre-allocated with half width/height.
 * @param width Input image width.
 * @param height Input image height.
 * @param inputPitch Input image pitch (bytes per row).
 * @param outputPitch Output image pitch (bytes per row).
 * @param d_kernel Pointer to the 5x5 Gaussian kernel data (float) on the GPU.
 * @param stream CUDA stream for asynchronous execution (optional, defaults to 0).
 */
void pyrDown_gpu(const float* d_input, float* d_output,
                 int width, int height, size_t inputPitch, size_t outputPitch,
                 const float* d_kernel, cudaStream_t stream = 0);

/**
 * @brief Performs upsampling by 2 and Gaussian filtering (custom pyrUp) on the GPU.
 * Assumes input is 3-channel float.
 *
 * @param d_input Pointer to input image data (float, 3 channels) on the GPU.
 * @param d_output Pointer to output image data (float, 3 channels) on the GPU. Must be pre-allocated with target dimensions.
 * @param width Input image width.
 * @param height Input image height.
 * @param outputWidth Target output image width.
 * @param outputHeight Target output image height.
 * @param inputPitch Input image pitch (bytes per row).
 * @param outputPitch Output image pitch (bytes per row).
 * @param d_kernel Pointer to the 5x5 Gaussian kernel data (float) on the GPU (kernel multiplied by 4).
 * @param stream CUDA stream for asynchronous execution (optional, defaults to 0).
 */
void pyrUp_gpu(const float* d_input, float* d_output,
               int width, int height, int outputWidth, int outputHeight,
               size_t inputPitch, size_t outputPitch,
               const float* d_kernel_x4, cudaStream_t stream = 0);

/**
 * @brief Performs 2D convolution using shared memory (internal helper).
 * Assumes input is 3-channel float and kernel is 5x5 float.
 *
 * @param d_input Pointer to input image data (float, 3 channels) on the GPU.
 * @param d_output Pointer to output image data (float, 3 channels) on the GPU. Must be pre-allocated.
 * @param width Image width.
 * @param height Image height.
 * @param inputPitch Input image pitch (bytes per row).
 * @param outputPitch Output image pitch (bytes per row).
 * @param d_kernel Pointer to the 5x5 kernel data (float) on the GPU.
 * @param stream CUDA stream for asynchronous execution (optional, defaults to 0).
 */
void conv2DKernelSharedMem_gpu(const float* d_input, float* d_output,
                               int width, int height, size_t inputPitch, size_t outputPitch,
                               const float* d_kernel, cudaStream_t stream = 0);


} // namespace evmcuda

#endif // EVMCUDA_PYRAMID_CUH