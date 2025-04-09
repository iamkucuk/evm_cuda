#ifndef EVMCUDA_COLOR_CONVERSION_CUH
#define EVMCUDA_COLOR_CONVERSION_CUH

#include <cuda_runtime.h> // For cudaStream_t
#include <cstddef>      // For size_t

namespace evmcuda {

/**
 * @brief Converts an RGB image stored on the GPU to YIQ format using CUDA.
 * Operates on raw device pointers. Input is 3-channel uint8 (RGB).
 *
 * @param d_inputRgb Pointer to input RGB image data (uint8, 3 channels) on the GPU.
 * @param d_outputYiq Pointer to output YIQ image data (float, 3 channels) on the GPU. Must be pre-allocated.
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @param inputPitch Input image pitch (bytes per row).
 * @param outputPitch Output image pitch (bytes per row).
 * @param stream CUDA stream for asynchronous execution (optional, defaults to 0).
 */
void rgb2yiq_gpu(const unsigned char* d_inputRgb, float* d_outputYiq,
                 int width, int height, size_t inputPitch, size_t outputPitch,
                 cudaStream_t stream = 0);

/**
 * @brief Converts an RGB image stored on the GPU to YIQ format using CUDA.
 * Operates on raw device pointers. Input is 3-channel float (RGB).
 *
 * @param d_inputRgb Pointer to input RGB image data (float, 3 channels) on the GPU.
 * @param d_outputYiq Pointer to output YIQ image data (float, 3 channels) on the GPU. Must be pre-allocated.
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @param inputPitch Input image pitch (bytes per row).
 * @param outputPitch Output image pitch (bytes per row).
 * @param stream CUDA stream for asynchronous execution (optional, defaults to 0).
 */
void rgb2yiq_gpu(const float* d_inputRgb, float* d_outputYiq,
                 int width, int height, size_t inputPitch, size_t outputPitch,
                 cudaStream_t stream = 0);

/**
 * @brief Converts a YIQ image stored on the GPU to RGB format using CUDA.
 * Operates on raw device pointers. Assumes input is 3-channel float (YIQ).
 *
 * @param d_inputYiq Pointer to input YIQ image data (float, 3 channels) on the GPU.
 * @param d_outputRgb Pointer to output RGB image data (float, 3 channels) on the GPU. Must be pre-allocated.
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @param inputPitch Input image pitch (bytes per row).
 * @param outputPitch Output image pitch (bytes per row).
 * @param stream CUDA stream for asynchronous execution (optional, defaults to 0).
 */
void yiq2rgb_gpu(const float* d_inputYiq, float* d_outputRgb,
                 int width, int height, size_t inputPitch, size_t outputPitch,
                 cudaStream_t stream = 0);

} // namespace evmcuda

#endif // EVMCUDA_COLOR_CONVERSION_CUH