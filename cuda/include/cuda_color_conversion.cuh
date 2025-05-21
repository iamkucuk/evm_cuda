#ifndef CUDA_COLOR_CONVERSION_CUH
#define CUDA_COLOR_CONVERSION_CUH

#include <cuda_runtime.h>
#include <stdexcept>

namespace evmcuda {

/**
 * @brief Converts an RGB image to YIQ color space using CUDA
 * 
 * @param d_rgb Input RGB image data in device memory (float3 format, row-major)
 * @param d_yiq Output YIQ image data in device memory (float3 format, row-major)
 * @param width Width of the image in pixels
 * @param height Height of the image in pixels
 * @param stream CUDA stream to use for the conversion (optional)
 * @throws std::runtime_error if CUDA operation fails
 */
void rgb_to_yiq_gpu(
    const float* d_rgb,
    float* d_yiq,
    int width,
    int height,
    cudaStream_t stream = nullptr);

/**
 * @brief Converts a YIQ image to RGB color space using CUDA
 * 
 * @param d_yiq Input YIQ image data in device memory (float3 format, row-major)
 * @param d_rgb Output RGB image data in device memory (float3 format, row-major)
 * @param width Width of the image in pixels
 * @param height Height of the image in pixels
 * @param stream CUDA stream to use for the conversion (optional)
 * @throws std::runtime_error if CUDA operation fails
 */
void yiq_to_rgb_gpu(
    const float* d_yiq,
    float* d_rgb,
    int width,
    int height,
    cudaStream_t stream = nullptr);

/**
 * @brief Host wrapper function that converts RGB data to YIQ
 *        Handles memory allocation and transfer
 * 
 * @param h_rgb Host RGB image data (float3 format, row-major)
 * @param h_yiq Host YIQ image data output buffer (float3 format, row-major)
 * @param width Width of the image in pixels
 * @param height Height of the image in pixels
 * @throws std::runtime_error if CUDA operation fails
 */
void rgb_to_yiq_wrapper(
    const float* h_rgb,
    float* h_yiq,
    int width,
    int height);

/**
 * @brief Host wrapper function that converts YIQ data to RGB
 *        Handles memory allocation and transfer
 * 
 * @param h_yiq Host YIQ image data (float3 format, row-major)
 * @param h_rgb Host RGB image data output buffer (float3 format, row-major)
 * @param width Width of the image in pixels
 * @param height Height of the image in pixels
 * @throws std::runtime_error if CUDA operation fails
 */
void yiq_to_rgb_wrapper(
    const float* h_yiq,
    float* h_rgb,
    int width,
    int height);

/**
 * @brief Initialize the color conversion module
 * This does any needed setup for the module (e.g., compiling kernels)
 * 
 * @return true if initialization succeeded, false otherwise
 */
bool init_color_conversion();

/**
 * @brief Cleanup resources used by the color conversion module
 */
void cleanup_color_conversion();

} // namespace evmcuda

#endif // CUDA_COLOR_CONVERSION_CUH