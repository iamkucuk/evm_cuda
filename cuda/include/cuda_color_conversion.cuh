#ifndef CUDA_COLOR_CONVERSION_CUH
#define CUDA_COLOR_CONVERSION_CUH

#include <cuda_runtime.h>

namespace evmcuda {

/**
 * @brief CUDA kernel for RGB to YIQ color space conversion
 * 
 * @param rgb_data Input RGB data (device memory, float3 per pixel)
 * @param yiq_data Output YIQ data (device memory, float3 per pixel)
 * @param num_pixels Total number of pixels to process
 */
__global__ void rgb_to_yiq_kernel(const float3* rgb_data, float3* yiq_data, int num_pixels);

/**
 * @brief CUDA kernel for YIQ to RGB color space conversion
 * 
 * @param yiq_data Input YIQ data (device memory, float3 per pixel)
 * @param rgb_data Output RGB data (device memory, float3 per pixel)
 * @param num_pixels Total number of pixels to process
 */
__global__ void yiq_to_rgb_kernel(const float3* yiq_data, float3* rgb_data, int num_pixels);

/**
 * @brief Host wrapper for RGB to YIQ conversion
 * 
 * @param rgb_data Input RGB data (device memory)
 * @param yiq_data Output YIQ data (device memory) 
 * @param width Image width
 * @param height Image height
 * @return cudaError_t CUDA error status
 */
cudaError_t rgb_to_yiq(const float3* rgb_data, float3* yiq_data, int width, int height);

/**
 * @brief Host wrapper for YIQ to RGB conversion
 * 
 * @param yiq_data Input YIQ data (device memory)
 * @param rgb_data Output RGB data (device memory)
 * @param width Image width
 * @param height Image height
 * @return cudaError_t CUDA error status
 */
cudaError_t yiq_to_rgb(const float3* yiq_data, float3* rgb_data, int width, int height);

} // namespace evmcuda

#endif // CUDA_COLOR_CONVERSION_CUH