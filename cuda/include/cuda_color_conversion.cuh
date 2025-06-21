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

// Additional kernels for cuda_evm namespace (for processing compatibility)
namespace cuda_evm {

/**
 * @brief CUDA kernel for planar RGB to YIQ conversion
 * Used by reconstruction pipeline
 */
__global__ void rgb_to_yiq_planar_kernel(const float* d_rgb, float* d_yiq, int width, int height, int channels);

/**
 * @brief CUDA kernel for planar YIQ to RGB conversion  
 * Used by reconstruction pipeline
 */
__global__ void yiq_to_rgb_planar_kernel(const float* d_yiq, float* d_rgb, int width, int height, int channels);

} // namespace cuda_evm

#endif // CUDA_COLOR_CONVERSION_CUH