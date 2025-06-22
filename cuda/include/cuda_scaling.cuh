#ifndef CUDA_SCALING_CUH
#define CUDA_SCALING_CUH

#include <cuda_runtime.h>

extern "C" {

/**
 * @brief Scale data from [0,255] to [0,1] range on GPU
 * @param d_input Input data in [0,255] range
 * @param d_output Output data in [0,1] range  
 * @param total_elements Total number of float elements to scale
 * @return cudaError_t CUDA error code
 */
cudaError_t gpu_scale_255_to_1(const float* d_input, float* d_output, int total_elements);

/**
 * @brief Scale data from [0,1] to [0,255] range on GPU
 * @param d_input Input data in [0,1] range
 * @param d_output Output data in [0,255] range
 * @param total_elements Total number of float elements to scale
 * @return cudaError_t CUDA error code
 */
cudaError_t gpu_scale_1_to_255(const float* d_input, float* d_output, int total_elements);

/**
 * @brief In-place scaling from [0,255] to [0,1] range on GPU
 * @param d_data Data to scale in-place
 * @param total_elements Total number of float elements to scale
 * @return cudaError_t CUDA error code
 */
cudaError_t gpu_scale_255_to_1_inplace(float* d_data, int total_elements);

/**
 * @brief In-place scaling from [0,1] to [0,255] range on GPU
 * @param d_data Data to scale in-place
 * @param total_elements Total number of float elements to scale
 * @return cudaError_t CUDA error code
 */
cudaError_t gpu_scale_1_to_255_inplace(float* d_data, int total_elements);

} // extern "C"

#endif // CUDA_SCALING_CUH