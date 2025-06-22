/**
 * GPU Scaling Kernels for Data Range Conversion
 * Critical for maintaining correct data ranges in GPU-resident pipeline
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * @brief Scale data from [0,255] to [0,1] range on GPU
 */
__global__ void scale_255_to_1_kernel(const float* input, float* output, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        output[idx] = input[idx] / 255.0f;
    }
}

/**
 * @brief Scale data from [0,1] to [0,255] range on GPU
 */
__global__ void scale_1_to_255_kernel(const float* input, float* output, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        output[idx] = input[idx] * 255.0f;
    }
}

/**
 * @brief In-place scaling from [0,255] to [0,1] range on GPU
 */
__global__ void scale_255_to_1_inplace_kernel(float* data, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        data[idx] = data[idx] / 255.0f;
    }
}

/**
 * @brief In-place scaling from [0,1] to [0,255] range on GPU
 */
__global__ void scale_1_to_255_inplace_kernel(float* data, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        data[idx] = data[idx] * 255.0f;
    }
}

extern "C" {

/**
 * @brief Scale data from [0,255] to [0,1] range on GPU
 */
cudaError_t gpu_scale_255_to_1(const float* d_input, float* d_output, int total_elements) {
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    scale_255_to_1_kernel<<<grid_size, block_size>>>(d_input, d_output, total_elements);
    
    return cudaGetLastError();
}

/**
 * @brief Scale data from [0,1] to [0,255] range on GPU
 */
cudaError_t gpu_scale_1_to_255(const float* d_input, float* d_output, int total_elements) {
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    scale_1_to_255_kernel<<<grid_size, block_size>>>(d_input, d_output, total_elements);
    
    return cudaGetLastError();
}

/**
 * @brief In-place scaling from [0,255] to [0,1] range on GPU
 */
cudaError_t gpu_scale_255_to_1_inplace(float* d_data, int total_elements) {
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    scale_255_to_1_inplace_kernel<<<grid_size, block_size>>>(d_data, total_elements);
    
    return cudaGetLastError();
}

/**
 * @brief In-place scaling from [0,1] to [0,255] range on GPU
 */
cudaError_t gpu_scale_1_to_255_inplace(float* d_data, int total_elements) {
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    scale_1_to_255_inplace_kernel<<<grid_size, block_size>>>(d_data, total_elements);
    
    return cudaGetLastError();
}

} // extern "C"