#include "cuda_format_conversion.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace evmcuda {

// CUDA kernel: Convert float3 array to flat float array
__global__ void convert_float3_to_flat_kernel(const float3* input, float* output, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_pixels) {
        // Direct memory copy with format conversion - no precision loss
        output[idx * 3 + 0] = input[idx].x;  // Y/R channel
        output[idx * 3 + 1] = input[idx].y;  // I/G channel  
        output[idx * 3 + 2] = input[idx].z;  // Q/B channel
    }
}

// CUDA kernel: Convert flat float array to float3 array
__global__ void convert_flat_to_float3_kernel(const float* input, float3* output, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_pixels) {
        // Direct memory copy with format conversion - no precision loss
        output[idx].x = input[idx * 3 + 0];  // Y/R channel
        output[idx].y = input[idx * 3 + 1];  // I/G channel
        output[idx].z = input[idx * 3 + 2];  // Q/B channel
    }
}

cudaError_t convert_float3_to_flat(const float3* input, float* output, int width, int height) {
    int num_pixels = width * height;
    
    // Define block size and grid size
    int block_size = 256;
    int grid_size = (num_pixels + block_size - 1) / block_size;
    
    // Launch conversion kernel
    convert_float3_to_flat_kernel<<<grid_size, block_size>>>(input, output, num_pixels);
    
    // Check for kernel launch errors
    return cudaGetLastError();
}

cudaError_t convert_flat_to_float3(const float* input, float3* output, int width, int height) {
    int num_pixels = width * height;
    
    // Define block size and grid size
    int block_size = 256;
    int grid_size = (num_pixels + block_size - 1) / block_size;
    
    // Launch conversion kernel
    convert_flat_to_float3_kernel<<<grid_size, block_size>>>(input, output, num_pixels);
    
    // Check for kernel launch errors
    return cudaGetLastError();
}

} // namespace evmcuda