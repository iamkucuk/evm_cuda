#include "cuda_color_conversion.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace evmcuda {

// RGB to YIQ conversion matrix stored in constant memory for fast access
__constant__ float c_rgb2yiq_matrix[9] = {
    0.299f,       0.587f,       0.114f,         // Y coefficients
    0.59590059f, -0.27455667f, -0.32134392f,   // I coefficients  
    0.21153661f, -0.52273617f,  0.31119955f    // Q coefficients
};

// YIQ to RGB conversion matrix stored in constant memory for fast access
__constant__ float c_yiq2rgb_matrix[9] = {
    1.0f,        0.9559863f,   0.6208248f,     // R coefficients
    1.0f,       -0.2720128f,  -0.6472042f,    // G coefficients
    1.0f,       -1.1067402f,   1.7042304f     // B coefficients
};

__global__ void rgb_to_yiq_kernel(const float3* rgb_data, float3* yiq_data, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_pixels) {
        float3 rgb = rgb_data[idx];
        float3 yiq;
        
        // Matrix multiplication: YIQ = RGB2YIQ_MATRIX * RGB
        yiq.x = c_rgb2yiq_matrix[0] * rgb.x + c_rgb2yiq_matrix[1] * rgb.y + c_rgb2yiq_matrix[2] * rgb.z; // Y
        yiq.y = c_rgb2yiq_matrix[3] * rgb.x + c_rgb2yiq_matrix[4] * rgb.y + c_rgb2yiq_matrix[5] * rgb.z; // I
        yiq.z = c_rgb2yiq_matrix[6] * rgb.x + c_rgb2yiq_matrix[7] * rgb.y + c_rgb2yiq_matrix[8] * rgb.z; // Q
        
        yiq_data[idx] = yiq;
    }
}

__global__ void yiq_to_rgb_kernel(const float3* yiq_data, float3* rgb_data, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_pixels) {
        float3 yiq = yiq_data[idx];
        float3 rgb;
        
        // Matrix multiplication: RGB = YIQ2RGB_MATRIX * YIQ
        rgb.x = c_yiq2rgb_matrix[0] * yiq.x + c_yiq2rgb_matrix[1] * yiq.y + c_yiq2rgb_matrix[2] * yiq.z; // R
        rgb.y = c_yiq2rgb_matrix[3] * yiq.x + c_yiq2rgb_matrix[4] * yiq.y + c_yiq2rgb_matrix[5] * yiq.z; // G
        rgb.z = c_yiq2rgb_matrix[6] * yiq.x + c_yiq2rgb_matrix[7] * yiq.y + c_yiq2rgb_matrix[8] * yiq.z; // B
        
        rgb_data[idx] = rgb;
    }
}

cudaError_t rgb_to_yiq(const float3* rgb_data, float3* yiq_data, int width, int height) {
    int num_pixels = width * height;
    
    // Define block size and grid size
    int block_size = 256;
    int grid_size = (num_pixels + block_size - 1) / block_size;
    
    // Launch kernel
    rgb_to_yiq_kernel<<<grid_size, block_size>>>(rgb_data, yiq_data, num_pixels);
    
    // Check for kernel launch errors
    return cudaGetLastError();
}

cudaError_t yiq_to_rgb(const float3* yiq_data, float3* rgb_data, int width, int height) {
    int num_pixels = width * height;
    
    // Define block size and grid size
    int block_size = 256;
    int grid_size = (num_pixels + block_size - 1) / block_size;
    
    // Launch kernel
    yiq_to_rgb_kernel<<<grid_size, block_size>>>(yiq_data, rgb_data, num_pixels);
    
    // Check for kernel launch errors
    return cudaGetLastError();
}

} // namespace evmcuda