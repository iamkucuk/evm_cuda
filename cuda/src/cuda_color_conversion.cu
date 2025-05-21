#include "cuda_color_conversion.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

namespace evmcuda {

// RGB to YIQ conversion matrix (float precision) based on ITU/NTSC specification
__constant__ float d_RGB2YIQ_MATRIX[9] = {
    0.299f,       0.587f,       0.114f,      // Y coefficients
    0.59590059f, -0.27455667f, -0.32134392f, // I coefficients
    0.21153661f, -0.52273617f,  0.31119955f  // Q coefficients
};

// YIQ to RGB conversion matrix (float precision) based on ITU/NTSC specification
__constant__ float d_YIQ2RGB_MATRIX[9] = {
    1.0f,        0.9559863f,   0.6208248f,  // R coefficients
    1.0f,       -0.2720128f,  -0.6472042f,  // G coefficients
    1.0f,       -1.1067402f,   1.7042304f   // B coefficients
};

// Kernel for RGB -> YIQ conversion
__global__ void rgb_to_yiq_kernel(
    const float* __restrict__ d_rgb,
    float* __restrict__ d_yiq,
    int width,
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = (y * width + x) * 3;
    
    // Load RGB values
    const float r = d_rgb[idx];
    const float g = d_rgb[idx + 1];
    const float b = d_rgb[idx + 2];
    
    // Matrix multiplication for RGB to YIQ conversion
    const float y_val = d_RGB2YIQ_MATRIX[0] * r + d_RGB2YIQ_MATRIX[1] * g + d_RGB2YIQ_MATRIX[2] * b;
    const float i_val = d_RGB2YIQ_MATRIX[3] * r + d_RGB2YIQ_MATRIX[4] * g + d_RGB2YIQ_MATRIX[5] * b;
    const float q_val = d_RGB2YIQ_MATRIX[6] * r + d_RGB2YIQ_MATRIX[7] * g + d_RGB2YIQ_MATRIX[8] * b;
    
    // Store YIQ values
    d_yiq[idx] = y_val;
    d_yiq[idx + 1] = i_val;
    d_yiq[idx + 2] = q_val;
}

// Kernel for YIQ -> RGB conversion
__global__ void yiq_to_rgb_kernel(
    const float* __restrict__ d_yiq,
    float* __restrict__ d_rgb,
    int width,
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = (y * width + x) * 3;
    
    // Load YIQ values
    const float y_val = d_yiq[idx];
    const float i_val = d_yiq[idx + 1];
    const float q_val = d_yiq[idx + 2];
    
    // Matrix multiplication for YIQ to RGB conversion
    const float r = d_YIQ2RGB_MATRIX[0] * y_val + d_YIQ2RGB_MATRIX[1] * i_val + d_YIQ2RGB_MATRIX[2] * q_val;
    const float g = d_YIQ2RGB_MATRIX[3] * y_val + d_YIQ2RGB_MATRIX[4] * i_val + d_YIQ2RGB_MATRIX[5] * q_val;
    const float b = d_YIQ2RGB_MATRIX[6] * y_val + d_YIQ2RGB_MATRIX[7] * i_val + d_YIQ2RGB_MATRIX[8] * q_val;
    
    // Store RGB values
    d_rgb[idx] = r;
    d_rgb[idx + 1] = g;
    d_rgb[idx + 2] = b;
}

// Helper function to check CUDA errors
inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        char errorMsg[256];
        snprintf(errorMsg, sizeof(errorMsg), "%s: %s", msg, cudaGetErrorString(err));
        throw std::runtime_error(errorMsg);
    }
}

// Device implementation of RGB to YIQ conversion
void rgb_to_yiq_gpu(
    const float* d_rgb,
    float* d_yiq,
    int width,
    int height,
    cudaStream_t stream)
{
    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);
    
    // Launch the kernel
    rgb_to_yiq_kernel<<<gridDim, blockDim, 0, stream>>>(d_rgb, d_yiq, width, height);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Failed to launch rgb_to_yiq_kernel");
}

// Device implementation of YIQ to RGB conversion
void yiq_to_rgb_gpu(
    const float* d_yiq,
    float* d_rgb,
    int width,
    int height,
    cudaStream_t stream)
{
    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);
    
    // Launch the kernel
    yiq_to_rgb_kernel<<<gridDim, blockDim, 0, stream>>>(d_yiq, d_rgb, width, height);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Failed to launch yiq_to_rgb_kernel");
}

// Host wrapper for RGB to YIQ conversion
void rgb_to_yiq_wrapper(
    const float* h_rgb,
    float* h_yiq,
    int width,
    int height)
{
    // Calculate data sizes
    const size_t dataSize = width * height * 3 * sizeof(float);
    
    // Allocate device memory
    float* d_rgb = nullptr;
    float* d_yiq = nullptr;
    
    cudaError_t err;
    
    // Allocate and copy input data
    err = cudaMalloc(&d_rgb, dataSize);
    checkCudaError(err, "Failed to allocate device memory for RGB data");
    
    err = cudaMalloc(&d_yiq, dataSize);
    checkCudaError(err, "Failed to allocate device memory for YIQ data");
    
    err = cudaMemcpy(d_rgb, h_rgb, dataSize, cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy RGB data to device");
    
    // Perform conversion
    rgb_to_yiq_gpu(d_rgb, d_yiq, width, height);
    
    // Copy result back to host
    err = cudaMemcpy(h_yiq, d_yiq, dataSize, cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy YIQ data from device");
    
    // Free device memory
    cudaFree(d_rgb);
    cudaFree(d_yiq);
}

// Host wrapper for YIQ to RGB conversion
void yiq_to_rgb_wrapper(
    const float* h_yiq,
    float* h_rgb,
    int width,
    int height)
{
    // Calculate data sizes
    const size_t dataSize = width * height * 3 * sizeof(float);
    
    // Allocate device memory
    float* d_yiq = nullptr;
    float* d_rgb = nullptr;
    
    cudaError_t err;
    
    // Allocate and copy input data
    err = cudaMalloc(&d_yiq, dataSize);
    checkCudaError(err, "Failed to allocate device memory for YIQ data");
    
    err = cudaMalloc(&d_rgb, dataSize);
    checkCudaError(err, "Failed to allocate device memory for RGB data");
    
    err = cudaMemcpy(d_yiq, h_yiq, dataSize, cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy YIQ data to device");
    
    // Perform conversion
    yiq_to_rgb_gpu(d_yiq, d_rgb, width, height);
    
    // Copy result back to host
    err = cudaMemcpy(h_rgb, d_rgb, dataSize, cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy RGB data from device");
    
    // Free device memory
    cudaFree(d_yiq);
    cudaFree(d_rgb);
}

// Initialize the color conversion module
bool init_color_conversion() {
    // Nothing to initialize for now
    return true;
}

// Cleanup resources
void cleanup_color_conversion() {
    // Nothing to clean up for now
}

} // namespace evmcuda