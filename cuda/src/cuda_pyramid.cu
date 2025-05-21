#include "cuda_pyramid.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core.hpp>
#include <stdio.h>
#include <cmath>

namespace evmcuda {

// Gaussian kernel constants (matching CPU implementation)
// 5x5 Gaussian kernel / 256
__constant__ float d_gaussian_kernel[25];

// Host-side Gaussian kernel for initialization
const float h_gaussian_kernel[25] = {
    1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f,  1.0f/256.0f,
    4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f,  4.0f/256.0f,
    6.0f/256.0f, 24.0f/256.0f, 36.0f/256.0f, 24.0f/256.0f,  6.0f/256.0f,
    4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f,  4.0f/256.0f,
    1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f,  1.0f/256.0f
};

// Kernel for applying 2D convolution with a 5x5 kernel
// This is a separable filter, but for simplicity we'll use the full 2D implementation first
// This kernel supports multi-channel images
__global__ void filter2D_kernel(
    const float* __restrict__ d_src,
    float* __restrict__ d_dst,
    int width,
    int height,
    int channels,
    int stride) // Stride in elements (for padding if needed)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = (y * stride + x) * channels;
    
    // For each channel
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        // Apply 5x5 filter
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                // Calculate source coordinates with border handling (REFLECT_101)
                int sx = x + kx;
                int sy = y + ky;
                
                // BORDER_REFLECT_101 (OpenCV's default and what CPU implementation uses)
                // For out-of-bounds: -1 -> 1, -2 -> 2, width -> width-2, width+1 -> width-3
                if (sx < 0) sx = -sx;
                if (sy < 0) sy = -sy;
                if (sx >= width) sx = 2 * width - sx - 2;
                if (sy >= height) sy = 2 * height - sy - 2;
                
                // Get kernel value (kernel is laid out row-major in constant memory)
                float k = d_gaussian_kernel[(ky + 2) * 5 + (kx + 2)];
                
                // Get source pixel value and multiply by kernel value
                sum += d_src[(sy * stride + sx) * channels + c] * k;
            }
        }
        
        // Store result
        d_dst[idx + c] = sum;
    }
}

// Kernel for downsampling by taking every other pixel
__global__ void downsample_kernel(
    const float* __restrict__ d_src,
    float* __restrict__ d_dst,
    int src_width,
    int src_height,
    int channels,
    int src_stride,  // Stride for source (for padding if needed)
    int dst_stride)  // Stride for destination
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= src_width/2 || dst_y >= src_height/2) return;
    
    // Calculate source and destination indices
    const int src_idx = ((dst_y * 2) * src_stride + (dst_x * 2)) * channels;
    const int dst_idx = (dst_y * dst_stride + dst_x) * channels;
    
    // Copy every other pixel from source to destination
    for (int c = 0; c < channels; c++) {
        d_dst[dst_idx + c] = d_src[src_idx + c];
    }
}

// Kernel for upsampling by placing pixels at even positions and filling with zeros
__global__ void upsample_kernel(
    const float* __restrict__ d_src,
    float* __restrict__ d_dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    int channels,
    int src_stride,  // Stride for source
    int dst_stride)  // Stride for destination
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= dst_width || dst_y >= dst_height) return;
    
    const int dst_idx = (dst_y * dst_stride + dst_x) * channels;
    
    // Initialize all pixels to zero
    for (int c = 0; c < channels; c++) {
        d_dst[dst_idx + c] = 0.0f;
    }
    
    // If this is an even coordinate in both dimensions, copy from source
    if (dst_x % 2 == 0 && dst_y % 2 == 0) {
        int src_x = dst_x / 2;
        int src_y = dst_y / 2;
        
        // Check if source coordinates are within bounds
        if (src_x < src_width && src_y < src_height) {
            const int src_idx = (src_y * src_stride + src_x) * channels;
            
            // Copy the source pixel
            for (int c = 0; c < channels; c++) {
                d_dst[dst_idx + c] = d_src[src_idx + c];
            }
        }
    }
}

// Helper function to check CUDA errors
inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        char errorMsg[256];
        snprintf(errorMsg, sizeof(errorMsg), "%s: %s", msg, cudaGetErrorString(err));
        throw std::runtime_error(errorMsg);
    }
}

// Initialize the Gaussian kernel in device memory
bool init_pyramid() {
    cudaError_t err = cudaMemcpyToSymbol(d_gaussian_kernel, h_gaussian_kernel, sizeof(h_gaussian_kernel));
    if (err != cudaSuccess) {
        printf("Failed to copy Gaussian kernel to device: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

// Implementation of pyramid downsampling
void pyr_down_gpu(
    const float* d_src,
    int src_width,
    int src_height,
    int src_channels,
    float* d_dst,
    cudaStream_t stream)
{
    // Calculate output dimensions
    int dst_width = src_width / 2;
    int dst_height = src_height / 2;
    
    // Allocate temporary buffer for filtered image
    float* d_filtered = nullptr;
    size_t filtered_size = src_width * src_height * src_channels * sizeof(float);
    cudaError_t err = cudaMalloc(&d_filtered, filtered_size);
    checkCudaError(err, "Failed to allocate device memory for filtered image in pyr_down_gpu");
    
    // Step 1: Apply Gaussian blur to input image
    dim3 blockSize(16, 16);
    dim3 gridSize((src_width + blockSize.x - 1) / blockSize.x, 
                 (src_height + blockSize.y - 1) / blockSize.y);
    
    filter2D_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_src, d_filtered, src_width, src_height, src_channels, src_width);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    checkCudaError(err, "Failed to launch filter2D_kernel in pyr_down_gpu");
    
    // Step 2: Downsample by taking every other pixel
    dim3 blockSizeDown(16, 16);
    dim3 gridSizeDown((dst_width + blockSizeDown.x - 1) / blockSizeDown.x, 
                     (dst_height + blockSizeDown.y - 1) / blockSizeDown.y);
    
    downsample_kernel<<<gridSizeDown, blockSizeDown, 0, stream>>>(
        d_filtered, d_dst, src_width, src_height, src_channels, src_width, dst_width);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    checkCudaError(err, "Failed to launch downsample_kernel in pyr_down_gpu");
    
    // Free temporary buffer
    cudaFree(d_filtered);
}

// Implementation of pyramid upsampling
void pyr_up_gpu(
    const float* d_src,
    int src_width,
    int src_height,
    int src_channels,
    float* d_dst,
    int dst_width,
    int dst_height,
    cudaStream_t stream)
{
    // Allocate temporary buffer for upsampled image (before filtering)
    float* d_upsampled = nullptr;
    size_t upsampled_size = dst_width * dst_height * src_channels * sizeof(float);
    cudaError_t err = cudaMalloc(&d_upsampled, upsampled_size);
    checkCudaError(err, "Failed to allocate device memory for upsampled image in pyr_up_gpu");
    
    // Step 1: Upsample by placing pixels at even positions and filling with zeros
    dim3 blockSize(16, 16);
    dim3 gridSize((dst_width + blockSize.x - 1) / blockSize.x, 
                 (dst_height + blockSize.y - 1) / blockSize.y);
    
    upsample_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_src, d_upsampled, src_width, src_height, dst_width, dst_height, 
        src_channels, src_width, dst_width);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    checkCudaError(err, "Failed to launch upsample_kernel in pyr_up_gpu");
    
    // Step 2: Apply Gaussian blur to upsampled image (scaled by 4 to match CPU implementation)
    // This is done by applying the filter2D_kernel with a kernel of 4 times the normal values
    // In this implementation, we'll modify the final output directly in the filter2D_kernel
    
    dim3 blockSizeFilter(16, 16);
    dim3 gridSizeFilter((dst_width + blockSizeFilter.x - 1) / blockSizeFilter.x, 
                       (dst_height + blockSizeFilter.y - 1) / blockSizeFilter.y);
    
    // Create a custom scaled kernel (4x normal kernel)
    float d_scaled_kernel[25];
    for (int i = 0; i < 25; i++) {
        d_scaled_kernel[i] = h_gaussian_kernel[i] * 4.0f;
    }
    
    // Copy to constant memory (temporary dedicated storage)
    const char* kernel_name = "d_scaled_kernel";
    err = cudaMemcpyToSymbol(d_gaussian_kernel, d_scaled_kernel, sizeof(d_scaled_kernel));
    checkCudaError(err, "Failed to copy scaled Gaussian kernel to device in pyr_up_gpu");
    
    filter2D_kernel<<<gridSizeFilter, blockSizeFilter, 0, stream>>>(
        d_upsampled, d_dst, dst_width, dst_height, src_channels, dst_width);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    checkCudaError(err, "Failed to launch filter2D_kernel for upsampled image in pyr_up_gpu");
    
    // Restore original kernel
    err = cudaMemcpyToSymbol(d_gaussian_kernel, h_gaussian_kernel, sizeof(h_gaussian_kernel));
    checkCudaError(err, "Failed to restore original Gaussian kernel in pyr_up_gpu");
    
    // Free temporary buffer
    cudaFree(d_upsampled);
}

// Host wrapper for pyramid downsampling
bool pyr_down_wrapper(
    const float* h_src,
    int src_width,
    int src_height,
    int src_channels,
    float* h_dst)
{
    // Calculate output dimensions
    int dst_width = src_width / 2;
    int dst_height = src_height / 2;
    
    // Calculate data sizes
    size_t src_size = src_width * src_height * src_channels * sizeof(float);
    size_t dst_size = dst_width * dst_height * src_channels * sizeof(float);
    
    // Allocate device memory
    float* d_src = nullptr;
    float* d_dst = nullptr;
    
    cudaError_t err;
    
    // Allocate and copy input data
    err = cudaMalloc(&d_src, src_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for source image: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    err = cudaMalloc(&d_dst, dst_size);
    if (err != cudaSuccess) {
        cudaFree(d_src);
        printf("Failed to allocate device memory for destination image: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    err = cudaMemcpy(d_src, h_src, src_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_dst);
        printf("Failed to copy source image to device: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    // Perform downsampling
    try {
        pyr_down_gpu(d_src, src_width, src_height, src_channels, d_dst);
    }
    catch (const std::exception& e) {
        cudaFree(d_src);
        cudaFree(d_dst);
        printf("Error in pyr_down_gpu: %s\n", e.what());
        return false;
    }
    
    // Copy result back to host
    err = cudaMemcpy(h_dst, d_dst, dst_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_dst);
        printf("Failed to copy result from device: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    // Free device memory
    cudaFree(d_src);
    cudaFree(d_dst);
    
    return true;
}

// Host wrapper for pyramid upsampling
bool pyr_up_wrapper(
    const float* h_src,
    int src_width,
    int src_height,
    int src_channels,
    float* h_dst,
    int dst_width,
    int dst_height)
{
    // Calculate data sizes
    size_t src_size = src_width * src_height * src_channels * sizeof(float);
    size_t dst_size = dst_width * dst_height * src_channels * sizeof(float);
    
    // Allocate device memory
    float* d_src = nullptr;
    float* d_dst = nullptr;
    
    cudaError_t err;
    
    // Allocate and copy input data
    err = cudaMalloc(&d_src, src_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for source image: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    err = cudaMalloc(&d_dst, dst_size);
    if (err != cudaSuccess) {
        cudaFree(d_src);
        printf("Failed to allocate device memory for destination image: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    err = cudaMemcpy(d_src, h_src, src_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_dst);
        printf("Failed to copy source image to device: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    // Perform upsampling
    try {
        pyr_up_gpu(d_src, src_width, src_height, src_channels, d_dst, dst_width, dst_height);
    }
    catch (const std::exception& e) {
        cudaFree(d_src);
        cudaFree(d_dst);
        printf("Error in pyr_up_gpu: %s\n", e.what());
        return false;
    }
    
    // Copy result back to host
    err = cudaMemcpy(h_dst, d_dst, dst_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_dst);
        printf("Failed to copy result from device: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    // Free device memory
    cudaFree(d_src);
    cudaFree(d_dst);
    
    return true;
}

// Convenience function: OpenCV Mat to device memory
bool mat_to_device(
    const cv::Mat& src_mat,
    float** d_src,
    size_t& src_size)
{
    // Check input
    if (src_mat.empty()) {
        printf("Input Mat is empty\n");
        return false;
    }
    if (src_mat.depth() != CV_32F) {
        printf("Input Mat must be CV_32F type\n");
        return false;
    }
    
    // Calculate size and allocate device memory
    int width = src_mat.cols;
    int height = src_mat.rows;
    int channels = src_mat.channels();
    src_size = width * height * channels * sizeof(float);
    
    cudaError_t err = cudaMalloc(d_src, src_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    // Copy data to device
    err = cudaMemcpy(*d_src, src_mat.data, src_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(*d_src);
        *d_src = nullptr;
        printf("Failed to copy data to device: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    return true;
}

// Convenience function: Device memory to OpenCV Mat
bool device_to_mat(
    const float* d_dst,
    int width,
    int height,
    int channels,
    cv::Mat& dst_mat)
{
    // Create output Mat
    int mat_type;
    if (channels == 1) {
        mat_type = CV_32FC1;
    } else if (channels == 3) {
        mat_type = CV_32FC3;
    } else {
        printf("Unsupported number of channels: %d\n", channels);
        return false;
    }
    
    dst_mat.create(height, width, mat_type);
    
    // Calculate data size
    size_t dst_size = width * height * channels * sizeof(float);
    
    // Copy data from device
    cudaError_t err = cudaMemcpy(dst_mat.data, d_dst, dst_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Failed to copy data from device: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    return true;
}

// OpenCV Mat wrapper for pyr_down
cv::Mat pyr_down(const cv::Mat& src) {
    // Check input
    if (src.empty()) {
        throw std::invalid_argument("Input image is empty");
    }
    if (src.depth() != CV_32F) {
        throw std::invalid_argument("Input image must be CV_32F type");
    }
    
    // Calculate output dimensions
    int dst_width = src.cols / 2;
    int dst_height = src.rows / 2;
    int channels = src.channels();
    
    // Create output Mat
    cv::Mat dst(dst_height, dst_width, src.type());
    
    // Allocate device memory
    float* d_src = nullptr;
    float* d_dst = nullptr;
    size_t src_size = 0;
    size_t dst_size = dst_width * dst_height * channels * sizeof(float);
    
    // Copy input data to device
    if (!mat_to_device(src, &d_src, src_size)) {
        throw std::runtime_error("Failed to copy input data to device");
    }
    
    // Allocate device memory for output
    cudaError_t err = cudaMalloc(&d_dst, dst_size);
    if (err != cudaSuccess) {
        cudaFree(d_src);
        throw std::runtime_error("Failed to allocate device memory for output");
    }
    
    // Perform downsampling
    try {
        pyr_down_gpu(d_src, src.cols, src.rows, channels, d_dst);
    }
    catch (const std::exception& e) {
        cudaFree(d_src);
        cudaFree(d_dst);
        throw;
    }
    
    // Copy result back to host
    if (!device_to_mat(d_dst, dst_width, dst_height, channels, dst)) {
        cudaFree(d_src);
        cudaFree(d_dst);
        throw std::runtime_error("Failed to copy result from device");
    }
    
    // Free device memory
    cudaFree(d_src);
    cudaFree(d_dst);
    
    return dst;
}

// OpenCV Mat wrapper for pyr_up
cv::Mat pyr_up(const cv::Mat& src, const cv::Size& dst_size) {
    // Check input
    if (src.empty()) {
        throw std::invalid_argument("Input image is empty");
    }
    if (src.depth() != CV_32F) {
        throw std::invalid_argument("Input image must be CV_32F type");
    }
    
    // Calculate output dimensions (if not specified)
    int dst_width = dst_size.width > 0 ? dst_size.width : src.cols * 2;
    int dst_height = dst_size.height > 0 ? dst_size.height : src.rows * 2;
    int channels = src.channels();
    
    // Create output Mat
    cv::Mat dst(dst_height, dst_width, src.type());
    
    // Allocate device memory
    float* d_src = nullptr;
    float* d_dst = nullptr;
    size_t src_size = 0;
    size_t dst_size_bytes = dst_width * dst_height * channels * sizeof(float);
    
    // Copy input data to device
    if (!mat_to_device(src, &d_src, src_size)) {
        throw std::runtime_error("Failed to copy input data to device");
    }
    
    // Allocate device memory for output
    cudaError_t err = cudaMalloc(&d_dst, dst_size_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_src);
        throw std::runtime_error("Failed to allocate device memory for output");
    }
    
    // Perform upsampling
    try {
        pyr_up_gpu(d_src, src.cols, src.rows, channels, d_dst, dst_width, dst_height);
    }
    catch (const std::exception& e) {
        cudaFree(d_src);
        cudaFree(d_dst);
        throw;
    }
    
    // Copy result back to host
    if (!device_to_mat(d_dst, dst_width, dst_height, channels, dst)) {
        cudaFree(d_src);
        cudaFree(d_dst);
        throw std::runtime_error("Failed to copy result from device");
    }
    
    // Free device memory
    cudaFree(d_src);
    cudaFree(d_dst);
    
    return dst;
}

// Clean up resources
void cleanup_pyramid() {
    // No resources to clean up for now
}

} // namespace evmcuda