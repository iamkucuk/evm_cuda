#include "../include/cuda_processing.cuh"
#include "../include/cuda_color_conversion.cuh"
#include <cuda_runtime.h>

namespace cuda_evm {

// Signal combination kernel for EVM reconstruction
__global__ void combine_yiq_signals_kernel(
    const float* d_original_yiq,
    const float* d_filtered_yiq,
    float* d_combined_yiq,
    int width, int height, int channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int spatial_idx = (y * width + x) * channels;
    
    for (int c = 0; c < channels; c++) {
        const int idx = spatial_idx + c;
        // EVM reconstruction: combined = original + filtered
        d_combined_yiq[idx] = d_original_yiq[idx] + d_filtered_yiq[idx];
    }
}

// Main reconstruction function
cudaError_t reconstruct_gaussian_frame_gpu(
    const float* d_original_rgb,
    const float* d_filtered_yiq_signal,
    float* d_output_rgb,
    int width, int height, int channels,
    float alpha, float chrom_attenuation)
{
    if (!d_original_rgb || !d_filtered_yiq_signal || !d_output_rgb || 
        width <= 0 || height <= 0 || channels != 3) {
        return cudaErrorInvalidValue;
    }
    
    const size_t image_size = width * height * channels * sizeof(float);
    
    // Allocate temporary buffers
    float *d_original_yiq = nullptr;
    float *d_combined_yiq = nullptr;
    
    cudaError_t err = cudaMalloc(&d_original_yiq, image_size);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&d_combined_yiq, image_size);
    if (err != cudaSuccess) {
        cudaFree(d_original_yiq);
        return err;
    }
    
    // Convert original RGB to YIQ
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    rgb_to_yiq_planar_kernel<<<gridSize, blockSize>>>(d_original_rgb, d_original_yiq, width, height, channels);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_original_yiq);
        cudaFree(d_combined_yiq);
        return err;
    }
    
    // Combine original and filtered signals: combined = original + filtered
    combine_yiq_signals_kernel<<<gridSize, blockSize>>>(
        d_original_yiq, d_filtered_yiq_signal, d_combined_yiq, width, height, channels);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_original_yiq);
        cudaFree(d_combined_yiq);
        return err;
    }
    
    // Convert back to RGB
    yiq_to_rgb_planar_kernel<<<gridSize, blockSize>>>(d_combined_yiq, d_output_rgb, width, height, channels);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_original_yiq);
        cudaFree(d_combined_yiq);
        return err;
    }
    
    // Cleanup
    cudaFree(d_original_yiq);
    cudaFree(d_combined_yiq);
    
    return cudaSuccess;
}

// GPU-resident batch reconstruction for all frames (optimal memory efficiency)
cudaError_t reconstruct_gaussian_batch_gpu(
    const float* d_original_rgb_batch,
    const float* d_filtered_yiq_batch,
    float* d_output_rgb_batch,
    int width, int height, int channels, int num_frames,
    float alpha, float chrom_attenuation)
{
    if (!d_original_rgb_batch || !d_filtered_yiq_batch || !d_output_rgb_batch || 
        width <= 0 || height <= 0 || channels != 3 || num_frames <= 0) {
        return cudaErrorInvalidValue;
    }
    
    // Process each frame using existing verified single-frame function
    // This reuses all the verified atomic components without modification
    for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        const float* d_original_rgb = d_original_rgb_batch + (frame_idx * width * height * channels);
        const float* d_filtered_yiq = d_filtered_yiq_batch + (frame_idx * width * height * channels);
        float* d_output_rgb = d_output_rgb_batch + (frame_idx * width * height * channels);
        
        // Apply verified atomic reconstruction
        cudaError_t err = reconstruct_gaussian_frame_gpu(
            d_original_rgb, d_filtered_yiq, d_output_rgb, 
            width, height, channels, alpha, chrom_attenuation);
        
        if (err != cudaSuccess) {
            return err;
        }
    }
    
    return cudaSuccess;
}

} // namespace cuda_evm