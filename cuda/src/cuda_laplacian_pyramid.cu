#include "../include/cuda_laplacian_pyramid.cuh"
#include "../include/cuda_color_conversion.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

namespace cuda_evm {

// ============================================================================
// LaplacianPyramidGPU Implementation
// ============================================================================

cudaError_t LaplacianPyramidGPU::allocate(int base_width, int base_height, int levels, int num_channels) {
    deallocate(); // Clean up any existing allocation
    
    num_levels = levels;
    channels = num_channels;
    
    d_levels.resize(levels);
    widths.resize(levels);
    heights.resize(levels);
    level_sizes.resize(levels);
    
    // Calculate dimensions and allocate memory for each level
    for (int level = 0; level < levels; level++) {
        calculatePyramidLevelDimensions(base_width, base_height, level, 
                                       widths[level], heights[level]);
        
        level_sizes[level] = widths[level] * heights[level] * channels;
        
        cudaError_t err = cudaMalloc(&d_levels[level], level_sizes[level] * sizeof(float));
        if (err != cudaSuccess) {
            deallocate(); // Clean up on failure
            return err;
        }
    }
    
    return cudaSuccess;
}

void LaplacianPyramidGPU::deallocate() {
    for (auto ptr : d_levels) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    d_levels.clear();
    widths.clear();
    heights.clear();
    level_sizes.clear();
    num_levels = 0;
}

// ============================================================================
// CUDA Kernels for Pyramid Operations
// ============================================================================

/**
 * @brief CUDA kernel for pyrDown operation (blur + downsample)
 */
__global__ void pyrDown_kernel(
    const float* d_input, float* d_output,
    int input_width, int input_height,
    int output_width, int output_height,
    int channels)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z;
    
    if (out_x >= output_width || out_y >= output_height || ch >= channels) return;
    
    // Gaussian kernel [1, 4, 6, 4, 1] / 16 (simplified)
    // For pyrDown, we sample at (2*out_x, 2*out_y) with blur
    int in_x = out_x * 2;
    int in_y = out_y * 2;
    
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Apply 5x5 Gaussian kernel around (in_x, in_y)
    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            int px = in_x + kx;
            int py = in_y + ky;
            
            // Border handling: reflect
            px = max(0, min(input_width - 1, px));
            py = max(0, min(input_height - 1, py));
            
            // Gaussian weights (simplified)
            float weight = 1.0f;
            if (abs(kx) == 2 || abs(ky) == 2) weight = 0.25f;
            else if (abs(kx) == 1 || abs(ky) == 1) weight = 0.5f;
            
            int input_idx = (py * input_width + px) * channels + ch;
            sum += d_input[input_idx] * weight;
            weight_sum += weight;
        }
    }
    
    int output_idx = (out_y * output_width + out_x) * channels + ch;
    d_output[output_idx] = sum / weight_sum;
}

/**
 * @brief CUDA kernel for pyrUp operation (upsample + blur)
 */
__global__ void pyrUp_kernel(
    const float* d_input, float* d_output,
    int input_width, int input_height,
    int output_width, int output_height,
    int channels)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z;
    
    if (out_x >= output_width || out_y >= output_height || ch >= channels) return;
    
    // Map output pixel to input coordinate
    float in_x_f = out_x * 0.5f;
    float in_y_f = out_y * 0.5f;
    
    int in_x = (int)floorf(in_x_f);
    int in_y = (int)floorf(in_y_f);
    
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Bilinear interpolation with Gaussian blur
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int px = in_x + kx;
            int py = in_y + ky;
            
            if (px >= 0 && px < input_width && py >= 0 && py < input_height) {
                float weight = 1.0f;
                if (kx != 0 || ky != 0) weight = 0.5f;
                
                int input_idx = (py * input_width + px) * channels + ch;
                sum += d_input[input_idx] * weight;
                weight_sum += weight;
            }
        }
    }
    
    int output_idx = (out_y * output_width + out_x) * channels + ch;
    d_output[output_idx] = (weight_sum > 0) ? sum / weight_sum : 0.0f;
}

/**
 * @brief CUDA kernel for computing Laplacian level (prev - upsampled)
 */
__global__ void computeLaplacianLevel_kernel(
    const float* d_prev, const float* d_upsampled, float* d_laplacian,
    int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z;
    
    if (x >= width || y >= height || ch >= channels) return;
    
    int idx = (y * width + x) * channels + ch;
    d_laplacian[idx] = d_prev[idx] - d_upsampled[idx];
}

// ============================================================================
// Helper Functions
// ============================================================================

void calculatePyramidLevelDimensions(int base_width, int base_height, int level,
                                   int& level_width, int& level_height) {
    level_width = base_width;
    level_height = base_height;
    
    for (int i = 0; i < level; i++) {
        level_width = (level_width + 1) / 2;  // Round up division
        level_height = (level_height + 1) / 2;
    }
    
    // Ensure minimum size
    level_width = max(1, level_width);
    level_height = max(1, level_height);
}

float calculateSpatialAttenuation(int level, float lambda_cutoff,
                                 int base_width, int base_height) {
    // Calculate attenuation based on spatial frequency
    // Higher pyramid levels represent lower spatial frequencies
    float level_scale = powf(2.0f, (float)level);
    float spatial_wavelength = level_scale * 2.0f; // Approximate wavelength
    
    if (spatial_wavelength > lambda_cutoff) {
        // Attenuate high spatial frequencies beyond cutoff
        return lambda_cutoff / spatial_wavelength;
    }
    
    return 1.0f; // No attenuation below cutoff
}

// ============================================================================
// Main Implementation Functions
// ============================================================================

cudaError_t generateLaplacianPyramid_gpu(
    const float* d_input_yiq,
    int width, int height,
    int pyramid_levels,
    LaplacianPyramidGPU& pyramid)
{
    cudaError_t err;
    
    // Allocate pyramid structure
    err = pyramid.allocate(width, height, pyramid_levels + 1, 3); // +1 for residual
    if (err != cudaSuccess) return err;
    
    // Temporary device memory for pyramid generation
    float* d_prev = nullptr;
    float* d_downsampled = nullptr;
    float* d_upsampled = nullptr;
    
    size_t prev_size = width * height * 3 * sizeof(float);
    err = cudaMalloc(&d_prev, prev_size);
    if (err != cudaSuccess) goto cleanup;
    
    // Copy input to d_prev
    err = cudaMemcpy(d_prev, d_input_yiq, prev_size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) goto cleanup;
    
    // Generate Laplacian levels
    for (int level = 0; level < pyramid_levels; level++) {
        int curr_width = pyramid.widths[level];
        int curr_height = pyramid.heights[level];
        int next_width = pyramid.widths[level + 1];
        int next_height = pyramid.heights[level + 1];
        
        // Allocate temporary memory for this level
        size_t down_size = next_width * next_height * 3 * sizeof(float);
        size_t up_size = curr_width * curr_height * 3 * sizeof(float);
        
        err = cudaMalloc(&d_downsampled, down_size);
        if (err != cudaSuccess) goto cleanup;
        
        err = cudaMalloc(&d_upsampled, up_size);
        if (err != cudaSuccess) goto cleanup;
        
        // 1. Downsample current level
        dim3 blockSize(16, 16, 1);
        dim3 gridSize((next_width + blockSize.x - 1) / blockSize.x,
                     (next_height + blockSize.y - 1) / blockSize.y,
                     3);
        
        pyrDown_kernel<<<gridSize, blockSize>>>(
            d_prev, d_downsampled,
            curr_width, curr_height,
            next_width, next_height, 3);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
        
        // 2. Upsample back to current level size
        gridSize = dim3((curr_width + blockSize.x - 1) / blockSize.x,
                       (curr_height + blockSize.y - 1) / blockSize.y,
                       3);
        
        pyrUp_kernel<<<gridSize, blockSize>>>(
            d_downsampled, d_upsampled,
            next_width, next_height,
            curr_width, curr_height, 3);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
        
        // 3. Compute Laplacian level (prev - upsampled)
        computeLaplacianLevel_kernel<<<gridSize, blockSize>>>(
            d_prev, d_upsampled, pyramid.d_levels[level],
            curr_width, curr_height, 3);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
        
        // Update d_prev for next iteration
        cudaFree(d_prev);
        d_prev = d_downsampled;
        d_downsampled = nullptr;
        
        cudaFree(d_upsampled);
        d_upsampled = nullptr;
    }
    
    // Store final residual level
    if (pyramid_levels < pyramid.num_levels) {
        size_t residual_size = pyramid.level_sizes[pyramid_levels];
        err = cudaMemcpy(pyramid.d_levels[pyramid_levels], d_prev, 
                        residual_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
cleanup:
    if (d_prev) cudaFree(d_prev);
    if (d_downsampled) cudaFree(d_downsampled);
    if (d_upsampled) cudaFree(d_upsampled);
    
    return err;
}

cudaError_t getLaplacianPyramids_gpu(
    const float* d_input_frames,
    int width, int height, int num_frames,
    int pyramid_levels,
    std::vector<LaplacianPyramidGPU>& pyramids)
{
    pyramids.resize(num_frames);
    
    size_t frame_size = width * height * 3;
    
    for (int frame = 0; frame < num_frames; frame++) {
        const float* d_frame = d_input_frames + frame * frame_size;
        
        cudaError_t err = generateLaplacianPyramid_gpu(
            d_frame, width, height, pyramid_levels, pyramids[frame]);
        
        if (err != cudaSuccess) {
            std::cerr << "Error generating pyramid for frame " << frame 
                     << ": " << cudaGetErrorString(err) << std::endl;
            return err;
        }
    }
    
    return cudaSuccess;
}

// Placeholder implementations for complex functions
// These will be implemented in subsequent steps

cudaError_t filterLaplacianPyramids_gpu(
    std::vector<LaplacianPyramidGPU>& pyramids,
    int num_frames, int pyramid_levels,
    float fps, float fl, float fh,
    float alpha, float lambda_cutoff, float chrom_attenuation)
{
    // TODO: Implement IIR temporal filtering per pyramid level
    // This is complex and will be implemented in next iteration
    std::cerr << "filterLaplacianPyramids_gpu: NOT YET IMPLEMENTED" << std::endl;
    return cudaErrorNotSupported;
}

/**
 * @brief CUDA kernel for adding upsampled pyramid level to reconstruction
 */
__global__ void addPyramidLevel_kernel(
    const float* d_upsampled, float* d_reconstructed,
    int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z;
    
    if (x >= width || y >= height || ch >= channels) return;
    
    int idx = (y * width + x) * channels + ch;
    d_reconstructed[idx] += d_upsampled[idx];
}

cudaError_t reconstructLaplacianImage_gpu(
    const float* d_original_yiq,
    const LaplacianPyramidGPU& filtered_pyramid,
    float* d_output_yiq,
    int width, int height)
{
    cudaError_t err;
    
    // Validate inputs
    if (filtered_pyramid.num_levels < 2) {
        // Not enough pyramid levels, just copy original
        size_t frame_size = width * height * 3 * sizeof(float);
        return cudaMemcpy(d_output_yiq, d_original_yiq, frame_size, cudaMemcpyDeviceToDevice);
    }
    
    // Start with original YIQ image as base for reconstruction
    size_t frame_size = width * height * 3 * sizeof(float);
    err = cudaMemcpy(d_output_yiq, d_original_yiq, frame_size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) return err;
    
    // Temporary memory for upsampling operations
    float* d_temp_upsampled = nullptr;
    float* d_current_level = nullptr;
    
    // First, add level 0 directly (no upsampling needed)
    if (filtered_pyramid.num_levels > 0) {
        dim3 blockSize(16, 16, 1);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y,
                     3);
        
        addPyramidLevel_kernel<<<gridSize, blockSize>>>(
            filtered_pyramid.d_levels[0], d_output_yiq, width, height, 3);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    }
    
    // Process pyramid levels from 1 to N-2 (skip final residual)
    // These need upsampling to base resolution
    for (int level = 1; level < filtered_pyramid.num_levels - 1; level++) {
        
        // Get current level data
        float* d_level_data = filtered_pyramid.d_levels[level];
        int level_width = filtered_pyramid.widths[level];
        int level_height = filtered_pyramid.heights[level];
        
        // Allocate memory for current level processing
        size_t level_size = level_width * level_height * 3 * sizeof(float);
        err = cudaMalloc(&d_current_level, level_size);
        if (err != cudaSuccess) goto cleanup;
        
        // Copy current pyramid level
        err = cudaMemcpy(d_current_level, d_level_data, level_size, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) goto cleanup;
        
        // Iteratively upsample from current level to base size
        float* d_source = d_current_level;
        int curr_width = level_width;
        int curr_height = level_height;
        
        for (int up_iter = 0; up_iter < level; up_iter++) {
            // Calculate target size for this upsampling step
            int target_level = level - up_iter - 1;
            int target_width = filtered_pyramid.widths[target_level];
            int target_height = filtered_pyramid.heights[target_level];
            
            // Allocate temporary memory for upsampled result
            size_t target_size = target_width * target_height * 3 * sizeof(float);
            err = cudaMalloc(&d_temp_upsampled, target_size);
            if (err != cudaSuccess) goto cleanup;
            
            // Perform pyrUp operation
            dim3 blockSize(16, 16, 1);
            dim3 gridSize((target_width + blockSize.x - 1) / blockSize.x,
                         (target_height + blockSize.y - 1) / blockSize.y,
                         3);
            
            pyrUp_kernel<<<gridSize, blockSize>>>(
                d_source, d_temp_upsampled,
                curr_width, curr_height,
                target_width, target_height, 3);
            
            err = cudaGetLastError();
            if (err != cudaSuccess) goto cleanup;
            
            // Update for next iteration
            if (d_source != d_current_level) {
                cudaFree(d_source);
            }
            d_source = d_temp_upsampled;
            curr_width = target_width;
            curr_height = target_height;
            d_temp_upsampled = nullptr; // Prevent double-free
        }
        
        // Now d_source contains the upsampled level at base resolution
        // Add it to the reconstruction
        if (curr_width == width && curr_height == height) {
            dim3 blockSize(16, 16, 1);
            dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                         (height + blockSize.y - 1) / blockSize.y,
                         3);
            
            addPyramidLevel_kernel<<<gridSize, blockSize>>>(
                d_source, d_output_yiq, width, height, 3);
            
            err = cudaGetLastError();
            if (err != cudaSuccess) goto cleanup;
        }
        
        // Cleanup current level
        if (d_source != d_current_level) {
            cudaFree(d_source);
        }
        cudaFree(d_current_level);
        d_current_level = nullptr;
    }
    
cleanup:
    if (d_temp_upsampled) cudaFree(d_temp_upsampled);
    if (d_current_level) cudaFree(d_current_level);
    
    return err;
}

} // namespace cuda_evm