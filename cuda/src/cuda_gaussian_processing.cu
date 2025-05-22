#include "cuda_gaussian_processing.cuh"
#include "cuda_color_conversion.cuh"
#include "cuda_pyramid.cuh"

#include <cuda_runtime.h>
#include <cufft.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <memory>

// Define logging for debug output
#define LOG_GAUSSIAN(message) std::cout << "[CUDA Gaussian] " << message << std::endl

namespace evmcuda {

// Global variables for FFT plans
static cufftHandle forward_plan = 0;
static cufftHandle inverse_plan = 0;
static bool fft_initialized = false;

// Error checking helper
static void checkCudaError(const char* operation) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + operation + ": " + cudaGetErrorString(error));
    }
}

// Error checking helper for CUFFT
static void checkCufftError(cufftResult result, const char* operation) {
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string("CUFFT error in ") + operation + ": " + std::to_string(result));
    }
}

// OpenCV Mat wrapper for RGB to YIQ conversion
cv::Mat rgb_to_yiq_opencv_wrapper(const cv::Mat& input_rgb) {
    cv::Mat input_float;
    cv::Mat result_yiq;
    
    // Convert to float if needed
    if (input_rgb.type() == CV_8UC3) {
        input_rgb.convertTo(input_float, CV_32FC3);
    } else {
        input_float = input_rgb;
    }
    
    // Create output matrix
    result_yiq = cv::Mat(input_float.size(), CV_32FC3);
    
    // Convert data to flat arrays
    std::vector<float> rgb_data(input_float.rows * input_float.cols * 3);
    std::vector<float> yiq_data(input_float.rows * input_float.cols * 3);
    
    // Copy input data
    if (input_float.isContinuous()) {
        std::memcpy(rgb_data.data(), input_float.data, rgb_data.size() * sizeof(float));
    } else {
        size_t idx = 0;
        for (int i = 0; i < input_float.rows; ++i) {
            const float* row_ptr = input_float.ptr<float>(i);
            for (int j = 0; j < input_float.cols * 3; ++j) {
                rgb_data[idx++] = row_ptr[j];
            }
        }
    }
    
    // Call CUDA wrapper
    rgb_to_yiq_wrapper(rgb_data.data(), yiq_data.data(), input_float.cols, input_float.rows);
    
    // Copy result back to Mat
    std::memcpy(result_yiq.data, yiq_data.data(), yiq_data.size() * sizeof(float));
    
    return result_yiq;
}

// OpenCV Mat wrapper for YIQ to RGB conversion
cv::Mat yiq_to_rgb_opencv_wrapper(const cv::Mat& input_yiq) {
    cv::Mat result_rgb(input_yiq.size(), CV_32FC3);
    
    // Convert data to flat arrays
    std::vector<float> yiq_data(input_yiq.rows * input_yiq.cols * 3);
    std::vector<float> rgb_data(input_yiq.rows * input_yiq.cols * 3);
    
    // Copy input data
    if (input_yiq.isContinuous()) {
        std::memcpy(yiq_data.data(), input_yiq.data, yiq_data.size() * sizeof(float));
    } else {
        size_t idx = 0;
        for (int i = 0; i < input_yiq.rows; ++i) {
            const float* row_ptr = input_yiq.ptr<float>(i);
            for (int j = 0; j < input_yiq.cols * 3; ++j) {
                yiq_data[idx++] = row_ptr[j];
            }
        }
    }
    
    // Call CUDA wrapper
    yiq_to_rgb_wrapper(yiq_data.data(), rgb_data.data(), input_yiq.cols, input_yiq.rows);
    
    // Copy result back to Mat
    std::memcpy(result_rgb.data, rgb_data.data(), rgb_data.size() * sizeof(float));
    
    return result_rgb;
}

// cuFFT kernel declarations
__global__ void reorganize_for_cufft_kernel(
    const float* __restrict__ d_frame_data,
    cufftReal* __restrict__ d_pixel_data,
    int width, int height, int channels, int num_frames, int dft_size
);

__global__ void apply_frequency_mask_kernel(
    cufftComplex* __restrict__ d_fft_data,
    int batch_size, int dft_size, float fps, float fl, float fh
);

__global__ void reorganize_from_cufft_kernel(
    const cufftReal* __restrict__ d_pixel_data,
    float* __restrict__ d_frame_data,
    int width, int height, int channels, int num_frames, int dft_size
);

// Initialize Gaussian processing module
bool init_gaussian_processing() {
    LOG_GAUSSIAN("Initializing Gaussian processing module...");
    
    // Initialize required dependencies
    if (!init_color_conversion()) {
        LOG_GAUSSIAN("Failed to initialize color conversion module");
        return false;
    }
    
    if (!init_pyramid()) {
        LOG_GAUSSIAN("Failed to initialize pyramid module");
        return false;
    }
    
    LOG_GAUSSIAN("Gaussian processing module initialized successfully");
    return true;
}

// Clean up Gaussian processing module
void cleanup_gaussian_processing() {
    LOG_GAUSSIAN("Cleaning up Gaussian processing module...");
    
    // Clean up FFT plans if initialized
    cleanup_fft_plans();
    
    // Clean up dependencies
    cleanup_pyramid();
    cleanup_color_conversion();
    
    LOG_GAUSSIAN("Gaussian processing module cleaned up");
}

// 5x5 Gaussian kernel (matching CPU implementation in processing.hpp)
__constant__ float d_gaussian_spatial_kernel[25] = {
    1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f,  1.0f/256.0f,
    4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f,  4.0f/256.0f,
    6.0f/256.0f, 24.0f/256.0f, 36.0f/256.0f, 24.0f/256.0f,  6.0f/256.0f,
    4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f,  4.0f/256.0f,
    1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f,  1.0f/256.0f
};

// CUDA kernel for Gaussian filtering (convolution with 5x5 kernel)
__global__ void gaussian_filter_kernel(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    int width,
    int height,
    int channels
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int output_idx = (y * width + x) * channels;
    
    // Apply 5x5 Gaussian kernel convolution
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        // Convolve with 5x5 kernel
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                // Calculate input coordinates with reflection padding (BORDER_REFLECT_101)
                int in_y = y + ky;
                int in_x = x + kx;
                
                // Reflect coordinates if out of bounds
                if (in_y < 0) in_y = -in_y;
                if (in_y >= height) in_y = 2 * height - in_y - 2;
                if (in_x < 0) in_x = -in_x;
                if (in_x >= width) in_x = 2 * width - in_x - 2;
                
                // Clamp to valid range (safety check)
                in_y = max(0, min(in_y, height - 1));
                in_x = max(0, min(in_x, width - 1));
                
                const int input_idx = (in_y * width + in_x) * channels + c;
                const int kernel_idx = (ky + 2) * 5 + (kx + 2);
                
                sum += d_input[input_idx] * d_gaussian_spatial_kernel[kernel_idx];
            }
        }
        
        d_output[output_idx + c] = sum;
    }
}

// CUDA kernel for spatial downsampling with proper Gaussian filtering
__global__ void spatial_downsample_kernel(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    int channels
) {
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (out_x >= output_width || out_y >= output_height) return;
    
    // Sample every other pixel from the Gaussian-filtered input
    const int in_x = out_x * 2;
    const int in_y = out_y * 2;
    
    const int input_idx = (in_y * input_width + in_x) * channels;
    const int output_idx = (out_y * output_width + out_x) * channels;
    
    // Copy all channels
    for (int c = 0; c < channels; c++) {
        d_output[output_idx + c] = d_input[input_idx + c];
    }
}

// CUDA kernel for spatial upsampling with zeros insertion
__global__ void spatial_upsample_zeros_kernel(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    int channels
) {
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (out_x >= output_width || out_y >= output_height) return;
    
    const int output_idx = (out_y * output_width + out_x) * channels;
    
    // Initialize to zero
    for (int c = 0; c < channels; c++) {
        d_output[output_idx + c] = 0.0f;
    }
    
    // If this corresponds to an even position, copy from input
    if (out_x % 2 == 0 && out_y % 2 == 0) {
        const int in_x = out_x / 2;
        const int in_y = out_y / 2;
        
        if (in_x < input_width && in_y < input_height) {
            const int input_idx = (in_y * input_width + in_x) * channels;
            
            for (int c = 0; c < channels; c++) {
                d_output[output_idx + c] = d_input[input_idx + c];
            }
        }
    }
}

// CUDA kernel for scaling by 4 and Gaussian filtering (for pyrUp)
__global__ void scale_and_filter_kernel(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    int width,
    int height,
    int channels
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int output_idx = (y * width + x) * channels;
    
    // Apply 4 * Gaussian kernel convolution (matching CPU pyrUp logic)
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        // Convolve with 4 * 5x5 kernel
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                // Calculate input coordinates with reflection padding
                int in_y = y + ky;
                int in_x = x + kx;
                
                // Reflect coordinates if out of bounds
                if (in_y < 0) in_y = -in_y;
                if (in_y >= height) in_y = 2 * height - in_y - 2;
                if (in_x < 0) in_x = -in_x;
                if (in_x >= width) in_x = 2 * width - in_x - 2;
                
                // Clamp to valid range
                in_y = max(0, min(in_y, height - 1));
                in_x = max(0, min(in_x, width - 1));
                
                const int input_idx = (in_y * width + in_x) * channels + c;
                const int kernel_idx = (ky + 2) * 5 + (kx + 2);
                
                sum += d_input[input_idx] * d_gaussian_spatial_kernel[kernel_idx] * 4.0f;
            }
        }
        
        d_output[output_idx + c] = sum;
    }
}

// CUDA Gaussian spatial filtering implementation
cv::Mat spatially_filter_gaussian_gpu(const cv::Mat& input_rgb, int levels) {
    LOG_GAUSSIAN("Starting spatial filtering with " + std::to_string(levels) + " levels");
    
    if (input_rgb.empty() || input_rgb.type() != CV_8UC3 || levels < 0) {
        throw std::runtime_error("Invalid input for spatial filtering");
    }
    
    // Convert RGB to YIQ first
    cv::Mat input_yiq_float = rgb_to_yiq_opencv_wrapper(input_rgb);
    
    if (levels == 0) {
        // No spatial filtering, just return YIQ conversion
        return input_yiq_float;
    }
    
    // Allocate device memory for current level
    int current_width = input_yiq_float.cols;
    int current_height = input_yiq_float.rows;
    int channels = 3;
    
    float* d_current = nullptr;
    float* d_temp = nullptr;
    float* d_filtered = nullptr;  // For Gaussian filtering
    
    size_t current_size = current_width * current_height * channels * sizeof(float);
    cudaMalloc(&d_current, current_size);
    cudaMalloc(&d_filtered, current_size);
    checkCudaError("malloc initial buffers");
    
    // Copy input to device
    cudaMemcpy(d_current, input_yiq_float.ptr<float>(), current_size, cudaMemcpyHostToDevice);
    checkCudaError("copy input to device");
    
    // Store original size for final upsampling
    int original_width = current_width;
    int original_height = current_height;
    
    // Downsample 'levels' times with proper Gaussian filtering
    std::vector<std::pair<int, int>> sizes;
    sizes.push_back({current_width, current_height});
    
    for (int level = 0; level < levels; level++) {
        // Step 1: Apply Gaussian filter to current level
        dim3 filter_block(16, 16);
        dim3 filter_grid((current_width + filter_block.x - 1) / filter_block.x, 
                        (current_height + filter_block.y - 1) / filter_block.y);
        
        gaussian_filter_kernel<<<filter_grid, filter_block>>>(
            d_current, d_filtered,
            current_width, current_height, channels
        );
        checkCudaError("gaussian_filter_kernel");
        
        // Step 2: Downsample the filtered result
        int new_width = current_width / 2;
        int new_height = current_height / 2;
        
        if (new_width < 1 || new_height < 1) break;
        
        size_t new_size = new_width * new_height * channels * sizeof(float);
        cudaMalloc(&d_temp, new_size);
        checkCudaError("malloc d_temp");
        
        dim3 down_block(16, 16);
        dim3 down_grid((new_width + down_block.x - 1) / down_block.x, 
                      (new_height + down_block.y - 1) / down_block.y);
        
        spatial_downsample_kernel<<<down_grid, down_block>>>(
            d_filtered, d_temp,  // Use filtered input for downsampling
            current_width, current_height,
            new_width, new_height,
            channels
        );
        checkCudaError("spatial_downsample_kernel");
        
        // Update buffer management
        cudaFree(d_current);
        cudaFree(d_filtered);
        d_current = d_temp;
        d_temp = nullptr;
        
        // Reallocate filtered buffer for next iteration
        current_width = new_width;
        current_height = new_height;
        current_size = new_size;
        if (level < levels - 1) {  // Don't allocate for last iteration
            cudaMalloc(&d_filtered, current_size);
            checkCudaError("realloc d_filtered");
        }
        
        sizes.push_back({current_width, current_height});
    }
    
    // Upsample back to original size with proper Gaussian filtering
    for (int level = levels - 1; level >= 0; level--) {
        int target_width = sizes[level].first;
        int target_height = sizes[level].second;
        
        size_t target_size = target_width * target_height * channels * sizeof(float);
        float* d_upsampled = nullptr;
        cudaMalloc(&d_temp, target_size);
        cudaMalloc(&d_upsampled, target_size);
        checkCudaError("malloc upsample buffers");
        
        // Step 1: Upsample with zero insertion
        dim3 up_block(16, 16);
        dim3 up_grid((target_width + up_block.x - 1) / up_block.x, 
                    (target_height + up_block.y - 1) / up_block.y);
        
        spatial_upsample_zeros_kernel<<<up_grid, up_block>>>(
            d_current, d_upsampled,
            current_width, current_height,
            target_width, target_height,
            channels
        );
        checkCudaError("spatial_upsample_zeros_kernel");
        
        // Step 2: Apply 4x Gaussian filter to upsampled result
        scale_and_filter_kernel<<<up_grid, up_block>>>(
            d_upsampled, d_temp,
            target_width, target_height, channels
        );
        checkCudaError("scale_and_filter_kernel");
        
        // Update buffers
        cudaFree(d_current);
        cudaFree(d_upsampled);
        d_current = d_temp;
        d_temp = nullptr;
        
        current_width = target_width;
        current_height = target_height;
    }
    
    // Copy result back to host
    cv::Mat result(original_height, original_width, CV_32FC3);
    size_t result_size = original_width * original_height * channels * sizeof(float);
    cudaMemcpy(result.ptr<float>(), d_current, result_size, cudaMemcpyDeviceToHost);
    checkCudaError("copy result to host");
    
    // Clean up
    cudaFree(d_current);
    
    LOG_GAUSSIAN("Spatial filtering completed");
    return result;
}

// Setup FFT plans for temporal filtering
void setup_fft_plans(int num_frames) {
    if (fft_initialized) {
        cleanup_fft_plans();
    }
    
    LOG_GAUSSIAN("Setting up FFT plans for " + std::to_string(num_frames) + " frames");
    
    // Create 1D FFT plans for temporal filtering
    cufftResult result;
    
    result = cufftPlan1d(&forward_plan, num_frames, CUFFT_R2C, 1);
    checkCufftError(result, "cufftPlan1d forward");
    
    result = cufftPlan1d(&inverse_plan, num_frames, CUFFT_C2R, 1);
    checkCufftError(result, "cufftPlan1d inverse");
    
    fft_initialized = true;
    LOG_GAUSSIAN("FFT plans set up successfully");
}

// Clean up FFT plans
void cleanup_fft_plans() {
    if (fft_initialized) {
        LOG_GAUSSIAN("Cleaning up FFT plans");
        
        if (forward_plan != 0) {
            cufftDestroy(forward_plan);
            forward_plan = 0;
        }
        
        if (inverse_plan != 0) {
            cufftDestroy(inverse_plan);
            inverse_plan = 0;
        }
        
        fft_initialized = false;
    }
}

// Kernel to create frequency mask for bandpass filtering
__global__ void create_frequency_mask_kernel(
    float* __restrict__ d_mask,
    int num_frames,
    float fps,
    float fl,
    float fh
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_frames) return;
    
    // Calculate frequency for this bin (like numpy.fft.fftfreq)
    float freq;
    if (idx <= num_frames / 2) {
        freq = static_cast<float>(idx) * fps / static_cast<float>(num_frames);
    } else {
        freq = static_cast<float>(idx - num_frames) * fps / static_cast<float>(num_frames);
    }
    
    // Set mask value: 1.0 if frequency is in passband, 0.0 otherwise
    float abs_freq = fabsf(freq);
    d_mask[idx] = (abs_freq >= fl && abs_freq <= fh) ? 1.0f : 0.0f;
}

// Kernel to apply FFT-based temporal filtering
__global__ void apply_fft_filter_kernel(
    cufftComplex* __restrict__ d_fft_data,
    const float* __restrict__ d_mask,
    int num_frames,
    float alpha,
    float chrom_attenuation,
    int channel
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_frames / 2 + 1) return; // FFT output size for real input
    
    float current_alpha = alpha;
    if (channel == 1 || channel == 2) { // I and Q channels
        current_alpha *= chrom_attenuation;
    }
    
    // Apply mask and amplification
    float mask_value = (idx < num_frames) ? d_mask[idx] : 0.0f;
    d_fft_data[idx].x *= mask_value * current_alpha;
    d_fft_data[idx].y *= mask_value * current_alpha;
}

// Parallel temporal filtering kernel with proper FFT - each thread processes one pixel across all frames
__global__ void parallel_temporal_filter_fft_kernel(
    const float* __restrict__ d_all_frames,  // [num_frames][height][width][channels]
    float* __restrict__ d_filtered_frames,   // Output: same layout
    cufftReal* __restrict__ d_temp_time_series,    // Temporary workspace: [total_pixels][num_frames]
    cufftComplex* __restrict__ d_temp_fft_data,    // Temporary workspace: [total_pixels][num_frames/2+1]
    const float* __restrict__ d_frequency_mask,    // [num_frames/2+1]
    int width,
    int height,
    int channels,
    int num_frames,
    float alpha,
    float chrom_attenuation,
    int total_pixels,
    float fps,
    float fl,
    float fh
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int pixel_idx = y * width + x;
    const int fft_size = num_frames / 2 + 1;
    
    // Each thread processes one pixel's time series across all channels
    for (int c = 0; c < channels; c++) {
        const int channel_offset = c * total_pixels;
        const int pixel_series_idx = channel_offset + pixel_idx;
        
        // Extract time series for this pixel and channel into temporary workspace
        for (int t = 0; t < num_frames; t++) {
            int frame_idx = t * (height * width * channels) + y * (width * channels) + x * channels + c;
            d_temp_time_series[pixel_series_idx * num_frames + t] = d_all_frames[frame_idx];
        }
        
        // Wait for all threads to finish loading data
        __syncthreads();
        
        // Apply cuFFT-based temporal filtering (matching CPU exactly but with proper FFT)
        float current_alpha = alpha;
        if (c > 0) { // Attenuate I and Q channels
            current_alpha *= chrom_attenuation;
        }
        
        // Use existing channel_offset and pixel_series_idx from loop start
        
        // Extract time series directly to workspace (padded to power of 2)
        cufftReal* pixel_time_series = &d_temp_time_series[pixel_series_idx * num_frames];
        cufftComplex* pixel_fft_data = &d_temp_fft_data[pixel_series_idx * fft_size];
        
        // Load time series from global memory to workspace
        for (int t = 0; t < num_frames; t++) {
            int frame_idx = t * (height * width * channels) + y * (width * channels) + x * channels + c;
            pixel_time_series[t] = d_all_frames[frame_idx];
        }
        
        // PLACEHOLDER: This will be replaced by proper cuFFT batched implementation
        // For now, indicate that cuFFT should be used here
        
        // Proper cuFFT implementation should:
        // 1. Use cufftExecR2C for forward transform
        // 2. Apply frequency mask to complex data
        // 3. Use cufftExecC2R for inverse transform
        // 4. Handle batched operations for all pixels simultaneously
        
        // Temporary fallback to simple approach for compilation
        for (int t = 0; t < num_frames; t++) {
            int frame_idx = t * (height * width * channels) + y * (width * channels) + x * channels + c;
            // Simple pass-through with amplification (placeholder)
            d_filtered_frames[frame_idx] = d_all_frames[frame_idx] * current_alpha * 0.01f;
        }
    }
}

// CUDA cuFFT-based temporal filtering implementation - PROPER BATCHED VERSION
std::vector<cv::Mat> temporal_filter_gaussian_batch_gpu(
    const std::vector<cv::Mat>& spatially_filtered_batch,
    float fps,
    float fl,
    float fh,
    float alpha,
    float chrom_attenuation
) {
    LOG_GAUSSIAN("Starting cuFFT BATCHED temporal filtering");
    
    if (spatially_filtered_batch.empty()) {
        throw std::runtime_error("Empty input batch for temporal filtering");
    }
    
    int num_frames = static_cast<int>(spatially_filtered_batch.size());
    if (num_frames <= 1) {
        throw std::runtime_error("Need more than 1 frame for temporal filtering");
    }
    
    cv::Size frame_size = spatially_filtered_batch[0].size();
    int width = frame_size.width;
    int height = frame_size.height;
    int channels = 3;
    
    // Find optimal DFT size (power of 2)
    int dft_size = 1;
    while (dft_size < num_frames) dft_size <<= 1;
    
    LOG_GAUSSIAN("Processing " + std::to_string(width * height) + " pixels with cuFFT batched operations");
    LOG_GAUSSIAN("FFT size: " + std::to_string(dft_size) + " (padded from " + std::to_string(num_frames) + ")");
    
    // Calculate sizes for cuFFT batched operations
    size_t total_pixels = width * height;
    size_t total_pixel_channels = total_pixels * channels;
    
    // Allocate device memory for cuFFT batched operations
    cufftReal* d_input_data = nullptr;    // [batch][dft_size] format for cuFFT
    cufftComplex* d_fft_data = nullptr;   // FFT output
    cufftReal* d_output_data = nullptr;   // IFFT output
    float* d_all_frames = nullptr;        // Original frame layout
    float* d_filtered_frames = nullptr;   // Final output
    
    size_t input_batch_size = total_pixel_channels * dft_size * sizeof(cufftReal);
    size_t fft_batch_size = total_pixel_channels * (dft_size / 2 + 1) * sizeof(cufftComplex);
    size_t frame_data_size = total_pixel_channels * sizeof(float);
    
    cudaMalloc(&d_input_data, input_batch_size);
    cudaMalloc(&d_fft_data, fft_batch_size);
    cudaMalloc(&d_output_data, input_batch_size);
    cudaMalloc(&d_all_frames, num_frames * frame_data_size);
    cudaMalloc(&d_filtered_frames, num_frames * frame_data_size);
    checkCudaError("malloc cuFFT buffers");
    
    // Copy frame data to GPU
    LOG_GAUSSIAN("Transferring frames to GPU");
    for (int t = 0; t < num_frames; t++) {
        size_t frame_offset = t * frame_data_size;
        cudaMemcpy(d_all_frames + frame_offset / sizeof(float), 
                  spatially_filtered_batch[t].ptr<float>(), 
                  frame_data_size, 
                  cudaMemcpyHostToDevice);
    }
    checkCudaError("copy frames to device");
    
    // Create cuFFT plans for batched 1D FFT
    cufftHandle forward_plan, inverse_plan;
    int batch_size = total_pixel_channels;
    
    // Create 1D R2C plan for forward transform
    cufftPlan1d(&forward_plan, dft_size, CUFFT_R2C, batch_size);
    checkCudaError("create forward cuFFT plan");
    
    // Create 1D C2R plan for inverse transform  
    cufftPlan1d(&inverse_plan, dft_size, CUFFT_C2R, batch_size);
    checkCudaError("create inverse cuFFT plan");
    
    // Prepare input data for cuFFT (reorganize from frame-major to pixel-major + zero padding)
    LOG_GAUSSIAN("Reorganizing data for cuFFT batched processing");
    dim3 prep_block(16, 16);
    dim3 prep_grid((width + prep_block.x - 1) / prep_block.x, (height + prep_block.y - 1) / prep_block.y);
    
    // Launch kernel to reorganize data for cuFFT
    reorganize_for_cufft_kernel<<<prep_grid, prep_block>>>(
        d_all_frames, d_input_data, 
        width, height, channels, num_frames, dft_size
    );
    checkCudaError("reorganize_for_cufft_kernel");
    
    // Execute forward FFT on all pixel time series simultaneously
    LOG_GAUSSIAN("Executing forward cuFFT on " + std::to_string(batch_size) + " pixel time series");
    cufftExecR2C(forward_plan, d_input_data, d_fft_data);
    checkCudaError("execute forward cuFFT");
    
    // Apply frequency mask (bandpass filter)
    LOG_GAUSSIAN("Applying frequency domain bandpass filter");
    dim3 mask_block(256);
    dim3 mask_grid((batch_size * (dft_size/2 + 1) + mask_block.x - 1) / mask_block.x);
    
    apply_frequency_mask_kernel<<<mask_grid, mask_block>>>(
        d_fft_data, batch_size, dft_size, fps, fl, fh
    );
    checkCudaError("apply_frequency_mask_kernel");
    
    // Execute inverse FFT
    LOG_GAUSSIAN("Executing inverse cuFFT");
    cufftExecC2R(inverse_plan, d_fft_data, d_output_data);
    checkCudaError("execute inverse cuFFT");
    
    // Reorganize output data back to frame format (pure bandpass filtering)
    LOG_GAUSSIAN("Reorganizing output from cuFFT format");
    reorganize_from_cufft_kernel<<<prep_grid, prep_block>>>(
        d_output_data, d_filtered_frames,
        width, height, channels, num_frames, dft_size
    );
    checkCudaError("reorganize_from_cufft_kernel");
    
    // Copy results back to host
    LOG_GAUSSIAN("Transferring filtered results back to host");
    std::vector<cv::Mat> filtered_batch(num_frames);
    for (int t = 0; t < num_frames; t++) {
        filtered_batch[t] = cv::Mat(height, width, CV_32FC3);
        size_t frame_offset = t * frame_data_size;
        cudaMemcpy(filtered_batch[t].ptr<float>(),
                  d_filtered_frames + frame_offset / sizeof(float),
                  frame_data_size,
                  cudaMemcpyDeviceToHost);
    }
    checkCudaError("copy filtered results to host");
    
    // Clean up cuFFT plans and memory
    cufftDestroy(forward_plan);
    cufftDestroy(inverse_plan);
    cudaFree(d_input_data);
    cudaFree(d_fft_data);
    cudaFree(d_output_data);
    cudaFree(d_all_frames);
    cudaFree(d_filtered_frames);
    
    LOG_GAUSSIAN("cuFFT BATCHED temporal filtering completed");
    return filtered_batch;
}

// cuFFT Helper Kernels

// Reorganize data from frame-major to pixel-major format for cuFFT batched operations
__global__ void reorganize_for_cufft_kernel(
    const float* __restrict__ d_frame_data,
    cufftReal* __restrict__ d_pixel_data,
    int width,
    int height,
    int channels,
    int num_frames,
    int dft_size
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Process all channels for this pixel
    for (int c = 0; c < channels; c++) {
        // Calculate pixel index in the batched layout
        int pixel_channel_idx = (y * width + x) * channels + c;
        int pixel_series_offset = pixel_channel_idx * dft_size;
        
        // Copy time series data for this pixel-channel
        for (int t = 0; t < num_frames; t++) {
            int frame_data_idx = t * (height * width * channels) + y * (width * channels) + x * channels + c;
            d_pixel_data[pixel_series_offset + t] = d_frame_data[frame_data_idx];
        }
        
        // Zero-pad the remaining elements
        for (int t = num_frames; t < dft_size; t++) {
            d_pixel_data[pixel_series_offset + t] = 0.0f;
        }
    }
}

// Apply frequency domain bandpass filter to cuFFT output
__global__ void apply_frequency_mask_kernel(
    cufftComplex* __restrict__ d_fft_data,
    int batch_size,
    int dft_size,
    float fps,
    float fl,
    float fh
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * (dft_size / 2 + 1);
    
    if (idx >= total_elements) return;
    
    // Extract batch index and frequency bin
    int freq_bin = idx % (dft_size / 2 + 1);
    
    // Calculate frequency for this bin
    // For R2C transforms, freq_bin ranges from 0 to dft_size/2
    // freq_bin=0 corresponds to DC (0 Hz)
    // freq_bin=dft_size/2 corresponds to Nyquist (fps/2 Hz)
    float freq = (float)freq_bin * fps / (float)dft_size;
    
    // Apply bandpass filter: zero out frequencies outside [fl, fh]
    if (freq < fl || freq > fh) {
        d_fft_data[idx].x = 0.0f;
        d_fft_data[idx].y = 0.0f;
    }
}

// Reorganize data from pixel-major back to frame-major format (pure bandpass filter)
__global__ void reorganize_from_cufft_kernel(
    const cufftReal* __restrict__ d_pixel_data,
    float* __restrict__ d_frame_data,
    int width,
    int height,
    int channels,
    int num_frames,
    int dft_size
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Process all channels for this pixel
    for (int c = 0; c < channels; c++) {
        // Calculate pixel index in the batched layout
        int pixel_channel_idx = (y * width + x) * channels + c;
        int pixel_series_offset = pixel_channel_idx * dft_size;
        
        // Copy filtered time series data back to frame format
        for (int t = 0; t < num_frames; t++) {
            int frame_data_idx = t * (height * width * channels) + y * (width * channels) + x * channels + c;
            // Apply only normalization (cuFFT doesn't normalize) - NO amplification here
            d_frame_data[frame_data_idx] = d_pixel_data[pixel_series_offset + t] / (float)dft_size;
        }
    }
}

// Frame reconstruction kernel
__global__ void reconstruct_frame_kernel(
    const float* __restrict__ d_original_yiq,
    const float* __restrict__ d_filtered_signal,
    float* __restrict__ d_combined_yiq,
    int width,
    int height,
    int channels
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = (y * width + x) * channels;
    
    // Add filtered signal to original YIQ
    for (int c = 0; c < channels; c++) {
        d_combined_yiq[idx + c] = d_original_yiq[idx + c] + d_filtered_signal[idx + c];
    }
}

// RGB clipping kernel
__global__ void clip_rgb_kernel(
    float* __restrict__ d_rgb,
    int width,
    int height,
    int channels
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = (y * width + x) * channels;
    
    // Clip RGB values to [0, 255] range
    for (int c = 0; c < channels; c++) {
        d_rgb[idx + c] = fmaxf(0.0f, fminf(255.0f, d_rgb[idx + c]));
    }
}

// CUDA Gaussian frame reconstruction implementation
cv::Mat reconstruct_gaussian_frame_gpu(
    const cv::Mat& original_rgb,
    const cv::Mat& filtered_yiq_signal
) {
    LOG_GAUSSIAN("Starting frame reconstruction");
    
    if (original_rgb.empty() || filtered_yiq_signal.empty() ||
        original_rgb.size() != filtered_yiq_signal.size()) {
        throw std::runtime_error("Invalid input for frame reconstruction");
    }
    
    // Convert original RGB to YIQ
    cv::Mat original_yiq = rgb_to_yiq_opencv_wrapper(original_rgb);
    
    int width = original_yiq.cols;
    int height = original_yiq.rows;
    int channels = 3;
    size_t data_size = width * height * channels * sizeof(float);
    
    // Allocate device memory
    float* d_original_yiq = nullptr;
    float* d_filtered_signal = nullptr;
    float* d_combined_yiq = nullptr;
    float* d_rgb_result = nullptr;
    
    cudaMalloc(&d_original_yiq, data_size);
    cudaMalloc(&d_filtered_signal, data_size);
    cudaMalloc(&d_combined_yiq, data_size);
    cudaMalloc(&d_rgb_result, data_size);
    checkCudaError("malloc reconstruction buffers");
    
    // Copy data to device
    cudaMemcpy(d_original_yiq, original_yiq.ptr<float>(), data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filtered_signal, filtered_yiq_signal.ptr<float>(), data_size, cudaMemcpyHostToDevice);
    checkCudaError("copy reconstruction data");
    
    // Combine original and filtered signals
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    reconstruct_frame_kernel<<<grid, block>>>(
        d_original_yiq, d_filtered_signal, d_combined_yiq, width, height, channels
    );
    checkCudaError("reconstruct_frame_kernel");
    
    // Convert combined YIQ back to RGB using existing function
    cv::Mat combined_yiq_host(height, width, CV_32FC3);
    cudaMemcpy(combined_yiq_host.ptr<float>(), d_combined_yiq, data_size, cudaMemcpyDeviceToHost);
    checkCudaError("copy combined YIQ");
    
    cv::Mat rgb_result_float = yiq_to_rgb_opencv_wrapper(combined_yiq_host);
    
    // Copy back to device for clipping
    cudaMemcpy(d_rgb_result, rgb_result_float.ptr<float>(), data_size, cudaMemcpyHostToDevice);
    checkCudaError("copy RGB for clipping");
    
    // Clip RGB values
    clip_rgb_kernel<<<grid, block>>>(d_rgb_result, width, height, channels);
    checkCudaError("clip_rgb_kernel");
    
    // Copy final result back
    cv::Mat rgb_clipped(height, width, CV_32FC3);
    cudaMemcpy(rgb_clipped.ptr<float>(), d_rgb_result, data_size, cudaMemcpyDeviceToHost);
    checkCudaError("copy final result");
    
    // Convert to uint8
    cv::Mat result_uint8;
    rgb_clipped.convertTo(result_uint8, CV_8UC3);
    
    // Clean up
    cudaFree(d_original_yiq);
    cudaFree(d_filtered_signal);
    cudaFree(d_combined_yiq);
    cudaFree(d_rgb_result);
    
    LOG_GAUSSIAN("Frame reconstruction completed");
    return result_uint8;
}

// Complete Gaussian mode processing pipeline
void process_video_gaussian(
    const std::string& input_filename,
    const std::string& output_filename,
    int levels,
    double alpha,
    double fl,
    double fh,
    double chrom_attenuation
) {
    LOG_GAUSSIAN("Starting Gaussian video processing for: " + input_filename);
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize Gaussian processing module
    if (!init_gaussian_processing()) {
        throw std::runtime_error("Failed to initialize Gaussian processing module");
    }
    
    // Open input video
    cv::VideoCapture cap(input_filename);
    if (!cap.isOpened()) {
        throw std::runtime_error("Error opening video file: " + input_filename);
    }
    
    // Get video properties
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::Size original_frame_size(frame_width, frame_height);
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    if (fps <= 0) {
        LOG_GAUSSIAN("Warning: Could not read FPS property. Using default value of 30.");
        fps = 30.0;
    }
    
    LOG_GAUSSIAN("Video properties: " + std::to_string(frame_width) + "x" + std::to_string(frame_height) + 
                ", FPS: " + std::to_string(fps));
    
    // Read all frames and perform spatial filtering
    std::vector<cv::Mat> original_rgb_frames;
    std::vector<cv::Mat> spatially_filtered_batch;
    cv::Mat frame_rgb_uint8;
    
    LOG_GAUSSIAN("Reading frames and performing spatial filtering...");
    int frame_num = 0;
    while (true) {
        cap >> frame_rgb_uint8;
        if (frame_rgb_uint8.empty()) {
            break; // End of video
        }
        
        // Store original RGB frame
        original_rgb_frames.push_back(frame_rgb_uint8.clone());
        
        // Apply spatial filtering
        cv::Mat spatially_filtered = spatially_filter_gaussian_gpu(frame_rgb_uint8, levels);
        spatially_filtered_batch.push_back(spatially_filtered);
        
        frame_num++;
        if (frame_num % 100 == 0) {
            LOG_GAUSSIAN("Read and spatially filtered " + std::to_string(frame_num) + " frames...");
        }
    }
    cap.release();
    LOG_GAUSSIAN("Finished reading and spatially filtering " + std::to_string(frame_num) + " frames.");
    
    if (original_rgb_frames.empty()) {
        LOG_GAUSSIAN("No frames read from video. Exiting.");
        return;
    }
    
    // Apply temporal filtering
    LOG_GAUSSIAN("Applying FFT-based temporal filtering...");
    std::vector<cv::Mat> filtered_amplified_batch = temporal_filter_gaussian_batch_gpu(
        spatially_filtered_batch,
        static_cast<float>(fps),
        static_cast<float>(fl),
        static_cast<float>(fh),
        static_cast<float>(alpha),
        static_cast<float>(chrom_attenuation)
    );
    LOG_GAUSSIAN("Temporal filtering completed.");
    
    // Reconstruct final video
    LOG_GAUSSIAN("Reconstructing final video...");
    std::vector<cv::Mat> output_video;
    output_video.reserve(original_rgb_frames.size());
    
    for (size_t i = 0; i < original_rgb_frames.size(); i++) {
        if (i < filtered_amplified_batch.size()) {
            cv::Mat reconstructed_frame = reconstruct_gaussian_frame_gpu(
                original_rgb_frames[i],
                filtered_amplified_batch[i]
            );
            output_video.push_back(reconstructed_frame);
        } else {
            LOG_GAUSSIAN("Warning: Missing filtered data for frame " + std::to_string(i) + 
                        ". Using original frame.");
            output_video.push_back(original_rgb_frames[i].clone());
        }
        
        if ((i + 1) % 100 == 0) {
            LOG_GAUSSIAN("Reconstructed " + std::to_string(i + 1) + " frames...");
        }
    }
    LOG_GAUSSIAN("Video reconstruction complete.");
    
    // Write output video
    cv::VideoWriter writer(output_filename,
                         cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                         fps,
                         original_frame_size);
    
    if (!writer.isOpened()) {
        throw std::runtime_error("Error opening video writer for: " + output_filename);
    }
    
    LOG_GAUSSIAN("Writing output video to: " + output_filename);
    for (size_t i = 0; i < output_video.size(); i++) {
        if (!output_video[i].empty()) {
            writer.write(output_video[i]);
        } else {
            LOG_GAUSSIAN("Warning: Skipping empty frame during writing at index " + std::to_string(i));
        }
        
        if ((i + 1) % 100 == 0) {
            LOG_GAUSSIAN("Wrote " + std::to_string(i + 1) + " frames...");
        }
    }
    writer.release();
    LOG_GAUSSIAN("Finished writing " + std::to_string(output_video.size()) + " frames.");
    
    // Clean up
    cleanup_gaussian_processing();
    
    // Calculate and report total processing time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    LOG_GAUSSIAN("Total processing time: " + std::to_string(duration.count()) + " ms");
    LOG_GAUSSIAN("CUDA Gaussian video processing complete for: " + output_filename);
}

} // namespace evmcuda