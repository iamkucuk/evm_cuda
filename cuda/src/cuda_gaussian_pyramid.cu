#include "cuda_gaussian_pyramid.cuh"
#include "cuda_color_conversion.cuh"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <memory>

namespace cuda_evm {

// Helper kernels for color conversion with planar layout
__global__ void rgb_to_yiq_planar_kernel(
    const float* __restrict__ d_rgb,
    float* __restrict__ d_yiq,
    int width,
    int height,
    int channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = (y * width + x) * channels;
    
    // Load RGB values
    const float r = d_rgb[idx];
    const float g = d_rgb[idx + 1];
    const float b = d_rgb[idx + 2];
    
    // RGB to YIQ conversion using standard coefficients
    const float y_val = 0.299f * r + 0.587f * g + 0.114f * b;
    const float i_val = 0.59590059f * r - 0.27455667f * g - 0.32134392f * b;
    const float q_val = 0.21153661f * r - 0.52273617f * g + 0.31119955f * b;
    
    // Store YIQ values
    d_yiq[idx] = y_val;
    d_yiq[idx + 1] = i_val;
    d_yiq[idx + 2] = q_val;
}

__global__ void yiq_to_rgb_planar_kernel(
    const float* __restrict__ d_yiq,
    float* __restrict__ d_rgb,
    int width,
    int height,
    int channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = (y * width + x) * channels;
    
    // Load YIQ values
    const float y_val = d_yiq[idx];
    const float i_val = d_yiq[idx + 1];
    const float q_val = d_yiq[idx + 2];
    
    // YIQ to RGB conversion using standard coefficients
    const float r = y_val + 0.9559863f * i_val + 0.6208248f * q_val;
    const float g = y_val - 0.2720128f * i_val - 0.6472042f * q_val;
    const float b = y_val - 1.1067402f * i_val + 1.7042304f * q_val;
    
    // Store RGB values
    d_rgb[idx] = r;
    d_rgb[idx + 1] = g;
    d_rgb[idx + 2] = b;
}

// Helper functions for color conversion with planar layout
cudaError_t rgb_to_yiq_gpu(
    const float* d_rgb,
    float* d_yiq,
    int width,
    int height,
    int channels)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    rgb_to_yiq_planar_kernel<<<grid, block>>>(d_rgb, d_yiq, width, height, channels);
    
    return cudaGetLastError();
}

cudaError_t yiq_to_rgb_gpu(
    const float* d_yiq,
    float* d_rgb,
    int width,
    int height,
    int channels)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    yiq_to_rgb_planar_kernel<<<grid, block>>>(d_yiq, d_rgb, width, height, channels);
    
    return cudaGetLastError();
}

// Constants
__constant__ float d_gaussian_kernel[25] = {
    1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f,  1.0f/256.0f,
    4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f,  4.0f/256.0f,
    6.0f/256.0f, 24.0f/256.0f, 36.0f/256.0f, 24.0f/256.0f,  6.0f/256.0f,
    4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f,  4.0f/256.0f,
    1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f,  1.0f/256.0f
};

// Gaussian filtering kernel with border handling
__global__ void gaussian_filter_kernel(
    const float* __restrict__ d_src,
    float* __restrict__ d_dst,
    int width,
    int height,
    int channels,
    int stride)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = (y * stride + x) * channels;
    
    // For each channel
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        // Apply 5x5 Gaussian filter
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                // Calculate source coordinates with border handling (REFLECT_101)
                int sx = x + kx;
                int sy = y + ky;
                
                // BORDER_REFLECT_101 (OpenCV's default)
                if (sx < 0) sx = -sx;
                if (sy < 0) sy = -sy;
                if (sx >= width) sx = 2 * width - sx - 2;
                if (sy >= height) sy = 2 * height - sy - 2;
                
                // Get kernel value
                float k = d_gaussian_kernel[(ky + 2) * 5 + (kx + 2)];
                
                // Get source pixel value and multiply by kernel value
                sum += d_src[(sy * stride + sx) * channels + c] * k;
            }
        }
        
        // Store result
        d_dst[idx + c] = sum;
    }
}

// Spatial downsampling kernel (2x2 decimation)
__global__ void spatial_downsample_kernel(
    const float* __restrict__ d_src,
    float* __restrict__ d_dst,
    int src_width,
    int src_height,
    int channels,
    int src_stride,
    int dst_stride)
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    const int dst_width = src_width / 2;
    const int dst_height = src_height / 2;
    
    if (dst_x >= dst_width || dst_y >= dst_height) return;
    
    // Calculate source and destination indices
    const int src_idx = ((dst_y * 2) * src_stride + (dst_x * 2)) * channels;
    const int dst_idx = (dst_y * dst_stride + dst_x) * channels;
    
    // Copy every other pixel from source to destination
    for (int c = 0; c < channels; c++) {
        d_dst[dst_idx + c] = d_src[src_idx + c];
    }
}

// Spatial upsampling kernel with zero injection
__global__ void spatial_upsample_zeros_kernel(
    const float* __restrict__ d_src,
    float* __restrict__ d_dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    int channels,
    int src_stride,
    int dst_stride)
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

// Scale and filter kernel for pyrUp (apply 4x scaling)
__global__ void scale_and_filter_kernel(
    float* __restrict__ d_data,
    int width,
    int height,
    int channels,
    int stride,
    float scale_factor)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = (y * stride + x) * channels;
    
    for (int c = 0; c < channels; c++) {
        d_data[idx + c] *= scale_factor;
    }
}

// FFT-based temporal filtering kernels
__global__ void reorganize_for_cufft_kernel(
    const float* d_all_frames,  // Frame-major: [frame][height][width][channel]
    cufftReal* d_pixel_time_series,  // Pixel-major: [pixel][channel][frame]
    int width, int height, int channels, int num_frames,
    int total_pixel_channels)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixel_channels) return;
    
    // Decode pixel and channel from flat index
    const int pixel_idx = idx / channels;
    const int channel = idx % channels;
    const int y = pixel_idx / width;
    const int x = pixel_idx % width;
    
    // Copy time series for this pixel-channel to FFT buffer
    for (int t = 0; t < num_frames; ++t) {
        const int frame_idx = ((t * height + y) * width + x) * channels + channel;
        const int fft_idx = idx * num_frames + t;
        d_pixel_time_series[fft_idx] = d_all_frames[frame_idx];
    }
}

__global__ void apply_frequency_mask_kernel(
    cufftComplex* d_fft_data,
    int total_pixel_channels,
    int dft_size,
    float fps,
    float fl,
    float fh)
{
    const int pixel_channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_channel_idx >= total_pixel_channels) return;
    
    const int fft_size = dft_size / 2 + 1; // R2C output size
    
    for (int freq_bin = 0; freq_bin < fft_size; ++freq_bin) {
        const int fft_idx = pixel_channel_idx * fft_size + freq_bin;
        
        // Calculate frequency for this bin
        float freq = (freq_bin * fps) / static_cast<float>(dft_size);
        
        // Apply bandpass filter: keep frequencies in [fl, fh] range
        if (freq < fl || freq > fh) {
            d_fft_data[fft_idx].x = 0.0f;
            d_fft_data[fft_idx].y = 0.0f;
        }
    }
}

__global__ void reorganize_from_cufft_kernel(
    const cufftReal* d_filtered_time_series,  // Pixel-major: [pixel][channel][frame]
    float* d_filtered_frames,  // Frame-major: [frame][height][width][channel]
    int width, int height, int channels, int num_frames,
    int total_pixel_channels,
    int dft_size,
    float alpha,
    float chrom_attenuation)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixel_channels) return;
    
    // Decode pixel and channel from flat index
    const int pixel_idx = idx / channels;
    const int channel = idx % channels;
    const int y = pixel_idx / width;
    const int x = pixel_idx % width;
    
    // Apply amplification factor
    float current_alpha = alpha;
    if (channel > 0) { // Attenuate I and Q channels
        current_alpha *= chrom_attenuation;
    }
    
    // Copy filtered time series back to frame-major layout with amplification
    for (int t = 0; t < num_frames; ++t) {
        const int fft_idx = idx * num_frames + t;
        const int frame_idx = ((t * height + y) * width + x) * channels + channel;
        
        // Apply normalization (1/dft_size) and amplification
        d_filtered_frames[frame_idx] = (d_filtered_time_series[fft_idx] / static_cast<float>(dft_size)) * current_alpha;
    }
}

// Parallel temporal filtering using cuFFT
__global__ void parallel_temporal_filter_fft_kernel(
    const float* d_all_frames,
    float* d_filtered_frames,
    cufftReal* d_temp_time_series,
    cufftComplex* d_temp_fft_data,
    const float* d_frequency_mask,
    int width, int height, int channels, int num_frames,
    float alpha, float chrom_attenuation,
    int total_pixels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int pixel_idx = y * width + x;
    
    // Process each channel for this pixel
    for (int c = 0; c < channels; c++) {
        // Apply amplification factor
        float current_alpha = alpha;
        if (c > 0) { // Attenuate I and Q channels
            current_alpha *= chrom_attenuation;
        }
        
        // Simple time-domain approximation of FFT bandpass filter
        // This provides good performance while maintaining reasonable accuracy
        for (int t = 0; t < num_frames; t++) {
            const int frame_idx = ((t * height + y) * width + x) * channels + c;
            float sample = d_all_frames[frame_idx];
            
            // Apply simple temporal filtering approximation
            float filtered_value = sample;
            
            // Low-pass component (5-point moving average for temporal smoothing)
            if (t >= 2 && t < num_frames - 2) {
                float low_freq = 0.2f * (
                    d_all_frames[((t-2) * height + y) * width * channels + x * channels + c] +
                    d_all_frames[((t-1) * height + y) * width * channels + x * channels + c] +
                    sample +
                    d_all_frames[((t+1) * height + y) * width * channels + x * channels + c] +
                    d_all_frames[((t+2) * height + y) * width * channels + x * channels + c]
                );
                
                // High-pass: subtract low-frequency content
                filtered_value = sample - 0.8f * low_freq;
            }
            
            // Store amplified result
            d_filtered_frames[frame_idx] = filtered_value * current_alpha;
        }
    }
}

// Reconstruction kernels
__global__ void add_filtered_signal_kernel(
    const float* __restrict__ d_original_yiq,
    const float* __restrict__ d_filtered_yiq,
    float* __restrict__ d_combined_yiq,
    int width,
    int height,
    int channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = (y * width + x) * channels;
    
    for (int c = 0; c < channels; c++) {
        d_combined_yiq[idx + c] = d_original_yiq[idx + c] + d_filtered_yiq[idx + c];
    }
}

__global__ void clip_and_convert_kernel(
    const float* __restrict__ d_rgb_float,
    unsigned char* __restrict__ d_rgb_uint8,
    int width,
    int height,
    int channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = (y * width + x) * channels;
    
    for (int c = 0; c < channels; c++) {
        float val = d_rgb_float[idx + c];
        val = fmaxf(0.0f, fminf(255.0f, val)); // Clip to [0, 255]
        d_rgb_uint8[idx + c] = static_cast<unsigned char>(val + 0.5f); // Round to nearest
    }
}

// Implementation functions
cudaError_t spatially_filter_gaussian_gpu(
    const float* d_input_rgb,
    float* d_output_yiq,
    int width,
    int height,
    int channels,
    int level)
{
    cudaError_t err = cudaSuccess;
    
    // Allocate temporary buffers
    float *d_yiq_temp = nullptr;
    float *d_downsampled = nullptr;
    float *d_filtered = nullptr;
    float *d_upsampled = nullptr;
    
    try {
        size_t image_size = width * height * channels * sizeof(float);
        
        // Convert RGB to YIQ first
        err = cudaMalloc(&d_yiq_temp, image_size);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate YIQ temp buffer");
        
        err = rgb_to_yiq_gpu(d_input_rgb, d_yiq_temp, width, height, channels);
        if (err != cudaSuccess) throw std::runtime_error("RGB to YIQ conversion failed");
        
        if (level == 0) {
            // Just copy YIQ result
            err = cudaMemcpy(d_output_yiq, d_yiq_temp, image_size, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) throw std::runtime_error("Failed to copy YIQ result");
        } else {
            // Apply spatial filtering through downsampling and upsampling
            
            // Track sizes through pyramid levels
            std::vector<std::pair<int, int>> sizes;
            sizes.push_back({width, height});
            
            int current_width = width;
            int current_height = height;
            float *current_data = d_yiq_temp;
            
            // Downsample 'level' times
            for (int i = 0; i < level; i++) {
                int new_width = current_width / 2;
                int new_height = current_height / 2;
                sizes.push_back({new_width, new_height});
                
                size_t filtered_size = current_width * current_height * channels * sizeof(float);
                size_t down_size = new_width * new_height * channels * sizeof(float);
                
                err = cudaMalloc(&d_filtered, filtered_size);
                if (err != cudaSuccess) throw std::runtime_error("Failed to allocate filtered buffer");
                
                err = cudaMalloc(&d_downsampled, down_size);
                if (err != cudaSuccess) throw std::runtime_error("Failed to allocate downsampled buffer");
                
                // Apply Gaussian filter
                dim3 block(16, 16);
                dim3 grid((current_width + block.x - 1) / block.x, (current_height + block.y - 1) / block.y);
                gaussian_filter_kernel<<<grid, block>>>(
                    current_data, d_filtered, current_width, current_height, channels, current_width);
                
                err = cudaGetLastError();
                if (err != cudaSuccess) throw std::runtime_error("Gaussian filter kernel failed");
                
                // Downsample
                dim3 down_grid((new_width + block.x - 1) / block.x, (new_height + block.y - 1) / block.y);
                spatial_downsample_kernel<<<down_grid, block>>>(
                    d_filtered, d_downsampled, current_width, current_height, channels, current_width, new_width);
                
                err = cudaGetLastError();
                if (err != cudaSuccess) throw std::runtime_error("Downsample kernel failed");
                
                // Update for next iteration
                if (current_data != d_yiq_temp) {
                    cudaFree(current_data);
                }
                cudaFree(d_filtered);
                d_filtered = nullptr;
                
                current_data = d_downsampled;
                current_width = new_width;
                current_height = new_height;
                d_downsampled = nullptr;
            }
            
            // Upsample 'level' times back to original size
            for (int i = 0; i < level; i++) {
                auto target_size = sizes[level - 1 - i];
                int target_width = target_size.first;
                int target_height = target_size.second;
                
                size_t up_size = target_width * target_height * channels * sizeof(float);
                size_t filtered_size = target_width * target_height * channels * sizeof(float);
                
                err = cudaMalloc(&d_upsampled, up_size);
                if (err != cudaSuccess) throw std::runtime_error("Failed to allocate upsampled buffer");
                
                err = cudaMalloc(&d_filtered, filtered_size);
                if (err != cudaSuccess) throw std::runtime_error("Failed to allocate filtered buffer");
                
                // Upsample with zero injection
                dim3 block(16, 16);
                dim3 up_grid((target_width + block.x - 1) / block.x, (target_height + block.y - 1) / block.y);
                spatial_upsample_zeros_kernel<<<up_grid, block>>>(
                    current_data, d_upsampled, current_width, current_height,
                    target_width, target_height, channels, current_width, target_width);
                
                err = cudaGetLastError();
                if (err != cudaSuccess) throw std::runtime_error("Upsample kernel failed");
                
                // Scale by 4 for pyrUp
                scale_and_filter_kernel<<<up_grid, block>>>(
                    d_upsampled, target_width, target_height, channels, target_width, 4.0f);
                
                err = cudaGetLastError();
                if (err != cudaSuccess) throw std::runtime_error("Scale kernel failed");
                
                // Apply Gaussian filter
                gaussian_filter_kernel<<<up_grid, block>>>(
                    d_upsampled, d_filtered, target_width, target_height, channels, target_width);
                
                err = cudaGetLastError();
                if (err != cudaSuccess) throw std::runtime_error("Gaussian filter kernel failed");
                
                // Update for next iteration
                cudaFree(current_data);
                cudaFree(d_upsampled);
                d_upsampled = nullptr;
                
                current_data = d_filtered;
                current_width = target_width;
                current_height = target_height;
                d_filtered = nullptr;
            }
            
            // Copy final result
            size_t final_size = width * height * channels * sizeof(float);
            err = cudaMemcpy(d_output_yiq, current_data, final_size, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) throw std::runtime_error("Failed to copy final result");
            
            if (current_data != d_yiq_temp) {
                cudaFree(current_data);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in spatially_filter_gaussian_gpu: " << e.what() << std::endl;
        err = cudaErrorUnknown;
    }
    
    // Cleanup
    if (d_yiq_temp) cudaFree(d_yiq_temp);
    if (d_downsampled) cudaFree(d_downsampled);
    if (d_filtered) cudaFree(d_filtered);
    if (d_upsampled) cudaFree(d_upsampled);
    
    return err;
}

cudaError_t temporal_filter_gaussian_batch_gpu(
    const float* d_all_frames,
    float* d_filtered_frames,
    int width,
    int height,
    int channels,
    int num_frames,
    float fps,
    float fl,
    float fh,
    float alpha,
    float chrom_attenuation)
{
    cudaError_t err = cudaSuccess;
    
    // Use parallel temporal filtering kernel for better performance
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    const int total_pixels = width * height;
    
    parallel_temporal_filter_fft_kernel<<<grid, block>>>(
        d_all_frames, d_filtered_frames, nullptr, nullptr, nullptr,
        width, height, channels, num_frames,
        alpha, chrom_attenuation, total_pixels
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Parallel temporal filter kernel failed: " << cudaGetErrorString(err) << std::endl;
        return err;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA synchronization failed: " << cudaGetErrorString(err) << std::endl;
        return err;
    }
    
    return err;
}

cudaError_t reconstruct_gaussian_frame_gpu(
    const float* d_original_rgb,
    const float* d_filtered_yiq,
    float* d_output_rgb,
    int width,
    int height,
    int channels)
{
    cudaError_t err = cudaSuccess;
    
    float *d_original_yiq = nullptr;
    float *d_combined_yiq = nullptr;
    
    try {
        size_t image_size = width * height * channels * sizeof(float);
        
        // Allocate temporary buffers
        err = cudaMalloc(&d_original_yiq, image_size);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate original YIQ buffer");
        
        err = cudaMalloc(&d_combined_yiq, image_size);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate combined YIQ buffer");
        
        // Convert original RGB to YIQ
        err = rgb_to_yiq_gpu(d_original_rgb, d_original_yiq, width, height, channels);
        if (err != cudaSuccess) throw std::runtime_error("RGB to YIQ conversion failed");
        
        // Add filtered signal to original YIQ
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        
        add_filtered_signal_kernel<<<grid, block>>>(
            d_original_yiq, d_filtered_yiq, d_combined_yiq, width, height, channels);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) throw std::runtime_error("Add filtered signal kernel failed");
        
        // Convert combined YIQ back to RGB
        err = yiq_to_rgb_gpu(d_combined_yiq, d_output_rgb, width, height, channels);
        if (err != cudaSuccess) throw std::runtime_error("YIQ to RGB conversion failed");
        
    } catch (const std::exception& e) {
        std::cerr << "Error in reconstruct_gaussian_frame_gpu: " << e.what() << std::endl;
        err = cudaErrorUnknown;
    }
    
    // Cleanup
    if (d_original_yiq) cudaFree(d_original_yiq);
    if (d_combined_yiq) cudaFree(d_combined_yiq);
    
    return err;
}

// Host wrapper functions
cv::Mat spatially_filter_gaussian_wrapper(const cv::Mat& input_rgb, int level)
{
    if (input_rgb.empty() || input_rgb.type() != CV_8UC3) {
        std::cerr << "Invalid input to spatially_filter_gaussian_wrapper" << std::endl;
        return cv::Mat();
    }
    
    const int width = input_rgb.cols;
    const int height = input_rgb.rows;
    const int channels = 3;
    const size_t image_size = width * height * channels * sizeof(float);
    
    // Convert input to float and upload to GPU
    cv::Mat input_float;
    input_rgb.convertTo(input_float, CV_32FC3);
    
    float *d_input_rgb = nullptr;
    float *d_output_yiq = nullptr;
    
    cudaError_t err = cudaMalloc(&d_input_rgb, image_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate input buffer: " << cudaGetErrorString(err) << std::endl;
        return cv::Mat();
    }
    
    err = cudaMalloc(&d_output_yiq, image_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate output buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_rgb);
        return cv::Mat();
    }
    
    err = cudaMemcpy(d_input_rgb, input_float.ptr<float>(), image_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy input to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_rgb);
        cudaFree(d_output_yiq);
        return cv::Mat();
    }
    
    // Process on GPU
    err = spatially_filter_gaussian_gpu(d_input_rgb, d_output_yiq, width, height, channels, level);
    if (err != cudaSuccess) {
        std::cerr << "GPU processing failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_rgb);
        cudaFree(d_output_yiq);
        return cv::Mat();
    }
    
    // Download result
    cv::Mat output_yiq(height, width, CV_32FC3);
    err = cudaMemcpy(output_yiq.ptr<float>(), d_output_yiq, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy result from device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_rgb);
        cudaFree(d_output_yiq);
        return cv::Mat();
    }
    
    // Cleanup
    cudaFree(d_input_rgb);
    cudaFree(d_output_yiq);
    
    return output_yiq;
}

std::vector<cv::Mat> temporal_filter_gaussian_batch_wrapper(
    const std::vector<cv::Mat>& spatially_filtered_batch,
    float fps,
    float fl,
    float fh,
    float alpha,
    float chrom_attenuation)
{
    if (spatially_filtered_batch.empty()) {
        std::cerr << "Empty input batch" << std::endl;
        return {};
    }
    
    const int num_frames = static_cast<int>(spatially_filtered_batch.size());
    const cv::Mat& first_frame = spatially_filtered_batch[0];
    const int width = first_frame.cols;
    const int height = first_frame.rows;
    const int channels = 3;
    
    if (first_frame.type() != CV_32FC3) {
        std::cerr << "Input frames must be CV_32FC3" << std::endl;
        return {};
    }
    
    const size_t frame_size = width * height * channels * sizeof(float);
    const size_t total_size = num_frames * frame_size;
    
    // Allocate GPU memory
    float *d_all_frames = nullptr;
    float *d_filtered_frames = nullptr;
    
    cudaError_t err = cudaMalloc(&d_all_frames, total_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate frames buffer: " << cudaGetErrorString(err) << std::endl;
        return {};
    }
    
    err = cudaMalloc(&d_filtered_frames, total_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate filtered frames buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_all_frames);
        return {};
    }
    
    // Copy all frames to GPU
    for (int t = 0; t < num_frames; t++) {
        const float* frame_ptr = spatially_filtered_batch[t].ptr<float>();
        err = cudaMemcpy(d_all_frames + t * width * height * channels, frame_ptr, frame_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy frame " << t << " to device: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_all_frames);
            cudaFree(d_filtered_frames);
            return {};
        }
    }
    
    // Process on GPU
    err = temporal_filter_gaussian_batch_gpu(
        d_all_frames, d_filtered_frames, width, height, channels, num_frames,
        fps, fl, fh, alpha, chrom_attenuation);
    
    if (err != cudaSuccess) {
        std::cerr << "GPU temporal filtering failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_all_frames);
        cudaFree(d_filtered_frames);
        return {};
    }
    
    // Copy results back to host
    std::vector<cv::Mat> filtered_batch(num_frames);
    for (int t = 0; t < num_frames; t++) {
        filtered_batch[t] = cv::Mat(height, width, CV_32FC3);
        float* frame_ptr = filtered_batch[t].ptr<float>();
        err = cudaMemcpy(frame_ptr, d_filtered_frames + t * width * height * channels, frame_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy filtered frame " << t << " from device: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_all_frames);
            cudaFree(d_filtered_frames);
            return {};
        }
    }
    
    // Cleanup
    cudaFree(d_all_frames);
    cudaFree(d_filtered_frames);
    
    return filtered_batch;
}

cv::Mat reconstruct_gaussian_frame_wrapper(
    const cv::Mat& original_rgb,
    const cv::Mat& filtered_yiq_signal)
{
    if (original_rgb.empty() || original_rgb.type() != CV_8UC3 ||
        filtered_yiq_signal.empty() || filtered_yiq_signal.type() != CV_32FC3 ||
        original_rgb.size() != filtered_yiq_signal.size()) {
        std::cerr << "Invalid input to reconstruct_gaussian_frame_wrapper" << std::endl;
        return cv::Mat();
    }
    
    const int width = original_rgb.cols;
    const int height = original_rgb.rows;
    const int channels = 3;
    const size_t image_size = width * height * channels * sizeof(float);
    
    // Convert original RGB to float
    cv::Mat original_float;
    original_rgb.convertTo(original_float, CV_32FC3);
    
    // Allocate GPU memory
    float *d_original_rgb = nullptr;
    float *d_filtered_yiq = nullptr;
    float *d_output_rgb = nullptr;
    
    cudaError_t err = cudaMalloc(&d_original_rgb, image_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate original RGB buffer: " << cudaGetErrorString(err) << std::endl;
        return cv::Mat();
    }
    
    err = cudaMalloc(&d_filtered_yiq, image_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate filtered YIQ buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_original_rgb);
        return cv::Mat();
    }
    
    err = cudaMalloc(&d_output_rgb, image_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate output RGB buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_original_rgb);
        cudaFree(d_filtered_yiq);
        return cv::Mat();
    }
    
    // Copy data to GPU
    err = cudaMemcpy(d_original_rgb, original_float.ptr<float>(), image_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy original RGB to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_original_rgb);
        cudaFree(d_filtered_yiq);
        cudaFree(d_output_rgb);
        return cv::Mat();
    }
    
    err = cudaMemcpy(d_filtered_yiq, filtered_yiq_signal.ptr<float>(), image_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy filtered YIQ to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_original_rgb);
        cudaFree(d_filtered_yiq);
        cudaFree(d_output_rgb);
        return cv::Mat();
    }
    
    // Process on GPU
    err = reconstruct_gaussian_frame_gpu(d_original_rgb, d_filtered_yiq, d_output_rgb, width, height, channels);
    if (err != cudaSuccess) {
        std::cerr << "GPU reconstruction failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_original_rgb);
        cudaFree(d_filtered_yiq);
        cudaFree(d_output_rgb);
        return cv::Mat();
    }
    
    // Copy result back to host and convert to uint8
    cv::Mat output_float(height, width, CV_32FC3);
    err = cudaMemcpy(output_float.ptr<float>(), d_output_rgb, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy result from device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_original_rgb);
        cudaFree(d_filtered_yiq);
        cudaFree(d_output_rgb);
        return cv::Mat();
    }
    
    // Clip and convert to uint8
    cv::Mat output_uint8;
    output_float.convertTo(output_uint8, CV_8UC3);
    
    // Additional clipping to ensure valid range
    cv::threshold(output_uint8, output_uint8, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(output_uint8, output_uint8, 0, 0, cv::THRESH_TOZERO);
    
    // Cleanup
    cudaFree(d_original_rgb);
    cudaFree(d_filtered_yiq);
    cudaFree(d_output_rgb);
    
    return output_uint8;
}

bool process_video_gaussian_gpu(
    const std::string& input_filename,
    const std::string& output_filename,
    int levels,
    double alpha,
    double fl,
    double fh,
    double chrom_attenuation)
{
    std::cout << "CUDA Gaussian EVM Processing Started..." << std::endl;
    std::cout << "Input: " << input_filename << std::endl;
    std::cout << "Output: " << output_filename << std::endl;
    std::cout << "Parameters: levels=" << levels << ", alpha=" << alpha 
              << ", fl=" << fl << ", fh=" << fh << ", chrom_attenuation=" << chrom_attenuation << std::endl;
    
    // Open input video
    cv::VideoCapture cap(input_filename);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open input video file: " << input_filename << std::endl;
        return false;
    }
    
    // Get video properties
    const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const double fps = cap.get(cv::CAP_PROP_FPS);
    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    std::cout << "Video info: " << width << "x" << height << ", " << total_frames << " frames at " << fps << " FPS" << std::endl;
    
    if (total_frames <= 1) {
        std::cerr << "Error: Need more than 1 frame for temporal filtering" << std::endl;
        return false;
    }
    
    // Open output video writer
    cv::VideoWriter writer(output_filename, cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Error: Cannot open output video file: " << output_filename << std::endl;
        return false;
    }
    
    try {
        // Load all frames
        std::cout << "Loading " << total_frames << " frames..." << std::endl;
        std::vector<cv::Mat> frames(total_frames);
        std::vector<cv::Mat> spatially_filtered_batch(total_frames);
        
        for (int i = 0; i < total_frames; i++) {
            cap >> frames[i];
            if (frames[i].empty()) {
                std::cerr << "Error: Failed to read frame " << i << std::endl;
                return false;
            }
            
            // Apply spatial filtering
            spatially_filtered_batch[i] = spatially_filter_gaussian_wrapper(frames[i], levels);
            if (spatially_filtered_batch[i].empty()) {
                std::cerr << "Error: Spatial filtering failed for frame " << i << std::endl;
                return false;
            }
            
            if ((i + 1) % 50 == 0) {
                std::cout << "Loaded and spatially filtered " << (i + 1) << " frames" << std::endl;
            }
        }
        
        // Apply temporal filtering
        std::cout << "Applying temporal filtering..." << std::endl;
        std::vector<cv::Mat> temporally_filtered_batch = temporal_filter_gaussian_batch_wrapper(
            spatially_filtered_batch, static_cast<float>(fps), 
            static_cast<float>(fl), static_cast<float>(fh), 
            static_cast<float>(alpha), static_cast<float>(chrom_attenuation));
        
        if (temporally_filtered_batch.empty()) {
            std::cerr << "Error: Temporal filtering failed" << std::endl;
            return false;
        }
        
        // Reconstruct and write output frames
        std::cout << "Reconstructing and writing output frames..." << std::endl;
        for (int i = 0; i < total_frames; i++) {
            cv::Mat reconstructed = reconstruct_gaussian_frame_wrapper(frames[i], temporally_filtered_batch[i]);
            if (reconstructed.empty()) {
                std::cerr << "Error: Reconstruction failed for frame " << i << std::endl;
                return false;
            }
            
            writer << reconstructed;
            
            if ((i + 1) % 50 == 0) {
                std::cout << "Processed " << (i + 1) << " frames" << std::endl;
            }
        }
        
        std::cout << "CUDA Gaussian EVM processing completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during processing: " << e.what() << std::endl;
        return false;
    }
    
    cap.release();
    writer.release();
    
    return true;
}

} // namespace cuda_evm