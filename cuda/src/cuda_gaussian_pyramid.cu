#include "cuda_gaussian_pyramid.cuh"
#include "cuda_color_conversion.cuh"
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
    
    // RGB to YIQ conversion
    const float r = d_rgb[idx + 0];
    const float g = d_rgb[idx + 1];  
    const float b = d_rgb[idx + 2];
    
    d_yiq[idx + 0] = 0.299f * r + 0.587f * g + 0.114f * b;        // Y
    d_yiq[idx + 1] = 0.596f * r - 0.274f * g - 0.322f * b;        // I  
    d_yiq[idx + 2] = 0.211f * r - 0.523f * g + 0.312f * b;        // Q
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
    
    // YIQ to RGB conversion
    const float y_val = d_yiq[idx + 0];
    const float i_val = d_yiq[idx + 1];
    const float q_val = d_yiq[idx + 2];
    
    d_rgb[idx + 0] = y_val + 0.956f * i_val + 0.621f * q_val;     // R
    d_rgb[idx + 1] = y_val - 0.272f * i_val - 0.647f * q_val;     // G
    d_rgb[idx + 2] = y_val - 1.106f * i_val + 1.703f * q_val;     // B
}

// Gaussian kernel for 5x5 filtering (stored in constant memory)
__constant__ float d_gaussian_kernel[25] = {
    1.0f/256, 4.0f/256, 6.0f/256, 4.0f/256, 1.0f/256,
    4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256,
    6.0f/256, 24.0f/256, 36.0f/256, 24.0f/256, 6.0f/256,
    4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256,
    1.0f/256, 4.0f/256, 6.0f/256, 4.0f/256, 1.0f/256
};

// Core CUDA kernels for Gaussian pyramid processing
__global__ void gaussian_filter_kernel(
    const float* __restrict__ d_src,
    float* __restrict__ d_dst,
    int width, int height, int channels, int stride)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = (y * stride + x) * channels;
    
    // Apply 5x5 Gaussian filter with REFLECT_101 border handling
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                int sx = x + kx, sy = y + ky;
                if (sx < 0) sx = -sx;
                if (sy < 0) sy = -sy;
                if (sx >= width) sx = 2 * width - sx - 2;
                if (sy >= height) sy = 2 * height - sy - 2;
                
                float k = d_gaussian_kernel[(ky + 2) * 5 + (kx + 2)];
                sum += d_src[(sy * stride + sx) * channels + c] * k;
            }
        }
        d_dst[idx + c] = sum;
    }
}

__global__ void spatial_downsample_kernel(
    const float* __restrict__ d_src,
    float* __restrict__ d_dst,
    int src_width, int src_height, int dst_width, int dst_height, int channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height) return;
    
    // Sample at 2x intervals
    const int src_x = x * 2;
    const int src_y = y * 2;
    
    if (src_x < src_width && src_y < src_height) {
        const int src_idx = (src_y * src_width + src_x) * channels;
        const int dst_idx = (y * dst_width + x) * channels;
        
        for (int c = 0; c < channels; c++) {
            d_dst[dst_idx + c] = d_src[src_idx + c];
        }
    }
}

__global__ void spatial_upsample_zeros_kernel(
    const float* __restrict__ d_src,
    float* __restrict__ d_dst,
    int src_width, int src_height, int dst_width, int dst_height, int channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height) return;
    
    const int dst_idx = (y * dst_width + x) * channels;
    
    // Zero injection: only copy values at even positions
    if (x % 2 == 0 && y % 2 == 0) {
        const int src_x = x / 2;
        const int src_y = y / 2;
        
        if (src_x < src_width && src_y < src_height) {
            const int src_idx = (src_y * src_width + src_x) * channels;
            for (int c = 0; c < channels; c++) {
                d_dst[dst_idx + c] = d_src[src_idx + c] * 4.0f; // 4x compensation
            }
        } else {
            for (int c = 0; c < channels; c++) {
                d_dst[dst_idx + c] = 0.0f;
            }
        }
    } else {
        for (int c = 0; c < channels; c++) {
            d_dst[dst_idx + c] = 0.0f;
        }
    }
}

// Temporal filtering kernels using cuFFT
__global__ void apply_butterworth_filter_kernel(
    cufftComplex* d_freq_data,
    int width, int height, int channels, int num_frames,
    float sample_rate, float low_freq, float high_freq)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || c >= channels) return;
    
    const int spatial_idx = (y * width + x) * channels + c;
    
    for (int f = 0; f < num_frames; f++) {
        const int freq_idx = spatial_idx * num_frames + f;
        
        // Calculate frequency
        float freq = (f < num_frames/2) ? f : f - num_frames;
        freq = freq * sample_rate / num_frames;
        freq = fabsf(freq);
        
        // Butterworth bandpass filter
        float response = 1.0f;
        if (freq < low_freq || freq > high_freq) {
            float ratio_low = (freq < low_freq) ? freq / low_freq : 1.0f;
            float ratio_high = (freq > high_freq) ? high_freq / freq : 1.0f;
            response = sqrtf(ratio_low * ratio_high);
        }
        
        d_freq_data[freq_idx].x *= response;
        d_freq_data[freq_idx].y *= response;
    }
}

// Main CUDA functions (pure CUDA, no OpenCV)
cudaError_t spatially_filter_gaussian_gpu(
    const float* d_input_rgb,
    float* d_output_yiq,
    int width, int height, int channels, int level)
{
    if (!d_input_rgb || !d_output_yiq || width <= 0 || height <= 0 || channels != 3 || level < 0) {
        return cudaErrorInvalidValue;
    }
    
    // Convert RGB to YIQ
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    rgb_to_yiq_planar_kernel<<<gridSize, blockSize>>>(d_input_rgb, d_output_yiq, width, height, channels);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    
    // Apply Gaussian filtering levels
    float *d_temp = nullptr;
    if (level > 0) {
        err = cudaMalloc(&d_temp, width * height * channels * sizeof(float));
        if (err != cudaSuccess) return err;
        
        for (int l = 0; l < level; l++) {
            float *d_src = (l == 0) ? d_output_yiq : d_temp;
            float *d_dst = (l == level - 1) ? d_output_yiq : d_temp;
            
            gaussian_filter_kernel<<<gridSize, blockSize>>>(d_src, d_dst, width, height, channels, width);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                cudaFree(d_temp);
                return err;
            }
        }
        
        cudaFree(d_temp);
    }
    
    return cudaDeviceSynchronize();
}

cudaError_t temporal_filter_gaussian_batch_gpu(
    const float* d_input_frames,
    float* d_output_frames,
    int width, int height, int channels, int num_frames,
    float fl, float fh, float fps)
{
    if (!d_input_frames || !d_output_frames || width <= 0 || height <= 0 || 
        channels != 3 || num_frames <= 1 || fl <= 0 || fh <= fl || fps <= 0) {
        return cudaErrorInvalidValue;
    }
    
    const size_t spatial_size = width * height * channels;
    const size_t total_size = spatial_size * num_frames;
    
    // Allocate frequency domain data
    cufftComplex *d_freq_data = nullptr;
    cudaError_t err = cudaMalloc(&d_freq_data, total_size * sizeof(cufftComplex));
    if (err != cudaSuccess) return err;
    
    // Create cuFFT plans
    cufftHandle plan_forward, plan_inverse;
    int fft_size = num_frames;
    
    if (cufftPlan1d(&plan_forward, fft_size, CUFFT_R2C, spatial_size) != CUFFT_SUCCESS ||
        cufftPlan1d(&plan_inverse, fft_size, CUFFT_C2R, spatial_size) != CUFFT_SUCCESS) {
        cudaFree(d_freq_data);
        return cudaErrorUnknown;
    }
    
    // Forward FFT
    if (cufftExecR2C(plan_forward, (cufftReal*)d_input_frames, d_freq_data) != CUFFT_SUCCESS) {
        cudaFree(d_freq_data);
        cufftDestroy(plan_forward);
        cufftDestroy(plan_inverse);
        return cudaErrorUnknown;
    }
    
    // Apply Butterworth filter
    dim3 blockSize(8, 8, 4);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y,
                  (channels + blockSize.z - 1) / blockSize.z);
    
    apply_butterworth_filter_kernel<<<gridSize, blockSize>>>(
        d_freq_data, width, height, channels, num_frames, fps, fl, fh);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_freq_data);
        cufftDestroy(plan_forward);
        cufftDestroy(plan_inverse);
        return err;
    }
    
    // Inverse FFT
    if (cufftExecC2R(plan_inverse, d_freq_data, (cufftReal*)d_output_frames) != CUFFT_SUCCESS) {
        cudaFree(d_freq_data);
        cufftDestroy(plan_forward);
        cufftDestroy(plan_inverse);
        return cudaErrorUnknown;
    }
    
    // Cleanup
    cudaFree(d_freq_data);
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
    
    return cudaDeviceSynchronize();
}

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
    
    // Combine original and filtered signals using a simple kernel
    // This is a placeholder - you'd implement the actual combination logic
    err = cudaMemcpy(d_combined_yiq, d_original_yiq, image_size, cudaMemcpyDeviceToDevice);
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
    if (d_original_yiq) cudaFree(d_original_yiq);
    if (d_combined_yiq) cudaFree(d_combined_yiq);
    
    return err;
}

// Video processing functions removed - no OpenCV allowed in core implementation
// Use external wrapper with OpenCV for video I/O if needed

} // namespace cuda_evm