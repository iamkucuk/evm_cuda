#include "../include/cuda_temporal_filter.cuh"
#include <cufft.h>
#include <cuda_runtime.h>
#include <cmath>

namespace cuda_evm {

// Temporal filtering kernels

// Binary bandpass filter kernel (fixed stride for cuFFT R2C output)
__global__ void apply_binary_bandpass_filter_kernel(
    cufftComplex* d_freq_data,
    int width, int height, int channels, int num_frames,
    float sample_rate, float low_freq, float high_freq)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || c >= channels) return;
    
    const int spatial_idx = (y * width + x) * channels + c;
    const int freq_size = (num_frames / 2) + 1;  // R2C output size
    
    for (int f = 0; f < freq_size; f++) {
        const int freq_idx = spatial_idx * freq_size + f;
        
        // Calculate frequency (matching CPU fftfreq logic exactly)
        float freqStep = sample_rate / static_cast<float>(num_frames);
        int n_over_2_ceil = (num_frames + 1) / 2; // Ceiling division like CPU
        
        float freq;
        if (f < n_over_2_ceil) {
            freq = static_cast<float>(f) * freqStep;
        } else {
            freq = static_cast<float>(f - num_frames) * freqStep;
        }
        freq = fabsf(freq);
        
        // Binary bandpass filter (like CPU implementation)
        // Keep frequencies if: fl <= abs(freq) <= fh, otherwise zero them out
        float response = 0.0f;  // Default: zero out
        if (freq >= low_freq && freq <= high_freq) {
            response = 1.0f;  // Keep this frequency
        }
        
        d_freq_data[freq_idx].x *= response;
        d_freq_data[freq_idx].y *= response;
    }
}

// FFT normalization kernel
__global__ void normalize_ifft_kernel(
    float* d_frames,
    int total_elements,
    float normalization_factor)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    d_frames[idx] *= normalization_factor;
}

// Alpha amplification kernel (frame-major layout from cuFFT C2R output)
__global__ void apply_alpha_amplification_kernel(
    float* d_filtered_frames,
    int width, int height, int channels, int num_frames,
    float alpha, float chrom_attenuation)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int f = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || f >= num_frames) return;
    
    // Frame-major layout from cuFFT C2R: [frame][spatial_location]
    const int spatial_idx = (y * width + x) * channels;
    const int frame_offset = f * width * height * channels;
    
    for (int c = 0; c < channels; c++) {
        const int idx = frame_offset + spatial_idx + c;
        
        // Apply amplification: alpha for Y channel, alpha*chrom_attenuation for I,Q channels
        float current_alpha = (c == 0) ? alpha : alpha * chrom_attenuation;
        d_filtered_frames[idx] *= current_alpha;
    }
}

// Main temporal filtering function
cudaError_t temporal_filter_gaussian_batch_gpu(
    const float* d_input_frames,  // Input: pixel-major layout [spatial_location][time_series]
    float* d_output_frames,       // Output: pixel-major layout [spatial_location][time_series]  
    int width, int height, int channels, int num_frames,
    float fl, float fh, float fps, float alpha, float chrom_attenuation)
{
    if (!d_input_frames || !d_output_frames || 
        width <= 0 || height <= 0 || channels <= 0 || num_frames <= 0) {
        return cudaErrorInvalidValue;
    }
    
    const size_t spatial_size = width * height * channels;
    const size_t total_size = spatial_size * num_frames;
    const int fft_size = num_frames;
    
    // Allocate frequency domain data
    cufftComplex *d_freq_data = nullptr;
    cudaError_t err = cudaMalloc(&d_freq_data, total_size * sizeof(cufftComplex));
    if (err != cudaSuccess) return err;
    
    // Create cuFFT plans
    cufftHandle plan_forward, plan_inverse;
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
    
    // Apply binary bandpass filter
    dim3 blockSize(8, 8, 4);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y,
                  (channels + blockSize.z - 1) / blockSize.z);
    
    apply_binary_bandpass_filter_kernel<<<gridSize, blockSize>>>(
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
    
    // CRITICAL: Normalize by N (like OpenCV's DFT_SCALE flag)
    // cuFFT does NOT auto-normalize, but OpenCV does with DFT_SCALE
    const float normalization_factor = 1.0f / static_cast<float>(num_frames);
    const int total_elements = static_cast<int>(total_size);
    
    dim3 normBlockSize(256);
    dim3 normGridSize((total_elements + normBlockSize.x - 1) / normBlockSize.x);
    
    normalize_ifft_kernel<<<normGridSize, normBlockSize>>>(
        d_output_frames, total_elements, normalization_factor);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_freq_data);
        cufftDestroy(plan_forward);
        cufftDestroy(plan_inverse);
        return err;
    }
    
    // Apply alpha amplification (equivalent to CPU: filteredTimeSeriesReal * currentAlpha)
    dim3 ampBlockSize(16, 16, 4);
    dim3 ampGridSize((width + ampBlockSize.x - 1) / ampBlockSize.x,
                     (height + ampBlockSize.y - 1) / ampBlockSize.y,
                     (num_frames + ampBlockSize.z - 1) / ampBlockSize.z);
    
    apply_alpha_amplification_kernel<<<ampGridSize, ampBlockSize>>>(
        d_output_frames, width, height, channels, num_frames, alpha, chrom_attenuation);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_freq_data);
        cufftDestroy(plan_forward);
        cufftDestroy(plan_inverse);
        return err;
    }
    
    // Cleanup
    cudaFree(d_freq_data);
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
    
    return cudaSuccess;
}

} // namespace cuda_evm