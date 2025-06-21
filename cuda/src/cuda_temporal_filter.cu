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

// =============================================================================
// LAPLACIAN PYRAMID IIR TEMPORAL FILTERING
// =============================================================================

/**
 * @brief Calculate Butterworth filter coefficients on device
 */
__device__ void calculate_butterworth_coefficients_device(
    float fl, float fh, float fps, int order,
    float* b_low, float* a_low, float* b_high, float* a_high)
{
    const float pi = 3.14159265358979323846f;
    
    // Calculate Butterworth coefficients for order-2 filters
    // Low-pass filter coefficients
    float wc_low = 2.0f * pi * fl;
    float T = 1.0f / fps;
    float wcd_low = (2.0f / T) * tanf(wc_low * T / 2.0f);
    float wd_low = wcd_low;
    float bd_low = 1.0f + sqrtf(2.0f) * wd_low + wd_low * wd_low;
    
    b_low[0] = wd_low * wd_low / bd_low;
    b_low[1] = 2.0f * b_low[0];
    b_low[2] = b_low[0];
    a_low[0] = 1.0f;
    a_low[1] = (2.0f * wd_low * wd_low - 2.0f) / bd_low;
    a_low[2] = (1.0f - sqrtf(2.0f) * wd_low + wd_low * wd_low) / bd_low;
    
    // High-pass filter coefficients  
    float wc_high = 2.0f * pi * fh;
    float wcd_high = (2.0f / T) * tanf(wc_high * T / 2.0f);
    float wd_high = wcd_high;
    float bd_high = 1.0f + sqrtf(2.0f) * wd_high + wd_high * wd_high;
    
    b_high[0] = 1.0f / bd_high;
    b_high[1] = -2.0f * b_high[0];
    b_high[2] = b_high[0];
    a_high[0] = 1.0f;
    a_high[1] = (2.0f * wd_high * wd_high - 2.0f) / bd_high;
    a_high[2] = (1.0f - sqrtf(2.0f) * wd_high + wd_high * wd_high) / bd_high;
}

/**
 * @brief Apply IIR Butterworth temporal filtering to pyramid levels with proper state management
 */
__global__ void apply_iir_temporal_filter_kernel(
    float** d_pyramid_levels,        // Input pyramid levels for current frame
    float** d_filtered_levels,       // Output filtered levels for current frame
    float** d_prev_input_levels,     // Previous frame input for IIR state
    float** d_lowpass_state_levels,  // Low-pass filter state memory
    float** d_highpass_state_levels, // High-pass filter state memory
    const int* level_widths,
    const int* level_heights,
    const size_t* level_sizes,
    int num_levels,
    int channels,
    float fl, float fh, float fps,
    float alpha, float lambda_cutoff, float chrom_attenuation,
    int frame_idx)
{
    const int level = blockIdx.z;
    const int spatial_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (level >= num_levels || ch >= channels) return;
    
    const int width = level_widths[level];
    const int height = level_heights[level];
    const int pixels_per_level = width * height;
    
    if (spatial_idx >= pixels_per_level) return;
    
    // Calculate memory index for this level/pixel/channel
    const int idx = spatial_idx * channels + ch;
    
    // Calculate Butterworth coefficients (order-1 for simplicity and performance)
    const float pi = 3.14159265358979323846f;
    const float T = 1.0f / fps;
    
    // Low-pass coefficients (order-1)
    float wc_low = 2.0f * pi * fl;
    float alpha_low = T * wc_low / (T * wc_low + 2.0f);
    float b_low[2] = {alpha_low, alpha_low};
    float a_low[2] = {1.0f, alpha_low - 1.0f};
    
    // High-pass coefficients (order-1)  
    float wc_high = 2.0f * pi * fh;
    float alpha_high = 2.0f / (T * wc_high + 2.0f);
    float b_high[2] = {alpha_high, -alpha_high};
    float a_high[2] = {1.0f, (T * wc_high - 2.0f) / (T * wc_high + 2.0f)};
    
    // Get current input value
    float current_input = d_pyramid_levels[level][idx];
    
    // For frame 0, initialize states
    if (frame_idx == 0) {
        // Initialize filter states with first frame values
        d_lowpass_state_levels[level][idx] = current_input;
        d_highpass_state_levels[level][idx] = current_input;
        d_prev_input_levels[level][idx] = current_input;
        d_filtered_levels[level][idx] = 0.0f; // First frame has no temporal change
        return;
    }
    
    // Get previous states
    float prev_input = d_prev_input_levels[level][idx];
    float prev_lowpass = d_lowpass_state_levels[level][idx];
    float prev_highpass = d_highpass_state_levels[level][idx];
    
    // Apply IIR low-pass filter: y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
    float lowpass_output = b_low[0] * current_input + b_low[1] * prev_input - a_low[1] * prev_lowpass;
    
    // Apply IIR high-pass filter: y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
    float highpass_output = b_high[0] * current_input + b_high[1] * prev_input - a_high[1] * prev_highpass;
    
    // Bandpass = highpass - lowpass
    float bandpass_result = highpass_output - lowpass_output;
    
    // Apply spatial attenuation based on pyramid level
    float current_alpha = alpha;
    if (level >= 1 && level < (num_levels - 1)) {
        // Calculate spatial wavelength-aware attenuation
        float lambd = sqrtf(static_cast<float>(height * height + width * width));
        float delta = lambda_cutoff;
        float new_alpha = (lambd / (8.0f * delta)) - 1.0f;
        current_alpha = fminf(alpha, new_alpha);
    }
    
    // Apply channel-specific amplification
    if (ch == 0) {
        // Y channel: full amplification
        bandpass_result *= current_alpha;
    } else {
        // I, Q channels: attenuated amplification
        bandpass_result *= current_alpha * chrom_attenuation;
    }
    
    // Update states for next frame
    d_lowpass_state_levels[level][idx] = lowpass_output;
    d_highpass_state_levels[level][idx] = highpass_output;
    d_prev_input_levels[level][idx] = current_input;
    
    // Store filtered result
    d_filtered_levels[level][idx] = bandpass_result;
}

/**
 * @brief Simplified streaming temporal filter for individual frame processing
 * This version processes one frame at a time and requires external state management
 */
cudaError_t temporal_filter_laplacian_frame_gpu(
    float** d_pyramid_levels,        // Input pyramid levels for current frame
    float** d_filtered_levels,       // Output filtered levels for current frame
    float** d_prev_input_levels,     // Previous frame input for IIR state
    float** d_lowpass_state_levels,  // Low-pass filter state memory
    float** d_highpass_state_levels, // High-pass filter state memory
    const int* level_widths,         // Width of each pyramid level
    const int* level_heights,        // Height of each pyramid level  
    const size_t* level_sizes,       // Total elements per level
    int num_levels,
    int channels,
    float fl, float fh, float fps,
    float alpha, float lambda_cutoff, float chrom_attenuation,
    int frame_idx)
{
    if (!d_pyramid_levels || !d_filtered_levels || !level_widths || !level_heights || !level_sizes ||
        num_levels <= 0 || channels <= 0) {
        return cudaErrorInvalidValue;
    }
    
    // Calculate kernel launch configuration
    int max_pixels = 0;
    for (int level = 0; level < num_levels; level++) {
        int pixels = level_widths[level] * level_heights[level];
        max_pixels = max(max_pixels, pixels);
    }
    
    dim3 blockSize(min(channels, 32), min(max_pixels, 32), 1);
    dim3 gridSize((channels + blockSize.x - 1) / blockSize.x,
                  (max_pixels + blockSize.y - 1) / blockSize.y,
                  num_levels);
    
    // Launch kernel for this frame
    apply_iir_temporal_filter_kernel<<<gridSize, blockSize>>>(
        d_pyramid_levels, d_filtered_levels,
        d_prev_input_levels, d_lowpass_state_levels, d_highpass_state_levels,
        level_widths, level_heights, level_sizes,
        num_levels, channels,
        fl, fh, fps, alpha, lambda_cutoff, chrom_attenuation,
        frame_idx);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    
    return cudaSuccess;
}

/**
 * @brief Main function for CUDA Laplacian pyramid temporal filtering (batch processing)
 */
cudaError_t temporal_filter_laplacian_pyramids_gpu(
    float** d_pyramid_levels,        // Array of device pointers to pyramid levels
    float** d_filtered_levels,       // Array of device pointers for filtered output levels
    const int* level_widths,         // Width of each pyramid level
    const int* level_heights,        // Height of each pyramid level  
    const size_t* level_sizes,       // Total elements per level
    int num_levels,
    int num_frames,
    int channels,
    float fl, float fh, float fps,
    float alpha, float lambda_cutoff, float chrom_attenuation)
{
    if (!d_pyramid_levels || !d_filtered_levels || !level_widths || !level_heights || !level_sizes ||
        num_levels <= 0 || num_frames <= 0 || channels <= 0) {
        return cudaErrorInvalidValue;
    }
    
    // Allocate state memory for IIR filters
    float** d_prev_input_levels = nullptr;
    float** d_lowpass_state_levels = nullptr;
    float** d_highpass_state_levels = nullptr;
    
    cudaError_t err = cudaMalloc(&d_prev_input_levels, num_levels * sizeof(float*));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&d_lowpass_state_levels, num_levels * sizeof(float*));
    if (err != cudaSuccess) {
        cudaFree(d_prev_input_levels);
        return err;
    }
    
    err = cudaMalloc(&d_highpass_state_levels, num_levels * sizeof(float*));
    if (err != cudaSuccess) {
        cudaFree(d_prev_input_levels);
        cudaFree(d_lowpass_state_levels);
        return err;
    }
    
    // Allocate memory for each level's state
    for (int level = 0; level < num_levels; level++) {
        size_t level_bytes = level_sizes[level] * sizeof(float);
        
        float* d_prev_input = nullptr;
        float* d_lowpass_state = nullptr;
        float* d_highpass_state = nullptr;
        
        err = cudaMalloc(&d_prev_input, level_bytes);
        if (err != cudaSuccess) goto cleanup;
        
        err = cudaMalloc(&d_lowpass_state, level_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_prev_input);
            goto cleanup;
        }
        
        err = cudaMalloc(&d_highpass_state, level_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_prev_input);
            cudaFree(d_lowpass_state);
            goto cleanup;
        }
        
        // Copy pointers to device
        err = cudaMemcpy(&d_prev_input_levels[level], &d_prev_input, sizeof(float*), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto cleanup;
        
        err = cudaMemcpy(&d_lowpass_state_levels[level], &d_lowpass_state, sizeof(float*), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto cleanup;
        
        err = cudaMemcpy(&d_highpass_state_levels[level], &d_highpass_state, sizeof(float*), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto cleanup;
    }
    
    // Process each frame sequentially (IIR requires sequential processing)
    for (int frame = 0; frame < num_frames; frame++) {
        err = temporal_filter_laplacian_frame_gpu(
            &d_pyramid_levels[frame * num_levels], &d_filtered_levels[frame * num_levels],
            d_prev_input_levels, d_lowpass_state_levels, d_highpass_state_levels,
            level_widths, level_heights, level_sizes,
            num_levels, channels,
            fl, fh, fps, alpha, lambda_cutoff, chrom_attenuation,
            frame);
        
        if (err != cudaSuccess) goto cleanup;
        
        // Synchronize after each frame (required for IIR temporal dependencies)
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto cleanup;
    }
    
cleanup:
    // Cleanup state memory
    if (d_prev_input_levels) {
        for (int level = 0; level < num_levels; level++) {
            float* d_ptr = nullptr;
            if (cudaMemcpy(&d_ptr, &d_prev_input_levels[level], sizeof(float*), cudaMemcpyDeviceToHost) == cudaSuccess) {
                cudaFree(d_ptr);
            }
        }
        cudaFree(d_prev_input_levels);
    }
    
    if (d_lowpass_state_levels) {
        for (int level = 0; level < num_levels; level++) {
            float* d_ptr = nullptr;
            if (cudaMemcpy(&d_ptr, &d_lowpass_state_levels[level], sizeof(float*), cudaMemcpyDeviceToHost) == cudaSuccess) {
                cudaFree(d_ptr);
            }
        }
        cudaFree(d_lowpass_state_levels);
    }
    
    if (d_highpass_state_levels) {
        for (int level = 0; level < num_levels; level++) {
            float* d_ptr = nullptr;
            if (cudaMemcpy(&d_ptr, &d_highpass_state_levels[level], sizeof(float*), cudaMemcpyDeviceToHost) == cudaSuccess) {
                cudaFree(d_ptr);
            }
        }
        cudaFree(d_highpass_state_levels);
    }
    
    return err;
}

} // namespace cuda_evm