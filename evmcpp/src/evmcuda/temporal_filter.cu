#include "evmcuda/temporal_filter.cuh"

#include <cufft.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <string>
#include <iostream> // For PRINT_WARNING/PRINT_ERROR
#include <complex> // Potentially needed if working with complex types directly

// Define PRINT_ERROR/PRINT_WARNING if not globally available via headers
#ifndef PRINT_ERROR
#include <iostream>
#define PRINT_ERROR(msg) std::cerr << "ERROR: " << msg << std::endl
#endif
#ifndef PRINT_WARNING
#define PRINT_WARNING(msg) std::cout << "WARNING: " << msg << std::endl
#endif

// Helper macro for checking cuFFT errors
#define CUFFT_CHECK(call)                                            \
    do {                                                             \
        cufftResult_t err = call;                                    \
        if (err != CUFFT_SUCCESS) {                                  \
            throw std::runtime_error(std::string("cuFFT error ") + std::to_string(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                            \
    } while (0)

// Helper macro for checking CUDA errors
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                           \
    } while (0)


namespace evmcuda {

// --- Kernel for Frequency Masking and Amplification ---
// This kernel operates on the complex output of the forward FFT (R2C format).
// It zeros out unwanted frequencies and applies amplification/attenuation.
// Note: Needs careful implementation for R2C's packed format.
__global__ void applyMaskAndAmplifyKernelR2C(cufftComplex* fftData, // cufftComplex is float2
                                          int N, // FFT size (numFrames)
                                          // int numFrames, // Duplicate removed - N is used
                                          int width, int height, // Image dimensions
                                          int batchSize, // width * height * 3
                                          const unsigned char* mask, // Mask (0=discard, 1=keep)
                                          float alpha, float chromAttenuation)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z; // Use z-dimension for channel (0=Y, 1=I, 2=Q)

    if (x >= width || y >= height || c >= 3) {
        return;
    }

    // Calculate the starting index for this pixel's time series in the batch FFT data
    // Batch FFT layout: Frame0_YIQ(x,y), Frame1_YIQ(x,y), ..., FrameN-1_YIQ(x,y)
    // For a specific pixel (x,y,c), its time series is interleaved across the batch.
    // cuFFT batch layout: (batch, z, y, x) where batch corresponds to pixel locations * channels.
    // The fftData pointer points to the beginning of the complex data.
    // We need to access the data for frequency 'f' of pixel (x,y,c).
    // Let N = numFrames. The total number of FFTs is width * height * 3.
    // The size of each FFT is N (complex numbers).
    // The index for pixel (x,y,c) starts at (c * height * width + y * width + x) * N.

    const int pixel_index = c * height * width + y * width + x;
    int complex_fft_size = N / 2 + 1; // Calculate outside loop
    cufftComplex* pixel_fft_start = fftData + (size_t)pixel_index * complex_fft_size; // Offset by complex size

    // Determine amplification factor for this channel
    float currentAlpha = alpha;
    if (c > 0) { // Attenuate I and Q channels
        currentAlpha *= chromAttenuation;
    }

    // Apply mask and amplification to each frequency component (f=0 to N/2)
    // This needs careful indexing based on the R2C output format and batch layout
    // int complex_fft_size = N / 2 + 1; // Moved outside loop
    for (int f = 0; f < complex_fft_size; ++f) {
        // TODO: Determine the correct frequency index 'freq_idx' corresponding to 'f'
        //       in the original full frequency spectrum (0 to N-1) to check the mask.
        //       This depends on the fftfreq mapping.
        int freq_idx = f; // Placeholder - needs correct mapping
        if (freq_idx >= N) continue; // Should not happen if mapping is correct

        if (mask[freq_idx] == 0) {
            // Zero out this frequency component
            pixel_fft_start[f].x = 0.0f;
            pixel_fft_start[f].y = 0.0f;
        } else {
            // Apply amplification
            pixel_fft_start[f].x *= currentAlpha;
            // Only apply to imaginary part if it exists (f > 0 and f < N/2 if N is even)
            if (f > 0 && (N % 2 == 1 || f < N / 2)) {
                 pixel_fft_start[f].y *= currentAlpha;
            } else {
                 // For DC (f=0) and Nyquist (f=N/2, if N even), imag part is conceptually zero
                 pixel_fft_start[f].y = 0.0f;
            }
        }
    }
}


// --- C++ Wrapper Function ---

void temporalFilterGaussianBatch_gpu(float* d_imageBatch, // In-place modification
                                     int numFrames, int width, int height, size_t pitch,
                                     float fps, float fl, float fh,
                                     float alpha, float chromAttenuation,
                                     cudaStream_t stream)
{
    // --- Input Validation ---
    if (!d_imageBatch || numFrames <= 1 || width <= 0 || height <= 0 || pitch == 0 || fps <= 0 || fl < 0 || fh <= fl) {
        throw std::invalid_argument("[temporalFilterGaussianBatch_gpu] Invalid arguments.");
    }

    // --- Prepare Frequency Mask (Host side first) ---
    std::vector<float> frequencies(numFrames);
    float freqStep = fps / static_cast<float>(numFrames);
    int n_over_2_ceil = (numFrames + 1) / 2;
    for (int i = 0; i < numFrames; ++i) {
        if (i < n_over_2_ceil) frequencies[i] = static_cast<float>(i) * freqStep;
        else frequencies[i] = static_cast<float>(i - numFrames) * freqStep;
    }

    std::vector<unsigned char> h_mask(numFrames); // Use unsigned char
    for (int i = 0; i < numFrames; ++i) {
        h_mask[i] = (std::abs(frequencies[i]) >= fl && std::abs(frequencies[i]) <= fh) ? 1 : 0; // Store 1 or 0
    }

    // --- Allocate and Copy Mask to Device ---
    unsigned char* d_mask = nullptr; // Use unsigned char*
    CUDA_CHECK(cudaMalloc((void**)&d_mask, numFrames * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpyAsync(d_mask, h_mask.data(), numFrames * sizeof(unsigned char), cudaMemcpyHostToDevice, stream));

    // --- Prepare cuFFT Plan ---
    cufftHandle plan;
    int n[] = {numFrames}; // Size of 1D FFT
    int batchSize = width * height * 3; // Total number of 1D FFTs

    // Create a batch plan for 1D FFTs
    // Input/output type: CUFFT_R2C for real-to-complex, CUFFT_C2R for complex-to-real
    // For in-place, input and output types must match (CUFFT_C2C)
    // Let's use out-of-place R2C and C2R for clarity first.
    // We need a temporary buffer for complex data.

    // Allocate temporary complex buffer on GPU
    cufftComplex* d_fftComplex = nullptr;
    // Size needed: batchSize * (numFrames / 2 + 1) complex numbers for R2C
    size_t complexSize = batchSize * (numFrames / 2 + 1);
    CUDA_CHECK(cudaMalloc((void**)&d_fftComplex, complexSize * sizeof(cufftComplex)));

    // Create R2C plan
    CUFFT_CHECK(cufftPlanMany(&plan, 1, n,
                              nullptr, 1, numFrames, // Input layout (batch, n)
                              nullptr, 1, numFrames / 2 + 1, // Output layout (batch, n/2+1)
                              CUFFT_R2C, batchSize));

    // Associate stream with the plan
    CUFFT_CHECK(cufftSetStream(plan, stream));

    // --- Execute Forward FFT (Real to Complex) ---
    // Input: d_imageBatch (float*), Output: d_fftComplex (cufftComplex*)
    CUFFT_CHECK(cufftExecR2C(plan, (cufftReal*)d_imageBatch, d_fftComplex));

    // Destroy R2C plan
    CUFFT_CHECK(cufftDestroy(plan));

    // --- Apply Mask and Amplification Kernel ---
    // Launch configuration: Grid covers all pixels (width x height) and channels (3)
    // Block dimension is (1, 1, 1) as each thread handles one pixel's full frequency spectrum.
    const dim3 gridMask(width, height, 3);
    const dim3 blockMask(1, 1, 1); // Each thread handles one FFT

    applyMaskAndAmplifyKernelR2C<<<gridMask, blockMask, 0, stream>>>(
        d_fftComplex, numFrames, width, height, batchSize, d_mask, alpha, chromAttenuation
    );
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Synchronize after masking kernel before inverse FFT
    CUDA_CHECK(cudaStreamSynchronize(stream));


    // --- Prepare Inverse FFT Plan (Complex to Real) ---
    // Create C2R plan
    // Input: d_fftComplex, Output: d_imageBatch (in-place is tricky with R2C/C2R)
    // Let's use the same temporary buffer for output of C2R for now, then copy back if needed.
    // Or, modify d_fftComplex in-place and use C2C inverse? No, need real output.
    // Plan for C2R out-of-place
    CUFFT_CHECK(cufftPlanMany(&plan, 1, n,
                              nullptr, 1, numFrames / 2 + 1, // Input layout (batch, n/2+1)
                              nullptr, 1, numFrames, // Output layout (batch, n)
                              CUFFT_C2R, batchSize));

    // Associate stream
    CUFFT_CHECK(cufftSetStream(plan, stream));

    // --- Execute Inverse FFT (Complex to Real) ---
    // Input: d_fftComplex, Output: d_imageBatch (overwriting original data)
    CUFFT_CHECK(cufftExecC2R(plan, d_fftComplex, (cufftReal*)d_imageBatch));

    // Destroy C2R plan
    CUFFT_CHECK(cufftDestroy(plan));

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_fftComplex));

    // Note: Amplification/Masking kernel needs proper implementation for R2C format.
    // The current code performs FFT -> IFFT without filtering/amplification.
}


// --- Kernel for Laplacian IIR Temporal Filtering (Single Frame, Bandpass) ---
// Processes one pixel for the current frame, updating state buffers for both stages.
__global__ void iirTemporalFilterFrameBandpassKernel(
    const float* d_inputLevelData,  // Input for current frame (x[n])
    float* d_outputLevelData, // Output for current frame ((yl[n]-yh[n]) * alpha)
    float* d_state_xl_1, // x[n-1] for low-pass stage, updated to x[n]
    float* d_state_yl_1, // yl[n-1] for low-pass stage, updated to yl[n]
    float* d_state_xh_1, // x[n-1] for high-pass stage, updated to x[n]
    float* d_state_yh_1, // yh[n-1] for high-pass stage, updated to yh[n]
    int width, int height, size_t pitch,
    float b0_l, float b1_l, float a1_l, // Low-pass coeffs
    float b0_h, float b1_h, float a1_h, // High-pass coeffs
    float alpha_spatially_attenuated, // Base alpha after spatial attenuation
    float chromAttenuation           // Chrominance attenuation factor
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const size_t pitchF = pitch / sizeof(float);
    const size_t pixelOffset = y * pitchF + x * 3;

    // Pointers to current frame data and state data for this pixel
    const float* currentInputPixel = d_inputLevelData + pixelOffset;
    float* currentOutputPixel = d_outputLevelData + pixelOffset;
    // Low-pass state pointers
    float* current_xl_1 = d_state_xl_1 + pixelOffset;
    float* current_yl_1 = d_state_yl_1 + pixelOffset;
    // High-pass state pointers
    float* current_xh_1 = d_state_xh_1 + pixelOffset;
    float* current_yh_1 = d_state_yh_1 + pixelOffset;

    // Process each channel
    for (int c = 0; c < 3; ++c) {
        float in_n = currentInputPixel[c]; // x[n]

        // --- Low-pass stage ---
        float xl_1 = current_xl_1[c]; // x[n-1] for low-pass
        float yl_1 = current_yl_1[c]; // yl[n-1]
        float yl_n = b0_l * in_n + b1_l * xl_1 - a1_l * yl_1; // yl[n]

        // --- High-pass stage ---
        float xh_1 = current_xh_1[c]; // x[n-1] for high-pass
        float yh_1 = current_yh_1[c]; // yh[n-1]
        float yh_n = b0_h * in_n + b1_h * xh_1 - a1_h * yh_1; // yh[n]

        // --- Bandpass result ---
        float bandpass_result = yh_n - yl_n; // y[n] = yh[n] - yl[n] (Corrected order)

        // --- Update state buffers for next frame ---
        current_xl_1[c] = in_n; // Store x[n] for next low-pass x[n-1]
        current_yl_1[c] = yl_n; // Store yl[n] for next low-pass y[n-1]
        current_xh_1[c] = in_n; // Store x[n] for next high-pass x[n-1]
        current_yh_1[c] = yh_n; // Store yh[n] for next high-pass y[n-1]

        // --- Apply amplification (with chrominance attenuation) ---
        float alpha_final = alpha_spatially_attenuated;
        if (c > 0) { // Attenuate I(c=1) and Q(c=2) channels
            alpha_final *= chromAttenuation;
        }
        currentOutputPixel[c] = bandpass_result * alpha_final;
    }
}


// --- C++ Wrapper for Laplacian Temporal Filter (Single Frame, Single Level, Bandpass) ---

void filterLaplacianLevelFrame_gpu(const float* d_inputLevelData,
                                   float* d_outputLevelData,
                                   float* d_state_xl_1, // Read & Write
                                   float* d_state_yl_1, // Read & Write
                                   float* d_state_xh_1, // Read & Write
                                   float* d_state_yh_1, // Read & Write
                                   int width, int height, size_t pitch,
                                   float b0_l, float b1_l, float a1_l, // Low-pass coeffs
                                   float b0_h, float b1_h, float a1_h, // High-pass coeffs
                                   int level, float fps, // For attenuation calc
                                   float alpha, float lambda_cutoff, float chromAttenuation, // Base params
                                   cudaStream_t stream)
{
    // --- Input Validation ---
    if (!d_inputLevelData || !d_outputLevelData || !d_state_xl_1 || !d_state_yl_1 ||
        !d_state_xh_1 || !d_state_yh_1 || width <= 0 || height <= 0 || pitch == 0 || fps <= 0) {
        throw std::invalid_argument("[filterLaplacianLevelFrame_gpu] Invalid arguments.");
    }

    // --- Calculate Spatially Attenuated Alpha (matching CPU logic) ---
    float alpha_spatially_attenuated = alpha; // Start with base alpha
    if (level > 0) { // Attenuation only applies to levels > 0
        // Calculate lambda based on level dimensions (matching CPU code)
        float lambda = sqrtf(static_cast<float>(height * height + width * width));
        // Calculate delta (ensure float division)
        float delta = lambda_cutoff / (8.0f * (1.0f + alpha));

        if (delta > 1e-6) { // Avoid division by zero or near-zero delta
             float new_alpha = (lambda / delta) - 1.0f;
             // Apply the minimum, ensuring it doesn't go below zero? CPU doesn't check.
             alpha_spatially_attenuated = fminf(alpha, new_alpha);
             // Clamp to zero if new_alpha was negative? CPU doesn't seem to.
             if (alpha_spatially_attenuated < 0.0f) alpha_spatially_attenuated = 0.0f;
        } else {
             // If delta is too small, lambda/delta is huge, so new_alpha is huge.
             // std::min(alpha, new_alpha) will just be alpha. No change needed.
             // Or, maybe attenuate completely if delta is near zero? Let's stick to CPU logic.
             alpha_spatially_attenuated = alpha;
        }

        // The CPU code applies attenuation only if lvl >= 1 and lvl < (level-1)
        // This seems inconsistent with the formula which depends on lambda_cutoff.
        // Let's follow the formula based on lambda_cutoff for now.
        // Re-checking CPU code: The condition `lvl >= 1 && lvl < (level - 1)` seems to be
        // applied *in addition* to the lambda check implicitly via loop bounds or logic.
        // The formula `min(alpha, new_alpha)` should handle the attenuation correctly
        // based on lambda vs lambda_cutoff via delta. Let's trust the formula.
    }

    // Override alpha for level 0 to match Python reference (no amplification at level 0)
    if (level == 0) {
        alpha_spatially_attenuated = 1.0f;
    }

    // Chrominance attenuation is handled inside the kernel now.
    // --- Launch Kernel ---
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y);

    iirTemporalFilterFrameBandpassKernel<<<gridSize, blockSize, 0, stream>>>(
        d_inputLevelData,
        d_outputLevelData,
        d_state_xl_1,
        d_state_yl_1,
        d_state_xh_1,
        d_state_yh_1,
        width, height, pitch,
        b0_l, b1_l, a1_l,
        b0_h, b1_h, a1_h,
        alpha_spatially_attenuated, // Pass spatially attenuated alpha
        chromAttenuation           // Pass chrominance attenuation factor
    );
    CUDA_CHECK(cudaGetLastError());
    // Synchronization should be handled by the caller
}



} // namespace evmcuda