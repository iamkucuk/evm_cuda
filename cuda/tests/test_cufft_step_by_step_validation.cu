#include <cuda_runtime.h>
#include <cufft.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Include our CUDA modules for color conversion and processing
#include "../include/cuda_color_conversion.cuh"
#include "../include/cuda_pyramid.cuh"

// Error checking helpers
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)

#define CHECK_CUFFT(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            std::cerr << "cuFFT error at " << __FILE__ << ":" << __LINE__ << " - " << err << std::endl; \
            exit(1); \
        } \
    } while (0)

// CPU reference implementation for temporal filtering (simplified)
std::vector<float> cpu_temporal_filter_reference(
    const std::vector<float>& time_series,
    float fps, float fl, float fh
) {
    int num_frames = time_series.size();
    
    // Create OpenCV Mat for time series
    cv::Mat signal(num_frames, 1, CV_32F);
    for (int i = 0; i < num_frames; i++) {
        signal.at<float>(i) = time_series[i];
    }
    
    // Prepare for DFT (optimal size)
    int dft_size = cv::getOptimalDFTSize(num_frames);
    cv::Mat padded_signal;
    cv::copyMakeBorder(signal, padded_signal, 0, dft_size - num_frames, 0, 0, cv::BORDER_CONSTANT, 0);
    
    // Convert to complex
    cv::Mat complex_signal;
    cv::Mat planes[] = {padded_signal, cv::Mat::zeros(padded_signal.size(), CV_32F)};
    cv::merge(planes, 2, complex_signal);
    
    // Forward DFT
    cv::dft(complex_signal, complex_signal);
    
    // Apply bandpass filter
    cv::split(complex_signal, planes);
    for (int i = 0; i < dft_size; i++) {
        float freq = i * fps / dft_size;
        if (i > dft_size/2) freq = (i - dft_size) * fps / dft_size;
        freq = std::abs(freq);
        
        if (freq < fl || freq > fh) {
            planes[0].at<float>(i) = 0;
            planes[1].at<float>(i) = 0;
        }
    }
    cv::merge(planes, 2, complex_signal);
    
    // Inverse DFT with scaling
    cv::idft(complex_signal, complex_signal, cv::DFT_SCALE);
    cv::split(complex_signal, planes);
    
    // Extract result
    std::vector<float> result(num_frames);
    for (int i = 0; i < num_frames; i++) {
        result[i] = planes[0].at<float>(i);
    }
    
    return result;
}

// CUDA cuFFT implementation for single time series (for validation)
std::vector<float> cuda_cufft_single_series(
    const std::vector<float>& time_series,
    float fps, float fl, float fh
) {
    int num_frames = time_series.size();
    
    // Find optimal DFT size (power of 2)
    int dft_size = 1;
    while (dft_size < num_frames) dft_size <<= 1;
    
    // Allocate device memory
    cufftReal* d_input;
    cufftComplex* d_fft;
    cufftReal* d_output;
    
    CHECK_CUDA(cudaMalloc(&d_input, dft_size * sizeof(cufftReal)));
    CHECK_CUDA(cudaMalloc(&d_fft, (dft_size/2 + 1) * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_output, dft_size * sizeof(cufftReal)));
    
    // Copy input data and zero-pad
    std::vector<float> padded_input(dft_size, 0.0f);
    for (int i = 0; i < num_frames; i++) {
        padded_input[i] = time_series[i];
    }
    CHECK_CUDA(cudaMemcpy(d_input, padded_input.data(), dft_size * sizeof(cufftReal), cudaMemcpyHostToDevice));
    
    // Create cuFFT plans
    cufftHandle forward_plan, inverse_plan;
    CHECK_CUFFT(cufftPlan1d(&forward_plan, dft_size, CUFFT_R2C, 1));
    CHECK_CUFFT(cufftPlan1d(&inverse_plan, dft_size, CUFFT_C2R, 1));
    
    // Forward FFT
    CHECK_CUFFT(cufftExecR2C(forward_plan, d_input, d_fft));
    
    // Apply frequency mask on device
    std::vector<cufftComplex> h_fft(dft_size/2 + 1);
    CHECK_CUDA(cudaMemcpy(h_fft.data(), d_fft, (dft_size/2 + 1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < dft_size/2 + 1; i++) {
        float freq = (float)i * fps / (float)dft_size;
        if (freq < fl || freq > fh) {
            h_fft[i].x = 0.0f;
            h_fft[i].y = 0.0f;
        }
    }
    
    CHECK_CUDA(cudaMemcpy(d_fft, h_fft.data(), (dft_size/2 + 1) * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    
    // Inverse FFT
    CHECK_CUFFT(cufftExecC2R(inverse_plan, d_fft, d_output));
    
    // Copy result back and apply normalization
    std::vector<float> padded_output(dft_size);
    CHECK_CUDA(cudaMemcpy(padded_output.data(), d_output, dft_size * sizeof(cufftReal), cudaMemcpyDeviceToHost));
    
    std::vector<float> result(num_frames);
    for (int i = 0; i < num_frames; i++) {
        result[i] = padded_output[i] / (float)dft_size; // Apply normalization only
    }
    
    // Cleanup
    CHECK_CUFFT(cufftDestroy(forward_plan));
    CHECK_CUFFT(cufftDestroy(inverse_plan));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_fft));
    CHECK_CUDA(cudaFree(d_output));
    
    return result;
}

// Helper function to calculate PSNR between two signals
double calculate_signal_psnr(const std::vector<float>& signal1, const std::vector<float>& signal2) {
    if (signal1.size() != signal2.size()) return -1.0;
    
    double mse = 0.0;
    double max_val = 0.0;
    
    for (size_t i = 0; i < signal1.size(); i++) {
        double diff = signal1[i] - signal2[i];
        mse += diff * diff;
        max_val = std::max(max_val, std::max((double)std::abs(signal1[i]), (double)std::abs(signal2[i])));
    }
    
    mse /= signal1.size();
    if (mse == 0.0) return 100.0; // Perfect match
    
    return 20.0 * log10(max_val / sqrt(mse));
}

// Helper function to print signal comparison
void print_signal_comparison(const std::vector<float>& cpu_signal, const std::vector<float>& cuda_signal, const std::string& name) {
    double psnr = calculate_signal_psnr(cpu_signal, cuda_signal);
    
    std::cout << "\n=== " << name << " ===" << std::endl;
    std::cout << "Signal length: " << cpu_signal.size() << std::endl;
    std::cout << "PSNR: " << std::fixed << std::setprecision(2) << psnr << " dB" << std::endl;
    
    // Print first few values for inspection
    std::cout << "First 10 values comparison:" << std::endl;
    std::cout << "Index\tCPU\t\tCUDA\t\tDiff" << std::endl;
    for (int i = 0; i < std::min(10, (int)cpu_signal.size()); i++) {
        float diff = cpu_signal[i] - cuda_signal[i];
        std::cout << i << "\t" << std::fixed << std::setprecision(6) 
                 << cpu_signal[i] << "\t" << cuda_signal[i] << "\t" << diff << std::endl;
    }
    
    if (psnr < 30.0) {
        std::cout << "❌ ACCURACY ISSUE: PSNR " << psnr << " dB < 30 dB target" << std::endl;
    } else {
        std::cout << "✅ ACCURACY OK: PSNR " << psnr << " dB >= 30 dB target" << std::endl;
    }
}

int main() {
    std::cout << "=== cuFFT Step-by-Step Validation Test ===" << std::endl;
    std::cout << "Testing each stage of cuFFT pipeline against CPU reference" << std::endl;
    
    // Initialize CUDA modules
    if (!evmcuda::init_color_conversion()) {
        std::cerr << "Failed to initialize color conversion" << std::endl;
        return 1;
    }
    
    if (!evmcuda::init_pyramid()) {
        std::cerr << "Failed to initialize pyramid" << std::endl;
        return 1;
    }
    
    // Test parameters
    const float fps = 30.0f;
    const float fl = 0.8f;  
    const float fh = 1.0f;
    const int num_frames = 100;  // Smaller test for detailed analysis
    
    std::cout << "\nTest Parameters:" << std::endl;
    std::cout << "FPS: " << fps << " Hz" << std::endl;
    std::cout << "Frequency Range: " << fl << " - " << fh << " Hz" << std::endl;
    std::cout << "Frames: " << num_frames << std::endl;
    
    // === STEP 1: Test Single Pixel Time Series ===
    std::cout << "\n=== STEP 1: Single Pixel Time Series Validation ===" << std::endl;
    
    // Create a test time series (simulated pulse signal)
    std::vector<float> test_time_series(num_frames);
    for (int t = 0; t < num_frames; t++) {
        // Simulate a pulse at 0.9 Hz (within passband)
        float time = t / fps;
        test_time_series[t] = 100.0f + 10.0f * sin(2.0f * M_PI * 0.9f * time);
    }
    
    // Process with CPU reference
    std::vector<float> cpu_result = cpu_temporal_filter_reference(test_time_series, fps, fl, fh);
    
    // Process with CUDA cuFFT
    std::vector<float> cuda_result = cuda_cufft_single_series(test_time_series, fps, fl, fh);
    
    print_signal_comparison(cpu_result, cuda_result, "Single Pixel Time Series");
    
    // === STEP 2: Test Multiple Pixel Time Series ===
    std::cout << "\n=== STEP 2: Multiple Pixel Time Series Validation ===" << std::endl;
    
    const int num_test_pixels = 5;
    std::vector<std::vector<float>> test_pixels(num_test_pixels);
    std::vector<std::vector<float>> cpu_results(num_test_pixels);
    std::vector<std::vector<float>> cuda_results(num_test_pixels);
    
    // Create different test signals for each pixel
    for (int p = 0; p < num_test_pixels; p++) {
        test_pixels[p].resize(num_frames);
        for (int t = 0; t < num_frames; t++) {
            float time = t / fps;
            // Different frequency for each pixel
            float freq = 0.8f + p * 0.05f; // 0.8, 0.85, 0.9, 0.95, 1.0 Hz
            test_pixels[p][t] = 100.0f + (10.0f + p) * sin(2.0f * M_PI * freq * time);
        }
        
        cpu_results[p] = cpu_temporal_filter_reference(test_pixels[p], fps, fl, fh);
        cuda_results[p] = cuda_cufft_single_series(test_pixels[p], fps, fl, fh);
        
        std::cout << "Pixel " << p << " (freq=" << (0.8f + p * 0.05f) << " Hz): ";
        double psnr = calculate_signal_psnr(cpu_results[p], cuda_results[p]);
        std::cout << "PSNR = " << std::fixed << std::setprecision(2) << psnr << " dB";
        if (psnr < 30.0) std::cout << " ❌"; else std::cout << " ✅";
        std::cout << std::endl;
    }
    
    // === STEP 3: Frequency Response Analysis ===
    std::cout << "\n=== STEP 3: Frequency Response Analysis ===" << std::endl;
    
    // Test pure frequency signals across the spectrum
    std::vector<float> test_frequencies = {0.5f, 0.8f, 0.9f, 1.0f, 1.2f}; // Some in, some out of passband
    
    for (float test_freq : test_frequencies) {
        std::vector<float> pure_signal(num_frames);
        for (int t = 0; t < num_frames; t++) {
            float time = t / fps;
            pure_signal[t] = sin(2.0f * M_PI * test_freq * time);
        }
        
        auto cpu_filtered = cpu_temporal_filter_reference(pure_signal, fps, fl, fh);
        auto cuda_filtered = cuda_cufft_single_series(pure_signal, fps, fl, fh);
        
        // Calculate RMS power of filtered signals
        float cpu_rms = 0.0f, cuda_rms = 0.0f;
        for (int i = 0; i < num_frames; i++) {
            cpu_rms += cpu_filtered[i] * cpu_filtered[i];
            cuda_rms += cuda_filtered[i] * cuda_filtered[i];
        }
        cpu_rms = sqrt(cpu_rms / num_frames);
        cuda_rms = sqrt(cuda_rms / num_frames);
        
        bool should_pass = (test_freq >= fl && test_freq <= fh);
        std::cout << "Freq " << test_freq << " Hz: CPU_RMS=" << std::fixed << std::setprecision(6) 
                 << cpu_rms << ", CUDA_RMS=" << cuda_rms 
                 << " (should " << (should_pass ? "pass" : "block") << ")";
        
        double psnr = calculate_signal_psnr(cpu_filtered, cuda_filtered);
        std::cout << " PSNR=" << std::fixed << std::setprecision(2) << psnr << " dB";
        if (psnr < 30.0) std::cout << " ❌"; else std::cout << " ✅";
        std::cout << std::endl;
    }
    
    // === SUMMARY ===
    std::cout << "\n=== VALIDATION SUMMARY ===" << std::endl;
    std::cout << "This test validates individual cuFFT operations against CPU reference." << std::endl;
    std::cout << "If single-pixel tests pass but full pipeline fails, the issue is likely in:" << std::endl;
    std::cout << "1. Data reorganization kernels (frame-major ↔ pixel-major)" << std::endl;
    std::cout << "2. Batched cuFFT plan configuration" << std::endl;
    std::cout << "3. Memory layout/indexing in helper kernels" << std::endl;
    
    // Cleanup
    evmcuda::cleanup_pyramid();
    evmcuda::cleanup_color_conversion();
    
    return 0;
}