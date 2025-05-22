#include <cuda_runtime.h>
#include <cufft.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)

#define CHECK_CUFFT(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            std::cerr << "cuFFT error: " << err << std::endl; \
            exit(1); \
        } \
    } while (0)

int main() {
    std::cout << "=== Minimal cuFFT Debug Test ===" << std::endl;
    
    // Test parameters
    const float fps = 30.0f;
    const float fl = 0.8f;  
    const float fh = 1.0f;
    const int num_frames = 16;  // Small power of 2 for easy debugging
    const int dft_size = 16;    // Already power of 2
    
    std::cout << "Parameters: fps=" << fps << ", range=" << fl << "-" << fh << " Hz, frames=" << num_frames << std::endl;
    
    // Create test signal: pure 0.9 Hz sine wave (should pass through filter)
    std::vector<float> input_signal(num_frames);
    for (int i = 0; i < num_frames; i++) {
        float time = i / fps;
        input_signal[i] = sin(2.0f * M_PI * 0.9f * time);
    }
    
    std::cout << "\nInput signal (0.9 Hz):" << std::endl;
    for (int i = 0; i < num_frames; i++) {
        std::cout << i << ": " << std::fixed << std::setprecision(6) << input_signal[i] << std::endl;
    }
    
    // === CPU REFERENCE using OpenCV ===
    cv::Mat signal_mat(num_frames, 1, CV_32F);
    for (int i = 0; i < num_frames; i++) {
        signal_mat.at<float>(i) = input_signal[i];
    }
    
    // OpenCV DFT with zero-padding
    cv::Mat padded_signal;
    cv::copyMakeBorder(signal_mat, padded_signal, 0, dft_size - num_frames, 0, 0, cv::BORDER_CONSTANT, 0);
    
    cv::Mat complex_signal;
    cv::Mat planes[] = {padded_signal, cv::Mat::zeros(padded_signal.size(), CV_32F)};
    cv::merge(planes, 2, complex_signal);
    
    cv::dft(complex_signal, complex_signal);
    
    // Print FFT coefficients before filtering
    cv::split(complex_signal, planes);
    std::cout << "\nCPU FFT coefficients (before filter):" << std::endl;
    for (int i = 0; i < dft_size; i++) {
        float real = planes[0].at<float>(i);
        float imag = planes[1].at<float>(i);
        float freq = i * fps / dft_size;
        if (i > dft_size/2) freq = (i - dft_size) * fps / dft_size;
        std::cout << "bin[" << i << "] freq=" << std::fixed << std::setprecision(3) << freq 
                 << " Hz: " << std::setprecision(6) << real << " + " << imag << "i" << std::endl;
    }
    
    // Apply bandpass filter
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
    
    // Print FFT coefficients after filtering
    cv::split(complex_signal, planes);
    std::cout << "\nCPU FFT coefficients (after filter):" << std::endl;
    for (int i = 0; i < dft_size; i++) {
        float real = planes[0].at<float>(i);
        float imag = planes[1].at<float>(i);
        float freq = i * fps / dft_size;
        if (i > dft_size/2) freq = (i - dft_size) * fps / dft_size;
        std::cout << "bin[" << i << "] freq=" << std::fixed << std::setprecision(3) << freq 
                 << " Hz: " << std::setprecision(6) << real << " + " << imag << "i" << std::endl;
    }
    
    // Inverse DFT
    cv::idft(complex_signal, complex_signal, cv::DFT_SCALE);
    cv::split(complex_signal, planes);
    
    std::vector<float> cpu_result(num_frames);
    for (int i = 0; i < num_frames; i++) {
        cpu_result[i] = planes[0].at<float>(i);
    }
    
    std::cout << "\nCPU result:" << std::endl;
    for (int i = 0; i < num_frames; i++) {
        std::cout << i << ": " << std::fixed << std::setprecision(6) << cpu_result[i] << std::endl;
    }
    
    // === CUDA cuFFT Implementation ===
    
    // Allocate device memory
    cufftReal* d_input;
    cufftComplex* d_fft;
    cufftReal* d_output;
    
    CHECK_CUDA(cudaMalloc(&d_input, dft_size * sizeof(cufftReal)));
    CHECK_CUDA(cudaMalloc(&d_fft, (dft_size/2 + 1) * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_output, dft_size * sizeof(cufftReal)));
    
    // Copy input data with zero-padding
    std::vector<float> padded_input(dft_size, 0.0f);
    for (int i = 0; i < num_frames; i++) {
        padded_input[i] = input_signal[i];
    }
    CHECK_CUDA(cudaMemcpy(d_input, padded_input.data(), dft_size * sizeof(cufftReal), cudaMemcpyHostToDevice));
    
    // Create cuFFT plans
    cufftHandle forward_plan, inverse_plan;
    CHECK_CUFFT(cufftPlan1d(&forward_plan, dft_size, CUFFT_R2C, 1));
    CHECK_CUFFT(cufftPlan1d(&inverse_plan, dft_size, CUFFT_C2R, 1));
    
    // Forward FFT
    CHECK_CUFFT(cufftExecR2C(forward_plan, d_input, d_fft));
    
    // Copy FFT result to host for inspection
    std::vector<cufftComplex> cuda_fft_before(dft_size/2 + 1);
    CHECK_CUDA(cudaMemcpy(cuda_fft_before.data(), d_fft, (dft_size/2 + 1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    
    std::cout << "\nCUDA FFT coefficients (before filter, R2C format):" << std::endl;
    for (int i = 0; i < dft_size/2 + 1; i++) {
        float freq = (float)i * fps / (float)dft_size;
        std::cout << "bin[" << i << "] freq=" << std::fixed << std::setprecision(3) << freq 
                 << " Hz: " << std::setprecision(6) << cuda_fft_before[i].x << " + " << cuda_fft_before[i].y << "i" << std::endl;
    }
    
    // Apply frequency mask
    std::vector<cufftComplex> cuda_fft_after = cuda_fft_before;
    for (int i = 0; i < dft_size/2 + 1; i++) {
        float freq = (float)i * fps / (float)dft_size;
        if (freq < fl || freq > fh) {
            cuda_fft_after[i].x = 0.0f;
            cuda_fft_after[i].y = 0.0f;
        }
    }
    
    std::cout << "\nCUDA FFT coefficients (after filter):" << std::endl;
    for (int i = 0; i < dft_size/2 + 1; i++) {
        float freq = (float)i * fps / (float)dft_size;
        std::cout << "bin[" << i << "] freq=" << std::fixed << std::setprecision(3) << freq 
                 << " Hz: " << std::setprecision(6) << cuda_fft_after[i].x << " + " << cuda_fft_after[i].y << "i" 
                 << (freq >= fl && freq <= fh ? " (PASS)" : " (BLOCK)") << std::endl;
    }
    
    // Copy filtered FFT back to device
    CHECK_CUDA(cudaMemcpy(d_fft, cuda_fft_after.data(), (dft_size/2 + 1) * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    
    // Inverse FFT
    CHECK_CUFFT(cufftExecC2R(inverse_plan, d_fft, d_output));
    
    // Copy result back and apply normalization
    std::vector<float> cuda_output_raw(dft_size);
    CHECK_CUDA(cudaMemcpy(cuda_output_raw.data(), d_output, dft_size * sizeof(cufftReal), cudaMemcpyDeviceToHost));
    
    std::cout << "\nCUDA result (before normalization):" << std::endl;
    for (int i = 0; i < num_frames; i++) {
        std::cout << i << ": " << std::fixed << std::setprecision(6) << cuda_output_raw[i] << std::endl;
    }
    
    std::vector<float> cuda_result(num_frames);
    for (int i = 0; i < num_frames; i++) {
        cuda_result[i] = cuda_output_raw[i] / (float)dft_size; // Apply normalization
    }
    
    std::cout << "\nCUDA result (after normalization):" << std::endl;
    for (int i = 0; i < num_frames; i++) {
        std::cout << i << ": " << std::fixed << std::setprecision(6) << cuda_result[i] << std::endl;
    }
    
    // Compare results
    std::cout << "\n=== COMPARISON ===" << std::endl;
    std::cout << "Index\tCPU\t\tCUDA\t\tDiff" << std::endl;
    double mse = 0.0;
    for (int i = 0; i < num_frames; i++) {
        float diff = cpu_result[i] - cuda_result[i];
        mse += diff * diff;
        std::cout << i << "\t" << std::fixed << std::setprecision(6) 
                 << cpu_result[i] << "\t" << cuda_result[i] << "\t" << diff << std::endl;
    }
    mse /= num_frames;
    double psnr = 20.0 * log10(1.0 / sqrt(mse)); // Assuming signal range [-1,1]
    std::cout << "PSNR: " << std::fixed << std::setprecision(2) << psnr << " dB" << std::endl;
    
    // Cleanup
    CHECK_CUFFT(cufftDestroy(forward_plan));
    CHECK_CUFFT(cufftDestroy(inverse_plan));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_fft));
    CHECK_CUDA(cudaFree(d_output));
    
    return 0;
}