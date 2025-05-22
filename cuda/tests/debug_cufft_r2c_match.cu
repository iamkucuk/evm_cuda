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

// CPU reference using R2C-equivalent processing
std::vector<float> cpu_r2c_equivalent(
    const std::vector<float>& time_series,
    float fps, float fl, float fh
) {
    int num_frames = time_series.size();
    int dft_size = 1;
    while (dft_size < num_frames) dft_size <<= 1;
    
    // Create OpenCV Mat for time series with zero padding
    cv::Mat signal(dft_size, 1, CV_32F, cv::Scalar(0));
    for (int i = 0; i < num_frames; i++) {
        signal.at<float>(i) = time_series[i];
    }
    
    // Convert to complex and do full DFT
    cv::Mat complex_signal;
    cv::Mat planes[] = {signal, cv::Mat::zeros(signal.size(), CV_32F)};
    cv::merge(planes, 2, complex_signal);
    cv::dft(complex_signal, complex_signal);
    
    // Extract R2C equivalent format (only positive frequencies)
    cv::split(complex_signal, planes);
    std::vector<cv::Vec2f> r2c_format(dft_size/2 + 1);
    
    for (int i = 0; i <= dft_size/2; i++) {
        r2c_format[i][0] = planes[0].at<float>(i); // Real
        r2c_format[i][1] = planes[1].at<float>(i); // Imag
    }
    
    std::cout << "CPU R2C-equivalent FFT coefficients:" << std::endl;
    for (int i = 0; i <= dft_size/2; i++) {
        float freq = (float)i * fps / (float)dft_size;
        std::cout << "bin[" << i << "] freq=" << std::fixed << std::setprecision(3) << freq 
                 << " Hz: " << std::setprecision(6) << r2c_format[i][0] << " + " << r2c_format[i][1] << "i" << std::endl;
    }
    
    // Apply frequency mask to R2C format
    for (int i = 0; i <= dft_size/2; i++) {
        float freq = (float)i * fps / (float)dft_size;
        if (freq < fl || freq > fh) {
            r2c_format[i][0] = 0.0f;
            r2c_format[i][1] = 0.0f;
        }
    }
    
    std::cout << "\nCPU R2C-equivalent FFT coefficients (after filter):" << std::endl;
    for (int i = 0; i <= dft_size/2; i++) {
        float freq = (float)i * fps / (float)dft_size;
        std::cout << "bin[" << i << "] freq=" << std::fixed << std::setprecision(3) << freq 
                 << " Hz: " << std::setprecision(6) << r2c_format[i][0] << " + " << r2c_format[i][1] << "i"
                 << (freq >= fl && freq <= fh ? " (PASS)" : " (BLOCK)") << std::endl;
    }
    
    // Reconstruct full complex spectrum from R2C format (Hermitian symmetry)
    for (int i = 0; i <= dft_size/2; i++) {
        planes[0].at<float>(i) = r2c_format[i][0];
        planes[1].at<float>(i) = r2c_format[i][1];
    }
    
    // Fill negative frequencies (Hermitian symmetry)
    for (int i = 1; i < dft_size/2; i++) {
        int neg_idx = dft_size - i;
        planes[0].at<float>(neg_idx) = r2c_format[i][0];  // Real part same
        planes[1].at<float>(neg_idx) = -r2c_format[i][1]; // Imaginary part conjugated
    }
    
    cv::merge(planes, 2, complex_signal);
    
    // Inverse DFT with scaling
    cv::idft(complex_signal, complex_signal, cv::DFT_SCALE);
    cv::split(complex_signal, planes);
    
    std::vector<float> result(num_frames);
    for (int i = 0; i < num_frames; i++) {
        result[i] = planes[0].at<float>(i);
    }
    
    return result;
}

int main() {
    std::cout << "=== R2C Format Matching Test ===" << std::endl;
    
    const float fps = 30.0f;
    const float fl = 0.8f;
    const float fh = 1.0f;
    const int num_frames = 16;  // Small for detailed analysis
    
    // Create test signal: 1.2 Hz (should be blocked)
    std::vector<float> test_signal(num_frames);
    for (int i = 0; i < num_frames; i++) {
        float time = i / fps;
        test_signal[i] = sin(2.0f * M_PI * 1.2f * time);
    }
    
    std::cout << "Test signal (1.2 Hz, should be BLOCKED):" << std::endl;
    for (int i = 0; i < num_frames; i++) {
        std::cout << i << ": " << std::fixed << std::setprecision(6) << test_signal[i] << std::endl;
    }
    
    // Process with CPU R2C-equivalent
    std::vector<float> cpu_result = cpu_r2c_equivalent(test_signal, fps, fl, fh);
    
    std::cout << "\nCPU R2C-equivalent result:" << std::endl;
    for (int i = 0; i < num_frames; i++) {
        std::cout << i << ": " << std::fixed << std::setprecision(6) << cpu_result[i] << std::endl;
    }
    
    // Calculate RMS of result
    float cpu_rms = 0.0f;
    for (int i = 0; i < num_frames; i++) {
        cpu_rms += cpu_result[i] * cpu_result[i];
    }
    cpu_rms = sqrt(cpu_rms / num_frames);
    
    std::cout << "\nCPU RMS power: " << std::fixed << std::setprecision(6) << cpu_rms << std::endl;
    std::cout << "Expected: Near 0 (since 1.2 Hz should be blocked)" << std::endl;
    
    if (cpu_rms < 0.001) {
        std::cout << "✅ CPU correctly blocks 1.2 Hz" << std::endl;
    } else {
        std::cout << "❌ CPU not blocking properly" << std::endl;
    }
    
    return 0;
}