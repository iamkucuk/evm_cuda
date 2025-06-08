#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>
#include <cuda_runtime.h>

// Include CPU implementation
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>

// Forward declare CPU functions we need for comparison
namespace evmcpp {
    std::pair<std::vector<double>, std::vector<double>> calculateButterworthCoeffs(
        int order, double cutoff_freq, const std::string& btype, double fs);
    
    class Butterworth {
    public:
        Butterworth(double Wn_low, double Wn_high);
        cv::Mat filter(const cv::Mat& input, cv::Mat& prev_input_state, cv::Mat& prev_output_state);
    private:
        int order_;
        std::vector<double> b_coeffs_;
        std::vector<double> a_coeffs_;
    };
}

// Include CUDA implementation
#include "../include/cuda_butterworth.cuh"

// Test utilities
struct TestResult {
    bool passed;
    double error;
    std::string message;
};

double calculatePSNR(const std::vector<float>& ref, const std::vector<float>& test) {
    if (ref.size() != test.size()) {
        throw std::runtime_error("Vector sizes don't match for PSNR calculation");
    }
    
    double mse = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double diff = ref[i] - test[i];
        mse += diff * diff;
    }
    mse /= ref.size();
    
    if (mse < 1e-12) return 100.0; // Perfect match
    
    double max_val = 1.0; // Assuming normalized values
    double psnr = 20.0 * std::log10(max_val / std::sqrt(mse));
    return psnr;
}

std::vector<double> loadReferenceCoeffs(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open reference file: " + filename);
    }
    
    std::string line;
    std::getline(file, line);
    
    std::vector<double> coeffs;
    std::stringstream ss(line);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        coeffs.push_back(std::stod(token));
    }
    
    return coeffs;
}

TestResult testCoefficientCalculation() {
    std::cout << "Testing coefficient calculation..." << std::endl;
    
    try {
        // Test parameters that should match reference data
        int order = 1;
        double cutoff_freq = 0.05; // Normalized frequency
        std::string btype = "low";
        double fs = 2.0; // Normalized sampling frequency
        
        // Load reference coefficients
        auto ref_a = loadReferenceCoeffs("../../cpp/tests/data/butter_low_a.txt");
        auto ref_b = loadReferenceCoeffs("../../cpp/tests/data/butter_low_b.txt");
        
        // Calculate coefficients using CPU implementation
        auto cpu_coeffs = evmcpp::calculateButterworthCoeffs(order, cutoff_freq, btype, fs);
        
        // Calculate coefficients using CUDA implementation
        auto cuda_coeffs = cuda_evm::calculateButterworthCoeffs(order, cutoff_freq, btype, fs);
        
        // Compare CPU vs reference
        double cpu_a_error = 0.0, cpu_b_error = 0.0;
        for (size_t i = 0; i < ref_a.size(); ++i) {
            cpu_a_error += std::abs(cpu_coeffs.second[i] - ref_a[i]);
        }
        for (size_t i = 0; i < ref_b.size(); ++i) {
            cpu_b_error += std::abs(cpu_coeffs.first[i] - ref_b[i]);
        }
        
        std::cout << "CPU vs Reference A coeffs error: " << cpu_a_error << std::endl;
        std::cout << "CPU vs Reference B coeffs error: " << cpu_b_error << std::endl;
        
        // Compare CUDA vs CPU
        double cuda_a_error = 0.0, cuda_b_error = 0.0;
        for (size_t i = 0; i < cpu_coeffs.second.size(); ++i) {
            cuda_a_error += std::abs(cuda_coeffs.second[i] - cpu_coeffs.second[i]);
        }
        for (size_t i = 0; i < cpu_coeffs.first.size(); ++i) {
            cuda_b_error += std::abs(cuda_coeffs.first[i] - cpu_coeffs.first[i]);
        }
        
        std::cout << "CUDA vs CPU A coeffs error: " << cuda_a_error << std::endl;
        std::cout << "CUDA vs CPU B coeffs error: " << cuda_b_error << std::endl;
        
        double total_error = cuda_a_error + cuda_b_error;
        bool passed = total_error < 1e-10;
        
        return {passed, total_error, passed ? "Coefficient calculation matches" : "Coefficient calculation mismatch"};
        
    } catch (const std::exception& e) {
        return {false, -1, std::string("Exception in coefficient test: ") + e.what()};
    }
}

TestResult testFilterOperation() {
    std::cout << "Testing filter operation..." << std::endl;
    
    try {
        // Test parameters
        int width = 64, height = 64, channels = 3;
        int total_pixels = width * height * channels;
        
        // Create test input data
        std::vector<float> input_data(total_pixels);
        std::vector<float> prev_input_data(total_pixels, 0.0f);
        std::vector<float> prev_output_data(total_pixels, 0.0f);
        
        // Initialize with some test pattern
        for (int i = 0; i < total_pixels; ++i) {
            input_data[i] = 0.5f + 0.3f * std::sin(2.0 * M_PI * i / 50.0);
        }
        
        // Test with CPU implementation
        evmcpp::Butterworth cpu_filter(0.04, 0.4); // Wn_low, Wn_high
        
        // Convert to OpenCV Mat for CPU processing
        cv::Mat cpu_input = cv::Mat(height, width, CV_32FC3, input_data.data()).clone();
        cv::Mat cpu_prev_input = cv::Mat(height, width, CV_32FC3, prev_input_data.data()).clone();
        cv::Mat cpu_prev_output = cv::Mat(height, width, CV_32FC3, prev_output_data.data()).clone();
        
        cv::Mat cpu_output = cpu_filter.filter(cpu_input, cpu_prev_input, cpu_prev_output);
        
        // Test with CUDA implementation
        cuda_evm::CudaButterworth cuda_filter(0.04, 0.4);
        
        // Allocate CUDA memory
        float *d_input, *d_prev_input, *d_prev_output, *d_output;
        size_t data_size = total_pixels * sizeof(float);
        
        cudaMalloc(&d_input, data_size);
        cudaMalloc(&d_prev_input, data_size);
        cudaMalloc(&d_prev_output, data_size);
        cudaMalloc(&d_output, data_size);
        
        // Copy input data to device
        cudaMemcpy(d_input, input_data.data(), data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_prev_input, prev_input_data.data(), data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_prev_output, prev_output_data.data(), data_size, cudaMemcpyHostToDevice);
        
        // Apply CUDA filter
        cuda_filter.filter(d_input, d_prev_input, d_prev_output, d_output, width, height, channels);
        
        // Copy result back to host
        std::vector<float> cuda_output_data(total_pixels);
        cudaMemcpy(cuda_output_data.data(), d_output, data_size, cudaMemcpyDeviceToHost);
        
        // Clean up CUDA memory
        cudaFree(d_input);
        cudaFree(d_prev_input);
        cudaFree(d_prev_output);
        cudaFree(d_output);
        
        // Convert CPU output to vector for comparison
        std::vector<float> cpu_output_data(total_pixels);
        std::memcpy(cpu_output_data.data(), cpu_output.ptr<float>(), data_size);
        
        // Calculate PSNR
        double psnr = calculatePSNR(cpu_output_data, cuda_output_data);
        
        std::cout << "Filter operation PSNR: " << std::fixed << std::setprecision(2) << psnr << " dB" << std::endl;
        
        bool passed = psnr > 30.0; // Target PSNR threshold
        
        return {passed, psnr, passed ? "Filter operation PSNR > 30 dB" : "Filter operation PSNR < 30 dB"};
        
    } catch (const std::exception& e) {
        return {false, -1, std::string("Exception in filter test: ") + e.what()};
    }
}

int main() {
    std::cout << "=== CUDA Butterworth Filter Validation ===" << std::endl;
    
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA initialization failed!" << std::endl;
        return -1;
    }
    
    bool all_passed = true;
    
    // Test 1: Coefficient calculation
    auto coeff_result = testCoefficientCalculation();
    std::cout << "Coefficient Test: " << (coeff_result.passed ? "PASSED" : "FAILED") 
              << " - " << coeff_result.message << std::endl;
    all_passed &= coeff_result.passed;
    
    // Test 2: Filter operation
    auto filter_result = testFilterOperation();
    std::cout << "Filter Test: " << (filter_result.passed ? "PASSED" : "FAILED") 
              << " - " << filter_result.message << std::endl;
    all_passed &= filter_result.passed;
    
    std::cout << "\n=== Overall Result ===" << std::endl;
    if (all_passed) {
        std::cout << "ALL TESTS PASSED - CUDA Butterworth achieves >30 dB PSNR" << std::endl;
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED - CUDA Butterworth does not meet >30 dB PSNR target" << std::endl;
        return -1;
    }
}