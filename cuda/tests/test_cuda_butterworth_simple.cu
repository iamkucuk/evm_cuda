#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>
#include <cuda_runtime.h>

// Include CUDA implementation
#include "../include/cuda_butterworth.cuh"

// Test utilities
struct TestResult {
    bool passed;
    double error;
    std::string message;
};

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

TestResult testCoefficientCalculation() {
    std::cout << "Testing CUDA coefficient calculation against reference..." << std::endl;
    
    try {
        // Test parameters that should match reference data (from CPU test)
        int order = 1;
        double cutoff_freq = 0.4; // low_cutoff from CPU test
        std::string btype = "low";
        double fs = 30.0; // test_fs from CPU test
        
        // Test both low-pass and high-pass coefficients
        bool all_coeff_tests_passed = true;
        double total_error = 0.0;
        
        // Test 1: Low-pass coefficients
        auto ref_a_low = loadReferenceCoeffs("../../cpp/tests/data/butter_low_a.txt");
        auto ref_b_low = loadReferenceCoeffs("../../cpp/tests/data/butter_low_b.txt");
        auto cuda_coeffs_low = cuda_evm::calculateButterworthCoeffs(order, cutoff_freq, btype, fs);
        
        std::cout << "LOW-PASS TEST:" << std::endl;
        std::cout << "Reference A coeffs: ";
        for (double val : ref_a_low) std::cout << val << " ";
        std::cout << std::endl;
        std::cout << "CUDA A coeffs: ";
        for (double val : cuda_coeffs_low.second) std::cout << val << " ";
        std::cout << std::endl;
        
        double low_a_error = 0.0, low_b_error = 0.0;
        for (size_t i = 0; i < ref_a_low.size() && i < cuda_coeffs_low.second.size(); ++i) {
            low_a_error += std::abs(cuda_coeffs_low.second[i] - ref_a_low[i]);
        }
        for (size_t i = 0; i < ref_b_low.size() && i < cuda_coeffs_low.first.size(); ++i) {
            low_b_error += std::abs(cuda_coeffs_low.first[i] - ref_b_low[i]);
        }
        std::cout << "Low-pass A error: " << low_a_error << ", B error: " << low_b_error << std::endl;
        
        // Test 2: High-pass coefficients (using low-pass design with high cutoff as per CPU test)
        double high_cutoff = 3.0; // from CPU test
        auto ref_a_high = loadReferenceCoeffs("../../cpp/tests/data/butter_high_a.txt");
        auto ref_b_high = loadReferenceCoeffs("../../cpp/tests/data/butter_high_b.txt");
        auto cuda_coeffs_high = cuda_evm::calculateButterworthCoeffs(order, high_cutoff, btype, fs);
        
        std::cout << "HIGH-PASS TEST:" << std::endl;
        std::cout << "Reference A coeffs: ";
        for (double val : ref_a_high) std::cout << val << " ";
        std::cout << std::endl;
        std::cout << "CUDA A coeffs: ";
        for (double val : cuda_coeffs_high.second) std::cout << val << " ";
        std::cout << std::endl;
        
        double high_a_error = 0.0, high_b_error = 0.0;
        for (size_t i = 0; i < ref_a_high.size() && i < cuda_coeffs_high.second.size(); ++i) {
            high_a_error += std::abs(cuda_coeffs_high.second[i] - ref_a_high[i]);
        }
        for (size_t i = 0; i < ref_b_high.size() && i < cuda_coeffs_high.first.size(); ++i) {
            high_b_error += std::abs(cuda_coeffs_high.first[i] - ref_b_high[i]);
        }
        std::cout << "High-pass A error: " << high_a_error << ", B error: " << high_b_error << std::endl;
        
        total_error = low_a_error + low_b_error + high_a_error + high_b_error;
        bool passed = total_error < 1e-10; // Very strict tolerance
        
        return {passed, total_error, passed ? "Coefficient calculation matches reference" : "Coefficient calculation differs from reference"};
        
    } catch (const std::exception& e) {
        return {false, -1, std::string("Exception in coefficient test: ") + e.what()};
    }
}

TestResult testFilterBasicOperation() {
    std::cout << "Testing basic CUDA filter operation..." << std::endl;
    
    try {
        // Test parameters
        int width = 32, height = 32, channels = 1; // Smaller test for simplicity
        int total_pixels = width * height * channels;
        
        // Create test input data
        std::vector<float> input_data(total_pixels);
        std::vector<float> prev_input_data(total_pixels, 0.0f);
        std::vector<float> prev_output_data(total_pixels, 0.0f);
        
        // Initialize with a simple test pattern
        for (int i = 0; i < total_pixels; ++i) {
            input_data[i] = 0.5f + 0.3f * std::sin(2.0 * M_PI * i / 10.0);
        }
        
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
        
        // Check for CUDA errors
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(cudaStatus));
        }
        
        // Copy result back to host
        std::vector<float> cuda_output_data(total_pixels);
        cudaMemcpy(cuda_output_data.data(), d_output, data_size, cudaMemcpyDeviceToHost);
        
        // Clean up CUDA memory
        cudaFree(d_input);
        cudaFree(d_prev_input);
        cudaFree(d_prev_output);
        cudaFree(d_output);
        
        // Basic sanity checks
        bool has_nan = false;
        bool has_inf = false;
        double mean_output = 0.0;
        
        for (float val : cuda_output_data) {
            if (std::isnan(val)) has_nan = true;
            if (std::isinf(val)) has_inf = true;
            mean_output += val;
        }
        mean_output /= total_pixels;
        
        std::cout << "Filter output statistics:" << std::endl;
        std::cout << "  Mean output: " << mean_output << std::endl;
        std::cout << "  Has NaN: " << (has_nan ? "YES" : "NO") << std::endl;
        std::cout << "  Has Inf: " << (has_inf ? "YES" : "NO") << std::endl;
        
        bool passed = !has_nan && !has_inf && std::abs(mean_output) < 10.0; // Reasonable output
        
        return {passed, mean_output, passed ? "Filter operation completed successfully" : "Filter operation produced invalid results"};
        
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
    if (coeff_result.error >= 0) {
        std::cout << "  Error: " << std::scientific << coeff_result.error << std::endl;
    }
    all_passed &= coeff_result.passed;
    
    std::cout << std::endl;
    
    // Test 2: Basic filter operation
    auto filter_result = testFilterBasicOperation();
    std::cout << "Filter Test: " << (filter_result.passed ? "PASSED" : "FAILED") 
              << " - " << filter_result.message << std::endl;
    all_passed &= filter_result.passed;
    
    std::cout << "\n=== Overall Result ===" << std::endl;
    if (all_passed) {
        std::cout << "ALL TESTS PASSED - CUDA Butterworth implementation is functional" << std::endl;
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED - CUDA Butterworth implementation needs fixes" << std::endl;
        return -1;
    }
}