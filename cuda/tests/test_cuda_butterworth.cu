#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cuda_runtime.h>
#include "cuda_butterworth.cuh"

namespace {

// Helper function to read test data from CSV file (single line)
std::vector<double> readTestDataVector(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};
    }
    
    std::vector<double> data;
    std::string line;
    
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value_str;
        
        while (std::getline(ss, value_str, ',')) {
            try {
                // Parse value (removing any whitespace)
                double value = std::stod(value_str);
                data.push_back(value);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing value: " << value_str << ": " << e.what() << std::endl;
            }
        }
    }
    
    return data;
}

// Helper function to compare results with tolerance
void compareVectors(const std::vector<double>& expected, const std::vector<double>& actual, 
                    const std::string& name, double epsilon = 1e-6) {
    if (expected.size() != actual.size()) {
        std::cerr << "Size mismatch for " << name << ": Expected = " << expected.size() 
                  << ", Actual = " << actual.size() << std::endl;
        std::cerr << "Test FAILED" << std::endl;
        return;
    }
    
    double maxError = 0.0;
    double meanError = 0.0;
    int maxErrorIdx = -1;
    
    for (size_t i = 0; i < expected.size(); ++i) {
        double error = std::abs(expected[i] - actual[i]);
        if (error > maxError) {
            maxError = error;
            maxErrorIdx = i;
        }
        meanError += error;
    }
    
    meanError /= expected.size();
    
    std::cout << "Comparison results for " << name << ":" << std::endl;
    std::cout << "  Data size: " << expected.size() << std::endl;
    std::cout << "  Max error: " << maxError << " at index " << maxErrorIdx;
    if (maxErrorIdx >= 0) {
        std::cout << " (Expected: " << expected[maxErrorIdx] 
                  << ", Actual: " << actual[maxErrorIdx] << ")";
    }
    std::cout << std::endl;
    std::cout << "  Mean error: " << meanError << std::endl;
    std::cout << "  Validation " << (maxError <= epsilon ? "PASSED" : "FAILED") 
              << " (epsilon = " << epsilon << ")" << std::endl;
}

// Helper function to print vector contents
void printVector(const std::vector<double>& vec, const std::string& name) {
    std::cout << name << ": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

} // namespace

// Test functions for Butterworth coefficient calculation
bool testLowPassCoefficients(const std::string& expectedBFile, const std::string& expectedAFile) {
    std::cout << "\nTesting Butterworth Low-Pass Coefficient Calculation" << std::endl;
    
    // Load expected coefficients
    std::vector<double> expectedB = readTestDataVector(expectedBFile);
    std::vector<double> expectedA = readTestDataVector(expectedAFile);
    
    if (expectedB.empty() || expectedA.empty()) {
        std::cerr << "Failed to load expected coefficient data" << std::endl;
        return false;
    }
    
    // Calculate coefficients using CUDA implementation
    const int order = 1;
    const double cutoff = 0.4; // Matches the test data
    const double fs = 30.0;    // Matches the test data
    
    try {
        auto coeffs = evmcuda::calculate_butterworth_coeffs(order, cutoff, "low", fs);
        std::vector<double> actualB = coeffs.first;
        std::vector<double> actualA = coeffs.second;
        
        // Compare with expected values
        printVector(expectedB, "Expected B");
        printVector(actualB, "Actual B");
        printVector(expectedA, "Expected A");
        printVector(actualA, "Actual A");
        
        compareVectors(expectedB, actualB, "Numerator (B) Coefficients");
        compareVectors(expectedA, actualA, "Denominator (A) Coefficients");
        
        // Return true if both comparisons pass
        return (std::abs(expectedB[0] - actualB[0]) <= 1e-6 &&
                std::abs(expectedB[1] - actualB[1]) <= 1e-6 &&
                std::abs(expectedA[0] - actualA[0]) <= 1e-6 &&
                std::abs(expectedA[1] - actualA[1]) <= 1e-6);
    } catch (const std::exception& e) {
        std::cerr << "Error in CUDA coefficient calculation: " << e.what() << std::endl;
        return false;
    }
}

bool testHighPassCoefficients(const std::string& expectedBFile, const std::string& expectedAFile) {
    std::cout << "\nTesting Butterworth High-Pass Coefficient Calculation (using low-pass design)" << std::endl;
    
    // Load expected coefficients
    std::vector<double> expectedB = readTestDataVector(expectedBFile);
    std::vector<double> expectedA = readTestDataVector(expectedAFile);
    
    if (expectedB.empty() || expectedA.empty()) {
        std::cerr << "Failed to load expected coefficient data" << std::endl;
        return false;
    }
    
    // Calculate coefficients using CUDA implementation
    // Note: The CPU test uses a low-pass design with the high cutoff, so we match that here
    const int order = 1;
    const double cutoff = 3.0; // Matches the test data
    const double fs = 30.0;    // Matches the test data
    
    try {
        auto coeffs = evmcuda::calculate_butterworth_coeffs(order, cutoff, "low", fs);
        std::vector<double> actualB = coeffs.first;
        std::vector<double> actualA = coeffs.second;
        
        // Compare with expected values
        printVector(expectedB, "Expected B");
        printVector(actualB, "Actual B");
        printVector(expectedA, "Expected A");
        printVector(actualA, "Actual A");
        
        compareVectors(expectedB, actualB, "Numerator (B) Coefficients");
        compareVectors(expectedA, actualA, "Denominator (A) Coefficients");
        
        // Return true if both comparisons pass
        return (std::abs(expectedB[0] - actualB[0]) <= 1e-6 &&
                std::abs(expectedB[1] - actualB[1]) <= 1e-6 &&
                std::abs(expectedA[0] - actualA[0]) <= 1e-6 &&
                std::abs(expectedA[1] - actualA[1]) <= 1e-6);
    } catch (const std::exception& e) {
        std::cerr << "Error in CUDA coefficient calculation: " << e.what() << std::endl;
        return false;
    }
}

// Test function for Butterworth filter application
bool testButterworthFilter() {
    std::cout << "\nTesting Butterworth Filter Application" << std::endl;
    
    // Create a simple test signal (10x10 image with a step function)
    const int width = 10;
    const int height = 10;
    const int channels = 3;
    const int numElements = width * height * channels;
    
    // Create host arrays
    std::vector<float> h_input(numElements, 0.0f);
    std::vector<float> h_prev_input(numElements, 0.0f);
    std::vector<float> h_prev_output(numElements, 0.0f);
    std::vector<float> h_output(numElements, 0.0f);
    std::vector<float> h_expected_output(numElements, 0.0f);
    
    // Initialize input with a step function (half 0s, half 1s)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                int idx = (y * width + x) * channels + c;
                h_input[idx] = (x >= width/2) ? 1.0f : 0.0f;
            }
        }
    }
    
    // Allocate device memory
    float *d_input = nullptr;
    float *d_prev_input = nullptr;
    float *d_prev_output = nullptr;
    float *d_output = nullptr;
    
    cudaError_t err = cudaMalloc(&d_input, numElements * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for input: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_prev_input, numElements * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_input);
        std::cerr << "Failed to allocate device memory for prev_input: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_prev_output, numElements * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_prev_input);
        std::cerr << "Failed to allocate device memory for prev_output: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_output, numElements * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_prev_input);
        cudaFree(d_prev_output);
        std::cerr << "Failed to allocate device memory for output: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Copy data to device
    err = cudaMemcpy(d_input, h_input.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy input to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_prev_input);
        cudaFree(d_prev_output);
        cudaFree(d_output);
        return false;
    }
    
    err = cudaMemcpy(d_prev_input, h_prev_input.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy prev_input to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_prev_input);
        cudaFree(d_prev_output);
        cudaFree(d_output);
        return false;
    }
    
    err = cudaMemcpy(d_prev_output, h_prev_output.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy prev_output to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_prev_input);
        cudaFree(d_prev_output);
        cudaFree(d_output);
        return false;
    }
    
    // Create Butterworth filter
    try {
        // Butterworth filter for test (0.05 - 0.4 Hz band)
        evmcuda::Butterworth butterFilter(0.05, 0.4);
        
        // Apply the filter
        butterFilter.filter(d_input, width, height, channels, d_prev_input, d_prev_output, d_output);
        
        // Synchronize to ensure the kernel has completed
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "Error during filter kernel execution: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_input);
            cudaFree(d_prev_input);
            cudaFree(d_prev_output);
            cudaFree(d_output);
            return false;
        }
        
        // Copy result back to host
        err = cudaMemcpy(h_output.data(), d_output, numElements * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy output from device: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_input);
            cudaFree(d_prev_input);
            cudaFree(d_prev_output);
            cudaFree(d_output);
            return false;
        }
        
        // Generate expected output (calculated using the filter coefficients)
        const auto& b_coeffs = butterFilter.get_b_coeffs();
        const auto& a_coeffs = butterFilter.get_a_coeffs();
        
        if (b_coeffs.size() != 2 || a_coeffs.size() != 2) {
            std::cerr << "Unexpected coefficient sizes. b_size=" << b_coeffs.size() 
                      << ", a_size=" << a_coeffs.size() << std::endl;
            cudaFree(d_input);
            cudaFree(d_prev_input);
            cudaFree(d_prev_output);
            cudaFree(d_output);
            return false;
        }
        
        // Calculate expected output manually using the IIR filter equation
        float b0 = static_cast<float>(b_coeffs[0]);
        float b1 = static_cast<float>(b_coeffs[1]);
        float a1 = static_cast<float>(a_coeffs[1]);
        
        for (int i = 0; i < numElements; ++i) {
            h_expected_output[i] = b0 * h_input[i] + b1 * h_prev_input[i] - a1 * h_prev_output[i];
        }
        
        // Compare the calculated output with the expected output
        float maxError = 0.0f;
        float meanError = 0.0f;
        int maxErrorIdx = -1;
        
        for (int i = 0; i < numElements; ++i) {
            float error = std::abs(h_expected_output[i] - h_output[i]);
            if (error > maxError) {
                maxError = error;
                maxErrorIdx = i;
            }
            meanError += error;
        }
        
        meanError /= numElements;
        
        std::cout << "Filter application comparison results:" << std::endl;
        std::cout << "  Max error: " << maxError;
        if (maxErrorIdx >= 0) {
            std::cout << " at index " << maxErrorIdx 
                      << " (Expected: " << h_expected_output[maxErrorIdx] 
                      << ", Actual: " << h_output[maxErrorIdx] << ")";
        }
        std::cout << std::endl;
        std::cout << "  Mean error: " << meanError << std::endl;
        
        // Cleanup
        cudaFree(d_input);
        cudaFree(d_prev_input);
        cudaFree(d_prev_output);
        cudaFree(d_output);
        
        // Test passes if maximum error is small
        return (maxError <= 1e-5);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in Butterworth filter test: " << e.what() << std::endl;
        cudaFree(d_input);
        cudaFree(d_prev_input);
        cudaFree(d_prev_output);
        cudaFree(d_output);
        return false;
    }
}

int main(int argc, char* argv[]) {
    // Test data paths
    std::string basePath = "../../../../cpp/tests/data/";
    if (argc > 1) {
        basePath = argv[1];
    }
    
    // File paths for test data
    std::string lowPassBFile = basePath + "butter_low_b.txt";
    std::string lowPassAFile = basePath + "butter_low_a.txt";
    std::string highPassBFile = basePath + "butter_high_b.txt";
    std::string highPassAFile = basePath + "butter_high_a.txt";
    
    std::cout << "Testing CUDA Butterworth Filter..." << std::endl;
    std::cout << "Using test data from:" << std::endl;
    std::cout << "  Low Pass B: " << lowPassBFile << std::endl;
    std::cout << "  Low Pass A: " << lowPassAFile << std::endl;
    std::cout << "  High Pass B: " << highPassBFile << std::endl;
    std::cout << "  High Pass A: " << highPassAFile << std::endl;
    
    // Initialize CUDA modules
    if (!evmcuda::init_butterworth()) {
        std::cerr << "Failed to initialize CUDA Butterworth module" << std::endl;
        return 1;
    }
    
    bool success = true;
    
    // Test Butterworth coefficient calculation
    bool lowPassCoeffsResult = testLowPassCoefficients(lowPassBFile, lowPassAFile);
    if (!lowPassCoeffsResult) {
        std::cerr << "Low-pass Butterworth coefficient test failed" << std::endl;
        success = false;
    }
    
    bool highPassCoeffsResult = testHighPassCoefficients(highPassBFile, highPassAFile);
    if (!highPassCoeffsResult) {
        std::cerr << "High-pass Butterworth coefficient test failed" << std::endl;
        success = false;
    }
    
    // Test Butterworth filter application
    bool filterResult = testButterworthFilter();
    if (!filterResult) {
        std::cerr << "Butterworth filter application test failed" << std::endl;
        success = false;
    }
    
    // Cleanup
    evmcuda::cleanup_butterworth();
    
    if (success) {
        std::cout << "\nAll Butterworth tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "\nSome Butterworth tests FAILED!" << std::endl;
        return 1;
    }
}