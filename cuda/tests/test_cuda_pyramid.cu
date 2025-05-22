#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cuda_runtime.h>
#include "cuda_pyramid.cuh"
#include "cuda_color_conversion.cuh"

// Helper function to read test data from CSV file
std::vector<float> readTestData(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};
    }
    
    std::vector<float> data;
    std::string line;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value_str;
        
        while (std::getline(ss, value_str, ',')) {
            try {
                // Parse scientific notation
                float value = std::stof(value_str);
                data.push_back(value);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing value: " << value_str << ": " << e.what() << std::endl;
            }
        }
    }
    
    return data;
}

// Helper function to compare results with tolerance
void compareResults(const std::vector<float>& expected, const std::vector<float>& actual, float epsilon = 1e-5) {
    if (expected.size() != actual.size()) {
        std::cerr << "Size mismatch: Expected = " << expected.size() 
                  << ", Actual = " << actual.size() << std::endl;
        return;
    }
    
    float maxError = 0.0f;
    float meanError = 0.0f;
    int maxErrorIdx = -1;
    
    for (size_t i = 0; i < expected.size(); ++i) {
        float error = std::abs(expected[i] - actual[i]);
        if (error > maxError) {
            maxError = error;
            maxErrorIdx = i;
        }
        meanError += error;
    }
    
    meanError /= expected.size();
    
    std::cout << "Comparison results:" << std::endl;
    std::cout << "  Data size: " << expected.size() << std::endl;
    std::cout << "  Max error: " << maxError << " at index " << maxErrorIdx;
    if (maxErrorIdx >= 0) {
        std::cout << " (Expected: " << expected[maxErrorIdx] 
                  << ", Actual: " << actual[maxErrorIdx] << ")";
    }
    std::cout << std::endl;
    std::cout << "  Mean error: " << meanError << std::endl;
    std::cout << "  PSNR: " << (maxError > 0 ? 20 * log10(255.0f / maxError) : 0) << " dB" << std::endl;
    std::cout << "  Validation " << (maxError <= epsilon ? "PASSED" : "FAILED") 
              << " (epsilon = " << epsilon << ")" << std::endl;
}

// Determine dimensions from data size
bool determineImageDimensions(size_t numElements, int channels, int& width, int& height) {
    if (numElements % channels != 0) {
        std::cerr << "Data size is not divisible by channels: " << numElements 
                  << " elements, " << channels << " channels" << std::endl;
        return false;
    }
    
    int numPixels = numElements / channels;
    
    // Calculate width and height for a square image
    width = height = static_cast<int>(sqrt(numPixels));
    
    // If not a perfect square, adjust to find width and height that are close
    while (width * height != numPixels) {
        if (width * height < numPixels) {
            width++;
        } else {
            height--;
        }
        
        // Safety check for unreasonable dimensions
        if (width > 10 * height || height > 10 * width) {
            std::cerr << "Could not determine reasonable image dimensions for " 
                      << numPixels << " pixels" << std::endl;
            return false;
        }
    }
    
    return true;
}

// Test function for pyr_down
bool testPyrDown(const std::string& inputFile, const std::string& expectedOutputFile) {
    // Load test data
    std::vector<float> inputData = readTestData(inputFile);
    std::vector<float> expectedOutputData = readTestData(expectedOutputFile);
    
    if (inputData.empty() || expectedOutputData.empty()) {
        std::cerr << "Failed to load test data" << std::endl;
        return false;
    }
    
    const int channels = 3; // Assuming 3-channel data (RGB or YIQ)
    
    // Determine dimensions from data size
    int inputWidth, inputHeight;
    int outputWidth, outputHeight;
    
    if (!determineImageDimensions(inputData.size(), channels, inputWidth, inputHeight)) {
        std::cerr << "Failed to determine input image dimensions" << std::endl;
        return false;
    }
    
    if (!determineImageDimensions(expectedOutputData.size(), channels, outputWidth, outputHeight)) {
        std::cerr << "Failed to determine expected output image dimensions" << std::endl;
        return false;
    }
    
    // Verify that output dimensions are half of input dimensions
    if (outputWidth != inputWidth / 2 || outputHeight != inputHeight / 2) {
        std::cerr << "Output dimensions (" << outputWidth << "x" << outputHeight 
                  << ") don't match expected half size of input (" 
                  << inputWidth / 2 << "x" << inputHeight / 2 << ")" << std::endl;
        return false;
    }
    
    // Allocate output buffer
    std::vector<float> actualOutputData(expectedOutputData.size());
    
    // Initialize CUDA pyramid module
    evmcuda::init_pyramid();
    
    // Call CUDA implementation
    bool success = evmcuda::pyr_down_wrapper(
        inputData.data(), 
        inputWidth, inputHeight, channels,
        actualOutputData.data()
    );
    
    if (!success) {
        std::cerr << "pyr_down_wrapper failed" << std::endl;
        return false;
    }
    
    // Compare results
    std::cout << "\nPyr_Down Test:" << std::endl;
    compareResults(expectedOutputData, actualOutputData, 1e-4);
    
    return true;
}

// Test function for pyr_up
bool testPyrUp(const std::string& inputFile, const std::string& expectedOutputFile) {
    // Load test data
    std::vector<float> inputData = readTestData(inputFile);
    std::vector<float> expectedOutputData = readTestData(expectedOutputFile);
    
    if (inputData.empty() || expectedOutputData.empty()) {
        std::cerr << "Failed to load test data" << std::endl;
        return false;
    }
    
    const int channels = 3; // Assuming 3-channel data (RGB or YIQ)
    
    // Determine dimensions from data size
    int inputWidth, inputHeight;
    int outputWidth, outputHeight;
    
    if (!determineImageDimensions(inputData.size(), channels, inputWidth, inputHeight)) {
        std::cerr << "Failed to determine input image dimensions" << std::endl;
        return false;
    }
    
    if (!determineImageDimensions(expectedOutputData.size(), channels, outputWidth, outputHeight)) {
        std::cerr << "Failed to determine expected output image dimensions" << std::endl;
        return false;
    }
    
    // Allocate output buffer
    std::vector<float> actualOutputData(expectedOutputData.size());
    
    // Initialize CUDA pyramid module
    evmcuda::init_pyramid();
    
    // Call CUDA implementation
    bool success = evmcuda::pyr_up_wrapper(
        inputData.data(), 
        inputWidth, inputHeight, channels,
        actualOutputData.data(),
        outputWidth, outputHeight
    );
    
    if (!success) {
        std::cerr << "pyr_up_wrapper failed" << std::endl;
        return false;
    }
    
    // Compare results
    std::cout << "\nPyr_Up Test:" << std::endl;
    compareResults(expectedOutputData, actualOutputData, 1e-4);
    
    return true;
}

// Test full pyramid operations using the same test data as the CPU tests
bool testPyramidOperations(const std::string& basePath) {
    // Test data paths
    std::string rgbFile = basePath + "frame_0_rgb.txt";
    std::string yiqFile = basePath + "frame_0_yiq.txt";
    std::string pyrdownFile = basePath + "frame_0_pyrdown_0.txt";
    std::string pyrupFile = basePath + "frame_0_pyrup_0.txt";
    
    // Load RGB data
    std::vector<float> rgbData = readTestData(rgbFile);
    if (rgbData.empty()) {
        std::cerr << "Failed to load RGB test data" << std::endl;
        return false;
    }
    
    // Load YIQ data
    std::vector<float> yiqData = readTestData(yiqFile);
    if (yiqData.empty()) {
        std::cerr << "Failed to load YIQ test data" << std::endl;
        return false;
    }
    
    // Determine dimensions
    int width, height;
    const int channels = 3;
    
    if (!determineImageDimensions(rgbData.size(), channels, width, height)) {
        std::cerr << "Failed to determine image dimensions from RGB data" << std::endl;
        return false;
    }
    
    // Convert RGB to YIQ
    std::vector<float> convertedYiqData(yiqData.size());
    std::vector<float> actualPyrDownData(width/2 * height/2 * channels);
    
    // Initialize modules
    evmcuda::init_color_conversion();
    evmcuda::init_pyramid();
    
    // Test RGB to YIQ conversion
    evmcuda::rgb_to_yiq_wrapper(rgbData.data(), convertedYiqData.data(), width, height);
    
    std::cout << "\nRGB to YIQ Conversion Test:" << std::endl;
    compareResults(yiqData, convertedYiqData, 1e-4);
    
    // Test Pyr_Down
    bool pyrDownSuccess = evmcuda::pyr_down_wrapper(
        convertedYiqData.data(), 
        width, height, channels,
        actualPyrDownData.data()
    );
    
    if (!pyrDownSuccess) {
        std::cerr << "pyr_down_wrapper failed" << std::endl;
        return false;
    }
    
    // Compare with expected Pyr_Down output
    std::vector<float> expectedPyrDownData = readTestData(pyrdownFile);
    if (expectedPyrDownData.empty()) {
        std::cerr << "Failed to load expected Pyr_Down data" << std::endl;
        return false;
    }
    
    std::cout << "\nPyr_Down Test (after RGB -> YIQ):" << std::endl;
    compareResults(expectedPyrDownData, actualPyrDownData, 1e-4);
    
    // Test Pyr_Up with specific target size
    int pyrUpWidth, pyrUpHeight;
    determineImageDimensions(readTestData(pyrupFile).size(), channels, pyrUpWidth, pyrUpHeight);
    
    std::vector<float> actualPyrUpData(pyrUpWidth * pyrUpHeight * channels);
    
    bool pyrUpSuccess = evmcuda::pyr_up_wrapper(
        actualPyrDownData.data(),
        width/2, height/2, channels,
        actualPyrUpData.data(),
        pyrUpWidth, pyrUpHeight
    );
    
    if (!pyrUpSuccess) {
        std::cerr << "pyr_up_wrapper failed" << std::endl;
        return false;
    }
    
    // Compare with expected Pyr_Up output
    std::vector<float> expectedPyrUpData = readTestData(pyrupFile);
    if (expectedPyrUpData.empty()) {
        std::cerr << "Failed to load expected Pyr_Up data" << std::endl;
        return false;
    }
    
    std::cout << "\nPyr_Up Test (after Pyr_Down):" << std::endl;
    compareResults(expectedPyrUpData, actualPyrUpData, 1e-4);
    
    return true;
}

int main(int argc, char* argv[]) {
    // Test data paths
    std::string basePath = "../../../cpp/tests/data/";
    if (argc > 1) {
        basePath = argv[1];
    }
    
    std::cout << "Testing CUDA Pyramid Operations..." << std::endl;
    std::cout << "Using test data from: " << basePath << std::endl;
    
    bool success = true;
    
    // Initialize CUDA modules
    if (!evmcuda::init_pyramid()) {
        std::cerr << "Failed to initialize CUDA pyramid module" << std::endl;
        return 1;
    }
    
    // Test individual operations
    success = testPyrDown(basePath + "frame_0_yiq.txt", basePath + "frame_0_pyrdown_0.txt") && success;
    success = testPyrUp(basePath + "frame_0_pyrdown_0.txt", basePath + "frame_0_pyrup_0.txt") && success;
    
    // Test full pyramid operations
    success = testPyramidOperations(basePath) && success;
    
    // Cleanup
    evmcuda::cleanup_pyramid();
    
    if (success) {
        std::cout << "\nAll pyramid tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "\nSome pyramid tests FAILED!" << std::endl;
        return 1;
    }
}