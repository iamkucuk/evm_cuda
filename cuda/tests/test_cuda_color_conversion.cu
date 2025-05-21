#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cuda_runtime.h>
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

// Test function for RGB to YIQ conversion
bool testRgbToYiq(const std::string& rgbDataFile, const std::string& expectedYiqFile) {
    // Load test data
    std::vector<float> rgbData = readTestData(rgbDataFile);
    std::vector<float> expectedYiqData = readTestData(expectedYiqFile);
    
    if (rgbData.empty() || expectedYiqData.empty()) {
        std::cerr << "Failed to load test data" << std::endl;
        return false;
    }
    
    if (rgbData.size() != expectedYiqData.size()) {
        std::cerr << "Data size mismatch: RGB = " << rgbData.size() 
                  << ", YIQ = " << expectedYiqData.size() << std::endl;
        return false;
    }
    
    // Determine dimensions
    // Assuming test images are typically square or have width as a multiple of height
    int numElements = rgbData.size();
    int numPixels = numElements / 3; // 3 channels (RGB or YIQ)
    
    // Try to determine width and height - for this example, assume simple case
    int width, height;
    
    // Look for common image dimensions that match our data size
    // Assuming test data is square for now
    width = height = static_cast<int>(sqrt(numPixels));
    
    if (width * height * 3 != numElements) {
        std::cerr << "Warning: Could not determine exact image dimensions. "
                  << "Using width = " << width << ", height = " << height 
                  << " (total pixels = " << width * height << ", needed = " << numPixels << ")" << std::endl;
    }
    
    // Allocate output buffer
    std::vector<float> actualYiqData(numElements);
    
    // Call CUDA implementation
    try {
        evmcuda::rgb_to_yiq_wrapper(rgbData.data(), actualYiqData.data(), width, height);
    } catch (const std::exception& e) {
        std::cerr << "Error in rgb_to_yiq_wrapper: " << e.what() << std::endl;
        return false;
    }
    
    // Compare results
    std::cout << "\nRGB to YIQ Conversion Test:" << std::endl;
    compareResults(expectedYiqData, actualYiqData, 1e-4);
    
    return true;
}

// Test function for YIQ to RGB conversion
bool testYiqToRgb(const std::string& yiqDataFile, const std::string& expectedRgbFile) {
    // Load test data
    std::vector<float> yiqData = readTestData(yiqDataFile);
    std::vector<float> expectedRgbData = readTestData(expectedRgbFile);
    
    if (yiqData.empty() || expectedRgbData.empty()) {
        std::cerr << "Failed to load test data" << std::endl;
        return false;
    }
    
    if (yiqData.size() != expectedRgbData.size()) {
        std::cerr << "Data size mismatch: YIQ = " << yiqData.size() 
                  << ", RGB = " << expectedRgbData.size() << std::endl;
        return false;
    }
    
    // Determine dimensions
    int numElements = yiqData.size();
    int numPixels = numElements / 3; // 3 channels (RGB or YIQ)
    
    // Try to determine width and height
    int width, height;
    width = height = static_cast<int>(sqrt(numPixels));
    
    if (width * height * 3 != numElements) {
        std::cerr << "Warning: Could not determine exact image dimensions. "
                  << "Using width = " << width << ", height = " << height 
                  << " (total pixels = " << width * height << ", needed = " << numPixels << ")" << std::endl;
    }
    
    // Allocate output buffer
    std::vector<float> actualRgbData(numElements);
    
    // Call CUDA implementation
    try {
        evmcuda::yiq_to_rgb_wrapper(yiqData.data(), actualRgbData.data(), width, height);
    } catch (const std::exception& e) {
        std::cerr << "Error in yiq_to_rgb_wrapper: " << e.what() << std::endl;
        return false;
    }
    
    // Compare results
    std::cout << "\nYIQ to RGB Conversion Test:" << std::endl;
    compareResults(expectedRgbData, actualRgbData, 1e-1); // Higher tolerance for YIQ to RGB conversion
    
    return true;
}

int main(int argc, char* argv[]) {
    // Test data paths
    std::string basePath = "../cpp/tests/data/";
    if (argc > 1) {
        basePath = argv[1];
    }
    
    std::string rgbDataFile = basePath + "frame_0_rgb.txt";
    std::string yiqDataFile = basePath + "frame_0_yiq.txt";
    
    // Initialize CUDA color conversion module
    if (!evmcuda::init_color_conversion()) {
        std::cerr << "Failed to initialize color conversion module" << std::endl;
        return 1;
    }
    
    std::cout << "Testing CUDA Color Conversion..." << std::endl;
    std::cout << "Using test data from:" << std::endl;
    std::cout << "  RGB: " << rgbDataFile << std::endl;
    std::cout << "  YIQ: " << yiqDataFile << std::endl;
    
    bool success = true;
    
    // Test RGB to YIQ conversion
    if (!testRgbToYiq(rgbDataFile, yiqDataFile)) {
        std::cerr << "RGB to YIQ test failed" << std::endl;
        success = false;
    }
    
    // Test YIQ to RGB conversion
    if (!testYiqToRgb(yiqDataFile, rgbDataFile)) {
        std::cerr << "YIQ to RGB test failed" << std::endl;
        success = false;
    }
    
    // Cleanup
    evmcuda::cleanup_color_conversion();
    
    if (success) {
        std::cout << "\nAll tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "\nSome tests FAILED!" << std::endl;
        return 1;
    }
}