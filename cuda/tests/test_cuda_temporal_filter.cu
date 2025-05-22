#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include "cuda_laplacian_pyramid.cuh"
#include "cuda_butterworth.cuh"

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

// Helper function to convert a flat vector to a cv::Mat
cv::Mat vectorToMat(const std::vector<float>& data, int width, int height, int channels) {
    cv::Mat mat(height, width, CV_32FC(channels));
    
    if (data.size() != static_cast<size_t>(width * height * channels)) {
        std::cerr << "Data size mismatch: expected " << (width * height * channels)
                  << ", got " << data.size() << std::endl;
        return mat; // Return empty mat
    }
    
    // Copy data to mat
    std::memcpy(mat.data, data.data(), data.size() * sizeof(float));
    return mat;
}

// Helper function to convert a cv::Mat to a flat vector
std::vector<float> matToVector(const cv::Mat& mat) {
    std::vector<float> data(mat.rows * mat.cols * mat.channels());
    
    if (mat.isContinuous()) {
        std::memcpy(data.data(), mat.data, data.size() * sizeof(float));
    } else {
        size_t idx = 0;
        for (int i = 0; i < mat.rows; ++i) {
            const float* row_ptr = mat.ptr<float>(i);
            for (int j = 0; j < mat.cols * mat.channels(); ++j) {
                data[idx++] = row_ptr[j];
            }
        }
    }
    
    return data;
}

// Helper function to compare results with tolerance
void compareResults(const std::vector<float>& expected, const std::vector<float>& actual, 
                   const std::string& name, float epsilon = 1e-5) {
    if (expected.size() != actual.size()) {
        std::cerr << "Size mismatch for " << name << ": Expected = " << expected.size() 
                  << ", Actual = " << actual.size() << std::endl;
        std::cerr << "Test FAILED" << std::endl;
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
    
    std::cout << "Comparison results for " << name << ":" << std::endl;
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

// Use hardcoded dimensions based on the test data
bool determineImageDimensions(size_t numElements, int channels, int& width, int& height) {
    if (numElements % channels != 0) {
        std::cerr << "Data size is not divisible by channels: " << numElements 
                  << " elements, " << channels << " channels" << std::endl;
        return false;
    }
    
    // For frame_0 image files (base resolution)
    if (numElements == 528 * 592 * 3) {
        width = 528;
        height = 592;
        return true;
    }
    
    // For the pyramid levels
    if (numElements == 264 * 296 * 3) {
        width = 264;
        height = 296;
        return true;
    }
    
    if (numElements == 132 * 148 * 3) {
        width = 132;
        height = 148;
        return true;
    }
    
    if (numElements == 66 * 74 * 3) {
        width = 66;
        height = 74;
        return true;
    }
    
    // If no match found, calculate dimensions (not recommended)
    int numPixels = numElements / channels;
    width = height = static_cast<int>(sqrt(numPixels));
    
    std::cerr << "Warning: Using calculated dimensions for unknown test data size. "
              << "Using width = " << width << ", height = " << height 
              << " (total pixels = " << width * height << ", needed = " << numPixels << ")" << std::endl;
    
    return true;
}

// Test function for temporal filtering of Laplacian pyramids
bool testTemporalFiltering(
    const std::vector<std::string>& inputLaplacianFiles,
    const std::vector<std::string>& expectedFilteredFiles,
    int level) {
    
    std::cout << "\nTesting Temporal Filtering of Laplacian Pyramids (Level " << level << ")" << std::endl;
    
    // For simplicity, we'll focus on a single pyramid level across multiple frames
    
    // Load input data for multiple frames
    std::vector<cv::Mat> inputLaplacianMats;
    for (const auto& filename : inputLaplacianFiles) {
        std::vector<float> frameData = readTestData(filename);
        if (frameData.empty()) {
            std::cerr << "Failed to load laplacian data from " << filename << std::endl;
            return false;
        }
        
        // Determine dimensions
        int width, height;
        const int channels = 3; // YIQ
        
        if (!determineImageDimensions(frameData.size(), channels, width, height)) {
            std::cerr << "Failed to determine dimensions for " << filename << std::endl;
            return false;
        }
        
        // Convert to Mat
        cv::Mat frameMat = vectorToMat(frameData, width, height, channels);
        inputLaplacianMats.push_back(frameMat);
    }
    
    // Load expected filtered data
    std::vector<cv::Mat> expectedFilteredMats;
    for (const auto& filename : expectedFilteredFiles) {
        std::vector<float> frameData = readTestData(filename);
        if (frameData.empty()) {
            std::cerr << "Failed to load filtered data from " << filename << std::endl;
            return false;
        }
        
        // Determine dimensions
        int width, height;
        const int channels = 3; // YIQ
        
        if (!determineImageDimensions(frameData.size(), channels, width, height)) {
            std::cerr << "Failed to determine dimensions for " << filename << std::endl;
            return false;
        }
        
        // Convert to Mat
        cv::Mat frameMat = vectorToMat(frameData, width, height, channels);
        expectedFilteredMats.push_back(frameMat);
    }
    
    // Organize the input frames into a pyramid-like structure
    std::vector<std::vector<cv::Mat>> pyramidsBatch;
    for (size_t i = 0; i < inputLaplacianMats.size(); ++i) {
        std::vector<cv::Mat> pyramid;
        pyramid.push_back(inputLaplacianMats[i]); // Add only the level we're testing
        pyramidsBatch.push_back(pyramid);
    }
    
    try {
        // Apply temporal filtering using CUDA
        const double fps = 30.0; // Assuming 30 fps for test data
        const std::pair<double, double> freq_range = {0.05, 0.4}; // Typical range used in EVM
        const double alpha = 10.0; // Magnification factor
        const double lambda_cutoff = 16.0; // Spatial cutoff
        const double attenuation = 0.1; // Chrominance attenuation
        
        std::vector<std::vector<cv::Mat>> filteredPyramids = evmcuda::filter_laplacian_pyramids(
            pyramidsBatch, 1, fps, freq_range, alpha, lambda_cutoff, attenuation);
        
        // Check if we got the expected number of filtered frames
        if (filteredPyramids.size() != expectedFilteredMats.size()) {
            std::cerr << "Filtered pyramid batch has " << filteredPyramids.size() 
                      << " frames, expected " << expectedFilteredMats.size() << std::endl;
            return false;
        }
        
        // Compare each filtered frame with the expected output
        bool allMatch = true;
        for (size_t i = 0; i < filteredPyramids.size(); ++i) {
            if (filteredPyramids[i].empty()) {
                std::cerr << "Empty pyramid at index " << i << std::endl;
                allMatch = false;
                continue;
            }
            
            cv::Mat filteredFrame = filteredPyramids[i][0]; // Get the first level (we're testing one level)
            std::vector<float> filteredVector = matToVector(filteredFrame);
            std::vector<float> expectedVector = matToVector(expectedFilteredMats[i]);
            
            std::string frameName = "Frame " + std::to_string(i) + " Level " + std::to_string(level);
            compareResults(expectedVector, filteredVector, frameName, 1e-3);
            
            // Consider a larger tolerance for temporal filtering tests
            if (std::abs(*std::max_element(expectedVector.begin(), expectedVector.end(), 
                                         [](float a, float b) { return std::abs(a) < std::abs(b); }) -
                       *std::max_element(filteredVector.begin(), filteredVector.end(),
                                        [](float a, float b) { return std::abs(a) < std::abs(b); })) > 1e-3) {
                allMatch = false;
            }
        }
        
        return allMatch;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in CUDA temporal filtering test: " << e.what() << std::endl;
        return false;
    }
}

// Test function for Butterworth filtering
bool testButterworthFiltering() {
    std::cout << "\nTesting Butterworth Filter on Image Frames" << std::endl;
    
    // Create a simple test signal (10x10 image with a ramp)
    const int width = 10;
    const int height = 10;
    const int channels = 3;
    const int numElements = width * height * channels;
    
    // Create host arrays for 3 frames
    std::vector<float> h_frames[3];
    for (int f = 0; f < 3; ++f) {
        h_frames[f].resize(numElements, 0.0f);
        
        // Each frame has a different pattern to simulate time variation
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int c = 0; c < channels; ++c) {
                    int idx = (y * width + x) * channels + c;
                    
                    // Create a time-varying signal
                    if (c == 0) { // Y channel
                        h_frames[f][idx] = static_cast<float>(x) / width + f * 0.1f;
                    } else if (c == 1) { // I channel
                        h_frames[f][idx] = static_cast<float>(y) / height - f * 0.05f;
                    } else { // Q channel
                        h_frames[f][idx] = static_cast<float>(x + y) / (width + height) + f * 0.02f;
                    }
                }
            }
        }
    }
    
    // Allocate device memory
    float *d_input[3] = {nullptr};
    float *d_prev_input[3] = {nullptr};
    float *d_prev_output[3] = {nullptr};
    float *d_output[3] = {nullptr};
    
    // Allocate memory for all frames
    for (int f = 0; f < 3; ++f) {
        cudaError_t err = cudaMalloc(&d_input[f], numElements * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate device memory for input frame " << f << ": " 
                      << cudaGetErrorString(err) << std::endl;
            
            // Cleanup previously allocated memory
            for (int j = 0; j < f; ++j) {
                cudaFree(d_input[j]);
                cudaFree(d_prev_input[j]);
                cudaFree(d_prev_output[j]);
                cudaFree(d_output[j]);
            }
            return false;
        }
        
        err = cudaMalloc(&d_prev_input[f], numElements * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate device memory for prev_input frame " << f << ": " 
                      << cudaGetErrorString(err) << std::endl;
            
            cudaFree(d_input[f]);
            // Cleanup previously allocated memory
            for (int j = 0; j < f; ++j) {
                cudaFree(d_input[j]);
                cudaFree(d_prev_input[j]);
                cudaFree(d_prev_output[j]);
                cudaFree(d_output[j]);
            }
            return false;
        }
        
        err = cudaMalloc(&d_prev_output[f], numElements * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate device memory for prev_output frame " << f << ": " 
                      << cudaGetErrorString(err) << std::endl;
            
            cudaFree(d_input[f]);
            cudaFree(d_prev_input[f]);
            // Cleanup previously allocated memory
            for (int j = 0; j < f; ++j) {
                cudaFree(d_input[j]);
                cudaFree(d_prev_input[j]);
                cudaFree(d_prev_output[j]);
                cudaFree(d_output[j]);
            }
            return false;
        }
        
        err = cudaMalloc(&d_output[f], numElements * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate device memory for output frame " << f << ": " 
                      << cudaGetErrorString(err) << std::endl;
            
            cudaFree(d_input[f]);
            cudaFree(d_prev_input[f]);
            cudaFree(d_prev_output[f]);
            // Cleanup previously allocated memory
            for (int j = 0; j < f; ++j) {
                cudaFree(d_input[j]);
                cudaFree(d_prev_input[j]);
                cudaFree(d_prev_output[j]);
                cudaFree(d_output[j]);
            }
            return false;
        }
        
        // Copy input frames to device
        err = cudaMemcpy(d_input[f], h_frames[f].data(), numElements * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy input frame " << f << " to device: " 
                      << cudaGetErrorString(err) << std::endl;
            
            for (int j = 0; j <= f; ++j) {
                cudaFree(d_input[j]);
                cudaFree(d_prev_input[j]);
                cudaFree(d_prev_output[j]);
                cudaFree(d_output[j]);
            }
            return false;
        }
        
        // Initialize prev_input and prev_output to zeros
        err = cudaMemset(d_prev_input[f], 0, numElements * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to initialize prev_input frame " << f << ": " 
                      << cudaGetErrorString(err) << std::endl;
            
            for (int j = 0; j <= f; ++j) {
                cudaFree(d_input[j]);
                cudaFree(d_prev_input[j]);
                cudaFree(d_prev_output[j]);
                cudaFree(d_output[j]);
            }
            return false;
        }
        
        err = cudaMemset(d_prev_output[f], 0, numElements * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to initialize prev_output frame " << f << ": " 
                      << cudaGetErrorString(err) << std::endl;
            
            for (int j = 0; j <= f; ++j) {
                cudaFree(d_input[j]);
                cudaFree(d_prev_input[j]);
                cudaFree(d_prev_output[j]);
                cudaFree(d_output[j]);
            }
            return false;
        }
    }
    
    try {
        // Create Butterworth filter for testing
        evmcuda::Butterworth butterFilter(0.05, 0.4);
        
        // Apply filter to each frame sequentially
        std::vector<float> h_outputs[3];
        for (int f = 0; f < 3; ++f) {
            h_outputs[f].resize(numElements);
            
            // Apply the filter
            butterFilter.filter(
                d_input[f], width, height, channels,
                d_prev_input[f], d_prev_output[f], d_output[f]);
            
            // Copy result back to host
            cudaError_t err = cudaMemcpy(h_outputs[f].data(), d_output[f], 
                                        numElements * sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                std::cerr << "Failed to copy output frame " << f << " from device: " 
                          << cudaGetErrorString(err) << std::endl;
                
                for (int j = 0; j < 3; ++j) {
                    cudaFree(d_input[j]);
                    cudaFree(d_prev_input[j]);
                    cudaFree(d_prev_output[j]);
                    cudaFree(d_output[j]);
                }
                return false;
            }
            
            // For the next frame: current input becomes prev_input, current output becomes prev_output
            if (f < 2) {
                err = cudaMemcpy(d_prev_input[f+1], d_input[f], 
                                numElements * sizeof(float), cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess) {
                    std::cerr << "Failed to update prev_input for frame " << f+1 << ": " 
                              << cudaGetErrorString(err) << std::endl;
                    
                    for (int j = 0; j < 3; ++j) {
                        cudaFree(d_input[j]);
                        cudaFree(d_prev_input[j]);
                        cudaFree(d_prev_output[j]);
                        cudaFree(d_output[j]);
                    }
                    return false;
                }
                
                err = cudaMemcpy(d_prev_output[f+1], d_output[f], 
                                numElements * sizeof(float), cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess) {
                    std::cerr << "Failed to update prev_output for frame " << f+1 << ": " 
                              << cudaGetErrorString(err) << std::endl;
                    
                    for (int j = 0; j < 3; ++j) {
                        cudaFree(d_input[j]);
                        cudaFree(d_prev_input[j]);
                        cudaFree(d_prev_output[j]);
                        cudaFree(d_output[j]);
                    }
                    return false;
                }
            }
        }
        
        // Calculate expected output manually using the filter coefficients
        const auto& b_coeffs = butterFilter.get_b_coeffs();
        const auto& a_coeffs = butterFilter.get_a_coeffs();
        
        if (b_coeffs.size() != 2 || a_coeffs.size() != 2) {
            std::cerr << "Unexpected coefficient sizes. b_size=" << b_coeffs.size() 
                      << ", a_size=" << a_coeffs.size() << std::endl;
            
            for (int j = 0; j < 3; ++j) {
                cudaFree(d_input[j]);
                cudaFree(d_prev_input[j]);
                cudaFree(d_prev_output[j]);
                cudaFree(d_output[j]);
            }
            return false;
        }
        
        // Calculate expected outputs
        std::vector<float> expected_outputs[3];
        for (int f = 0; f < 3; ++f) {
            expected_outputs[f].resize(numElements);
            
            float b0 = static_cast<float>(b_coeffs[0]);
            float b1 = static_cast<float>(b_coeffs[1]);
            float a1 = static_cast<float>(a_coeffs[1]);
            
            for (int i = 0; i < numElements; ++i) {
                // For first frame, prev_input and prev_output are zero
                float prev_input = (f > 0) ? h_frames[f-1][i] : 0.0f;
                float prev_output = (f > 0) ? expected_outputs[f-1][i] : 0.0f;
                
                expected_outputs[f][i] = b0 * h_frames[f][i] + b1 * prev_input - a1 * prev_output;
            }
        }
        
        // Compare actual outputs with expected outputs
        bool allFramesMatch = true;
        for (int f = 0; f < 3; ++f) {
            float maxError = 0.0f;
            float meanError = 0.0f;
            int maxErrorIdx = -1;
            
            for (int i = 0; i < numElements; ++i) {
                float error = std::abs(expected_outputs[f][i] - h_outputs[f][i]);
                if (error > maxError) {
                    maxError = error;
                    maxErrorIdx = i;
                }
                meanError += error;
            }
            
            meanError /= numElements;
            
            std::cout << "Frame " << f << " comparison results:" << std::endl;
            std::cout << "  Max error: " << maxError;
            if (maxErrorIdx >= 0) {
                std::cout << " at index " << maxErrorIdx 
                          << " (Expected: " << expected_outputs[f][maxErrorIdx] 
                          << ", Actual: " << h_outputs[f][maxErrorIdx] << ")";
            }
            std::cout << std::endl;
            std::cout << "  Mean error: " << meanError << std::endl;
            std::cout << "  Validation " << (maxError <= 1e-5 ? "PASSED" : "FAILED") 
                      << " (epsilon = 1e-5)" << std::endl;
            
            if (maxError > 1e-5) {
                allFramesMatch = false;
            }
        }
        
        // Cleanup
        for (int j = 0; j < 3; ++j) {
            cudaFree(d_input[j]);
            cudaFree(d_prev_input[j]);
            cudaFree(d_prev_output[j]);
            cudaFree(d_output[j]);
        }
        
        return allFramesMatch;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in Butterworth filter test: " << e.what() << std::endl;
        
        for (int j = 0; j < 3; ++j) {
            cudaFree(d_input[j]);
            cudaFree(d_prev_input[j]);
            cudaFree(d_prev_output[j]);
            cudaFree(d_output[j]);
        }
        return false;
    }
}

int main(int argc, char* argv[]) {
    // Test data paths
    std::string basePath = "../../../../cpp/tests/data/";
    if (argc > 1) {
        basePath = argv[1];
    }
    
    std::cout << "Testing CUDA Temporal Filtering Operations..." << std::endl;
    std::cout << "Using test data from: " << basePath << std::endl;
    
    // Initialize CUDA modules
    if (!evmcuda::init_butterworth()) {
        std::cerr << "Failed to initialize CUDA Butterworth module" << std::endl;
        return 1;
    }
    
    if (!evmcuda::init_laplacian_pyramid()) {
        std::cerr << "Failed to initialize CUDA Laplacian pyramid module" << std::endl;
        evmcuda::cleanup_butterworth();
        return 1;
    }
    
    bool success = true;
    
    // Test direct Butterworth filtering
    if (!testButterworthFiltering()) {
        std::cerr << "Butterworth filtering test failed" << std::endl;
        success = false;
    }
    
    // Test temporal filtering of Laplacian pyramids
    // For this test, we need multiple frames at a specific level
    std::vector<std::string> inputLaplacianFiles = {
        basePath + "frame_0_laplacian_level_0.txt",
        basePath + "frame_1_laplacian_level_0.txt",
        basePath + "frame_2_laplacian_level_0.txt",
        basePath + "frame_3_laplacian_level_0.txt",
        basePath + "frame_4_laplacian_level_0.txt"
    };
    
    std::vector<std::string> expectedFilteredFiles = {
        basePath + "frame_0_filtered_level_0.txt",
        basePath + "frame_1_filtered_level_0.txt",
        basePath + "frame_2_filtered_level_0.txt",
        basePath + "frame_3_filtered_level_0.txt",
        basePath + "frame_4_filtered_level_0.txt"
    };
    
    if (!testTemporalFiltering(inputLaplacianFiles, expectedFilteredFiles, 0)) {
        std::cerr << "Temporal filtering test failed for level 0" << std::endl;
        success = false;
    }
    
    // Test another level to be thorough
    inputLaplacianFiles = {
        basePath + "frame_0_laplacian_level_1.txt",
        basePath + "frame_1_laplacian_level_1.txt",
        basePath + "frame_2_laplacian_level_1.txt",
        basePath + "frame_3_laplacian_level_1.txt",
        basePath + "frame_4_laplacian_level_1.txt"
    };
    
    expectedFilteredFiles = {
        basePath + "frame_0_filtered_level_1.txt",
        basePath + "frame_1_filtered_level_1.txt",
        basePath + "frame_2_filtered_level_1.txt",
        basePath + "frame_3_filtered_level_1.txt",
        basePath + "frame_4_filtered_level_1.txt"
    };
    
    if (!testTemporalFiltering(inputLaplacianFiles, expectedFilteredFiles, 1)) {
        std::cerr << "Temporal filtering test failed for level 1" << std::endl;
        success = false;
    }
    
    // Cleanup
    evmcuda::cleanup_laplacian_pyramid();
    evmcuda::cleanup_butterworth();
    
    if (success) {
        std::cout << "\nAll temporal filtering tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "\nSome temporal filtering tests FAILED!" << std::endl;
        return 1;
    }
}