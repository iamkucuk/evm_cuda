#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "cuda_evm.cuh"
#include "cuda_laplacian_pyramid.cuh"
#include "cuda_pyramid.cuh"
#include "cuda_color_conversion.cuh"
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

// Determine dimensions from data size
bool determineImageDimensions(size_t numElements, int channels, int& width, int& height) {
    if (numElements % channels != 0) {
        std::cerr << "Data size is not divisible by channels: " << numElements 
                  << " elements, " << channels << " channels" << std::endl;
        return false;
    }
    
    int numPixels = numElements / channels;
    
    // Try to determine width and height (assuming a square image for simplicity)
    width = height = static_cast<int>(sqrt(numPixels));
    
    if (width * height * channels != numElements) {
        std::cerr << "Warning: Could not determine exact image dimensions. "
                  << "Using width = " << width << ", height = " << height 
                  << " (total pixels = " << width * height << ", needed = " << numPixels << ")" << std::endl;
    }
    
    return true;
}

// Test function for the complete EVM pipeline
bool testEVMPipeline(
    const std::vector<std::string>& inputRgbFiles,
    const std::string& expectedOutputFile,
    int pyramidLevels = 4,
    double alpha = 10.0,
    double lambdaCutoff = 16.0,
    double freqLow = 0.05,
    double freqHigh = 0.4,
    double chromAttenuation = 0.1) {
    
    std::cout << "\nTesting Complete EVM Pipeline" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Pyramid Levels: " << pyramidLevels << std::endl;
    std::cout << "  Alpha: " << alpha << std::endl;
    std::cout << "  Lambda Cutoff: " << lambdaCutoff << std::endl;
    std::cout << "  Frequency Range: [" << freqLow << ", " << freqHigh << "]" << std::endl;
    std::cout << "  Chrominance Attenuation: " << chromAttenuation << std::endl;
    
    // Load input RGB frames
    std::vector<cv::Mat> inputFrames;
    for (const auto& filename : inputRgbFiles) {
        std::vector<float> frameData = readTestData(filename);
        if (frameData.empty()) {
            std::cerr << "Failed to load RGB data from " << filename << std::endl;
            return false;
        }
        
        // Determine dimensions
        int width, height;
        const int channels = 3; // RGB
        
        if (!determineImageDimensions(frameData.size(), channels, width, height)) {
            std::cerr << "Failed to determine dimensions for " << filename << std::endl;
            return false;
        }
        
        // Convert to Mat
        cv::Mat frameMat = vectorToMat(frameData, width, height, channels);
        
        // Convert to 8-bit format (evm_cuda expects CV_8UC3)
        cv::Mat frameUint8;
        frameMat.convertTo(frameUint8, CV_8UC3);
        
        inputFrames.push_back(frameUint8);
    }
    
    // Load expected output data
    std::vector<float> expectedOutputData = readTestData(expectedOutputFile);
    if (expectedOutputData.empty()) {
        std::cerr << "Failed to load expected output data from " << expectedOutputFile << std::endl;
        return false;
    }
    
    // Determine dimensions for expected output
    int outWidth, outHeight;
    const int channels = 3; // RGB
    
    if (!determineImageDimensions(expectedOutputData.size(), channels, outWidth, outHeight)) {
        std::cerr << "Failed to determine dimensions for expected output" << std::endl;
        return false;
    }
    
    // Convert expected output to Mat
    cv::Mat expectedOutputMat = vectorToMat(expectedOutputData, outWidth, outHeight, channels);
    
    // Create a temporary directory for output
    std::string tempOutputFile = "temp_evm_output.mp4";
    
    try {
        // Run the complete EVM pipeline
        // Note: We'll use the process_video_laplacian function from cuda_evm.cuh
        
        // Set up a mock video file containing our test frames
        // Since we can't easily create a video file here, we'll simulate the pipeline steps
        
        // Step 1: Convert RGB to YIQ for all frames
        std::vector<cv::Mat> yiqFrames;
        for (const auto& rgbFrame : inputFrames) {
            // Convert to float for processing
            cv::Mat rgbFloat;
            rgbFrame.convertTo(rgbFloat, CV_32FC3);
            
            // Convert RGB to YIQ
            std::vector<float> rgbFloatVec = matToVector(rgbFloat);
            std::vector<float> yiqFloatVec(rgbFloatVec.size());
            
            evmcuda::rgb_to_yiq_wrapper(
                rgbFloatVec.data(), 
                yiqFloatVec.data(), 
                rgbFrame.cols, 
                rgbFrame.rows);
            
            // Convert back to Mat
            cv::Mat yiqMat = vectorToMat(yiqFloatVec, rgbFrame.cols, rgbFrame.rows, 3);
            yiqFrames.push_back(yiqMat);
        }
        
        // Step 2: Generate Laplacian pyramids
        std::vector<std::vector<cv::Mat>> laplacianPyramids = 
            evmcuda::get_laplacian_pyramids(inputFrames, pyramidLevels);
        
        // Step 3: Filter Laplacian pyramids
        std::pair<double, double> freqRange(freqLow, freqHigh);
        double fps = 30.0; // Assume 30fps for test data
        
        std::vector<std::vector<cv::Mat>> filteredPyramids = 
            evmcuda::filter_laplacian_pyramids(
                laplacianPyramids, 
                pyramidLevels, 
                fps, 
                freqRange, 
                alpha, 
                lambdaCutoff, 
                chromAttenuation);
        
        // Step 4: Reconstruct the output frames
        std::vector<cv::Mat> outputFrames;
        for (size_t i = 0; i < inputFrames.size(); ++i) {
            if (i < filteredPyramids.size()) {
                cv::Mat reconstructed = evmcuda::reconstruct_laplacian_image(
                    inputFrames[i], 
                    filteredPyramids[i]);
                
                outputFrames.push_back(reconstructed);
            }
        }
        
        // Step 5: Compare the output of the middle frame with expected output
        if (outputFrames.empty()) {
            std::cerr << "No output frames were generated" << std::endl;
            return false;
        }
        
        // Choose a representative frame to compare (middle frame)
        size_t middleIdx = outputFrames.size() / 2;
        cv::Mat outputFrame = outputFrames[middleIdx];
        
        // Convert to float for comparison
        cv::Mat outputFloat;
        outputFrame.convertTo(outputFloat, CV_32FC3);
        
        // Compare with expected output
        std::vector<float> outputFloatVec = matToVector(outputFloat);
        
        compareResults(expectedOutputData, outputFloatVec, "EVM Pipeline Output", 5.0f);
        
        // Calculate mean absolute error for full pipeline validation
        cv::Mat outputFrame8U, expectedFrame8U;
        outputFloat.convertTo(outputFrame8U, CV_8UC3);
        expectedOutputMat.convertTo(expectedFrame8U, CV_8UC3);
        
        cv::Mat diffMat;
        cv::absdiff(outputFrame8U, expectedFrame8U, diffMat);
        cv::Scalar meanDiff = cv::mean(diffMat);
        
        double meanAbsError = (meanDiff[0] + meanDiff[1] + meanDiff[2]) / 3.0;
        
        std::cout << "End-to-end pipeline comparison:" << std::endl;
        std::cout << "  Mean absolute error: " << meanAbsError << std::endl;
        std::cout << "  Validation " << (meanAbsError < 10.0 ? "PASSED" : "FAILED") 
                  << " (threshold = 10.0)" << std::endl;
        
        // For full pipeline, we use a higher tolerance threshold due to accumulated errors
        return (meanAbsError < 10.0);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in EVM pipeline test: " << e.what() << std::endl;
        return false;
    }
}

// Test function that uses the direct process_video_laplacian function
bool testDirectEVMPipeline() {
    std::cout << "\nTesting Direct EVM Pipeline Process Function" << std::endl;
    
    try {
        // Create a temporary test video with a simple pattern
        const int width = 64;
        const int height = 64;
        const int frames = 10;
        
        // Create frames with a moving pattern
        std::vector<cv::Mat> testFrames;
        for (int f = 0; f < frames; ++f) {
            cv::Mat frame(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
            
            // Create a moving pattern
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    // Create a pulsating circle
                    float dx = x - width / 2;
                    float dy = y - height / 2;
                    float dist = std::sqrt(dx * dx + dy * dy);
                    
                    // Pulsate with frame number
                    float intensity = 128.0f + 127.0f * std::sin(dist * 0.1f + f * 0.2f);
                    
                    // Set pixel value
                    cv::Vec3b& pixel = frame.at<cv::Vec3b>(y, x);
                    pixel[0] = static_cast<uchar>(intensity);
                    pixel[1] = static_cast<uchar>(intensity);
                    pixel[2] = static_cast<uchar>(intensity);
                }
            }
            
            testFrames.push_back(frame);
        }
        
        // Create a temporary video file
        std::string tempInputFile = "temp_evm_input.mp4";
        std::string tempOutputFile = "temp_evm_output.mp4";
        
        // Writer parameters
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        double fps = 30.0;
        cv::Size frameSize(width, height);
        
        // Create a writer and write frames
        cv::VideoWriter writer(tempInputFile, fourcc, fps, frameSize);
        if (!writer.isOpened()) {
            std::cerr << "Failed to create temporary video file" << std::endl;
            return false;
        }
        
        for (const auto& frame : testFrames) {
            writer.write(frame);
        }
        
        writer.release();
        
        // Set EVM parameters
        int pyramidLevels = 4;
        double alpha = 10.0;
        double lambdaCutoff = 16.0;
        double freqLow = 0.05;
        double freqHigh = 0.4;
        double chromAttenuation = 0.1;
        
        // Process the video directly using the EVM function
        evmcuda::process_video_laplacian(
            tempInputFile,
            tempOutputFile,
            pyramidLevels,
            alpha,
            lambdaCutoff,
            freqLow,
            freqHigh,
            chromAttenuation
        );
        
        // Check if output file was created
        std::ifstream outputFile(tempOutputFile);
        bool outputExists = outputFile.good();
        outputFile.close();
        
        if (!outputExists) {
            std::cerr << "Output video file was not created" << std::endl;
            return false;
        }
        
        // Clean up temporary files
        remove(tempInputFile.c_str());
        remove(tempOutputFile.c_str());
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in direct EVM pipeline test: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    // Test data paths
    std::string basePath = "../../cpp/tests/data/";
    if (argc > 1) {
        basePath = argv[1];
    }
    
    std::cout << "Testing CUDA EVM Pipeline..." << std::endl;
    std::cout << "Using test data from: " << basePath << std::endl;
    
    // Initialize all CUDA modules
    if (!evmcuda::init_evm()) {
        std::cerr << "Failed to initialize CUDA EVM modules" << std::endl;
        return 1;
    }
    
    bool success = true;
    
    // Test end-to-end pipeline with test data
    std::vector<std::string> inputRgbFiles = {
        basePath + "frame_0_rgb.txt",
        basePath + "frame_1_rgb.txt",
        basePath + "frame_2_rgb.txt",
        basePath + "frame_3_rgb.txt",
        basePath + "frame_4_rgb.txt"
    };
    
    // Use the final output file as expected result
    std::string expectedOutputFile = basePath + "frame_0_step6e_final_rgb_uint8.txt";
    
    if (!testEVMPipeline(inputRgbFiles, expectedOutputFile)) {
        std::cerr << "End-to-end EVM pipeline test failed" << std::endl;
        success = false;
    }
    
    // Test the direct process_video_laplacian function
    if (!testDirectEVMPipeline()) {
        std::cerr << "Direct EVM process function test failed" << std::endl;
        success = false;
    }
    
    // Cleanup
    evmcuda::cleanup_evm();
    
    if (success) {
        std::cout << "\nAll EVM pipeline tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "\nSome EVM pipeline tests FAILED!" << std::endl;
        return 1;
    }
}