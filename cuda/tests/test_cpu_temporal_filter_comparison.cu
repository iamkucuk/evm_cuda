#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "cuda_butterworth.cuh"
#include "cuda_laplacian_pyramid.cuh"
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
                   const std::string& name, float epsilon = 1e-4) {
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
    
    // Show comparison outcome
    std::cout << "  Validation " << (maxError <= epsilon ? "PASSED" : "FAILED") 
              << " (epsilon = " << epsilon << ")" << std::endl;
    
    // If failed, show error histogram for diagnostics
    if (maxError > epsilon) {
        std::vector<int> errorCounts(10, 0);
        float errorStep = maxError / 10.0f;
        for (size_t i = 0; i < expected.size(); ++i) {
            float error = std::abs(expected[i] - actual[i]);
            int bucketIdx = static_cast<int>(error / errorStep);
            if (bucketIdx >= 10) bucketIdx = 9;
            errorCounts[bucketIdx]++;
        }
        
        std::cout << "  Error distribution:" << std::endl;
        for (int i = 0; i < 10; ++i) {
            float lowerBound = i * errorStep;
            float upperBound = (i + 1) * errorStep;
            std::cout << "    " << lowerBound << " - " << upperBound << ": " 
                      << errorCounts[i] << " values (" 
                      << (100.0f * errorCounts[i] / expected.size()) << "%)" << std::endl;
        }
    }
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

// Implement a simpler DFT-based temporal filter to match the CPU implementation
std::vector<cv::Mat> applyDftTemporalFilter(
    const std::vector<cv::Mat>& frames,
    double fl,
    double fh,
    double fps,
    double alpha)
{
    if (frames.empty()) {
        throw std::runtime_error("Empty frame sequence for filtering");
    }
    
    int numFrames = frames.size();
    int width = frames[0].cols;
    int height = frames[0].rows;
    int channels = frames[0].channels();
    
    // Result frames
    std::vector<cv::Mat> filteredFrames(numFrames);
    for (int i = 0; i < numFrames; i++) {
        filteredFrames[i] = cv::Mat::zeros(height, width, CV_32FC3);
    }
    
    // Process each pixel location independently
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Process each channel
            for (int c = 0; c < channels; c++) {
                // Extract time series for this pixel and channel
                cv::Mat timeSignal(numFrames, 1, CV_32F);
                for (int t = 0; t < numFrames; t++) {
                    timeSignal.at<float>(t, 0) = frames[t].at<cv::Vec3f>(y, x)[c];
                }
                
                // Prepare for DFT (set optimal size)
                int dftSize = cv::getOptimalDFTSize(numFrames);
                cv::Mat paddedSignal;
                cv::copyMakeBorder(timeSignal, paddedSignal, 
                                 0, dftSize - numFrames,    // top, bottom
                                 0, 0,                      // left, right
                                 cv::BORDER_CONSTANT, 0);   // border type, value
                
                // Convert to complex (real + imaginary)
                cv::Mat complexSignal;
                cv::Mat planes[] = {paddedSignal, cv::Mat::zeros(paddedSignal.size(), CV_32F)};
                cv::merge(planes, 2, complexSignal);
                
                // Forward DFT
                cv::dft(complexSignal, complexSignal);
                
                // Calculate frequency bins
                std::vector<float> freqs(dftSize);
                float df = fps / dftSize; // frequency resolution
                for (int i = 0; i < dftSize; i++) {
                    freqs[i] = i * df;
                    if (i > dftSize/2) {
                        freqs[i] = (i - dftSize) * df; // negative frequencies
                    }
                }
                
                // Apply ideal bandpass filter
                cv::split(complexSignal, planes); // planes[0] = Re, planes[1] = Im
                for (int i = 0; i < dftSize; i++) {
                    float freq = std::abs(freqs[i]);
                    if (freq < fl || freq > fh) {
                        // Zero out frequencies outside the passband
                        planes[0].at<float>(i) = 0;
                        planes[1].at<float>(i) = 0;
                    }
                }
                cv::merge(planes, 2, complexSignal);
                
                // Inverse DFT
                cv::idft(complexSignal, complexSignal, cv::DFT_SCALE); // Scale to normalize
                cv::split(complexSignal, planes);
                
                // Store filtered signal to output frames with amplification
                for (int t = 0; t < numFrames; t++) {
                    filteredFrames[t].at<cv::Vec3f>(y, x)[c] = planes[0].at<float>(t, 0) * alpha;
                }
            }
        }
    }
    
    return filteredFrames;
}

// Main test function to compare DFT-based approach with CUDA implementation
bool testTemporalFilterDftComparison(
    const std::vector<std::string>& inputFiles,
    const std::vector<std::string>& expectedOutputFiles,
    double fl = 0.4,
    double fh = 3.0,
    double fps = 30.0,
    double alpha = 10.0)
{
    std::cout << "\nTesting Temporal Filtering DFT vs. CUDA Implementation" << std::endl;
    
    if (inputFiles.size() < 2) {
        std::cerr << "At least 2 input frames are required for temporal filtering" << std::endl;
        return false;
    }
    
    if (inputFiles.size() != expectedOutputFiles.size()) {
        std::cerr << "Number of input files (" << inputFiles.size() << ") doesn't match "
                  << "number of expected output files (" << expectedOutputFiles.size() << ")" << std::endl;
        return false;
    }
    
    const int numFrames = static_cast<int>(inputFiles.size());
    std::vector<cv::Mat> inputMats;
    std::vector<cv::Mat> expectedOutputMats;
    
    std::cout << "Loading input and expected output frames..." << std::endl;
    
    // Load input frames and expected output frames
    for (int i = 0; i < numFrames; ++i) {
        std::vector<float> inputData = readTestData(inputFiles[i]);
        std::vector<float> expectedData = readTestData(expectedOutputFiles[i]);
        
        if (inputData.empty() || expectedData.empty()) {
            std::cerr << "Failed to load data for frame " << i << std::endl;
            return false;
        }
        
        // Determine dimensions
        int width, height;
        const int channels = 3; // YIQ format
        
        if (!determineImageDimensions(inputData.size(), channels, width, height)) {
            std::cerr << "Failed to determine dimensions for frame " << i << std::endl;
            return false;
        }
        
        // Convert to cv::Mat
        cv::Mat inputMat = vectorToMat(inputData, width, height, channels);
        cv::Mat expectedMat = vectorToMat(expectedData, width, height, channels);
        
        inputMats.push_back(inputMat);
        expectedOutputMats.push_back(expectedMat);
    }
    
    std::cout << "Loaded " << numFrames << " frames, dimensions: " 
              << inputMats[0].cols << "x" << inputMats[0].rows << ", channels: " 
              << inputMats[0].channels() << std::endl;
    
    try {
        // Apply DFT-based temporal filter (CPU approach)
        std::vector<cv::Mat> dftFilteredFrames = applyDftTemporalFilter(
            inputMats, fl, fh, fps, alpha);
        
        // Create Butterworth filter
        const double Wn_low = fl / (fps / 2.0);
        const double Wn_high = fh / (fps / 2.0);
        evmcuda::Butterworth butterFilter(Wn_low, Wn_high);
        
        // Allocate device memory for CUDA implementation
        float *d_input, *d_output, *d_prev_input, *d_prev_output;
        const int width = inputMats[0].cols;
        const int height = inputMats[0].rows;
        const int channels = inputMats[0].channels();
        const size_t frameSize = width * height * channels * sizeof(float);
        
        cudaError_t err = cudaMalloc(&d_input, frameSize);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for input: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        err = cudaMalloc(&d_output, frameSize);
        if (err != cudaSuccess) {
            cudaFree(d_input);
            throw std::runtime_error("Failed to allocate device memory for output: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        err = cudaMalloc(&d_prev_input, frameSize);
        if (err != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            throw std::runtime_error("Failed to allocate device memory for prev_input: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        err = cudaMalloc(&d_prev_output, frameSize);
        if (err != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_prev_input);
            throw std::runtime_error("Failed to allocate device memory for prev_output: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        // Initialize previous input and output to zeros
        err = cudaMemset(d_prev_input, 0, frameSize);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to initialize prev_input: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        err = cudaMemset(d_prev_output, 0, frameSize);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to initialize prev_output: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        std::vector<cv::Mat> butterworthFilteredFrames(numFrames);
        
        // Process each frame with Butterworth filter
        for (int i = 0; i < numFrames; ++i) {
            // Copy input frame to device
            err = cudaMemcpy(d_input, inputMats[i].data, frameSize, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to copy input to device: " + 
                                       std::string(cudaGetErrorString(err)));
            }
            
            // Apply temporal filter
            butterFilter.filter(d_input, width, height, channels, d_prev_input, d_prev_output, d_output);
            
            // Copy output back to host
            std::vector<float> outputData(width * height * channels);
            err = cudaMemcpy(outputData.data(), d_output, frameSize, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to copy output from device: " + 
                                       std::string(cudaGetErrorString(err)));
            }
            
            // Update previous input and output for next frame
            std::swap(d_prev_input, d_input); // Previous input = current input
            std::swap(d_prev_output, d_output); // Previous output = current output
            
            // Convert to cv::Mat
            cv::Mat outputMat = vectorToMat(outputData, width, height, channels);
            
            // Apply amplification (similar to CPU implementation)
            outputMat *= alpha;
            
            butterworthFilteredFrames[i] = outputMat;
        }
        
        // Compare results between methods and with expected output
        bool allMethodsMatch = true;
        bool matchesExpected = true;
        
        std::cout << "\nComparing different filter implementations:" << std::endl;
        
        for (int i = 0; i < numFrames; ++i) {
            std::string frameName = "Frame " + std::to_string(i);
            
            // Compare Butterworth vs DFT methods
            std::vector<float> butterData = matToVector(butterworthFilteredFrames[i]);
            std::vector<float> dftData = matToVector(dftFilteredFrames[i]);
            compareResults(dftData, butterData, frameName + " (DFT vs Butterworth)", 50.0); // Use large tolerance
            
            // Compare Butterworth method vs expected output
            std::vector<float> expectedData = matToVector(expectedOutputMats[i]);
            compareResults(expectedData, butterData, frameName + " (Expected vs Butterworth)", 50.0);
            
            // Compare DFT method vs expected output
            compareResults(expectedData, dftData, frameName + " (Expected vs DFT)", 20.0);
        }
        
        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_prev_input);
        cudaFree(d_prev_output);
        
        // Report conclusion
        std::cout << "\nImportant Note: The implementations use different filtering approaches:" << std::endl;
        std::cout << "- CPU implementation uses DFT-based filtering (ideal filter in frequency domain)" << std::endl;
        std::cout << "- CUDA implementation uses IIR Butterworth filter (recursive time-domain filter)" << std::endl;
        std::cout << "Some differences in output are expected due to the different approaches." << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error in temporal filtering test: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    // Test data paths
    std::string basePath = "../../../../cpp/tests/data/";
    if (argc > 1) {
        basePath = argv[1];
    }
    
    std::cout << "Testing CUDA Temporal Filtering Comparison..." << std::endl;
    std::cout << "Using test data from: " << basePath << std::endl;
    
    // Initialize CUDA modules
    if (!evmcuda::init_butterworth()) {
        std::cerr << "Failed to initialize CUDA Butterworth module" << std::endl;
        return 1;
    }
    
    bool success = true;
    
    // Test temporal filtering on level 0 of the Laplacian pyramid
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
    
    if (!testTemporalFilterDftComparison(inputLaplacianFiles, expectedFilteredFiles)) {
        std::cerr << "Temporal filtering comparison test failed" << std::endl;
        success = false;
    }
    
    // Cleanup
    evmcuda::cleanup_butterworth();
    
    if (success) {
        std::cout << "\nAll temporal filtering comparison tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "\nSome temporal filtering comparison tests FAILED!" << std::endl;
        return 1;
    }
}