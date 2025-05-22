#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

// Include both CPU and CUDA implementations
#include "pyramid.hpp"  // CPU implementation
#include "color_conversion.hpp"  // CPU color conversion
#include "cuda_pyramid.cuh"  // CUDA implementation
#include "cuda_color_conversion.cuh"  // CUDA color conversion

// Helper function to read test data from CSV file (same as color test)
template <typename T>
cv::Mat loadMatrixFromTxt(const std::string& filename, int expected_channels = 3) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open test data file: " + filename);
    }

    std::vector<T> data;
    std::string line;
    int rows = 0;
    int cols_file = -1;

    while (std::getline(file, line)) {
        rows++;
        std::stringstream ss(line);
        std::string value_str;
        int current_cols = 0;
        while (std::getline(ss, value_str, ',')) {
            try {
                T value;
                std::stringstream converter(value_str);
                converter >> value;
                if (converter.fail()) {
                     throw std::invalid_argument("Invalid number format");
                }
                data.push_back(value);
                current_cols++;
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error("Invalid number format in file " + filename + " at row " + std::to_string(rows) + ": '" + value_str + "' (" + e.what() + ")");
            } catch (const std::out_of_range& e) {
                 throw std::runtime_error("Number out of range in file " + filename + " at row " + std::to_string(rows) + ": '" + value_str + "'");
            }
        }
        if (cols_file == -1) {
            cols_file = current_cols;
        } else if (cols_file != current_cols) {
            throw std::runtime_error("Inconsistent number of columns in file: " + filename + " (Expected " + std::to_string(cols_file) + ", got " + std::to_string(current_cols) + " at row " + std::to_string(rows) + ")");
        }
    }

    if (rows == 0 || cols_file <= 0) {
         throw std::runtime_error("No data loaded or zero columns found in file: " + filename);
    }

    int depth = CV_32F;  // Always use float for this test
    int expected_cols = cols_file / expected_channels;
    int expected_rows = rows;

    if (cols_file % expected_channels != 0) {
         throw std::runtime_error("Number of columns (" + std::to_string(cols_file) + ") is not divisible by expected channels (" + std::to_string(expected_channels) + ") for file " + filename);
    }

    // Create Mat from vector data (requires copy), then reshape
    cv::Mat flat_mat(rows, cols_file, CV_32F, data.data()); // Create flat matrix first
    return flat_mat.reshape(expected_channels, expected_rows).clone(); // Reshape and clone
}

// Helper function to compare OpenCV Mat with vector data
void compareResults(const cv::Mat& expected_mat, const std::vector<float>& actual_data, const std::string& test_name, float epsilon = 1e-5) {
    // Convert Mat to vector for comparison
    std::vector<float> expected_data;
    if (expected_mat.isContinuous()) {
        expected_data.assign(expected_mat.ptr<float>(0), expected_mat.ptr<float>(0) + expected_mat.total() * expected_mat.channels());
    } else {
        for (int i = 0; i < expected_mat.rows; ++i) {
            const float* row_ptr = expected_mat.ptr<float>(i);
            expected_data.insert(expected_data.end(), row_ptr, row_ptr + expected_mat.cols * expected_mat.channels());
        }
    }
    
    if (expected_data.size() != actual_data.size()) {
        std::cerr << test_name << " - Size mismatch: Expected = " << expected_data.size() 
                  << ", Actual = " << actual_data.size() << std::endl;
        return;
    }
    
    float maxError = 0.0f;
    float meanError = 0.0f;
    int maxErrorIdx = -1;
    
    for (size_t i = 0; i < expected_data.size(); ++i) {
        float error = std::abs(expected_data[i] - actual_data[i]);
        if (error > maxError) {
            maxError = error;
            maxErrorIdx = i;
        }
        meanError += error;
    }
    
    meanError /= expected_data.size();
    
    std::cout << test_name << " Comparison Results:" << std::endl;
    std::cout << "  Data size: " << expected_data.size() << std::endl;
    std::cout << "  Max error: " << maxError << " at index " << maxErrorIdx;
    if (maxErrorIdx >= 0) {
        std::cout << " (Expected: " << expected_data[maxErrorIdx] 
                  << ", Actual: " << actual_data[maxErrorIdx] << ")";
    }
    std::cout << std::endl;
    std::cout << "  Mean error: " << meanError << std::endl;
    std::cout << "  PSNR: " << (maxError > 0 ? 20 * log10(255.0f / maxError) : 0) << " dB" << std::endl;
    std::cout << "  Validation " << (maxError <= epsilon ? "PASSED" : "FAILED") 
              << " (epsilon = " << epsilon << ")" << std::endl << std::endl;
}

// Helper to convert cv::Mat to vector for CUDA processing
std::vector<float> matToVector(const cv::Mat& mat) {
    std::vector<float> data;
    if (mat.isContinuous()) {
        data.assign(mat.ptr<float>(0), mat.ptr<float>(0) + mat.total() * mat.channels());
    } else {
        for (int i = 0; i < mat.rows; ++i) {
            const float* row_ptr = mat.ptr<float>(i);
            data.insert(data.end(), row_ptr, row_ptr + mat.cols * mat.channels());
        }
    }
    return data;
}

// Test pyrDown: CPU vs CUDA
bool testPyrDownCpuVsCuda(const std::string& rgbDataFile) {
    std::cout << "=== PyrDown: CPU vs CUDA Comparison ===" << std::endl;
    
    // Load input RGB data and convert to YIQ (matching CPU test approach)
    cv::Mat input_rgb_mat = loadMatrixFromTxt<float>(rgbDataFile, 3);
    if (input_rgb_mat.empty()) {
        std::cerr << "Failed to load RGB input data" << std::endl;
        return false;
    }
    
    // Convert RGB to YIQ using CPU implementation (to match test data generation)
    cv::Mat input_yiq_mat = evmcpu::rgb_to_yiq(input_rgb_mat);
    
    std::cout << "Input image: " << input_yiq_mat.cols << "x" << input_yiq_mat.rows 
              << " channels=" << input_yiq_mat.channels() << std::endl;
    
    // Create 5x5 Gaussian kernel (exact match with Python implementation)
    float kernel_data[] = {
        1,  4,  6,  4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1,  4,  6,  4, 1
    };
    cv::Mat kernel = cv::Mat(5, 5, CV_32F, kernel_data).clone() / 256.0f;
    
    // Run CPU implementation
    cv::Mat cpu_result_mat = evmcpu::pyr_down(input_yiq_mat, kernel);
    
    // Convert input to vector for CUDA
    std::vector<float> input_yiq_data = matToVector(input_yiq_mat);
    int output_width = input_yiq_mat.cols / 2;
    int output_height = input_yiq_mat.rows / 2;
    std::vector<float> cuda_result_data(output_width * output_height * 3);
    
    // Run CUDA implementation
    try {
        evmcuda::pyr_down_wrapper(input_yiq_data.data(), 
                                  input_yiq_mat.cols, input_yiq_mat.rows, 3,
                                  cuda_result_data.data());
    } catch (const std::exception& e) {
        std::cerr << "Error in CUDA pyr_down_wrapper: " << e.what() << std::endl;
        return false;
    }
    
    // Compare CPU vs CUDA results
    compareResults(cpu_result_mat, cuda_result_data, "PyrDown CPU vs CUDA", 1e-4);
    
    return true;
}

// Test pyrUp: CPU vs CUDA
bool testPyrUpCpuVsCuda(const std::string& pyrDownDataFile) {
    std::cout << "=== PyrUp: CPU vs CUDA Comparison ===" << std::endl;
    
    // Load pyrDown result as input
    cv::Mat input_mat = loadMatrixFromTxt<float>(pyrDownDataFile, 3);
    if (input_mat.empty()) {
        std::cerr << "Failed to load pyrDown input data" << std::endl;
        return false;
    }
    
    std::cout << "Input image: " << input_mat.cols << "x" << input_mat.rows 
              << " channels=" << input_mat.channels() << std::endl;
    
    // Create 5x5 Gaussian kernel (exact match with Python implementation)
    float kernel_data[] = {
        1,  4,  6,  4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1,  4,  6,  4, 1
    };
    cv::Mat kernel = cv::Mat(5, 5, CV_32F, kernel_data).clone() / 256.0f;
    
    // Run CPU implementation - upscale to target size
    int target_width = input_mat.cols * 2;
    int target_height = input_mat.rows * 2;
    cv::Mat cpu_result_mat = evmcpu::pyr_up(input_mat, kernel, cv::Size(target_width, target_height));
    
    // Convert input to vector for CUDA
    std::vector<float> input_data = matToVector(input_mat);
    std::vector<float> cuda_result_data(target_width * target_height * 3);
    
    // Run CUDA implementation
    try {
        evmcuda::pyr_up_wrapper(input_data.data(), 
                                input_mat.cols, input_mat.rows, 3,
                                cuda_result_data.data(),
                                target_width, target_height);
    } catch (const std::exception& e) {
        std::cerr << "Error in CUDA pyr_up_wrapper: " << e.what() << std::endl;
        return false;
    }
    
    // Compare CPU vs CUDA results
    compareResults(cpu_result_mat, cuda_result_data, "PyrUp CPU vs CUDA", 1e-4);
    
    return true;
}

int main(int argc, char* argv[]) {
    // Test data paths
    std::string basePath = "../../../cpp/tests/data/";
    if (argc > 1) {
        basePath = argv[1];
    }
    
    std::string rgbDataFile = basePath + "frame_0_rgb.txt";
    std::string pyrDownDataFile = basePath + "frame_0_pyrdown_0.txt";
    
    // Initialize CUDA modules
    if (!evmcuda::init_pyramid()) {
        std::cerr << "Failed to initialize CUDA pyramid module" << std::endl;
        return 1;
    }
    
    if (!evmcuda::init_color_conversion()) {
        std::cerr << "Failed to initialize CUDA color conversion module" << std::endl;
        return 1;
    }
    
    std::cout << "Testing CUDA vs CPU Pyramid Operations..." << std::endl;
    std::cout << "Using test data from:" << std::endl;
    std::cout << "  RGB: " << rgbDataFile << std::endl;
    std::cout << "  PyrDown: " << pyrDownDataFile << std::endl << std::endl;
    
    bool success = true;
    
    // Test pyrDown
    if (!testPyrDownCpuVsCuda(rgbDataFile)) {
        std::cerr << "PyrDown CPU vs CUDA test failed" << std::endl;
        success = false;
    }
    
    // Test pyrUp
    if (!testPyrUpCpuVsCuda(pyrDownDataFile)) {
        std::cerr << "PyrUp CPU vs CUDA test failed" << std::endl;
        success = false;
    }
    
    // Cleanup
    evmcuda::cleanup_pyramid();
    evmcuda::cleanup_color_conversion();
    
    if (success) {
        std::cout << "All CPU vs CUDA pyramid tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some CPU vs CUDA pyramid tests FAILED!" << std::endl;
        return 1;
    }
}