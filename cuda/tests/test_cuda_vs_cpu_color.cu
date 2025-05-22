#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

// Include both CPU and CUDA implementations
#include "color_conversion.hpp"  // CPU implementation
#include "cuda_color_conversion.cuh"  // CUDA implementation

// Helper function to read test data from CSV file (matching CPU tests)
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

    // Determine matrix type based on template type T and expected channels
    int depth = -1;
    if (std::is_same<T, float>::value) depth = CV_32F;
    else if (std::is_same<T, double>::value) depth = CV_64F;
    else if (std::is_same<T, uint8_t>::value) depth = CV_8U;
    else throw std::runtime_error("Unsupported template type for loadMatrixFromTxt");

    int mat_type = CV_MAKETYPE(depth, expected_channels);
    int expected_cols = cols_file / expected_channels;
    int expected_rows = rows;

    if (cols_file % expected_channels != 0) {
         throw std::runtime_error("Number of columns (" + std::to_string(cols_file) + ") is not divisible by expected channels (" + std::to_string(expected_channels) + ") for file " + filename);
    }

    if (data.size() != static_cast<size_t>(rows * cols_file)) {
         throw std::runtime_error("Data size mismatch after loading file: " + filename);
    }

    // Create Mat from vector data (requires copy), then reshape
    cv::Mat flat_mat(rows, cols_file, CV_MAKETYPE(depth, 1), data.data()); // Create flat matrix first
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

// Test RGB to YIQ conversion: CPU vs CUDA
bool testRgbToYiqCpuVsCuda(const std::string& rgbDataFile) {
    std::cout << "=== RGB to YIQ: CPU vs CUDA Comparison ===" << std::endl;
    
    // Load input RGB data
    cv::Mat input_rgb_mat = loadMatrixFromTxt<float>(rgbDataFile, 3);
    if (input_rgb_mat.empty()) {
        std::cerr << "Failed to load RGB input data" << std::endl;
        return false;
    }
    
    std::cout << "Input image: " << input_rgb_mat.cols << "x" << input_rgb_mat.rows 
              << " channels=" << input_rgb_mat.channels() << std::endl;
    
    // Run CPU implementation
    cv::Mat cpu_result_mat = evmcpu::rgb_to_yiq(input_rgb_mat);
    
    // Convert input to vector for CUDA
    std::vector<float> input_rgb_data = matToVector(input_rgb_mat);
    std::vector<float> cuda_result_data(input_rgb_data.size());
    
    // Run CUDA implementation
    try {
        evmcuda::rgb_to_yiq_wrapper(input_rgb_data.data(), cuda_result_data.data(), 
                                    input_rgb_mat.cols, input_rgb_mat.rows);
    } catch (const std::exception& e) {
        std::cerr << "Error in CUDA rgb_to_yiq_wrapper: " << e.what() << std::endl;
        return false;
    }
    
    // Compare CPU vs CUDA results
    compareResults(cpu_result_mat, cuda_result_data, "RGB->YIQ CPU vs CUDA", 1e-5);
    
    return true;
}

// Test YIQ to RGB conversion: CPU vs CUDA
bool testYiqToRgbCpuVsCuda(const std::string& yiqDataFile) {
    std::cout << "=== YIQ to RGB: CPU vs CUDA Comparison ===" << std::endl;
    
    // Load input YIQ data
    cv::Mat input_yiq_mat = loadMatrixFromTxt<float>(yiqDataFile, 3);
    if (input_yiq_mat.empty()) {
        std::cerr << "Failed to load YIQ input data" << std::endl;
        return false;
    }
    
    std::cout << "Input image: " << input_yiq_mat.cols << "x" << input_yiq_mat.rows 
              << " channels=" << input_yiq_mat.channels() << std::endl;
    
    // Run CPU implementation
    cv::Mat cpu_result_mat;
    evmcpu::yiq_to_rgb(input_yiq_mat, cpu_result_mat);
    
    // Convert input to vector for CUDA
    std::vector<float> input_yiq_data = matToVector(input_yiq_mat);
    std::vector<float> cuda_result_data(input_yiq_data.size());
    
    // Run CUDA implementation
    try {
        evmcuda::yiq_to_rgb_wrapper(input_yiq_data.data(), cuda_result_data.data(), 
                                    input_yiq_mat.cols, input_yiq_mat.rows);
    } catch (const std::exception& e) {
        std::cerr << "Error in CUDA yiq_to_rgb_wrapper: " << e.what() << std::endl;
        return false;
    }
    
    // Compare CPU vs CUDA results
    compareResults(cpu_result_mat, cuda_result_data, "YIQ->RGB CPU vs CUDA", 1e-4);
    
    return true;
}

int main(int argc, char* argv[]) {
    // Test data paths
    std::string basePath = "../../../cpp/tests/data/";
    if (argc > 1) {
        basePath = argv[1];
    }
    
    std::string rgbDataFile = basePath + "frame_0_rgb.txt";
    std::string yiqDataFile = basePath + "frame_0_yiq.txt";
    
    // Initialize CUDA color conversion module
    if (!evmcuda::init_color_conversion()) {
        std::cerr << "Failed to initialize CUDA color conversion module" << std::endl;
        return 1;
    }
    
    std::cout << "Testing CUDA vs CPU Color Conversion..." << std::endl;
    std::cout << "Using test data from:" << std::endl;
    std::cout << "  RGB: " << rgbDataFile << std::endl;
    std::cout << "  YIQ: " << yiqDataFile << std::endl << std::endl;
    
    bool success = true;
    
    // Test RGB to YIQ conversion
    if (!testRgbToYiqCpuVsCuda(rgbDataFile)) {
        std::cerr << "RGB to YIQ CPU vs CUDA test failed" << std::endl;
        success = false;
    }
    
    // Test YIQ to RGB conversion
    if (!testYiqToRgbCpuVsCuda(yiqDataFile)) {
        std::cerr << "YIQ to RGB CPU vs CUDA test failed" << std::endl;
        success = false;
    }
    
    // Cleanup
    evmcuda::cleanup_color_conversion();
    
    if (success) {
        std::cout << "All CPU vs CUDA color conversion tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some CPU vs CUDA color conversion tests FAILED!" << std::endl;
        return 1;
    }
}