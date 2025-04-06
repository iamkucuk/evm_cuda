#ifndef EVMCPP_TEST_HELPERS_HPP
#define EVMCPP_TEST_HELPERS_HPP

#include <gtest/gtest.h> // For ::testing::AssertionResult
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept> // For runtime_error
#include <limits>    // For numeric_limits
#include <iostream>  // For error messages

// Compile-time definition set by CMake in tests/CMakeLists.txt
#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "." // Default fallback if not defined
#endif

// Template function to load a matrix from a CSV text file saved by numpy.savetxt
// Definition must be in the header for template instantiation.
template <typename T>
cv::Mat loadMatrixFromTxt(const std::string& filename, int expected_channels = 3) {
    // Construct the full path using the compile definition
    std::string full_path = std::string(TEST_DATA_DIR) + "/" + filename;
    std::ifstream file(full_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open test data file: " + full_path + " (Original filename: " + filename + ")");
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
    else if (std::is_same<T, int8_t>::value) depth = CV_8S;
    else if (std::is_same<T, uint16_t>::value) depth = CV_16U;
    else if (std::is_same<T, int16_t>::value) depth = CV_16S;
    else if (std::is_same<T, int32_t>::value) depth = CV_32S;
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

// Function to compare two float matrices element-wise with tolerance
// Note: CompareMatrices definition remains in test_helpers.cpp
::testing::AssertionResult CompareMatrices(const cv::Mat& mat1, const cv::Mat& mat2, double tolerance = 1e-4); // Increased default tolerance

// Function to apply FFT-based temporal bandpass filter and amplify (implementation in test_helpers.cpp)
// Mimics the Python filterGaussianPyramids output saved as steps 4, 5, and 6b reference data.
std::vector<cv::Mat> applyFftTemporalFilterAndAmplify(
    const std::vector<cv::Mat>& spatial_filtered_sequence, // Input for filtering AND adding back
    double fl,
    double fh,
    double samplingRate,
    double alpha,
    double chromAttenuation
);

// upsamplePyramidLevel declaration removed (moved to gaussian_pyramid.hpp)

#endif // EVMCPP_TEST_HELPERS_HPP