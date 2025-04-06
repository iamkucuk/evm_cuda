#ifndef EVMCPP_TEST_HELPERS_HPP
#define EVMCPP_TEST_HELPERS_HPP

#include <gtest/gtest.h> // For ::testing::AssertionResult
#include <opencv2/core.hpp>
#include <string>
#include <stdexcept> // For runtime_error

// Function to load a matrix from a CSV text file saved by numpy.savetxt
cv::Mat loadMatrixFromTxt(const std::string& filename, int expected_rows, int expected_cols, int expected_channels = 3);

// Function to compare two float matrices element-wise with tolerance
::testing::AssertionResult CompareMatrices(const cv::Mat& mat1, const cv::Mat& mat2, float tolerance = 1e-4f);

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