#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "evmcpu/pyramid.hpp"
#include "evmcpu/color_conversion.hpp"
#include "test_helpers.hpp"

class CpuPyramidTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create 5x5 Gaussian kernel (exact match with Python implementation)
        float kernel_data[] = {
            1,  4,  6,  4, 1,
            4, 16, 24, 16, 4,
            6, 24, 36, 24, 6,
            4, 16, 24, 16, 4,
            1,  4,  6,  4, 1
        };
        kernel_ = cv::Mat(5, 5, CV_32F, kernel_data).clone() / 256.0f;
    }

    cv::Mat kernel_;
};

TEST_F(CpuPyramidTest, PyrDown) {
    // Load input RGB image
    cv::Mat rgb_input = loadMatrixFromTxt<float>("frame_0_rgb.txt", 3);
    ASSERT_FALSE(rgb_input.empty()) << "Failed to load input RGB image";
    ASSERT_EQ(rgb_input.type(), CV_32FC3);

    // Convert RGB to YIQ (matching Python's processing)
    cv::Mat input = evmcpu::rgb_to_yiq(rgb_input);

    // Load expected output after pyrDown (which was generated from YIQ input)
    cv::Mat expected = loadMatrixFromTxt<float>("frame_0_pyrdown_0.txt", 3);
    ASSERT_FALSE(expected.empty()) << "Failed to load expected output";
    ASSERT_EQ(expected.type(), CV_32FC3);

    // Apply pyrDown
    cv::Mat result = evmcpu::pyr_down(input, kernel_);

    // Compare results
    ASSERT_TRUE(CompareMatrices(result, expected, 1e-4))
        << "PyrDown output doesn't match expected values";
}

TEST_F(CpuPyramidTest, PyrUp) {
    // Load input downsampled YIQ image
    cv::Mat input = loadMatrixFromTxt<float>("frame_0_pyrdown_0.txt", 3);
    ASSERT_FALSE(input.empty()) << "Failed to load input YIQ image";
    ASSERT_EQ(input.type(), CV_32FC3);

    // Load expected YIQ output after pyrUp
    cv::Mat expected = loadMatrixFromTxt<float>("frame_0_pyrup_0.txt", 3);
    ASSERT_FALSE(expected.empty()) << "Failed to load expected YIQ output";
    ASSERT_EQ(expected.type(), CV_32FC3);

    // Apply pyrUp with target shape from expected output
    cv::Mat result = evmcpu::pyr_up(input, kernel_, expected.size());

    // Verify dimensions match
    ASSERT_EQ(result.size(), expected.size())
        << "PyrUp output dimensions don't match expected dimensions";

    // Compare results
    ASSERT_TRUE(CompareMatrices(result, expected, 1e-4))
        << "PyrUp output doesn't match expected values";
}

// Test error cases
TEST_F(CpuPyramidTest, EmptyInput) {
    cv::Mat empty;
    EXPECT_THROW(evmcpu::pyr_down(empty, kernel_), std::invalid_argument);
    EXPECT_THROW(evmcpu::pyr_up(empty, kernel_), std::invalid_argument);
}

TEST_F(CpuPyramidTest, WrongType) {
    cv::Mat uint8_mat(10, 10, CV_8UC3);
    EXPECT_THROW(evmcpu::pyr_down(uint8_mat, kernel_), std::invalid_argument);
    EXPECT_THROW(evmcpu::pyr_up(uint8_mat, kernel_), std::invalid_argument);
}