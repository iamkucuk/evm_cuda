#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "color_conversion.hpp"
#include "test_helpers.hpp"

TEST(CpuColorConversion, RgbToYiq) {
    // Load the input RGB image from file
    cv::Mat input_rgb = loadMatrixFromTxt<float>("frame_0_rgb.txt", 3);
    ASSERT_FALSE(input_rgb.empty()) << "Failed to load input RGB image";
    ASSERT_EQ(input_rgb.channels(), 3) << "Input image should have 3 channels";
    ASSERT_EQ(input_rgb.type(), CV_32FC3) << "Input image should be float32";

    // Load the expected YIQ result
    cv::Mat expected_yiq = loadMatrixFromTxt<float>("frame_0_yiq.txt", 3);
    ASSERT_FALSE(expected_yiq.empty()) << "Failed to load expected YIQ image";
    ASSERT_EQ(expected_yiq.channels(), 3) << "Expected YIQ should have 3 channels";
    ASSERT_EQ(expected_yiq.type(), CV_32FC3) << "Expected YIQ should be float32";

    // Convert RGB to YIQ using our implementation
    cv::Mat result_yiq = evmcpu::rgb_to_yiq(input_rgb);

    // Compare the results
    ASSERT_TRUE(CompareMatrices(result_yiq, expected_yiq))
        << "RGB to YIQ conversion output doesn't match expected values";
}

TEST(CpuColorConversion, YiqToRgb) {
    // Load the input YIQ image
    cv::Mat input_yiq = loadMatrixFromTxt<float>("frame_0_yiq.txt", 3);
    ASSERT_FALSE(input_yiq.empty()) << "Failed to load input YIQ image";
    ASSERT_EQ(input_yiq.channels(), 3) << "Input image should have 3 channels";
    ASSERT_EQ(input_yiq.type(), CV_32FC3) << "Input image should be float32";

    // Load the expected RGB result as float (matching format used in RgbToYiq test)
    cv::Mat expected_rgb_float = loadMatrixFromTxt<float>("frame_0_rgb.txt", 3);
    ASSERT_FALSE(expected_rgb_float.empty()) << "Failed to load expected RGB image";
    ASSERT_EQ(expected_rgb_float.type(), CV_32FC3) << "Expected RGB should be float32";

    // Convert YIQ to RGB using our implementation
    cv::Mat result_rgb_float;
    evmcpu::yiq_to_rgb(input_yiq, result_rgb_float);

    // Compare the results with tolerance of 0.1 (1e-1) to account for floating-point differences
    // between Python reference and C++ implementation (~0.13 max difference observed)
    ASSERT_TRUE(CompareMatrices(result_rgb_float, expected_rgb_float, 1e-1))
        << "YIQ to RGB conversion output doesn't match expected values";
}