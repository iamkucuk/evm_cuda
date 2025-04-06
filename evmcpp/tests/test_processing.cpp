#include <gtest/gtest.h>
#include "evmcpp/processing.hpp" // Include the header for the functions we want to test
#include "test_helpers.hpp"      // Include common test helpers
#include <opencv2/core.hpp> // Include OpenCV for basic types like cv::Mat
#include <opencv2/imgproc.hpp> // Include OpenCV for image processing functions like getGaussianKernel
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <limits> // Required for numeric_limits

// Helper functions loadMatrixFromTxt and CompareMatrices are now in test_helpers.cpp/.hpp

// --- Test Fixture ---
class ProcessingTest : public ::testing::Test {
protected:
    // Define expected dimensions based on face.mp4 frame 0
    const int frame_rows = 592;
    const int frame_cols = 528;
    const int down_rows = frame_rows / 2; // 296
    const int down_cols = frame_cols / 2; // 264

    // const std::string data_dir = "data/"; // No longer needed, path handled by TEST_DATA_DIR macro

    cv::Mat rgb_frame0;
    cv::Mat yiq_ref0;
    cv::Mat pyrdown_ref0;
    cv::Mat pyrup_ref0;

    void SetUp() override {
        // Load reference data for frame 0
        try {
            // Use template version: loadMatrixFromTxt<DataType>(filename, channels)
            rgb_frame0 = loadMatrixFromTxt<float>("frame_0_rgb.txt", 3);
            yiq_ref0 = loadMatrixFromTxt<float>("frame_0_yiq.txt", 3);
            pyrdown_ref0 = loadMatrixFromTxt<float>("frame_0_pyrdown_0.txt", 3);
            // Note: pyrUp output should match the size *before* downsampling
            pyrup_ref0 = loadMatrixFromTxt<float>("frame_0_pyrup_0.txt", 3);
        } catch (const std::exception& e) {
            // Fail the test immediately if data loading fails
            GTEST_FAIL() << "Failed to load test data: " << e.what();
        }
    }
};

// --- Test Cases ---

TEST_F(ProcessingTest, Rgb2YiqNumerical) {
    ASSERT_FALSE(rgb_frame0.empty()) << "Input RGB frame data is empty.";
    ASSERT_FALSE(yiq_ref0.empty()) << "Reference YIQ data is empty.";

    cv::Mat yiq_result;
    ASSERT_NO_THROW(yiq_result = evmcpp::rgb2yiq(rgb_frame0));

    ASSERT_FALSE(yiq_result.empty());
    ASSERT_EQ(yiq_result.size(), yiq_ref0.size());
    ASSERT_EQ(yiq_result.type(), yiq_ref0.type());

    // Compare numerically
    EXPECT_TRUE(CompareMatrices(yiq_result, yiq_ref0)); // Use default tolerance (now 2e-5f)
}

TEST_F(ProcessingTest, PyrDownNumerical) {
    // pyrDown operates on the YIQ image in the pipeline
    ASSERT_FALSE(yiq_ref0.empty()) << "Input YIQ frame data is empty.";
    ASSERT_FALSE(pyrdown_ref0.empty()) << "Reference pyrDown data is empty.";

    // Define the kernel used in Python reference data generation
    float kernel_data[25] = {
        1,  4,  6,  4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1,  4,  6,  4, 1
    };
    cv::Mat kernel = cv::Mat(5, 5, CV_32F, kernel_data) / 256.0f;
    cv::Mat down_result;
    // Use cv::pyrDown - it modifies the output argument directly
    ASSERT_NO_THROW(cv::pyrDown(yiq_ref0, down_result));

    ASSERT_FALSE(down_result.empty());
    ASSERT_EQ(down_result.size(), pyrdown_ref0.size());
    ASSERT_EQ(down_result.type(), pyrdown_ref0.type());

    EXPECT_TRUE(CompareMatrices(down_result, pyrdown_ref0)); // Use default tolerance
}

TEST_F(ProcessingTest, PyrUpNumerical) {
    // pyrUp operates on the downsampled image, target size is the original YIQ size
    ASSERT_FALSE(pyrdown_ref0.empty()) << "Input pyrDown frame data is empty.";
    ASSERT_FALSE(pyrup_ref0.empty()) << "Reference pyrUp data is empty.";
    ASSERT_FALSE(yiq_ref0.empty()) << "Reference YIQ frame needed for target size.";

    cv::Size target_size = yiq_ref0.size(); // Target size is the size before downsampling

    // Define the kernel used in Python reference data generation
    float kernel_data[25] = {
        1,  4,  6,  4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1,  4,  6,  4, 1
    };
    cv::Mat kernel = cv::Mat(5, 5, CV_32F, kernel_data) / 256.0f;
    cv::Mat up_result;
    // Use cv::pyrUp - it modifies the output argument directly
    ASSERT_NO_THROW(cv::pyrUp(pyrdown_ref0, up_result, target_size));

    ASSERT_FALSE(up_result.empty());
    ASSERT_EQ(up_result.size(), pyrup_ref0.size());
    ASSERT_EQ(up_result.type(), pyrup_ref0.type());

    EXPECT_TRUE(CompareMatrices(up_result, pyrup_ref0)); // Use default tolerance
}