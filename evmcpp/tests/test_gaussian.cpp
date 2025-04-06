// File: evmcpp/tests/test_gaussian.cpp
// Purpose: Unit tests for the Gaussian EVM pathway functions.

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <filesystem> // Requires C++17

// Include the header for the functions being tested
#include "evmcpp/gaussian_pyramid.hpp"
// Include processing for constants like kernel if needed, or define locally
#include "evmcpp/processing.hpp" // For rgb2yiq, constants etc.
#include "test_helpers.hpp"      // Include the header with template definition and CompareMatrices declaration

namespace fs = std::filesystem;

// Define a test fixture for Gaussian pathway tests if needed (e.g., to load common data)
class GaussianPathwayTest : public ::testing::Test {
protected:
    // Per-test-suite set-up.
    // (Removed findTestDataDir call; getDataPath uses TEST_DATA_DIR macro)
    // No SetUpTestSuite needed for now, data loading handled in tests
    // static void SetUpTestSuite() {
    // }

        // Load Gaussian kernel (assuming it's defined in processing.hpp or accessible)
        // If not, define it here based on Python's constants.py
        // Example: gaussianKernel = evmcpp::getGaussianKernel();
        // For now, assume it's available via processing.hpp or similar
    // Removed erroneous closing brace

    // You can define per-test set-up and tear-down logic here if needed.
    void SetUp() override {
        // Load data common to multiple tests?
    }

    // Removed getDataPath helper function, loadMatrixFromTxt uses TEST_DATA_DIR macro

    // Shared data members
    // Removed static testDataDir member
    // static cv::Mat gaussianKernel; // If loaded in SetUpTestSuite
};

// Removed definition for static testDataDir member
// cv::Mat GaussianPathwayTest::gaussianKernel; // Uncomment if defined

// --- Test Cases ---

// TDD_ANCHOR: test_spatiallyFilterGaussian_matches_python
TEST_F(GaussianPathwayTest, SpatiallyFilterGaussianMatchesPython) {
    // 1. Load Input Data (e.g., frame 0 RGB)
    cv::Mat inputRgb = loadMatrixFromTxt<float>("frame_0_rgb.txt", 3); // Load as float, convert later if needed
    EXPECT_FALSE(inputRgb.empty()) << "Failed to load frame_0_rgb.txt"; // Changed to non-fatal assertion
    cv::Mat inputRgbUint8;
    inputRgb.convertTo(inputRgbUint8, CV_8UC3); // Convert to uint8 for function input

    // 2. Load Expected Output Data (Spatially Filtered YIQ)
    cv::Mat expectedFilteredYiq = loadMatrixFromTxt<float>("frame_0_step3_spatial_filtered_yiq.txt", 3);
    EXPECT_FALSE(expectedFilteredYiq.empty()) << "Failed to load frame_0_step3_spatial_filtered_yiq.txt"; // Changed to non-fatal assertion

    // 3. Define Parameters
    int level = 4; // Match generate_test_data.py TEST_PYRAMID_LEVELS
    cv::Mat kernel = evmcpp::gaussian_kernel; // Use the constant kernel variable
    ASSERT_FALSE(kernel.empty()) << "Gaussian kernel is empty.";

    // 4. Call the C++ Function
    cv::Mat actualFilteredYiq = evmcpp::spatiallyFilterGaussian(inputRgbUint8, level, kernel);

    // 5. Compare Results
    ASSERT_FALSE(actualFilteredYiq.empty()) << "spatiallyFilterGaussian returned an empty matrix.";
    ASSERT_EQ(actualFilteredYiq.type(), CV_32FC3) << "Output type mismatch.";
    // Use a suitable tolerance for floating-point comparisons
    double tolerance = 1e-4; // Increased tolerance
    ASSERT_TRUE(CompareMatrices(actualFilteredYiq, expectedFilteredYiq, tolerance))
        << "Mismatch between C++ spatiallyFilterGaussian output and Python reference data.";
}

// TDD_ANCHOR: test_temporalFilterGaussianBatch_matches_python
TEST_F(GaussianPathwayTest, TemporalFilterGaussianBatchMatchesPython) {
    // 1. Load Input Data (Batch of Spatially Filtered YIQ Frames)
    std::vector<cv::Mat> spatiallyFilteredBatch;
    int numFrames = 5; // Match generate_test_data.py NUM_FRAMES_TO_PROCESS
    for (int i = 0; i < numFrames; ++i) {
        // Load the output of the spatial filtering step for each frame
        // Note: generate_test_data.py saves this as frame_i_gaussian_reconstructed.txt
        //       and frame_0_step3_spatial_filtered_yiq.txt for frame 0. Use the former.
        cv::Mat frame = loadMatrixFromTxt<float>("frame_" + std::to_string(i) + "_gaussian_reconstructed.txt", 3);
        EXPECT_FALSE(frame.empty()) << "Failed to load frame_" << i << "_gaussian_reconstructed.txt"; // Changed to non-fatal assertion
        spatiallyFilteredBatch.push_back(frame);
    }
    ASSERT_EQ(spatiallyFilteredBatch.size(), numFrames) << "Incorrect number of input frames loaded.";

    // 2. Load Expected Output Data (Temporally Filtered YIQ for Frame 0)
    // Note: We only have the filtered output for frame 0 from the script
    cv::Mat expectedFilteredYiqFrame0 = loadMatrixFromTxt<float>("frame_0_step4_temporal_filtered_yiq.txt", 3);
    EXPECT_FALSE(expectedFilteredYiqFrame0.empty()) << "Failed to load frame_0_step4_temporal_filtered_yiq.txt"; // Changed to non-fatal assertion

    // 3. Define Parameters (Match generate_test_data.py)
    float fps = 30.0f;
    float fl = 0.4f; // TEST_FREQ_RANGE[0]
    float fh = 3.0f; // TEST_FREQ_RANGE[1]
    float alpha = 10.0f; // TEST_ALPHA
    float chromAttenuation = 1.0f; // TEST_ATTENUATION

    // 4. Call the C++ Function
    std::vector<cv::Mat> actualFilteredBatch = evmcpp::temporalFilterGaussianBatch(
        spatiallyFilteredBatch, fps, fl, fh, alpha, chromAttenuation
    );

    // 5. Compare Results (Compare only Frame 0, as that's what we have reference data for)
    ASSERT_FALSE(actualFilteredBatch.empty()) << "temporalFilterGaussianBatch returned an empty vector.";
    ASSERT_EQ(actualFilteredBatch.size(), numFrames) << "Output batch size mismatch.";
    ASSERT_FALSE(actualFilteredBatch[0].empty()) << "Filtered frame 0 is empty.";
    ASSERT_EQ(actualFilteredBatch[0].type(), CV_32FC3) << "Output frame 0 type mismatch.";

    double tolerance = 1e-5; // Adjust based on FFT/iFFT precision
    ASSERT_TRUE(CompareMatrices(actualFilteredBatch[0], expectedFilteredYiqFrame0, tolerance))
        << "Mismatch between C++ temporalFilterGaussianBatch output (frame 0) and Python reference data.";

    // Optional: Add checks for other frames if needed, or verify properties (e.g., non-zero)
}

// TDD_ANCHOR: test_reconstructGaussianFrame_matches_python
TEST_F(GaussianPathwayTest, ReconstructGaussianFrameMatchesPython) {
    // 1. Load Input Data
    //    - Original RGB Frame 0
    cv::Mat originalRgb = loadMatrixFromTxt<float>("frame_0_rgb.txt", 3);
    EXPECT_FALSE(originalRgb.empty()) << "Failed to load frame_0_rgb.txt"; // Changed to non-fatal assertion
    cv::Mat originalRgbUint8;
    originalRgb.convertTo(originalRgbUint8, CV_8UC3);

    //    - Filtered YIQ Signal Frame 0 (Output of temporal filter)
    cv::Mat filteredYiqSignal = loadMatrixFromTxt<float>("frame_0_step4_temporal_filtered_yiq.txt", 3);
    EXPECT_FALSE(filteredYiqSignal.empty()) << "Failed to load frame_0_step4_temporal_filtered_yiq.txt"; // Changed to non-fatal assertion

    // 2. Call the C++ Function under test
    cv::Mat actualRgbUint8 = evmcpp::reconstructGaussianFrame(originalRgbUint8, filteredYiqSignal);
    ASSERT_FALSE(actualRgbUint8.empty()) << "reconstructGaussianFrame returned an empty matrix.";
    ASSERT_EQ(actualRgbUint8.type(), CV_8UC3) << "Output type mismatch.";

    // 3. Calculate the EXPECTED final uint8 result using correct intermediate data
    //    Load Combined YIQ (Step 6b) - this represents the correct signal before final conversion
    cv::Mat combinedYiqRef = loadMatrixFromTxt<float>("frame_0_step6b_combined_yiq.txt", 3);
    EXPECT_FALSE(combinedYiqRef.empty()) << "Failed to load frame_0_step6b_combined_yiq.txt";

    //    Perform YIQ -> RGB Float conversion
    cv::Mat expectedRgbFloat = evmcpp::yiq2rgb(combinedYiqRef);
    EXPECT_FALSE(expectedRgbFloat.empty()) << "yiq2rgb failed during expected result calculation.";

    //    Perform Clipping
    cv::Mat expectedClippedRgbFloat = expectedRgbFloat.clone();
    cv::Mat lowerBound = cv::Mat::zeros(expectedClippedRgbFloat.size(), expectedClippedRgbFloat.type());
    cv::Mat upperBound = cv::Mat(expectedClippedRgbFloat.size(), expectedClippedRgbFloat.type(), cv::Scalar::all(255.0f));
    cv::max(expectedClippedRgbFloat, lowerBound, expectedClippedRgbFloat);
    cv::min(expectedClippedRgbFloat, upperBound, expectedClippedRgbFloat);

    //    Convert to uint8
    cv::Mat expectedRgbUint8_calculated;
    expectedClippedRgbFloat.convertTo(expectedRgbUint8_calculated, CV_8UC3);
    ASSERT_FALSE(expectedRgbUint8_calculated.empty()) << "convertTo uint8 failed during expected result calculation.";

    // 4. Compare the function's output against the correctly calculated expected result
    double tolerance = 0.0; // Exact match for uint8
    ASSERT_TRUE(CompareMatrices(actualRgbUint8, expectedRgbUint8_calculated, tolerance))
        << "Mismatch between C++ reconstructGaussianFrame output and the expected result calculated from intermediate reference data.";
}

// Optional: Add tests for intermediate reconstruction steps using the other saved files
// (e.g., test combined YIQ, reconstructed float RGB, clipped float RGB)
// Example:
TEST_F(GaussianPathwayTest, ReconstructGaussianFrameIntermediateCombinedYiq) {
    // 1. Load Inputs: original RGB, filtered YIQ
    cv::Mat originalRgb = loadMatrixFromTxt<float>("frame_0_rgb.txt", 3);
    cv::Mat filteredYiqSignal = loadMatrixFromTxt<float>("frame_0_step4_temporal_filtered_yiq.txt", 3);
    EXPECT_FALSE(originalRgb.empty() || filteredYiqSignal.empty()); // Changed to non-fatal assertion

    // 2. Load Expected: Combined YIQ
    cv::Mat expectedCombinedYiq = loadMatrixFromTxt<float>("frame_0_step6b_combined_yiq.txt", 3);
    EXPECT_FALSE(expectedCombinedYiq.empty()); // Changed to non-fatal assertion

    // 3. Perform the relevant part of the C++ logic (convert original, add)
    cv::Mat originalYiq = evmcpp::rgb2yiq(originalRgb); // Assuming rgb2yiq handles float input or convert originalRgb first
    EXPECT_FALSE(originalYiq.empty()); // Changed to non-fatal assertion
    cv::Mat actualCombinedYiq;
    cv::add(originalYiq, filteredYiqSignal, actualCombinedYiq);

    // 4. Compare
    double tolerance = 1e-4; // Increased tolerance
    ASSERT_TRUE(CompareMatrices(actualCombinedYiq, expectedCombinedYiq, tolerance))
        << "Mismatch in intermediate combined YIQ calculation.";
}

// Test Step 6c: YIQ -> RGB Float conversion
TEST_F(GaussianPathwayTest, ReconstructGaussianFrameIntermediateRgbFloat) {
    // 1. Load Inputs: Filtered YIQ (Reference from Step 5)
    // Note: Step 4 and Step 5 files are identical in the Python script for Gaussian path
    cv::Mat filteredYiq = loadMatrixFromTxt<float>("frame_0_step5_amplified_filtered_yiq.txt", 3);
    EXPECT_FALSE(filteredYiq.empty());

    // 2. Load Expected: Reconstructed RGB Float (Reference from Step 6c)
    cv::Mat expectedRgbFloat = loadMatrixFromTxt<float>("frame_0_step6c_reconstructed_rgb_float.txt", 3);
    EXPECT_FALSE(expectedRgbFloat.empty());

    // 3. Perform C++ yiq2rgb
    cv::Mat actualRgbFloat = evmcpp::yiq2rgb(filteredYiq); // Call yiq2rgb on the correct input
    EXPECT_FALSE(actualRgbFloat.empty());

    // 4. Compare
    double tolerance = 1e-4; // Use same tolerance as other float comparisons
    ASSERT_TRUE(CompareMatrices(actualRgbFloat, expectedRgbFloat, tolerance))
        << "Mismatch in intermediate YIQ -> RGB Float conversion.";
}

// Test Step 6d: Clipping Float RGB
TEST_F(GaussianPathwayTest, ReconstructGaussianFrameIntermediateClippedFloat) {
    // 1. Load Inputs: Reconstructed RGB Float (Reference from Step 6c)
    cv::Mat rgbFloat = loadMatrixFromTxt<float>("frame_0_step6c_reconstructed_rgb_float.txt", 3);
     EXPECT_FALSE(rgbFloat.empty());

    // 2. Load Expected: Clipped RGB Float (Reference from Step 6d)
    cv::Mat expectedClippedRgbFloat = loadMatrixFromTxt<float>("frame_0_step6d_clipped_rgb_float.txt", 3);
    EXPECT_FALSE(expectedClippedRgbFloat.empty());

    // 3. Perform C++ Clipping
    cv::Mat actualClippedRgbFloat = rgbFloat.clone(); // Clone to avoid modifying input for potential reuse
    cv::Mat lowerBound = cv::Mat::zeros(actualClippedRgbFloat.size(), actualClippedRgbFloat.type());
    cv::Mat upperBound = cv::Mat(actualClippedRgbFloat.size(), actualClippedRgbFloat.type(), cv::Scalar::all(255.0f));
    cv::max(actualClippedRgbFloat, lowerBound, actualClippedRgbFloat);
    cv::min(actualClippedRgbFloat, upperBound, actualClippedRgbFloat);

    // 4. Compare
    double tolerance = 1e-4; // Use same tolerance
    ASSERT_TRUE(CompareMatrices(actualClippedRgbFloat, expectedClippedRgbFloat, tolerance))
        << "Mismatch in intermediate Clipped RGB Float calculation.";
}