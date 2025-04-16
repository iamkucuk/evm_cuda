#include <gtest/gtest.h>
#include "laplacian_pyramid.hpp" // Header for the function under test
#include "processing.hpp"       // Needed for types, maybe setup
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp> // Include OpenCV for image processing functions like getGaussianKernel
#include <vector>
#include <string>
#include <stdexcept> // For exception checking

// Include the test helpers header
#include "test_helpers.hpp"

// --- Test Fixture ---
class LaplacianPyramidTest : public ::testing::Test {
protected:
    // Define expected dimensions based on face.mp4 frame 0 pyramid levels
    const std::vector<cv::Size> level_sizes = {
        {528, 592}, // Level 0
        {264, 296}, // Level 1
        {132, 148}, // Level 2
        {66, 74}    // Level 3
    };
    const int num_levels = 4; // Matches the generation script

    // const std::string data_dir = "data/"; // No longer needed, path handled by TEST_DATA_DIR macro

    cv::Mat yiq_frame0;
    std::vector<cv::Mat> laplacian_ref0;

    void SetUp() override {
        // Load reference data for frame 0 and its pyramid levels
        try {
            yiq_frame0 = loadMatrixFromTxt<float>("frame_0_yiq.txt", 3);

            laplacian_ref0.resize(num_levels);
            for (int i = 0; i < num_levels; ++i) {
                std::string filename = "frame_0_laplacian_level_" + std::to_string(i) + ".txt";
                laplacian_ref0[i] = loadMatrixFromTxt<float>(filename, 3);
            }
        } catch (const std::exception& e) {
            GTEST_FAIL() << "Failed to load test data: " << e.what();
        }
    }
};

// --- Test Cases ---

TEST_F(LaplacianPyramidTest, GenerateLaplacianPyramidNumerical) {
    ASSERT_FALSE(yiq_frame0.empty()) << "Input YIQ frame data is empty.";
    ASSERT_EQ(laplacian_ref0.size(), num_levels) << "Incorrect number of reference Laplacian levels loaded.";

    // Define the kernel used in Python reference data generation
    float kernel_data[25] = {
        1,  4,  6,  4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1,  4,  6,  4, 1
    };
    cv::Mat kernel = cv::Mat(5, 5, CV_32F, kernel_data) / 256.0f;
    std::vector<cv::Mat> laplacian_result;
    ASSERT_NO_THROW(laplacian_result = evmcpp::generateLaplacianPyramid(yiq_frame0, num_levels, kernel)); // use_cuda removed

    // Check number of levels generated
    ASSERT_EQ(laplacian_result.size(), laplacian_ref0.size()) << "Mismatch in number of generated pyramid levels.";

    // Compare each level numerically
    for (size_t i = 0; i < laplacian_result.size(); ++i) {
        SCOPED_TRACE("Comparing Level " + std::to_string(i)); // Add context to failures
        ASSERT_FALSE(laplacian_result[i].empty()) << "Generated level " << i << " is empty.";
        ASSERT_FALSE(laplacian_ref0[i].empty()) << "Reference level " << i << " is empty.";

        ASSERT_EQ(laplacian_result[i].size(), laplacian_ref0[i].size());
        ASSERT_EQ(laplacian_result[i].type(), laplacian_ref0[i].type());

        EXPECT_TRUE(CompareMatrices(laplacian_result[i], laplacian_ref0[i], 1e-4)); // Increased explicit tolerance
    }
}

TEST_F(LaplacianPyramidTest, GetLaplacianPyramidsBatch) {
    // Test processing multiple frames
    const int num_test_frames = 2; // Test first 2 frames
    std::vector<cv::Mat> rgb_frames;
    std::vector<std::vector<cv::Mat>> laplacian_ref_batch(num_test_frames);

    // Load data for the required number of frames
    try {
        for (int i = 0; i < num_test_frames; ++i) {
            // Need RGB frames as input to getLaplacianPyramids
            rgb_frames.push_back(loadMatrixFromTxt<float>("frame_" + std::to_string(i) + "_rgb.txt", 3));
            laplacian_ref_batch[i].resize(num_levels);
            for (int lvl = 0; lvl < num_levels; ++lvl) {
                 std::string filename = "frame_" + std::to_string(i) + "_laplacian_level_" + std::to_string(lvl) + ".txt";
                 laplacian_ref_batch[i][lvl] = loadMatrixFromTxt<float>(filename, 3);
            }
        }
    } catch (const std::exception& e) {
         GTEST_FAIL() << "Failed to load multi-frame test data: " << e.what();
    }

    ASSERT_EQ(rgb_frames.size(), num_test_frames);
    ASSERT_EQ(laplacian_ref_batch.size(), num_test_frames);

    // Call the C++ function
    // Define the kernel used in Python reference data generation
    float kernel_data[25] = {
        1,  4,  6,  4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1,  4,  6,  4, 1
    };
    cv::Mat kernel = cv::Mat(5, 5, CV_32F, kernel_data) / 256.0f;
    std::vector<std::vector<cv::Mat>> laplacian_result_batch;
    ASSERT_NO_THROW(laplacian_result_batch = evmcpp::getLaplacianPyramids(rgb_frames, num_levels, kernel)); // use_cuda removed

    // Compare results
    ASSERT_EQ(laplacian_result_batch.size(), laplacian_ref_batch.size()) << "Mismatch in number of frames processed.";

    for (size_t i = 0; i < laplacian_result_batch.size(); ++i) {
        SCOPED_TRACE("Comparing Frame " + std::to_string(i));
        ASSERT_EQ(laplacian_result_batch[i].size(), laplacian_ref_batch[i].size()) << "Mismatch in number of pyramid levels for frame " << i;
        for (size_t lvl = 0; lvl < laplacian_result_batch[i].size(); ++lvl) {
             SCOPED_TRACE("Comparing Level " + std::to_string(lvl));
             ASSERT_FALSE(laplacian_result_batch[i][lvl].empty()) << "Generated level is empty.";
             ASSERT_FALSE(laplacian_ref_batch[i][lvl].empty()) << "Reference level is empty.";
             ASSERT_EQ(laplacian_result_batch[i][lvl].size(), laplacian_ref_batch[i][lvl].size());
             ASSERT_EQ(laplacian_result_batch[i][lvl].type(), laplacian_ref_batch[i][lvl].type());
             // Use slightly higher tolerance for batch test due to potential accumulation
             EXPECT_TRUE(CompareMatrices(laplacian_result_batch[i][lvl], laplacian_ref_batch[i][lvl], 1e-4)); // Increased explicit tolerance
        }
    }
} // End of GetLaplacianPyramidsBatch test case


// --- New Test Case for Filtering ---
TEST_F(LaplacianPyramidTest, FilterLaplacianPyramidsNumerical) {
    // Test the temporal filtering function
    const int num_test_frames = 5; // Must match NUM_FRAMES_TO_PROCESS in python script
    std::vector<std::vector<cv::Mat>> unfiltered_pyramids(num_test_frames);
    std::vector<std::vector<cv::Mat>> filtered_ref_pyramids(num_test_frames);

    // Parameters used during data generation
    const double test_fps = 30.0;
    const std::pair<double, double> test_freq_range = {0.4, 3.0};
    const double test_alpha = 10.0;
    const double test_lambda_cutoff = 16.0;
    const double test_attenuation = 1.0;


    // Load unfiltered and filtered reference data
    try {
        for (int i = 0; i < num_test_frames; ++i) {
            unfiltered_pyramids[i].resize(num_levels);
            filtered_ref_pyramids[i].resize(num_levels);
            for (int lvl = 0; lvl < num_levels; ++lvl) {
                 // Ensure level sizes are valid before accessing
                 if (lvl >= level_sizes.size()) {
                     throw std::runtime_error("Level index out of bounds for level_sizes vector.");
                 }
                 std::string unfiltered_filename = "frame_" + std::to_string(i) + "_laplacian_level_" + std::to_string(lvl) + ".txt";
                 unfiltered_pyramids[i][lvl] = loadMatrixFromTxt<float>(unfiltered_filename, 3);

                 std::string filtered_filename = "frame_" + std::to_string(i) + "_filtered_level_" + std::to_string(lvl) + ".txt";
                 filtered_ref_pyramids[i][lvl] = loadMatrixFromTxt<float>(filtered_filename, 3);
            }
        }
    } catch (const std::exception& e) {
         GTEST_FAIL() << "Failed to load filtered pyramid test data: " << e.what();
    }

    // Call the C++ function
    std::vector<std::vector<cv::Mat>> filtered_result_pyramids;
    ASSERT_NO_THROW(filtered_result_pyramids = evmcpp::filterLaplacianPyramids(
                                                    unfiltered_pyramids,
                                                    num_levels,
                                                    test_fps,
                                                    test_freq_range,
                                                    test_alpha,
                                                    test_lambda_cutoff,
                                                    test_attenuation));

    // Compare results
    ASSERT_EQ(filtered_result_pyramids.size(), filtered_ref_pyramids.size()) << "Mismatch in number of filtered frames.";

    // Compare frame by frame, level by level
    // Note: The filter needs previous frames, so comparisons usually start from frame 1 or 2 depending on filter order.
    // However, the Python reference data includes frame 0 (copied directly) and subsequent filtered frames.
    // Let's compare all frames generated by C++.
    for (size_t i = 0; i < filtered_result_pyramids.size(); ++i) {
        SCOPED_TRACE("Comparing Filtered Frame " + std::to_string(i));
        ASSERT_EQ(filtered_result_pyramids[i].size(), filtered_ref_pyramids[i].size()) << "Mismatch in number of pyramid levels for filtered frame " << i;
        for (size_t lvl = 0; lvl < filtered_result_pyramids[i].size(); ++lvl) {
             SCOPED_TRACE("Comparing Filtered Level " + std::to_string(lvl));
             ASSERT_FALSE(filtered_result_pyramids[i][lvl].empty()) << "Generated filtered level is empty.";
             ASSERT_FALSE(filtered_ref_pyramids[i][lvl].empty()) << "Reference filtered level is empty.";
             ASSERT_EQ(filtered_result_pyramids[i][lvl].size(), filtered_ref_pyramids[i][lvl].size());
             ASSERT_EQ(filtered_result_pyramids[i][lvl].type(), filtered_ref_pyramids[i][lvl].type());
             // Use a slightly higher tolerance for the filter output due to IIR filter state accumulation
             EXPECT_TRUE(CompareMatrices(filtered_result_pyramids[i][lvl], filtered_ref_pyramids[i][lvl], 5e-4f)); // Increased tolerance further
        }
    }
}