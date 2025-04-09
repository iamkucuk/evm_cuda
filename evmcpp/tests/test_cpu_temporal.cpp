#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "evmcpu/temporal_filter.hpp"
#include "test_helpers.hpp"

class CpuTemporalFilterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load sequence of YIQ frames
        frames_.clear();
        frames_.reserve(5); // We have 5 test frames
        
        for (int i = 0; i < 5; ++i) {
            cv::Mat frame = loadMatrixFromTxt<float>(
                "frame_" + std::to_string(i) + "_yiq.txt", 3);
            ASSERT_FALSE(frame.empty()) << "Failed to load frame " << i;
            frames_.push_back(frame);
        }

        // Create 4D tensor (num_frames x height x width x channels)
        const int num_frames = frames_.size();
        const int height = frames_[0].rows;
        const int width = frames_[0].cols;
        const int channels = frames_[0].channels();

        // Create tensor with proper dimensions and type (CV_32FC3)
        const int tensor_dims[4] = {num_frames, height, width, channels};
        frames_tensor_.create(4, tensor_dims, CV_32F);

        // Copy data from frames to tensor
        for (int t = 0; t < num_frames; t++) {
            const cv::Mat& frame = frames_[t];
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    const cv::Vec3f& pixel = frame.at<cv::Vec3f>(h, w);
                    for (int c = 0; c < channels; c++) {
                        const int idx[] = {t, h, w, c};
                        frames_tensor_.at<float>(idx) = pixel[c];
                    }
                }
            }
        }
    }

    std::vector<cv::Mat> frames_;     // Original frames
    cv::Mat frames_tensor_;           // 4D tensor for processing
};

TEST_F(CpuTemporalFilterTest, MatchesPythonReference) {
    double fps = 30.0;
    cv::Point2d freq_range(0.4, 3.0);

    // Apply temporal filter
    cv::Mat filtered = evmcpu::ideal_temporal_bandpass_filter(frames_tensor_, fps, freq_range);
    ASSERT_FALSE(filtered.empty()) << "Filtered output is empty";
    
    // Load and compare with Python reference output for frame 0
    cv::Mat expected = loadMatrixFromTxt<float>("frame_0_step4_temporal_filtered_yiq.txt", 3);
    ASSERT_FALSE(expected.empty()) << "Failed to load reference output";
    
    // Extract frame 0 from filtered tensor
    std::vector<int> frame0_size = {filtered.size[1], filtered.size[2]};  // height, width
    cv::Mat filtered_frame0(frame0_size, CV_32FC3);  // Create as 3-channel float matrix
    
    // Copy frame 0 data
    for (int h = 0; h < frame0_size[0]; h++) {
        for (int w = 0; w < frame0_size[1]; w++) {
            cv::Vec3f& pixel = filtered_frame0.at<cv::Vec3f>(h, w);
            for (int c = 0; c < 3; c++) {
                const int src_idx[] = {0, h, w, c};  // Frame 0
                pixel[c] = filtered.at<float>(src_idx);
            }
        }
    }
    
    // Compare results
    ASSERT_TRUE(CompareMatrices(filtered_frame0, expected, 1e-4))
        << "Temporal filter output doesn't match Python reference";
}

TEST_F(CpuTemporalFilterTest, InvalidInputs) {
    double fps = 30.0;
    cv::Point2d freq_range(0.4, 3.0);
    
    // Empty input
    cv::Mat empty;
    EXPECT_THROW(
        evmcpu::ideal_temporal_bandpass_filter(empty, fps, freq_range),
        std::invalid_argument
    );
    
    // Invalid frequency range
    cv::Point2d invalid_range(-1.0, 0.5);
    EXPECT_THROW(
        evmcpu::ideal_temporal_bandpass_filter(frames_tensor_, fps, invalid_range),
        std::invalid_argument
    );
    
    // Invalid axis
    EXPECT_THROW(
        evmcpu::ideal_temporal_bandpass_filter(frames_tensor_, fps, freq_range, 1),
        std::invalid_argument
    );
}