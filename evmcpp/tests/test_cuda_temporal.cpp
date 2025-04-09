// File: evmcpp/tests/test_cuda_temporal.cpp
// Purpose: Unit tests for CUDA temporal filtering functions.

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <numeric> // For std::iota

// Include the header for the CUDA function being tested
#include "evmcuda/temporal_filter.cuh"
// Include the header for test helpers (loading data, comparing matrices)
#include "test_helpers.hpp"
// Include CPU implementation for reference comparison
#include "evmcpp/laplacian_pyramid.hpp"
#include "evmcpp/butterworth.hpp"

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err); \
    } while (0)

// Test fixture for CUDA Temporal Filter tests
class CudaTemporalTest : public ::testing::Test {
    // Add setup/teardown if needed
};

// Test case for temporalFilterGaussianBatch_gpu (FFT-based)
TEST_F(CudaTemporalTest, TemporalFilterGpuMatchesCpuReference) {
    // 1. Load CPU Reference Data
    int numFrames = 5; // Match generate_test_data.py
    std::vector<cv::Mat> input_batch_cpu(numFrames);
    cv::Size frameSize;
    size_t hostPitch = 0;

    for (int i = 0; i < numFrames; ++i) {
        // Use the spatially filtered Gaussian output as input to temporal filter
        input_batch_cpu[i] = loadMatrixFromTxt<float>("frame_" + std::to_string(i) + "_gaussian_reconstructed.txt", 3);
        EXPECT_FALSE(input_batch_cpu[i].empty()) << "Failed to load frame_" << i << "_gaussian_reconstructed.txt";
        if (i == 0) {
            frameSize = input_batch_cpu[i].size();
            hostPitch = input_batch_cpu[i].step;
            ASSERT_TRUE(input_batch_cpu[i].isContinuous()) << "Frame 0 must be continuous for pitch calculation";
            ASSERT_EQ(input_batch_cpu[i].type(), CV_32FC3);
        } else {
            ASSERT_EQ(input_batch_cpu[i].size(), frameSize) << "Frame size mismatch in input batch";
            ASSERT_EQ(input_batch_cpu[i].type(), CV_32FC3) << "Frame type mismatch in input batch";
        }
        // Ensure continuity for easier memory copy later
        if (!input_batch_cpu[i].isContinuous()) {
            input_batch_cpu[i] = input_batch_cpu[i].clone();
        }
    }

    // Load expected output for frame 0 (after temporal filtering + amplification in Python)
    cv::Mat filtered_ref_cpu = loadMatrixFromTxt<float>("frame_0_step4_temporal_filtered_yiq.txt", 3);
    EXPECT_FALSE(filtered_ref_cpu.empty()) << "Failed to load frame_0_step4_temporal_filtered_yiq.txt";

    int width = frameSize.width;
    int height = frameSize.height;
    size_t frameSizeInBytes = hostPitch * height;
    size_t totalBatchBytes = frameSizeInBytes * numFrames;

    // 2. Allocate Contiguous GPU Memory for the Batch
    float* d_imageBatch = nullptr;
    size_t devicePitch = 0;
    // Allocate potentially padded memory for the entire batch height
    CUDA_CHECK(cudaMallocPitch((void**)&d_imageBatch, &devicePitch, width * 3 * sizeof(float), height * numFrames));

    // 3. Copy Input Batch Host -> Device (frame by frame, row by row)
    for (int i = 0; i < numFrames; ++i) {
        // Calculate destination pointer for this frame within the batch allocation
        char* frameDestPtr = (char*)d_imageBatch + i * height * devicePitch; // Offset by full frame height * pitch
        CUDA_CHECK(cudaMemcpy2D(frameDestPtr, devicePitch, input_batch_cpu[i].ptr<float>(), hostPitch,
                                width * 3 * sizeof(float), height, cudaMemcpyHostToDevice));
    }

    // 4. Define Parameters
    float fps = 30.0f;
    float fl = 0.4f;
    float fh = 3.0f;
    float alpha = 10.0f;
    float chromAttenuation = 1.0f;

    // 5. Call the CUDA function (operates in-place)
    try {
        evmcuda::temporalFilterGaussianBatch_gpu(d_imageBatch, numFrames, width, height, devicePitch,
                                                 fps, fl, fh, alpha, chromAttenuation, 0);
    } catch (const std::exception& e) {
         cudaFree(d_imageBatch);
         GTEST_FAIL() << "temporalFilterGaussianBatch_gpu threw an exception: " << e.what();
    } catch (...) {
         cudaFree(d_imageBatch);
         GTEST_FAIL() << "temporalFilterGaussianBatch_gpu threw an unknown exception.";
    }

    // Synchronize device before copying back
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. Copy Result (Frame 0) Device -> Host
    cv::Mat result_frame0_cpu(height, width, CV_32FC3);
    CUDA_CHECK(cudaMemcpy2D(result_frame0_cpu.ptr<float>(), result_frame0_cpu.step,
                            d_imageBatch, devicePitch, // Read from the start of the batch buffer
                            width * 3 * sizeof(float), height, cudaMemcpyDeviceToHost));

    // 7. Free GPU Memory
    cudaFree(d_imageBatch);

    // 8. Compare GPU result (Frame 0) with CPU reference
    ASSERT_FALSE(result_frame0_cpu.empty()) << "Downloaded result frame 0 is empty.";
    ASSERT_EQ(result_frame0_cpu.size(), filtered_ref_cpu.size());
    ASSERT_EQ(result_frame0_cpu.type(), filtered_ref_cpu.type());

    double tolerance = 1e-4; // Adjusted tolerance
    EXPECT_TRUE(CompareMatrices(result_frame0_cpu, filtered_ref_cpu, tolerance))
        << "Mismatch between CUDA Gaussian temporal filter output and CPU reference data.";

}


// Test case for filterLaplacianLevelFrame_gpu (IIR Bandpass filter)
// Tests processing of frame 1 using state from frame 0 for Level 0.
TEST_F(CudaTemporalTest, FilterLaplacianBandpassGpuFrame1MatchesCpuReference_Level0) {
    // 1. Load CPU Reference Data (Level 0, Frames 0 and 1)
    int level_to_test = 0;
    cv::Mat input_f0_cpu = loadMatrixFromTxt<float>("frame_0_laplacian_level_0.txt", 3);
    cv::Mat input_f1_cpu = loadMatrixFromTxt<float>("frame_1_laplacian_level_0.txt", 3);
    cv::Mat filtered_f1_ref_cpu = loadMatrixFromTxt<float>("frame_1_filtered_level_0.txt", 3);

    EXPECT_FALSE(input_f0_cpu.empty()) << "Failed to load frame_0_laplacian_level_0.txt";
    EXPECT_FALSE(input_f1_cpu.empty()) << "Failed to load frame_1_laplacian_level_0.txt";
    EXPECT_FALSE(filtered_f1_ref_cpu.empty()) << "Failed to load frame_1_filtered_level_0.txt";

    ASSERT_EQ(input_f0_cpu.size(), input_f1_cpu.size());
    ASSERT_EQ(input_f0_cpu.type(), CV_32FC3);
    ASSERT_EQ(input_f1_cpu.type(), CV_32FC3);
    if (!input_f0_cpu.isContinuous()) input_f0_cpu = input_f0_cpu.clone();
    if (!input_f1_cpu.isContinuous()) input_f1_cpu = input_f1_cpu.clone();
    ASSERT_TRUE(input_f0_cpu.isContinuous());
    ASSERT_TRUE(input_f1_cpu.isContinuous());

    int width = input_f0_cpu.cols;
    int height = input_f0_cpu.rows;
    size_t hostPitch = input_f0_cpu.step;
    size_t frameSizeElements = (size_t)width * height * 3;
    size_t frameSizeBytes = frameSizeElements * sizeof(float);

    // 2. Allocate GPU Memory (Input F1, Output F1, 4 State Buffers)
    float* d_inputF1 = nullptr;
    float* d_outputF1 = nullptr;
    float* d_state_xl_1 = nullptr; // x[n-1] for low-pass
    float* d_state_yl_1 = nullptr; // yl[n-1] for low-pass
    float* d_state_xh_1 = nullptr; // x[n-1] for high-pass
    float* d_state_yh_1 = nullptr; // yh[n-1] for high-pass
    size_t devicePitch = 0; // Assume same pitch for simplicity here

    CUDA_CHECK(cudaMallocPitch((void**)&d_inputF1, &devicePitch, width * 3 * sizeof(float), height));
    CUDA_CHECK(cudaMallocPitch((void**)&d_outputF1, &devicePitch, width * 3 * sizeof(float), height));
    CUDA_CHECK(cudaMallocPitch((void**)&d_state_xl_1, &devicePitch, width * 3 * sizeof(float), height));
    CUDA_CHECK(cudaMallocPitch((void**)&d_state_yl_1, &devicePitch, width * 3 * sizeof(float), height));
    CUDA_CHECK(cudaMallocPitch((void**)&d_state_xh_1, &devicePitch, width * 3 * sizeof(float), height));
    CUDA_CHECK(cudaMallocPitch((void**)&d_state_yh_1, &devicePitch, width * 3 * sizeof(float), height));


    // 3. Initialize/Copy Data Host -> Device
    // Copy input frame 1
    CUDA_CHECK(cudaMemcpy2D(d_inputF1, devicePitch, input_f1_cpu.ptr<float>(), hostPitch, width * 3 * sizeof(float), height, cudaMemcpyHostToDevice));
    // Copy input frame 0 to previous input state buffers (xl_1, xh_1)
    CUDA_CHECK(cudaMemcpy2D(d_state_xl_1, devicePitch, input_f0_cpu.ptr<float>(), hostPitch, width * 3 * sizeof(float), height, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_state_xh_1, devicePitch, input_f0_cpu.ptr<float>(), hostPitch, width * 3 * sizeof(float), height, cudaMemcpyHostToDevice));
    // Initialize previous output state buffers (yl_1, yh_1) with frame 0 data (matching Python's implicit init)
    CUDA_CHECK(cudaMemcpy2D(d_state_yl_1, devicePitch, input_f0_cpu.ptr<float>(), hostPitch, width * 3 * sizeof(float), height, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_state_yh_1, devicePitch, input_f0_cpu.ptr<float>(), hostPitch, width * 3 * sizeof(float), height, cudaMemcpyHostToDevice));

    // 4. Define Parameters
    // Load coefficients
    cv::Mat b_low_mat = loadMatrixFromTxt<double>("butter_low_b.txt", 1);
    cv::Mat a_low_mat = loadMatrixFromTxt<double>("butter_low_a.txt", 1);
    cv::Mat b_high_mat = loadMatrixFromTxt<double>("butter_high_b.txt", 1);
    cv::Mat a_high_mat = loadMatrixFromTxt<double>("butter_high_a.txt", 1);
    ASSERT_EQ(b_low_mat.total(), 2); ASSERT_EQ(a_low_mat.total(), 2);
    ASSERT_EQ(b_high_mat.total(), 2); ASSERT_EQ(a_high_mat.total(), 2);

    // Extract low-pass coefficients
    float b0_l = static_cast<float>(b_low_mat.at<double>(0));
    float b1_l = static_cast<float>(b_low_mat.at<double>(1));
    float a1_l = static_cast<float>(a_low_mat.at<double>(1));
    // Extract high-pass coefficients
    float b0_h = static_cast<float>(b_high_mat.at<double>(0));
    float b1_h = static_cast<float>(b_high_mat.at<double>(1));
    float a1_h = static_cast<float>(a_high_mat.at<double>(1));

    float alpha = 10.0f; // Base alpha from CPU test data generation
    float lambda_cutoff = 16.0f; // From CPU test data generation
    float chromAttenuation = 1.0f; // From CPU test data generation
    float fps = 30.0f; // From CPU test data generation
    int level = level_to_test;

    // 5. Call the CUDA function
    try {
        evmcuda::filterLaplacianLevelFrame_gpu(d_inputF1, d_outputF1,
                                               d_state_xl_1, d_state_yl_1,
                                               d_state_xh_1, d_state_yh_1,
                                               width, height, devicePitch,
                                               b0_l, b1_l, a1_l,
                                               b0_h, b1_h, a1_h,
                                               level, fps,
                                               alpha, lambda_cutoff, chromAttenuation,
                                               0); // Use default stream 0
    } catch (const std::exception& e) {
         cudaFree(d_inputF1); cudaFree(d_outputF1); cudaFree(d_state_xl_1);
         cudaFree(d_state_yl_1); cudaFree(d_state_xh_1); cudaFree(d_state_yh_1);
         GTEST_FAIL() << "filterLaplacianLevelFrame_gpu threw an exception: " << e.what();
    } catch (...) {
         cudaFree(d_inputF1); cudaFree(d_outputF1); cudaFree(d_state_xl_1);
         cudaFree(d_state_yl_1); cudaFree(d_state_xh_1); cudaFree(d_state_yh_1);
         GTEST_FAIL() << "filterLaplacianLevelFrame_gpu threw an unknown exception.";
    }

    // Synchronize device before copying back
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. Copy Result (Frame 1) Device -> Host
    cv::Mat result_frame1_cpu(height, width, CV_32FC3);
    CUDA_CHECK(cudaMemcpy2D(result_frame1_cpu.ptr<float>(), result_frame1_cpu.step,
                            d_outputF1, devicePitch,
                            width * 3 * sizeof(float), height, cudaMemcpyDeviceToHost));

    // 7. Free GPU Memory
    cudaFree(d_inputF1);
    cudaFree(d_outputF1);
    cudaFree(d_state_xl_1);
    cudaFree(d_state_yl_1);
    cudaFree(d_state_xh_1);
    cudaFree(d_state_yh_1);

    // 8. Compare GPU result (Frame 1) with CPU reference (Frame 1)
    ASSERT_FALSE(result_frame1_cpu.empty()) << "Downloaded result frame 1 is empty.";
    ASSERT_EQ(result_frame1_cpu.size(), filtered_f1_ref_cpu.size());
    ASSERT_EQ(result_frame1_cpu.type(), filtered_f1_ref_cpu.type());

    double tolerance = 1e-4; // Keep tolerance from previous steps
    ASSERT_TRUE(CompareMatrices(result_frame1_cpu, filtered_f1_ref_cpu, tolerance))
        << "Mismatch between CUDA IIR filter output (Level 0, Frame 1) and CPU reference.";
}

// New Test: Process multiple frames sequentially for Laplacian IIR filter
TEST_F(CudaTemporalTest, FilterLaplacianBandpassGpuMultiFrameMatchesCpuReference_Level0) {
    // 1. Load CPU Reference Data (Level 0, Frames 0-4)
    int level_to_test = 0;
    int num_frames_to_test = 5;
    std::vector<cv::Mat> input_frames_cpu(num_frames_to_test);
    std::vector<cv::Mat> filtered_ref_frames_cpu(num_frames_to_test);
    cv::Size frame_size;
    size_t host_pitch = 0;

    for (int i = 0; i < num_frames_to_test; ++i) {
        input_frames_cpu[i] = loadMatrixFromTxt<float>("frame_" + std::to_string(i) + "_laplacian_level_" + std::to_string(level_to_test) + ".txt", 3);
        filtered_ref_frames_cpu[i] = loadMatrixFromTxt<float>("frame_" + std::to_string(i) + "_filtered_level_" + std::to_string(level_to_test) + ".txt", 3);

        EXPECT_FALSE(input_frames_cpu[i].empty()) << "Failed to load input frame " << i;
        EXPECT_FALSE(filtered_ref_frames_cpu[i].empty()) << "Failed to load filtered reference frame " << i;

        if (i == 0) {
            frame_size = input_frames_cpu[i].size();
            host_pitch = input_frames_cpu[i].step;
            ASSERT_TRUE(input_frames_cpu[i].isContinuous()) << "Frame 0 must be continuous";
            ASSERT_EQ(input_frames_cpu[i].type(), CV_32FC3);
        } else {
            ASSERT_EQ(input_frames_cpu[i].size(), frame_size) << "Frame size mismatch frame " << i;
            ASSERT_EQ(input_frames_cpu[i].type(), CV_32FC3) << "Frame type mismatch frame " << i;
        }
        if (!input_frames_cpu[i].isContinuous()) input_frames_cpu[i] = input_frames_cpu[i].clone();
    }

    int width = frame_size.width;
    int height = frame_size.height;

    // 2. Allocate GPU Memory (Input, Output, 4 State Buffers)
    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_state_xl_1 = nullptr;
    float* d_state_yl_1 = nullptr;
    float* d_state_xh_1 = nullptr;
    float* d_state_yh_1 = nullptr;
    size_t device_pitch = 0;

    CUDA_CHECK(cudaMallocPitch((void**)&d_input, &device_pitch, width * 3 * sizeof(float), height));
    CUDA_CHECK(cudaMallocPitch((void**)&d_output, &device_pitch, width * 3 * sizeof(float), height));
    CUDA_CHECK(cudaMallocPitch((void**)&d_state_xl_1, &device_pitch, width * 3 * sizeof(float), height));
    CUDA_CHECK(cudaMallocPitch((void**)&d_state_yl_1, &device_pitch, width * 3 * sizeof(float), height));
    CUDA_CHECK(cudaMallocPitch((void**)&d_state_xh_1, &device_pitch, width * 3 * sizeof(float), height));
    CUDA_CHECK(cudaMallocPitch((void**)&d_state_yh_1, &device_pitch, width * 3 * sizeof(float), height));

    // 3. Initialize State Buffers using Frame 0
    CUDA_CHECK(cudaMemcpy2D(d_state_xl_1, device_pitch, input_frames_cpu[0].ptr<float>(), host_pitch, width * 3 * sizeof(float), height, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_state_xh_1, device_pitch, input_frames_cpu[0].ptr<float>(), host_pitch, width * 3 * sizeof(float), height, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_state_yl_1, device_pitch, input_frames_cpu[0].ptr<float>(), host_pitch, width * 3 * sizeof(float), height, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_state_yh_1, device_pitch, input_frames_cpu[0].ptr<float>(), host_pitch, width * 3 * sizeof(float), height, cudaMemcpyHostToDevice));

    // 4. Define Parameters (same as single frame test)
    cv::Mat b_low_mat = loadMatrixFromTxt<double>("butter_low_b.txt", 1);
    cv::Mat a_low_mat = loadMatrixFromTxt<double>("butter_low_a.txt", 1);
    cv::Mat b_high_mat = loadMatrixFromTxt<double>("butter_high_b.txt", 1);
    cv::Mat a_high_mat = loadMatrixFromTxt<double>("butter_high_a.txt", 1);
    float b0_l = static_cast<float>(b_low_mat.at<double>(0));
    float b1_l = static_cast<float>(b_low_mat.at<double>(1));
    float a1_l = static_cast<float>(a_low_mat.at<double>(1));
    float b0_h = static_cast<float>(b_high_mat.at<double>(0));
    float b1_h = static_cast<float>(b_high_mat.at<double>(1));
    float a1_h = static_cast<float>(a_high_mat.at<double>(1));
    float alpha = 10.0f;
    float lambda_cutoff = 16.0f;
    float chromAttenuation = 1.0f;
    float fps = 30.0f;
    int level = level_to_test;

    // 5. Process Frames 1 to N-1 Sequentially
    for (int i = 1; i < num_frames_to_test; ++i) {
        // Copy current input frame to GPU
        CUDA_CHECK(cudaMemcpy2D(d_input, device_pitch, input_frames_cpu[i].ptr<float>(), host_pitch, width * 3 * sizeof(float), height, cudaMemcpyHostToDevice));

        // Call the CUDA function (state buffers are updated in-place by the kernel)
        try {
            evmcuda::filterLaplacianLevelFrame_gpu(d_input, d_output,
                                                   d_state_xl_1, d_state_yl_1,
                                                   d_state_xh_1, d_state_yh_1,
                                                   width, height, device_pitch,
                                                   b0_l, b1_l, a1_l,
                                                   b0_h, b1_h, a1_h,
                                                   level, fps,
                                                   alpha, lambda_cutoff, chromAttenuation,
                                                   0);
        } catch (const std::exception& e) {
             // Cleanup before failing
             cudaFree(d_input); cudaFree(d_output); cudaFree(d_state_xl_1);
             cudaFree(d_state_yl_1); cudaFree(d_state_xh_1); cudaFree(d_state_yh_1);
             GTEST_FAIL() << "filterLaplacianLevelFrame_gpu threw exception on frame " << i << ": " << e.what();
        } catch (...) {
             cudaFree(d_input); cudaFree(d_output); cudaFree(d_state_xl_1);
             cudaFree(d_state_yl_1); cudaFree(d_state_xh_1); cudaFree(d_state_yh_1);
             GTEST_FAIL() << "filterLaplacianLevelFrame_gpu threw unknown exception on frame " << i;
        }

        // Synchronize after kernel launch for this frame
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result for current frame back to CPU
        cv::Mat result_frame_cpu(height, width, CV_32FC3);
        CUDA_CHECK(cudaMemcpy2D(result_frame_cpu.ptr<float>(), result_frame_cpu.step,
                                d_output, device_pitch,
                                width * 3 * sizeof(float), height, cudaMemcpyDeviceToHost));

        // Compare GPU result for frame i with CPU reference for frame i
        ASSERT_FALSE(result_frame_cpu.empty()) << "Downloaded result frame " << i << " is empty.";
        ASSERT_EQ(result_frame_cpu.size(), filtered_ref_frames_cpu[i].size()) << "Size mismatch frame " << i;
        ASSERT_EQ(result_frame_cpu.type(), filtered_ref_frames_cpu[i].type()) << "Type mismatch frame " << i;

        double tolerance = 1e-4;
        ASSERT_TRUE(CompareMatrices(result_frame_cpu, filtered_ref_frames_cpu[i], tolerance))
            << "Mismatch between CUDA IIR filter output and CPU reference for Frame " << i << " Level " << level;
    }

    // 6. Free GPU Memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_state_xl_1);
    cudaFree(d_state_yl_1);
    cudaFree(d_state_xh_1);
    cudaFree(d_state_yh_1);
}