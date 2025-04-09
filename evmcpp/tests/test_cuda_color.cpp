// File: evmcpp/tests/test_cuda_color.cpp
// Purpose: Unit tests for CUDA color conversion functions.

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
// #include <opencv2/core/cuda.hpp>        // No longer using GpuMat
// #include <opencv2/core/cuda/common.hpp> // No longer using StreamAccessor
#include <cuda_runtime.h> // For CUDA API calls
#include <string>
#include <vector>

// Include the header for the CUDA function being tested
#include "evmcuda/color_conversion.cuh"
// Include the header for test helpers (loading data, comparing matrices)
#include "test_helpers.hpp"

// Test fixture for CUDA tests
class CudaColorTest : public ::testing::Test {
protected:
    // Optional: Add setup/teardown for common CUDA operations if needed
    // static void SetUpTestSuite() {
    //     // Basic check if CUDA runtime initializes
    //     cudaError_t err = cudaFree(0); // Simple API call to check initialization
    //     if (err != cudaSuccess && err != cudaErrorInvalidDevicePointer) { // cudaFree(0) is okay
    //          GTEST_SKIP() << "CUDA Runtime API error during setup: " << cudaGetErrorString(err);
    //     }
    // }
};

// Test case for rgb2yiq_gpu
TEST_F(CudaColorTest, Rgb2YiqGpuMatchesCpu) {
    // 1. Load CPU Reference Data
    cv::Mat rgb_ref_cpu = loadMatrixFromTxt<float>("frame_0_rgb.txt", 3);
    EXPECT_FALSE(rgb_ref_cpu.empty()) << "Failed to load frame_0_rgb.txt";
    cv::Mat yiq_ref_cpu = loadMatrixFromTxt<float>("frame_0_yiq.txt", 3);
    EXPECT_FALSE(yiq_ref_cpu.empty()) << "Failed to load frame_0_yiq.txt";

    // Ensure input is CV_32FC3 and continuous for easy memcpy
    if (!rgb_ref_cpu.isContinuous()) {
        rgb_ref_cpu = rgb_ref_cpu.clone(); // Make it continuous
    }
    ASSERT_EQ(rgb_ref_cpu.type(), CV_32FC3);
    ASSERT_TRUE(rgb_ref_cpu.isContinuous());

    int width = rgb_ref_cpu.cols;
    int height = rgb_ref_cpu.rows;
    size_t hostPitch = rgb_ref_cpu.step; // Bytes per row on host

    // 2. Allocate GPU Memory
    float* d_inputRgb = nullptr;
    float* d_outputYiq = nullptr;
    size_t deviceInputPitch = 0;
    size_t deviceOutputPitch = 0;
    cudaError_t err;

    err = cudaMallocPitch((void**)&d_inputRgb, &deviceInputPitch, width * sizeof(float) * 3, height);
    ASSERT_EQ(err, cudaSuccess) << "cudaMallocPitch failed for input: " << cudaGetErrorString(err);
    err = cudaMallocPitch((void**)&d_outputYiq, &deviceOutputPitch, width * sizeof(float) * 3, height);
    ASSERT_EQ(err, cudaSuccess) << "cudaMallocPitch failed for output: " << cudaGetErrorString(err);

    // 3. Copy Input Data Host -> Device
    err = cudaMemcpy2D(d_inputRgb, deviceInputPitch, rgb_ref_cpu.ptr<float>(), hostPitch,
                       width * sizeof(float) * 3, height, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy2D H2D failed: " << cudaGetErrorString(err);

    // 4. Call the CUDA function (using default stream 0)
    try {
        evmcuda::rgb2yiq_gpu(d_inputRgb, d_outputYiq, width, height, deviceInputPitch, deviceOutputPitch, 0);
    } catch (const std::exception& e) {
         // Clean up allocated memory before failing
         cudaFree(d_inputRgb);
         cudaFree(d_outputYiq);
         GTEST_FAIL() << "rgb2yiq_gpu function threw an exception: " << e.what();
    } catch (...) {
         cudaFree(d_inputRgb);
         cudaFree(d_outputYiq);
         GTEST_FAIL() << "rgb2yiq_gpu function threw an unknown exception.";
    }

    // 5. Copy Result Device -> Host
    cv::Mat yiq_result_cpu(height, width, CV_32FC3); // Pre-allocate host matrix
    size_t resultHostPitch = yiq_result_cpu.step;
    err = cudaMemcpy2D(yiq_result_cpu.ptr<float>(), resultHostPitch, d_outputYiq, deviceOutputPitch,
                       width * sizeof(float) * 3, height, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy2D D2H failed: " << cudaGetErrorString(err);

    // 6. Free GPU Memory
    cudaFree(d_inputRgb);
    cudaFree(d_outputYiq);

    // 7. Compare GPU result with CPU reference
    ASSERT_FALSE(yiq_result_cpu.empty()) << "Downloaded YIQ result is empty.";
    ASSERT_EQ(yiq_result_cpu.size(), yiq_ref_cpu.size()) << "Dimension mismatch between GPU result and CPU reference.";
    ASSERT_EQ(yiq_result_cpu.type(), yiq_ref_cpu.type()) << "Type mismatch between GPU result and CPU reference.";

    double tolerance = 1e-4; // Use tolerance consistent with CPU tests
    ASSERT_TRUE(CompareMatrices(yiq_result_cpu, yiq_ref_cpu, tolerance))
        << "Mismatch between CUDA rgb2yiq output and CPU reference data.";
}

// Test case for yiq2rgb_gpu
TEST_F(CudaColorTest, Yiq2RgbGpuMatchesCpu) {
    // 1. Load CPU Reference Data
    cv::Mat yiq_ref_cpu = loadMatrixFromTxt<float>("frame_0_yiq.txt", 3);
    EXPECT_FALSE(yiq_ref_cpu.empty()) << "Failed to load frame_0_yiq.txt";
    cv::Mat rgb_ref_cpu = loadMatrixFromTxt<float>("frame_0_rgb.txt", 3);
    EXPECT_FALSE(rgb_ref_cpu.empty()) << "Failed to load frame_0_rgb.txt";

    // Ensure input is CV_32FC3 and continuous
    if (!yiq_ref_cpu.isContinuous()) {
        yiq_ref_cpu = yiq_ref_cpu.clone();
    }
    ASSERT_EQ(yiq_ref_cpu.type(), CV_32FC3);
    ASSERT_TRUE(yiq_ref_cpu.isContinuous());

    int width = yiq_ref_cpu.cols;
    int height = yiq_ref_cpu.rows;
    size_t hostPitch = yiq_ref_cpu.step;

    // 2. Allocate GPU Memory
    float* d_inputYiq = nullptr;
    float* d_outputRgb = nullptr;
    size_t deviceInputPitch = 0;
    size_t deviceOutputPitch = 0;
    cudaError_t err;

    err = cudaMallocPitch((void**)&d_inputYiq, &deviceInputPitch, width * sizeof(float) * 3, height);
    ASSERT_EQ(err, cudaSuccess) << "cudaMallocPitch failed for input: " << cudaGetErrorString(err);
    err = cudaMallocPitch((void**)&d_outputRgb, &deviceOutputPitch, width * sizeof(float) * 3, height);
    ASSERT_EQ(err, cudaSuccess) << "cudaMallocPitch failed for output: " << cudaGetErrorString(err);

    // 3. Copy Input Data Host -> Device
    err = cudaMemcpy2D(d_inputYiq, deviceInputPitch, yiq_ref_cpu.ptr<float>(), hostPitch,
                       width * sizeof(float) * 3, height, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy2D H2D failed: " << cudaGetErrorString(err);

    // 4. Call the CUDA function (using default stream 0)
    try {
        evmcuda::yiq2rgb_gpu(d_inputYiq, d_outputRgb, width, height, deviceInputPitch, deviceOutputPitch, 0);
    } catch (const std::exception& e) {
         cudaFree(d_inputYiq);
         cudaFree(d_outputRgb);
         GTEST_FAIL() << "yiq2rgb_gpu function threw an exception: " << e.what();
    } catch (...) {
         cudaFree(d_inputYiq);
         cudaFree(d_outputRgb);
         GTEST_FAIL() << "yiq2rgb_gpu function threw an unknown exception.";
    }

    // 5. Copy Result Device -> Host
    cv::Mat rgb_result_cpu(height, width, CV_32FC3);
    size_t resultHostPitch = rgb_result_cpu.step;
    err = cudaMemcpy2D(rgb_result_cpu.ptr<float>(), resultHostPitch, d_outputRgb, deviceOutputPitch,
                       width * sizeof(float) * 3, height, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy2D D2H failed: " << cudaGetErrorString(err);

    // 6. Free GPU Memory
    cudaFree(d_inputYiq);
    cudaFree(d_outputRgb);

    // 7. Compare GPU result with CPU reference
    ASSERT_FALSE(rgb_result_cpu.empty()) << "Downloaded RGB result is empty.";
    ASSERT_EQ(rgb_result_cpu.size(), rgb_ref_cpu.size()) << "Dimension mismatch.";
    ASSERT_EQ(rgb_result_cpu.type(), rgb_ref_cpu.type()) << "Type mismatch.";

    double tolerance = 1e-4;
    ASSERT_TRUE(CompareMatrices(rgb_result_cpu, rgb_ref_cpu, tolerance))
        << "Mismatch between CUDA yiq2rgb output and CPU reference data.";
}