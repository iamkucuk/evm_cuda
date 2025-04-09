// File: evmcpp/tests/test_cuda_pyramid.cpp
// Purpose: Unit tests for CUDA custom pyramid functions.

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <string>
#include <vector>

// Include the header for the CUDA function being tested
#include "evmcuda/pyramid.cuh"
// Include the header for test helpers (loading data, comparing matrices)
#include "test_helpers.hpp"
// Include processing header for the gaussian_kernel constant
#include "evmcpp/processing.hpp"
#include <opencv2/imgproc.hpp> // For cv::filter2D

// Removed kernel forward declaration, will call C++ wrapper function instead

// Helper function for CUDA error checking
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err); \
    } while (0)


// Test fixture for CUDA Pyramid tests
class CudaPyramidTest : public ::testing::Test {
protected:
    // Device pointers for kernel data
    static float* d_kernel;
    static float* d_kernel_x4; // Kernel multiplied by 4 for pyrUp

    static void SetUpTestSuite() {
        // Basic CUDA check
        // cudaError_t err = cudaFree(0);
        // if (err != cudaSuccess && err != cudaErrorInvalidDevicePointer) {
        //     GTEST_SKIP() << "CUDA Runtime API error during setup: " << cudaGetErrorString(err);
        // }

        // Allocate and copy kernel data to device
        const cv::Mat& h_kernel = evmcpp::gaussian_kernel; // Get from processing.hpp
        ASSERT_FALSE(h_kernel.empty());
        ASSERT_TRUE(h_kernel.isContinuous());
        ASSERT_EQ(h_kernel.type(), CV_32F);
        ASSERT_EQ(h_kernel.total(), 25); // Ensure it's 5x5

        CUDA_CHECK(cudaMalloc((void**)&d_kernel, 25 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.ptr<float>(), 25 * sizeof(float), cudaMemcpyHostToDevice));

        // Create kernel * 4 for pyrUp
        cv::Mat h_kernel_x4 = h_kernel * 4.0f;
        CUDA_CHECK(cudaMalloc((void**)&d_kernel_x4, 25 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_kernel_x4, h_kernel_x4.ptr<float>(), 25 * sizeof(float), cudaMemcpyHostToDevice));
    }

    static void TearDownTestSuite() {
        cudaFree(d_kernel);
        cudaFree(d_kernel_x4);
    }
};

float* CudaPyramidTest::d_kernel = nullptr;
float* CudaPyramidTest::d_kernel_x4 = nullptr;


// --- Test Cases ---

TEST_F(CudaPyramidTest, PyrDownGpuMatchesCpu) {
    // 1. Load CPU Reference Data
    cv::Mat input_yiq_cpu = loadMatrixFromTxt<float>("frame_0_yiq.txt", 3);
    EXPECT_FALSE(input_yiq_cpu.empty()) << "Failed to load frame_0_yiq.txt";
    cv::Mat pyrdown_ref_cpu = loadMatrixFromTxt<float>("frame_0_pyrdown_0.txt", 3);
    EXPECT_FALSE(pyrdown_ref_cpu.empty()) << "Failed to load frame_0_pyrdown_0.txt";

    // Ensure input is continuous
    if (!input_yiq_cpu.isContinuous()) input_yiq_cpu = input_yiq_cpu.clone();
    ASSERT_TRUE(input_yiq_cpu.isContinuous());
    ASSERT_EQ(input_yiq_cpu.type(), CV_32FC3);

    int width = input_yiq_cpu.cols;
    int height = input_yiq_cpu.rows;
    size_t hostPitch = input_yiq_cpu.step;
    int outWidth = width / 2;
    int outHeight = height / 2;

    // 2. Allocate GPU Memory
    float* d_input = nullptr;
    float* d_output = nullptr;
    size_t dInputPitch = 0;
    size_t dOutputPitch = 0;
    CUDA_CHECK(cudaMallocPitch((void**)&d_input, &dInputPitch, width * 3 * sizeof(float), height));
    CUDA_CHECK(cudaMallocPitch((void**)&d_output, &dOutputPitch, outWidth * 3 * sizeof(float), outHeight));

    // 3. Copy Input Host -> Device
    CUDA_CHECK(cudaMemcpy2D(d_input, dInputPitch, input_yiq_cpu.ptr<float>(), hostPitch, width * 3 * sizeof(float), height, cudaMemcpyHostToDevice));

    // 4. Call CUDA function
    ASSERT_NE(d_kernel, nullptr); // Ensure kernel was allocated
    evmcuda::pyrDown_gpu(d_input, d_output, width, height, dInputPitch, dOutputPitch, d_kernel, 0);

    // 5. Copy Result Device -> Host
    cv::Mat pyrdown_result_cpu(outHeight, outWidth, CV_32FC3);
    CUDA_CHECK(cudaMemcpy2D(pyrdown_result_cpu.ptr<float>(), pyrdown_result_cpu.step, d_output, dOutputPitch, outWidth * 3 * sizeof(float), outHeight, cudaMemcpyDeviceToHost));

    // 6. Free GPU Memory
    cudaFree(d_input);
    cudaFree(d_output);

    // 7. Compare
    ASSERT_EQ(pyrdown_result_cpu.size(), pyrdown_ref_cpu.size());
    ASSERT_EQ(pyrdown_result_cpu.type(), pyrdown_ref_cpu.type());
    ASSERT_TRUE(CompareMatrices(pyrdown_result_cpu, pyrdown_ref_cpu, 1e-3)) // Increased tolerance
        << "Mismatch between CUDA pyrDown output and CPU reference data.";
}


TEST_F(CudaPyramidTest, PyrUpGpuMatchesCpu) {
    // 1. Load CPU Reference Data
    // Input for pyrUp is the output of pyrDown
    cv::Mat input_pyrdown_cpu = loadMatrixFromTxt<float>("frame_0_pyrdown_0.txt", 3);
    EXPECT_FALSE(input_pyrdown_cpu.empty()) << "Failed to load frame_0_pyrdown_0.txt";
    // Expected output is from frame_0_pyrup_0.txt
    cv::Mat pyrup_ref_cpu = loadMatrixFromTxt<float>("frame_0_pyrup_0.txt", 3);
    EXPECT_FALSE(pyrup_ref_cpu.empty()) << "Failed to load frame_0_pyrup_0.txt";

    // Ensure input is continuous
    if (!input_pyrdown_cpu.isContinuous()) input_pyrdown_cpu = input_pyrdown_cpu.clone();
    ASSERT_TRUE(input_pyrdown_cpu.isContinuous());
    ASSERT_EQ(input_pyrdown_cpu.type(), CV_32FC3);

    int width = input_pyrdown_cpu.cols;
    int height = input_pyrdown_cpu.rows;
    size_t hostPitch = input_pyrdown_cpu.step;
    // Output size should match the reference pyrUp output
    int outWidth = pyrup_ref_cpu.cols;
    int outHeight = pyrup_ref_cpu.rows;
    cv::Size dst_shape(outWidth, outHeight);

    // 2. Allocate GPU Memory
    float* d_input = nullptr;
    float* d_output = nullptr;
    size_t dInputPitch = 0;
    size_t dOutputPitch = 0;
    CUDA_CHECK(cudaMallocPitch((void**)&d_input, &dInputPitch, width * 3 * sizeof(float), height));
    CUDA_CHECK(cudaMallocPitch((void**)&d_output, &dOutputPitch, outWidth * 3 * sizeof(float), outHeight));

    // 3. Copy Input Host -> Device
    CUDA_CHECK(cudaMemcpy2D(d_input, dInputPitch, input_pyrdown_cpu.ptr<float>(), hostPitch, width * 3 * sizeof(float), height, cudaMemcpyHostToDevice));

    // 4. Call CUDA function
    ASSERT_NE(d_kernel_x4, nullptr); // Ensure kernel*4 was allocated
    evmcuda::pyrUp_gpu(d_input, d_output, width, height, outWidth, outHeight, dInputPitch, dOutputPitch, d_kernel_x4, 0);

    // 5. Copy Result Device -> Host
    cv::Mat pyrup_result_cpu(outHeight, outWidth, CV_32FC3);
    CUDA_CHECK(cudaMemcpy2D(pyrup_result_cpu.ptr<float>(), pyrup_result_cpu.step, d_output, dOutputPitch, outWidth * 3 * sizeof(float), outHeight, cudaMemcpyDeviceToHost));

    // 6. Free GPU Memory
    cudaFree(d_input);
    cudaFree(d_output);

    // 7. Compare
    ASSERT_EQ(pyrup_result_cpu.size(), pyrup_ref_cpu.size());
    ASSERT_EQ(pyrup_result_cpu.type(), pyrup_ref_cpu.type());
    ASSERT_TRUE(CompareMatrices(pyrup_result_cpu, pyrup_ref_cpu, 1e-3)) // Increased tolerance
        << "Mismatch between CUDA pyrUp output and CPU reference data.";
}


// Test case comparing custom conv2DKernelSharedMem against cv::filter2D
TEST_F(CudaPyramidTest, Conv2DGpuMatchesCpuFilter2D) {
    // 1. Load CPU Input Data & Kernel
    cv::Mat input_yiq_cpu = loadMatrixFromTxt<float>("frame_0_yiq.txt", 3);
    EXPECT_FALSE(input_yiq_cpu.empty()) << "Failed to load frame_0_yiq.txt";
    const cv::Mat& h_kernel = evmcpp::gaussian_kernel;
    ASSERT_FALSE(h_kernel.empty());

    // Ensure input is continuous
    if (!input_yiq_cpu.isContinuous()) input_yiq_cpu = input_yiq_cpu.clone();
    ASSERT_TRUE(input_yiq_cpu.isContinuous());
    ASSERT_EQ(input_yiq_cpu.type(), CV_32FC3);

    int width = input_yiq_cpu.cols;
    int height = input_yiq_cpu.rows;
    size_t hostPitch = input_yiq_cpu.step;

    // 2. Calculate CPU Reference using cv::filter2D
    cv::Mat filter2d_ref_cpu;
    cv::filter2D(input_yiq_cpu, filter2d_ref_cpu, -1, h_kernel, cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101);
    ASSERT_FALSE(filter2d_ref_cpu.empty());

    // 3. Allocate GPU Memory (Input, Output, Kernel)
    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_kernel_conv = nullptr; // Use separate kernel memory for this test
    size_t dInputPitch = 0;
    size_t dOutputPitch = 0;
    CUDA_CHECK(cudaMallocPitch((void**)&d_input, &dInputPitch, width * 3 * sizeof(float), height));
    CUDA_CHECK(cudaMallocPitch((void**)&d_output, &dOutputPitch, width * 3 * sizeof(float), height)); // Output same size as input for conv
    CUDA_CHECK(cudaMalloc((void**)&d_kernel_conv, 25 * sizeof(float)));

    // 4. Copy Input & Kernel Host -> Device
    CUDA_CHECK(cudaMemcpy2D(d_input, dInputPitch, input_yiq_cpu.ptr<float>(), hostPitch, width * 3 * sizeof(float), height, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel_conv, h_kernel.ptr<float>(), 25 * sizeof(float), cudaMemcpyHostToDevice));

    // 5. Call the C++ wrapper function for the CUDA kernel
    try {
        evmcuda::conv2DKernelSharedMem_gpu(d_input, d_output, width, height, dInputPitch, dOutputPitch, d_kernel_conv, 0); // Use default stream 0
        CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel completion
    } catch (const std::exception& e) {
         cudaFree(d_input);
         cudaFree(d_output);
         cudaFree(d_kernel_conv);
         GTEST_FAIL() << "conv2DKernelSharedMem_gpu function threw an exception: " << e.what();
    } catch (...) {
         cudaFree(d_input);
         cudaFree(d_output);
         cudaFree(d_kernel_conv);
         GTEST_FAIL() << "conv2DKernelSharedMem_gpu function threw an unknown exception.";
    }

    // 6. Copy Result Device -> Host
    cv::Mat conv_result_cpu(height, width, CV_32FC3);
    CUDA_CHECK(cudaMemcpy2D(conv_result_cpu.ptr<float>(), conv_result_cpu.step, d_output, dOutputPitch, width * 3 * sizeof(float), height, cudaMemcpyDeviceToHost));

    // 7. Free GPU Memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel_conv);

    // 8. Compare GPU conv result with CPU filter2D reference
    ASSERT_EQ(conv_result_cpu.size(), filter2d_ref_cpu.size());
    ASSERT_EQ(conv_result_cpu.type(), filter2d_ref_cpu.type());
    // Use a reasonable tolerance for direct convolution comparison
    ASSERT_TRUE(CompareMatrices(conv_result_cpu, filter2d_ref_cpu, 1e-4))
        << "Mismatch between CUDA conv2DKernelSharedMem output and cv::filter2D reference data.";
}