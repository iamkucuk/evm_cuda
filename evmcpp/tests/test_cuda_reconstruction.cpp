#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "evmcuda/reconstruction.cuh"
#include "evmcuda/color_conversion.cuh"
#include "evmcpu/color_conversion.hpp"
#include "test_helpers.hpp"

class CudaReconstructionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load reference data from files specified in test_gaussian.cpp and generate_test_data.py
        original_rgb_ = loadMatrixFromTxt<unsigned char>("frame_0_rgb.txt", 3);
        filtered_yiq_ = loadMatrixFromTxt<float>("frame_0_step4_temporal_filtered_yiq.txt", 3);
        expected_rgb_ = loadMatrixFromTxt<unsigned char>("frame_0_step6e_final_rgb_uint8.txt", 3);
        
        ASSERT_FALSE(original_rgb_.empty()) << "Failed to load original RGB frame (frame_0_rgb.txt)";
        ASSERT_FALSE(filtered_yiq_.empty()) << "Failed to load filtered YIQ frame (frame_0_step4_temporal_filtered_yiq.txt)";
        ASSERT_FALSE(expected_rgb_.empty()) << "Failed to load expected RGB frame (frame_0_step6e_final_rgb_uint8.txt)";

        // Allocate pitched device memory for filtered YIQ data (input to reconstruction)
        const int width = filtered_yiq_.cols;
        const int height = filtered_yiq_.rows;
        
        ASSERT_EQ(cudaMallocPitch(&d_filtered_yiq_, &filtered_pitch_,
                                 width * 3 * sizeof(float), height),
                  cudaSuccess) << "Failed to allocate device memory for filtered YIQ";

        // Copy filtered YIQ data to device
        ASSERT_EQ(cudaMemcpy2D(d_filtered_yiq_, filtered_pitch_,
                              filtered_yiq_.ptr<float>(), filtered_yiq_.step,
                              width * 3 * sizeof(float), height,
                              cudaMemcpyHostToDevice),
                  cudaSuccess) << "Failed to copy filtered YIQ to device";
    }

    void TearDown() override {
        if (d_filtered_yiq_) {
            cudaFree(d_filtered_yiq_);
            d_filtered_yiq_ = nullptr;
        }
    }

    cv::Mat original_rgb_;     // Original RGB frame (uint8) - Host
    cv::Mat filtered_yiq_;     // Filtered YIQ signal (float) - Host (also copied to device)
    cv::Mat expected_rgb_;     // Expected output RGB frame (uint8) - Host
    
    float* d_filtered_yiq_ = nullptr;  // Device pointer to filtered YIQ
    size_t filtered_pitch_ = 0;        // Pitch of filtered YIQ in bytes
};

// Test the addFilteredSignal GPU kernel against CPU cv::add
TEST_F(CudaReconstructionTest, AddFilteredSignalGpuMatchesCpu) {
    const int width = original_rgb_.cols;
    const int height = original_rgb_.rows;
    
    // --- Prepare CPU Reference ---
    // 1. Convert original RGB (uint8) to YIQ (float) on CPU
    cv::Mat cpu_original_yiq;
    cv::Mat tmp_rgb_float;
    original_rgb_.convertTo(tmp_rgb_float, CV_32F);
    cpu_original_yiq = evmcpu::rgb_to_yiq(tmp_rgb_float);
    ASSERT_FALSE(cpu_original_yiq.empty()) << "CPU rgb_to_yiq failed";

    // 2. Add the filtered signal (already loaded as float YIQ) on CPU
    cv::Mat cpu_result_yiq;
    cv::add(cpu_original_yiq, filtered_yiq_, cpu_result_yiq);
    ASSERT_FALSE(cpu_result_yiq.empty()) << "CPU cv::add failed";

    // --- Prepare GPU Execution ---
    // Allocate device memory for original YIQ and result YIQ
    float *d_original_yiq = nullptr, *d_result_yiq = nullptr;
    size_t yiq_pitch = 0;
    
    ASSERT_EQ(cudaMallocPitch(&d_original_yiq, &yiq_pitch, width * 3 * sizeof(float), height), cudaSuccess);
    ASSERT_EQ(cudaMallocPitch(&d_result_yiq, &yiq_pitch, width * 3 * sizeof(float), height), cudaSuccess);
    
    // Copy CPU's original YIQ to device
    ASSERT_EQ(cudaMemcpy2D(d_original_yiq, yiq_pitch,
                          cpu_original_yiq.ptr<float>(), cpu_original_yiq.step,
                          width * 3 * sizeof(float), height,
                          cudaMemcpyHostToDevice), cudaSuccess);
    
    // --- Run GPU addFilteredSignal ---
    cudaError_t gpu_err = evmcuda::addFilteredSignal(
        d_original_yiq, yiq_pitch,
        d_filtered_yiq_, filtered_pitch_,
        d_result_yiq, yiq_pitch,
        width, height);
    ASSERT_EQ(gpu_err, cudaSuccess) << "GPU addFilteredSignal kernel failed: " << cudaGetErrorString(gpu_err);
    
    // --- Copy GPU result back to host ---
    cv::Mat gpu_result_yiq(height, width, CV_32FC3);
    ASSERT_EQ(cudaMemcpy2D(gpu_result_yiq.ptr<float>(), gpu_result_yiq.step,
                          d_result_yiq, yiq_pitch,
                          width * 3 * sizeof(float), height,
                          cudaMemcpyDeviceToHost), cudaSuccess);
    
    // --- Compare GPU vs CPU ---
    // Use a small tolerance for floating-point comparisons
    ASSERT_TRUE(CompareMatrices(gpu_result_yiq, cpu_result_yiq, 1e-4f))
        << "GPU addFilteredSignal output doesn't match CPU reference (cv::add)";
    
    // --- Cleanup ---
    cudaFree(d_original_yiq);
    cudaFree(d_result_yiq);
}

// Test the convertYiqToRgbClipAndCast GPU kernel against CPU steps
TEST_F(CudaReconstructionTest, ConvertYiqToRgbClipAndCastGpuMatchesCpu) {
    const int width = filtered_yiq_.cols;
    const int height = filtered_yiq_.rows;
    
    // --- Prepare CPU Reference ---
    // 1. Convert YIQ (float) to RGB (float) on CPU
    cv::Mat cpu_rgb_float;
    evmcpu::yiq_to_rgb(filtered_yiq_, cpu_rgb_float);
    ASSERT_FALSE(cpu_rgb_float.empty()) << "CPU yiq_to_rgb failed";
    
    // 2. Clip values to [0, 255] on CPU
    cv::Mat cpu_clipped;
    cv::max(cv::min(cpu_rgb_float, 255.0f), 0.0f, cpu_clipped);
    
    // 3. Convert to uint8 on CPU
    cv::Mat cpu_result_rgb_uint8;
    cpu_clipped.convertTo(cpu_result_rgb_uint8, CV_8UC3);
    ASSERT_FALSE(cpu_result_rgb_uint8.empty()) << "CPU convertTo uint8 failed";

    // --- Prepare GPU Execution ---
    // Allocate device memory for output RGB (uint8)
    unsigned char* d_rgb_result = nullptr;
    size_t rgb_pitch = 0;
    ASSERT_EQ(cudaMallocPitch(&d_rgb_result, &rgb_pitch, width * 3, height), cudaSuccess);
    
    // --- Run GPU convertYiqToRgbClipAndCast ---
    // Use the pre-loaded d_filtered_yiq_ from SetUp
    cudaError_t gpu_err = evmcuda::convertYiqToRgbClipAndCast(
        d_filtered_yiq_, filtered_pitch_,
        d_rgb_result, rgb_pitch,
        width, height);
    ASSERT_EQ(gpu_err, cudaSuccess) << "GPU convertYiqToRgbClipAndCast kernel failed: " << cudaGetErrorString(gpu_err);
    
    // --- Copy GPU result back to host ---
    cv::Mat gpu_result_rgb_uint8(height, width, CV_8UC3);
    ASSERT_EQ(cudaMemcpy2D(gpu_result_rgb_uint8.ptr(), gpu_result_rgb_uint8.step,
                          d_rgb_result, rgb_pitch,
                          width * 3, height,
                          cudaMemcpyDeviceToHost), cudaSuccess);
    
    // --- Compare GPU vs CPU ---
    // Allow tolerance of 1 due to potential minor differences in float -> uint8 conversion/rounding
    ASSERT_TRUE(CompareMatrices(gpu_result_rgb_uint8, cpu_result_rgb_uint8, 1)) 
        << "GPU YIQ->RGB conversion with clipping/casting doesn't match CPU reference";
    
    // --- Cleanup ---
    cudaFree(d_rgb_result);
}

// Test the full reconstructGaussianFrame GPU function against CPU pipeline
TEST_F(CudaReconstructionTest, FullGaussianReconstructionMatchesCpu) {
    cv::Mat output_rgb_gpu(original_rgb_.size(), CV_8UC3);
    cv::Mat original_rgb_float;
    original_rgb_.convertTo(original_rgb_float, CV_32F);

    // --- Run GPU reconstruction ---
    cudaError_t gpu_err = evmcuda::reconstructGaussianFrame(
        original_rgb_,
        d_filtered_yiq_, filtered_pitch_,
        output_rgb_gpu);
    ASSERT_EQ(gpu_err, cudaSuccess) << "GPU reconstructGaussianFrame function failed: " << cudaGetErrorString(gpu_err);
    
    // --- Calculate CPU reference result using same steps as test_gaussian.cpp ---
    // 1. Convert original RGB to YIQ
    cv::Mat originalYiq = evmcpu::rgb_to_yiq(original_rgb_float);
    ASSERT_FALSE(originalYiq.empty()) << "CPU rgb_to_yiq failed";

    // 2. Add filtered signal
    cv::Mat combinedYiq;
    cv::add(originalYiq, filtered_yiq_, combinedYiq);
    ASSERT_FALSE(combinedYiq.empty()) << "CPU cv::add failed";

    // 3. Convert combined YIQ back to RGB
    cv::Mat expectedRgbFloat;
    evmcpu::yiq_to_rgb(combinedYiq, expectedRgbFloat);
    ASSERT_FALSE(expectedRgbFloat.empty()) << "CPU yiq_to_rgb failed";

    // 4. Clip RGB values to [0, 255]
    cv::Mat expectedClippedRgbFloat = expectedRgbFloat.clone();
    cv::max(cv::min(expectedClippedRgbFloat, 255.0f), 0.0f, expectedClippedRgbFloat);

    // 5. Convert to uint8
    cv::Mat expected_rgb_cpu;
    expectedClippedRgbFloat.convertTo(expected_rgb_cpu, CV_8UC3);
    ASSERT_FALSE(expected_rgb_cpu.empty()) << "CPU convertTo uint8 failed";

    // --- Compare GPU vs CPU results ---
    // Allow tolerance of 1 for uint8 comparison due to potential rounding differences
    ASSERT_TRUE(CompareMatrices(output_rgb_gpu, expected_rgb_cpu, 1))
        << "GPU Full Gaussian reconstruction output doesn't match CPU reference pipeline";
}