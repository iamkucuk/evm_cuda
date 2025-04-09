#ifndef EVMCPP_EVMCUDA_RECONSTRUCTION_CUH
#define EVMCPP_EVMCUDA_RECONSTRUCTION_CUH

#include <opencv2/core.hpp> // For cv::Mat
#include <cuda_runtime.h>   // For cudaError_t, size_t, cudaStream_t

namespace evmcuda {

/**
 * @brief Adds the filtered YIQ signal back to the original YIQ signal on the GPU.
 *        result_yiq = original_yiq + filtered_yiq
 * 
 * @param original_yiq Device pointer to the original YIQ frame (CV_32FC3).
 * @param original_pitch Pitch of original_yiq in bytes.
 * @param filtered_yiq Device pointer to the filtered YIQ signal (CV_32FC3).
 * @param filtered_pitch Pitch of filtered_yiq in bytes.
 * @param result_yiq Device pointer to store the combined YIQ result (CV_32FC3).
 * @param result_pitch Pitch of result_yiq in bytes.
 * @param width Frame width in pixels.
 * @param height Frame height in pixels.
 * @param stream CUDA stream for asynchronous execution (default: 0).
 * @return cudaError_t Error code (cudaSuccess on success).
 */
cudaError_t addFilteredSignal(
    const float* original_yiq,
    size_t original_pitch,
    const float* filtered_yiq,
    size_t filtered_pitch,
    float* result_yiq,
    size_t result_pitch,
    int width,
    int height,
    cudaStream_t stream = 0
);

/**
 * @brief Converts a YIQ frame (float) to RGB (uint8) on the GPU,
 *        including clipping to [0, 255] and casting.
 * 
 * @param yiq_frame Device pointer to the input YIQ frame (CV_32FC3).
 * @param yiq_pitch Pitch of yiq_frame in bytes.
 * @param rgb_frame Device pointer to store the output RGB frame (CV_8UC3).
 * @param rgb_pitch Pitch of rgb_frame in bytes.
 * @param width Frame width in pixels.
 * @param height Frame height in pixels.
 * @param stream CUDA stream for asynchronous execution (default: 0).
 * @return cudaError_t Error code (cudaSuccess on success).
 */
cudaError_t convertYiqToRgbClipAndCast(
    const float* yiq_frame,
    size_t yiq_pitch,
    unsigned char* rgb_frame,
    size_t rgb_pitch,
    int width,
    int height,
    cudaStream_t stream = 0
);


/**
 * @brief Reconstructs a single frame for the Gaussian EVM pathway using CUDA.
 *        Performs: original RGB -> YIQ -> Add Filtered Signal -> YIQ -> RGB (clipped, uint8)
 * 
 * @param original_rgb Host cv::Mat containing the original RGB frame (CV_8UC3).
 * @param filtered_yiq Device pointer to the temporally filtered YIQ signal (CV_32FC3).
 * @param filtered_pitch Pitch of filtered_yiq in bytes.
 * @param output_rgb Host cv::Mat to store the final reconstructed RGB frame (CV_8UC3). Will be allocated/resized if necessary.
 * @param stream CUDA stream for asynchronous execution (default: 0).
 * @return cudaError_t Error code (cudaSuccess on success).
 */
cudaError_t reconstructGaussianFrame(
    const cv::Mat& original_rgb,
    const float* filtered_yiq,
    size_t filtered_pitch,
    cv::Mat& output_rgb,
    cudaStream_t stream = 0
);


} // namespace evmcuda

#endif // EVMCPP_EVMCUDA_RECONSTRUCTION_CUH