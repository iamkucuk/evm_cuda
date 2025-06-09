#ifndef CUDA_GAUSSIAN_PYRAMID_CUH
#define CUDA_GAUSSIAN_PYRAMID_CUH

#include <cuda_runtime.h>
#include <cufft.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace cuda_evm {

/**
 * @brief CUDA implementation of spatially filter Gaussian
 * Equivalent to CPU spatiallyFilterGaussian function
 * @param d_input_rgb Input RGB image data on device (float)
 * @param d_output_yiq Output YIQ filtered data on device (float)
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels (3 for RGB/YIQ)
 * @param level Number of pyramid levels for down/up sampling
 * @return cudaError_t error code
 */
cudaError_t spatially_filter_gaussian_gpu(
    const float* d_input_rgb,
    float* d_output_yiq,
    int width,
    int height,
    int channels,
    int level
);

/**
 * @brief CUDA implementation of temporal Gaussian batch filtering using FFT
 * Equivalent to CPU temporalFilterGaussianBatch function
 * @param d_all_frames All spatially filtered frames (frame-major layout)
 * @param d_filtered_frames Output filtered frames
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels (3 for YIQ)
 * @param num_frames Number of frames in batch
 * @param fps Frames per second
 * @param fl Low cutoff frequency (Hz)
 * @param fh High cutoff frequency (Hz)
 * @param alpha Amplification factor
 * @param chrom_attenuation Chrominance attenuation factor
 * @return cudaError_t error code
 */
cudaError_t temporal_filter_gaussian_batch_gpu(
    const float* d_all_frames,
    float* d_filtered_frames,
    int width,
    int height,
    int channels,
    int num_frames,
    float fps,
    float fl,
    float fh,
    float alpha,
    float chrom_attenuation
);

/**
 * @brief CUDA implementation of Gaussian frame reconstruction
 * Equivalent to CPU reconstructGaussianFrame function
 * @param d_original_rgb Original RGB frame on device
 * @param d_filtered_yiq Filtered YIQ signal on device
 * @param d_output_rgb Reconstructed RGB frame on device
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels (3)
 * @return cudaError_t error code
 */
cudaError_t reconstruct_gaussian_frame_gpu(
    const float* d_original_rgb,
    const float* d_filtered_yiq,
    float* d_output_rgb,
    int width,
    int height,
    int channels
);

/**
 * @brief Process video using CUDA Gaussian pyramid method
 * Complete end-to-end GPU implementation equivalent to CPU processVideoGaussianBatch
 * @param input_filename Input video file path
 * @param output_filename Output video file path
 * @param levels Number of pyramid levels
 * @param alpha Amplification factor
 * @param fl Low cutoff frequency (Hz)
 * @param fh High cutoff frequency (Hz)
 * @param chrom_attenuation Chrominance attenuation factor
 * @return bool success status
 */
bool process_video_gaussian_gpu(
    const std::string& input_filename,
    const std::string& output_filename,
    int levels,
    double alpha,
    double fl,
    double fh,
    double chrom_attenuation
);

// Host wrapper functions for easier integration
cv::Mat spatially_filter_gaussian_wrapper(const cv::Mat& input_rgb, int level);
std::vector<cv::Mat> temporal_filter_gaussian_batch_wrapper(
    const std::vector<cv::Mat>& spatially_filtered_batch,
    float fps,
    float fl,
    float fh,
    float alpha,
    float chrom_attenuation
);
cv::Mat reconstruct_gaussian_frame_wrapper(
    const cv::Mat& original_rgb,
    const cv::Mat& filtered_yiq_signal
);

} // namespace cuda_evm

#endif // CUDA_GAUSSIAN_PYRAMID_CUH