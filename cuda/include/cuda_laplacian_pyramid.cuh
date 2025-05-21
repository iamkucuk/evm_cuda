#ifndef CUDA_LAPLACIAN_PYRAMID_CUH
#define CUDA_LAPLACIAN_PYRAMID_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <vector>
#include <opencv2/core.hpp>
#include <utility> // For std::pair

namespace evmcuda {

// Kernel to subtract two images (exposed for use in other functions)
__global__ void subtract_kernel(
    const float* __restrict__ d_src,
    const float* __restrict__ d_upsampled,
    float* __restrict__ d_laplacian,
    int width,
    int height,
    int channels,
    int stride);

/**
 * @brief Generate a Laplacian pyramid for an image using CUDA
 * 
 * @param input_image Input image (assumed to be in YIQ format, CV_32FC3)
 * @param level Number of pyramid levels to generate
 * @return A vector of cv::Mat, where each Mat is a level of the Laplacian pyramid
 */
std::vector<cv::Mat> generate_laplacian_pyramid(
    const cv::Mat& input_image,
    int level);

/**
 * @brief Generates Laplacian pyramids for a sequence of images (video frames) using CUDA
 * 
 * @param input_images Sequence of RGB image frames
 * @param level Number of pyramid levels
 * @return A vector where each element is a Laplacian pyramid (std::vector<cv::Mat>) for a frame
 */
std::vector<std::vector<cv::Mat>> get_laplacian_pyramids(
    const std::vector<cv::Mat>& input_images,
    int level);

/**
 * @brief Apply temporal bandpass filtering and spatial attenuation to Laplacian pyramids using CUDA
 * 
 * @param pyramids_batch Collection of Laplacian pyramids (Time x Level x H x W x C)
 * @param level Number of pyramid levels
 * @param fps Frames per second of the video
 * @param freq_range Pair [low_freq, high_freq] for bandpass filter
 * @param alpha Magnification factor
 * @param lambda_cutoff Spatial wavelength cutoff for attenuation
 * @param attenuation Factor to attenuate chrominance channels
 * @return The modified pyramids_batch after filtering and attenuation
 */
std::vector<std::vector<cv::Mat>> filter_laplacian_pyramids(
    const std::vector<std::vector<cv::Mat>>& pyramids_batch,
    int level,
    double fps,
    const std::pair<double, double>& freq_range,
    double alpha,
    double lambda_cutoff,
    double attenuation);

/**
 * @brief Reconstruct a magnified image from the original and filtered pyramid using CUDA
 * 
 * @param original_rgb_image Original input RGB image
 * @param filtered_pyramid Filtered Laplacian pyramid
 * @return Reconstructed magnified RGB image (CV_8UC3)
 */
cv::Mat reconstruct_laplacian_image(
    const cv::Mat& original_rgb_image,
    const std::vector<cv::Mat>& filtered_pyramid);

/**
 * @brief Device function to construct a Laplacian pyramid level from source and downsampled images
 * 
 * @param d_src Source image in device memory
 * @param src_width Width of source image
 * @param src_height Height of source image
 * @param d_downsampled Downsampled image in device memory
 * @param d_upsampled Upsampled image in device memory (derived from downsampled)
 * @param d_laplacian Laplacian level output in device memory
 * @param stream CUDA stream to use for the operation
 */
void construct_laplacian_level_gpu(
    const float* d_src,
    int src_width,
    int src_height,
    int channels,
    const float* d_upsampled,
    float* d_laplacian,
    cudaStream_t stream = nullptr);

/**
 * @brief Initialize resources needed by the Laplacian pyramid module
 * 
 * @return true if initialization succeeded, false otherwise
 */
bool init_laplacian_pyramid();

/**
 * @brief Clean up resources used by the Laplacian pyramid module
 */
void cleanup_laplacian_pyramid();

/**
 * @brief Helper function to get an estimated lambda for a pyramid level
 * @param width Width of the level
 * @param height Height of the level
 * @return Estimated lambda value
 */
inline double estimate_lambda(int width, int height) {
    return std::sqrt(static_cast<double>(height * height + width * width));
}

/**
 * @brief Compute the appropriate alpha value based on spatial lambda cutoff
 * @param lambda Estimated lambda for the level
 * @param delta Spatial delta value calculated from lambda_cutoff
 * @param max_alpha Maximum alpha value to cap the result
 * @return Computed alpha value
 */
inline double compute_alpha(double lambda, double delta, double max_alpha) {
    double new_alpha = (lambda / (8.0 * delta)) - 1.0;
    return std::min(max_alpha, new_alpha);
}

} // namespace evmcuda

#endif // CUDA_LAPLACIAN_PYRAMID_CUH