#ifndef CUDA_PYRAMID_CUH
#define CUDA_PYRAMID_CUH

#include <cuda_runtime.h>
#include <stdexcept>
#include <opencv2/core.hpp>

namespace evmcuda {

/**
 * @brief Constant data for the 5x5 Gaussian kernel used in pyramid operations
 */
extern __constant__ float d_gaussian_kernel[25];

/**
 * @brief Host-side function to initialize the Gaussian kernel in device constant memory
 * This must be called before using any pyramid operations
 * 
 * @return true if initialization succeeded, false otherwise
 */
bool init_pyramid();

/**
 * @brief Downsamples an image using Gaussian filtering followed by 2x2 decimation
 * 
 * @param d_src Input image data in device memory (float format, row-major)
 * @param src_width Width of input image
 * @param src_height Height of input image
 * @param src_channels Number of channels in input image (usually 3 for YIQ/RGB)
 * @param d_dst Output image data in device memory (float format, row-major)
 * @param stream CUDA stream to use for the operation (optional)
 * @throws std::runtime_error if CUDA operation fails
 */
void pyr_down_gpu(
    const float* d_src,
    int src_width,
    int src_height,
    int src_channels,
    float* d_dst,
    cudaStream_t stream = nullptr);

/**
 * @brief Upsamples an image by 2x with zeros and applies Gaussian filtering
 * 
 * @param d_src Input image data in device memory (float format, row-major)
 * @param src_width Width of input image
 * @param src_height Height of input image
 * @param src_channels Number of channels in input image (usually 3 for YIQ/RGB)
 * @param d_dst Output image data in device memory (float format, row-major)
 * @param dst_width Width of output image
 * @param dst_height Height of output image
 * @param stream CUDA stream to use for the operation (optional)
 * @throws std::runtime_error if CUDA operation fails
 */
void pyr_up_gpu(
    const float* d_src,
    int src_width,
    int src_height,
    int src_channels,
    float* d_dst,
    int dst_width,
    int dst_height,
    cudaStream_t stream = nullptr);

/**
 * @brief Host wrapper function for pyramid downsampling
 * 
 * @param h_src Host input image data (float format, row-major)
 * @param src_width Width of input image
 * @param src_height Height of input image
 * @param src_channels Number of channels in input image (usually 3 for YIQ/RGB)
 * @param h_dst Host output image data (pre-allocated buffer)
 * @return true if operation succeeded, false otherwise
 */
bool pyr_down_wrapper(
    const float* h_src,
    int src_width,
    int src_height,
    int src_channels,
    float* h_dst);

/**
 * @brief Host wrapper function for pyramid upsampling
 * 
 * @param h_src Host input image data (float format, row-major)
 * @param src_width Width of input image
 * @param src_height Height of input image
 * @param src_channels Number of channels in input image (usually 3 for YIQ/RGB)
 * @param h_dst Host output image data (pre-allocated buffer)
 * @param dst_width Width of output image
 * @param dst_height Height of output image
 * @return true if operation succeeded, false otherwise
 */
bool pyr_up_wrapper(
    const float* h_src,
    int src_width,
    int src_height,
    int src_channels,
    float* h_dst,
    int dst_width,
    int dst_height);

/**
 * @brief Convenience function that converts OpenCV Mat to device memory for pyramid operations
 * 
 * @param src_mat Input OpenCV Mat (CV_32FC1 or CV_32FC3)
 * @param d_src Pointer to device memory for input (will be allocated)
 * @param src_size Size of the allocated device memory in bytes
 * @return true if operation succeeded, false otherwise
 */
bool mat_to_device(
    const cv::Mat& src_mat,
    float** d_src,
    size_t& src_size);

/**
 * @brief Convenience function that converts device memory back to OpenCV Mat
 * 
 * @param d_dst Device memory pointer with output data
 * @param width Width of the output image
 * @param height Height of the output image
 * @param channels Number of channels in the output image
 * @param dst_mat Output OpenCV Mat (will be allocated)
 * @return true if operation succeeded, false otherwise
 */
bool device_to_mat(
    const float* d_dst,
    int width,
    int height,
    int channels,
    cv::Mat& dst_mat);

/**
 * @brief Apply pyramid downsampling directly to an OpenCV Mat
 * 
 * @param src Input image (CV_32FC1 or CV_32FC3)
 * @return Downsampled image (half width and height)
 */
cv::Mat pyr_down(const cv::Mat& src);

/**
 * @brief Apply pyramid upsampling directly to an OpenCV Mat
 * 
 * @param src Input image (CV_32FC1 or CV_32FC3)
 * @param dst_size Target size for the output image
 * @return Upsampled and filtered image
 */
cv::Mat pyr_up(const cv::Mat& src, const cv::Size& dst_size);

/**
 * @brief Clean up resources used by the pyramid module
 */
void cleanup_pyramid();

} // namespace evmcuda

#endif // CUDA_PYRAMID_CUH