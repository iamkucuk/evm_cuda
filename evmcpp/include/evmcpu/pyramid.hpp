#ifndef EVMCPP_EVMCPU_PYRAMID_HPP
#define EVMCPP_EVMCPU_PYRAMID_HPP

#include <opencv2/core.hpp>

namespace evmcpu {

/**
 * @brief Downsamples an image using Gaussian filtering followed by 2x2 decimation.
 * @param image Input image (CV_32FC1 or CV_32FC3)
 * @param kernel Gaussian kernel for filtering before downsampling
 * @return Downsampled image (half width and height)
 * @throws std::invalid_argument if image is empty or not CV_32F type
 */
cv::Mat pyr_down(const cv::Mat& image, const cv::Mat& kernel);

/**
 * @brief Upsamples an image by 2x with zeros and applies Gaussian filtering.
 * @param image Input image (CV_32FC1 or CV_32FC3)
 * @param kernel Gaussian kernel for filtering after upsampling
 * @param dst_shape Optional target shape (height, width) for the output. If not provided,
 *                 output will be (2*height+1, 2*width+1). If provided, output will be
 *                 adjusted to match the destination shape if needed.
 * @return Upsampled and filtered image
 * @throws std::invalid_argument if image is empty or not CV_32F type
 */
cv::Mat pyr_up(const cv::Mat& image, const cv::Mat& kernel, const cv::Size& dst_shape = cv::Size());

} // namespace evmcpu

#endif // EVMCPP_EVMCPU_PYRAMID_HPP