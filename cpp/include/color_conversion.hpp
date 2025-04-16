#ifndef COLOR_CONVERSION_HPP
#define COLOR_CONVERSION_HPP

#include <opencv2/core.hpp>

namespace evmcpu {

/**
 * @brief Converts an RGB image to YIQ color space
 * 
 * @param rgb_image Input RGB image (CV_8UC3 or CV_32FC3)
 * @return cv::Mat Output YIQ image (CV_32FC3)
 * @throws std::invalid_argument if input image is empty or not 3-channel
 */
cv::Mat rgb_to_yiq(const cv::Mat& rgb_image);

/**
 * @brief Converts a YIQ image to RGB color space
 *
 * @param yiq_image Input YIQ image (CV_32FC3)
 * @param rgb_image Output RGB image (CV_32FC3)
 * @throws std::invalid_argument if input image is empty, not 3-channel, or not CV_32FC3
 */
void yiq_to_rgb(const cv::Mat& yiq_image, cv::Mat& rgb_image);

} // namespace evmcpu

#endif // COLOR_CONVERSION_HPP