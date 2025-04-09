#include "evmcpu/pyramid.hpp"
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace evmcpu {

cv::Mat pyr_down(const cv::Mat& image, const cv::Mat& kernel) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty.");
    }
    if (image.depth() != CV_32F) {
        throw std::invalid_argument("Input image must be CV_32F type.");
    }

    // Apply Gaussian filtering
    cv::Mat filtered;
    cv::filter2D(image, filtered, CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101);

    // Downsample by taking every other pixel (matches Python's [::2, ::2])
    cv::Mat result(filtered.rows/2, filtered.cols/2, filtered.type());
    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {
            if (image.channels() == 1) {
                result.at<float>(y, x) = filtered.at<float>(2*y, 2*x);
            } else {
                result.at<cv::Vec3f>(y, x) = filtered.at<cv::Vec3f>(2*y, 2*x);
            }
        }
    }
    return result;
}

cv::Mat pyr_up(const cv::Mat& image, const cv::Mat& kernel, const cv::Size& dst_shape) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty.");
    }
    if (image.depth() != CV_32F) {
        throw std::invalid_argument("Input image must be CV_32F type.");
    }

    // Calculate target dimensions
    int dst_height = dst_shape.height > 0 ? dst_shape.height : 2 * image.rows;
    int dst_width = dst_shape.width > 0 ? dst_shape.width : 2 * image.cols;

    // Create upsampled image initialized to zeros
    cv::Mat upsampled = cv::Mat::zeros(dst_height, dst_width, image.type());

    // Copy source pixels to every other position (matches Python's insert logic)
    for (int y = 0; y < image.rows && 2*y < dst_height; ++y) {
        for (int x = 0; x < image.cols && 2*x < dst_width; ++x) {
            if (image.channels() == 1) {
                upsampled.at<float>(2*y, 2*x) = image.at<float>(y, x);
            } else {
                upsampled.at<cv::Vec3f>(2*y, 2*x) = image.at<cv::Vec3f>(y, x);
            }
        }
    }

    // Apply Gaussian filtering with 4x kernel (matches Python implementation)
    cv::Mat result;
    cv::filter2D(upsampled, result, CV_32F, 4 * kernel, cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101);
    return result;
}

} // namespace evmcpu