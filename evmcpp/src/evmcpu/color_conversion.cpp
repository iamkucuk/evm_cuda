#include "evmcpu/color_conversion.hpp"
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace evmcpu {

// RGB to YIQ conversion matrix (float precision)
const cv::Matx33f RGB2YIQ_MATRIX = {
    0.299f,       0.587f,       0.114f,
    0.59590059f, -0.27455667f, -0.32134392f,
    0.21153661f, -0.52273617f,  0.31119955f
};

// YIQ to RGB conversion matrix (float precision)
// Standard values from ITU/NTSC specification
const cv::Matx33f YIQ2RGB_MATRIX = {
    1.0000f,     0.9563f,     0.6210f,
    1.0000f,    -0.2721f,    -0.6474f,
    1.0000f,    -1.1070f,     1.7046f
};

cv::Mat rgb_to_yiq(const cv::Mat& rgb_image) {
    if (rgb_image.empty()) {
        throw std::invalid_argument("Input RGB image is empty.");
    }
    if (rgb_image.channels() != 3) {
        throw std::invalid_argument("Input image must have 3 channels (RGB).");
    }

    cv::Mat float_image;
    if (rgb_image.depth() != CV_32F) {
        // Convert type without scaling to match Python's astype(np.float32)
        rgb_image.convertTo(float_image, CV_32F);
    } else {
        float_image = rgb_image.clone();
    }

    cv::Mat yiq_image_float;
    cv::transform(float_image, yiq_image_float, RGB2YIQ_MATRIX);
    return yiq_image_float;
}

void yiq_to_rgb(const cv::Mat& yiq_image, cv::Mat& rgb_image) {
    if (yiq_image.empty()) {
        throw std::invalid_argument("Input YIQ image is empty.");
    }
    if (yiq_image.channels() != 3) {
        throw std::invalid_argument("Input image must have 3 channels (YIQ).");
    }
    if (yiq_image.type() != CV_32FC3) {
        throw std::invalid_argument("Input YIQ image must be type CV_32FC3.");
    }

    // Transform YIQ to RGB using the conversion matrix
    cv::transform(yiq_image, rgb_image, YIQ2RGB_MATRIX);
}

} // namespace evmcpu