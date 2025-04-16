#ifndef EVMCPP_PROCESSING_HPP
#define EVMCPP_PROCESSING_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
// #include "butterworth.hpp"         // No longer needed here
// #include "gaussian_pyramid.hpp" // No longer needed here after refactor
#include "laplacian_pyramid.hpp" // Keep if laplacian processing is still used

namespace evmcpp {

// --- Gaussian Kernel (from Python constants.py) ---
const float gaussian_kernel_data[25] = {
    1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f,  1.0f/256.0f,
    4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f,  4.0f/256.0f,
    6.0f/256.0f, 24.0f/256.0f, 36.0f/256.0f, 24.0f/256.0f,  6.0f/256.0f,
    4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f,  4.0f/256.0f,
    1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f,  1.0f/256.0f
};
const cv::Mat gaussian_kernel(5, 5, CV_32F, (void*)gaussian_kernel_data);


// --- Color Space Conversions ---
// Keep signatures simple, handle debug data capture in the orchestrating function.
cv::Mat rgb2yiq(const cv::Mat& rgbFrame); // Corrected
cv::Mat yiq2rgb(const cv::Mat& yiqFrame); // Corrected

// --- Video Processing ---

// Process video using Laplacian Pyramid method
void processVideoLaplacian(const std::string& inputFilename, const std::string& outputFilename, // Corrected
                           int levels, double alpha, double lambda_c, double fl, double fh,
                           double chromAttenuation);

// Process video using Gaussian Pyramid method (Batch version)
void processVideoGaussianBatch(const std::string& inputFilename, const std::string& outputFilename,
                             int levels, double alpha, double fl, double fh,
                             double chromAttenuation);


// --- Custom Pyramid Functions (Mirroring Python's filter2D approach) ---
// Version with explicit kernel
cv::Mat pyrDown(const cv::Mat& image, const cv::Mat& kernel);
// Version using default Gaussian kernel
cv::Mat pyrDown(const cv::Mat& image);

// Version with explicit kernel
cv::Mat pyrUp(const cv::Mat& image, const cv::Mat& kernel, const cv::Size& dst_shape);
// Version using default Gaussian kernel
cv::Mat pyrUp(const cv::Mat& image, const cv::Size& dst_shape);


} // namespace evmcpp

#endif // EVMCPP_PROCESSING_HPP