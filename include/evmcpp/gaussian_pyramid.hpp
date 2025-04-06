#ifndef EVMCPP_GAUSSIAN_PYRAMID_HPP
#define EVMCPP_GAUSSIAN_PYRAMID_HPP

#include <opencv2/core.hpp> // Use core instead of full opencv.hpp
#include <vector>

namespace evmcpp {

// Forward declarations for functions potentially defined in processing.hpp/cpp
cv::Mat rgb2yiq(const cv::Mat& rgb_image);
cv::Mat yiq2rgb(const cv::Mat& yiq_image);
cv::Mat pyrDown(const cv::Mat& image); // Assuming default kernel for now
cv::Mat pyrUp(const cv::Mat& image, const cv::Size& dst_shape); // Assuming default kernel

// Spatially filters a single YIQ frame by downsampling then upsampling 'levels' times.
// Matches the spatial filtering part of Python's generateGaussianPyramid.
cv::Mat spatiallyFilterGaussian(const cv::Mat& yiq_frame, int levels);

// Applies FFT-based temporal filtering and amplification to a batch of spatially filtered YIQ frames.
// Matches Python's filterGaussianPyramids logic.
std::vector<cv::Mat> temporalFilterGaussianBatch(
    const std::vector<cv::Mat>& spatially_filtered_batch, // Batch of blurred YIQ frames
    double fl,
    double fh,
    double samplingRate,
    double alpha,
    double chromAttenuation);

// Reconstructs the final video by adding the filtered/amplified signal back to the original frames.
// Matches Python's getGaussianOutputVideo logic.
std::vector<cv::Mat> reconstructGaussianVideo(
    const std::vector<cv::Mat>& original_rgb_frames,
    const std::vector<cv::Mat>& filtered_amplified_batch); // Result from temporalFilterGaussianBatch

// --- Helper functions (potentially moved from processing.cpp or implemented here) ---
// These might be needed by spatiallyFilterGaussian
// cv::Mat buildGaussianPyramidLowestLevel(const cv::Mat& frame, int levels);
// cv::Mat upsamplePyramidLevel(const cv::Mat& lowest_level, const cv::Size& original_size, int levels);
// --- FFT Filter function (potentially moved from processing.cpp or implemented here) ---
// std::vector<cv::Mat> idealTemporalBandpassFilter(
//     const std::vector<cv::Mat>& images,
//     double fl, double fh, double samplingRate);

} // namespace evmcpp

#endif // EVMCPP_GAUSSIAN_PYRAMID_HPP