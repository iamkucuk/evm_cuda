// File: include/evmcpp/gaussian_pyramid.hpp
// Purpose: Declares functions for the Gaussian EVM pathway.

#ifndef EVMCPPGUASSIANPYRAMID_HPP
#define EVMCPPGUASSIANPYRAMID_HPP

#include <opencv2/core.hpp> // Include necessary OpenCV headers
#include <vector>
#include <string>

namespace evmcpp {

// Forward declaration from processing.hpp if needed, or include processing.hpp
// #include "processing.hpp" // Consider including if types/constants are needed directly

/**
 * @brief Applies spatial filtering (Gaussian pyramid down/up) to a single frame.
 * Mimics the effect of Python's generateGaussianPyramid by applying pyrDown then pyrUp.
 * @param inputRgb Input RGB frame (CV_8UC3).
 * @param level Number of pyramid levels for down/up sampling (controls blur amount).
 * @param kernel Gaussian kernel (e.g., 5x5 CV_32F) used by custom pyrDown/pyrUp.
 * @return Spatially filtered YIQ frame (CV_32FC3). Returns empty Mat on error.
 * TDD_ANCHOR: test_spatiallyFilterGaussian_matches_python
 */
cv::Mat spatiallyFilterGaussian(const cv::Mat& inputRgb, int level, const cv::Mat& kernel);


/**
 * @brief Applies temporal bandpass filtering (FFT) and amplification to a batch of spatially filtered frames.
 * Implements the core logic of Python's filterGaussianPyramids and idealTemporalBandpassFilter.
 * @param spatiallyFilteredBatch Vector of spatially filtered YIQ frames (CV_32FC3). Input data might be modified in place by FFT functions if DFT_INPLACE is used.
 * @param fps Frames per second of the video.
 * @param fl Low cutoff frequency (Hz).
 * @param fh High cutoff frequency (Hz).
 * @param alpha Amplification factor.
 * @param chromAttenuation Attenuation factor for chrominance channels (IQ).
 * @return Vector of amplified, temporally filtered YIQ frames (CV_32FC3). Returns empty vector on error.
 * TDD_ANCHOR: test_temporalFilterGaussianBatch_matches_python
 */
std::vector<cv::Mat> temporalFilterGaussianBatch(
    const std::vector<cv::Mat>& spatiallyFilteredBatch, // Pass by const&, copy inside if needed for DFT
    float fps,
    float fl,
    float fh,
    float alpha,
    float chromAttenuation
);


/**
 * @brief Reconstructs the final output frame by adding the filtered signal to the original.
 * Implements the logic of Python's reconstructGaussianImage.
 * @param originalRgb Original input RGB frame (CV_8UC3).
 * @param filteredYiqSignal The corresponding amplified, temporally filtered YIQ signal (CV_32FC3).
 * @return Reconstructed RGB frame (CV_8UC3), clipped to [0, 255]. Returns empty Mat on error.
 * TDD_ANCHOR: test_reconstructGaussianFrame_matches_python
 */
cv::Mat reconstructGaussianFrame(
    const cv::Mat& originalRgb,
    const cv::Mat& filteredYiqSignal
);

} // namespace evmcpp

#endif // EVMCPPGUASSIANPYRAMID_HPP