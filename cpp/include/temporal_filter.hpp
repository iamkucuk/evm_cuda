#ifndef TEMPORAL_FILTER_HPP
#define TEMPORAL_FILTER_HPP

#include <opencv2/core.hpp>

namespace evmcpu {

/**
 * @brief Apply ideal temporal bandpass filter to an image sequence
 * @param images Input image sequence batch (NxHxWxC Matrix, where N is number of frames)
 * @param fps Frame rate of the video
 * @param freq_range Low and high frequency bounds [low_freq, high_freq]
 * @param axis Axis to perform FFT on (default=0, time axis)
 * @return Filtered image sequence
 */
cv::Mat ideal_temporal_bandpass_filter(
    const cv::Mat& images,
    double fps,
    const cv::Point2d& freq_range,
    int axis = 0
);

} // namespace evmcpu

#endif // TEMPORAL_FILTER_HPP