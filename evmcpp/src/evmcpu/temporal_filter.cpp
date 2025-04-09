#include "evmcpu/temporal_filter.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>
#include <complex>
#include <stdexcept>

namespace evmcpu {

cv::Mat ideal_temporal_bandpass_filter(
    const cv::Mat& images,
    double fps,
    const cv::Point2d& freq_range,
    int axis)
{
    if (images.empty()) {
        throw std::invalid_argument("Input images batch is empty");
    }
    if (freq_range.x < 0 || freq_range.y < freq_range.x || freq_range.y >= fps/2) {
        throw std::invalid_argument("Invalid frequency range");
    }
    if (axis != 0) {
        throw std::invalid_argument("Currently only time axis (0) is supported");
    }

    // Get dimensions
    const int num_frames = images.size[0];
    const int height = images.size[1];
    const int width = images.size[2];
    const int channels = images.size[3];
    
    // Create output tensor with same shape as input
    cv::Mat filtered(4, images.size, images.type());
    
    // Process each spatial location independently
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            // Extract temporal signals for all channels at this pixel
            std::vector<cv::Mat> channel_signals(channels);
            for (int c = 0; c < channels; c++) {
                channel_signals[c] = cv::Mat(num_frames, 1, CV_32F);
                for (int t = 0; t < num_frames; t++) {
                    const int idx[] = {t, h, w, c};
                    channel_signals[c].at<float>(t) = images.at<float>(idx);
                }
            }

            // Process each channel independently
            for (int c = 0; c < channels; c++) {
                cv::Mat& signal = channel_signals[c];

                // Prepare for DFT (need optimal size)
                int dft_size = cv::getOptimalDFTSize(num_frames);
                cv::Mat padded_signal;
                cv::copyMakeBorder(signal, padded_signal, 
                                 0, dft_size - num_frames,  // top, bottom
                                 0, 0,                      // left, right
                                 cv::BORDER_CONSTANT, 0);

                // Convert to complex (2 channels: real and imaginary)
                cv::Mat complex_signal;
                cv::Mat planes[] = {padded_signal, cv::Mat::zeros(padded_signal.size(), CV_32F)};
                cv::merge(planes, 2, complex_signal);

                // Forward DFT
                cv::dft(complex_signal, complex_signal);

                // Calculate frequency bins
                std::vector<float> freqs(dft_size);
                float df = fps / dft_size;  // frequency resolution
                for (int i = 0; i < dft_size; i++) {
                    freqs[i] = i * df;
                    if (i > dft_size/2) {
                        freqs[i] = (i - dft_size) * df;  // negative frequencies
                    }
                }

                // Apply ideal bandpass filter
                cv::split(complex_signal, planes);  // planes[0] = Re, planes[1] = Im
                for (int i = 0; i < dft_size; i++) {
                    float freq = std::abs(freqs[i]);
                    if (freq < freq_range.x || freq > freq_range.y) {
                        // Zero out frequencies outside the passband
                        planes[0].at<float>(i) = 0;
                        planes[1].at<float>(i) = 0;
                    }
                }
                cv::merge(planes, 2, complex_signal);

                // Inverse DFT
                cv::idft(complex_signal, complex_signal, cv::DFT_SCALE);  // Scale to normalize
                cv::split(complex_signal, planes);

                // Copy real part back to output tensor
                for (int t = 0; t < num_frames; t++) {
                    const int idx[] = {t, h, w, c};
                    filtered.at<float>(idx) = planes[0].at<float>(t);
                }
            }
        }
    }

    return filtered;
}

} // namespace evmcpu
