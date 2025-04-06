#include "evmcpp/gaussian_pyramid.hpp"
#include "evmcpp/processing.hpp" // For rgb2yiq, yiq2rgb, pyrDown, pyrUp

#include <opencv2/imgproc.hpp> // For pyrDown, pyrUp, resize
#include <opencv2/core.hpp>
// #include <opencv2/dft.hpp> // dft/idft should be included via core.hpp when linking core component
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <limits> // For numeric_limits

namespace evmcpp {

// Helper to build the Gaussian pyramid down to the specified level
// Returns only the lowest level.
// Adapted from the previous implementation.
cv::Mat buildGaussianPyramidLowestLevel(const cv::Mat& frame, int levels) {
    cv::Mat current_level;
    if (frame.type() != CV_32FC3) {
         frame.convertTo(current_level, CV_32FC3); // Ensure float
    } else {
        current_level = frame.clone();
    }

    if (levels < 0) {
        throw std::invalid_argument("Number of pyramid levels cannot be negative.");
    }
    if (levels == 0) {
        return current_level; // No downsampling needed
    }

    for (int i = 0; i < levels; ++i) {
        cv::Mat next_level;
        // Use the pyrDown function potentially defined in processing.cpp
        // If not defined there, use cv::pyrDown directly.
        // Assuming evmcpp::pyrDown exists and matches cv::pyrDown behavior for now.
        next_level = evmcpp::pyrDown(current_level);
        // cv::pyrDown(current_level, next_level); // Alternative if evmcpp::pyrDown isn't defined

        if (next_level.empty() || next_level.rows < 1 || next_level.cols < 1) {
            std::cerr << "Warning: Image became too small during pyrDown at level " << i + 1 << ". Using previous level." << std::endl;
            break; // Stop if image gets too small
        }
        current_level = next_level;
    }
    return current_level;
}

// Helper function to upsample a pyramid level back towards an original size.
// Needs the target size for each step. Simulates the Python pyrUp logic more closely.
// Adapted from the previous implementation.
cv::Mat upsamplePyramidLevel(const cv::Mat& level_to_upsample, const cv::Size& original_size, int total_levels, int current_level_depth) {
     if (level_to_upsample.empty()) {
        throw std::runtime_error("Input level_to_upsample for upsampling is empty.");
    }
    if (total_levels < 0) {
         throw std::runtime_error("Total number of levels cannot be negative.");
    }
    if (current_level_depth > total_levels || current_level_depth < 0) {
         throw std::runtime_error("Current level depth is invalid.");
    }
    if (current_level_depth == 0) {
        // Already at original size (or should be), resize just in case
        cv::Mat result;
        if (level_to_upsample.size() != original_size) {
             cv::resize(level_to_upsample, result, original_size, 0, 0, cv::INTER_LINEAR);
        } else {
            result = level_to_upsample.clone();
        }
        return result;
    }

    cv::Mat current_level = level_to_upsample.clone();
    cv::Mat temp_size_ref = cv::Mat::zeros(original_size, level_to_upsample.type()); // Reference for size calculation

    // Upsample step by step
    for (int i = 0; i < current_level_depth; ++i) {
        cv::Mat temp_up;
        // Calculate the target size for this specific pyrUp step
        cv::Mat target_size_ref = temp_size_ref;
        int downs_needed = current_level_depth - 1 - i; // How many downs from original to get target size
         for(int j = 0; j < downs_needed; ++j) {
            cv::Mat next_down;
            // Use evmcpp::pyrDown or cv::pyrDown
            next_down = evmcpp::pyrDown(target_size_ref);
            // cv::pyrDown(target_size_ref, next_down);
             if (next_down.empty() || next_down.rows < 1 || next_down.cols < 1) {
                 throw std::runtime_error("Intermediate downsampling failed while calculating target size for pyrUp.");
             }
            target_size_ref = next_down;
        }
        cv::Size target_size = target_size_ref.size();

        // Perform the upsampling using evmcpp::pyrUp or cv::pyrUp
        // Assuming evmcpp::pyrUp exists and matches Python logic (handles dst_shape)
        temp_up = evmcpp::pyrUp(current_level, target_size);
        // cv::pyrUp(current_level, temp_up, target_size); // Alternative

        // OpenCV's pyrUp might not perfectly match the target size, resize if needed
        if (temp_up.size() != target_size) {
            cv::resize(temp_up, temp_up, target_size, 0, 0, cv::INTER_LINEAR);
        }
        current_level = temp_up;
    }

     // Final resize to ensure it matches original frame size exactly
     if (current_level.size() != original_size) {
        cv::resize(current_level, current_level, original_size, 0, 0, cv::INTER_LINEAR);
     }

    return current_level;
}


// Spatially filters a single YIQ frame by downsampling then upsampling 'levels' times.
cv::Mat spatiallyFilterGaussian(const cv::Mat& yiq_frame, int levels) {
    if (yiq_frame.empty()) {
        throw std::invalid_argument("Input YIQ frame for spatial filtering is empty.");
    }
    if (yiq_frame.type() != CV_32FC3) {
         throw std::invalid_argument("Input YIQ frame must be CV_32FC3 for spatial filtering.");
    }
    if (levels < 0) {
        throw std::invalid_argument("Number of levels cannot be negative.");
    }
    if (levels == 0) {
        return yiq_frame.clone(); // No filtering needed
    }

    // 1. Downsample 'levels' times
    cv::Mat lowest_level = buildGaussianPyramidLowestLevel(yiq_frame, levels);

    // 2. Upsample back 'levels' times to original size
    cv::Mat reconstructed = upsamplePyramidLevel(lowest_level, yiq_frame.size(), levels, levels);

    return reconstructed;
}


// Helper function for FFT-based temporal bandpass filter (adapted from previous implementation)
std::vector<cv::Mat> idealTemporalBandpassFilter(
    const std::vector<cv::Mat>& images,
    double fl, double fh, double samplingRate)
{
    if (images.empty()) {
        throw std::runtime_error("Input sequence for FFT filtering is empty.");
    }

    int num_frames = static_cast<int>(images.size());
    if (num_frames <= 1) {
         std::cerr << "Warning: FFT filtering requires more than one frame. Returning input sequence." << std::endl;
         return images; // Cannot filter with <= 1 frame
    }

    cv::Size frame_size = images[0].size();
    int frame_type = images[0].type();
    int channels = images[0].channels();

    // Validate input consistency
    for (const auto& frame : images) {
        if (frame.size() != frame_size || frame.type() != frame_type) {
            throw std::runtime_error("Inconsistent frame size or type in input sequence for FFT filtering.");
        }
        if (frame.depth() != CV_32F) {
             throw std::runtime_error("FFT filtering currently requires CV_32F input frames.");
        }
         if (channels != 3) {
             throw std::runtime_error("FFT filtering currently requires 3-channel (YIQ) input frames.");
        }
    }

    // Prepare output sequence
    std::vector<cv::Mat> filtered_sequence(num_frames);
    for (int i = 0; i < num_frames; ++i) {
        filtered_sequence[i] = cv::Mat::zeros(frame_size, frame_type);
    }

    // Process each pixel location independently
    for (int r = 0; r < frame_size.height; ++r) {
        for (int c = 0; c < frame_size.width; ++c) {
            for (int ch = 0; ch < channels; ++ch) {
                // 1. Extract time series for this pixel/channel
                cv::Mat pixel_timeseries_mat(num_frames, 1, CV_32F);
                for (int t = 0; t < num_frames; ++t) {
                    pixel_timeseries_mat.at<float>(t, 0) = images[t].at<cv::Vec3f>(r, c)[ch];
                }

                // 2. Perform forward DFT
                cv::Mat dft_result;
                cv::dft(pixel_timeseries_mat, dft_result, cv::DFT_COMPLEX_OUTPUT);

                // 3. Create and apply frequency mask (mimicking numpy.fft.fftfreq and slicing)
                int N = num_frames;
                std::vector<double> frequencies(N);
                // Calculate frequencies corresponding to DFT output bins
                for(int i = 0; i < N; ++i) {
                    frequencies[i] = static_cast<double>(i) / N * samplingRate;
                }

                // Find indices corresponding to frequency range [fl, fh]
                // Note: Python uses closest frequency, this uses a simple range check.
                // Adjust if exact Python behavior is critical.
                std::vector<bool> mask(N, false);
                int low_idx = -1, high_idx = -1;

                for(int i = 0; i < N; ++i) {
                    double freq = frequencies[i];
                    // Handle Nyquist frequency and negative frequencies implicitly represented
                    double effective_freq = (i <= N / 2) ? freq : freq - samplingRate;
                    if (std::abs(effective_freq) >= fl && std::abs(effective_freq) <= fh) {
                         mask[i] = true;
                         if (low_idx == -1) low_idx = i; // First index in band
                         high_idx = i; // Last index in band
                    }
                }


                // Zero out frequencies outside the bandpass mask
                // dft_result has N rows, 1 col, 2 channels (complex: CV_32FC2)
                for (int k = 0; k < N; ++k) {
                    if (!mask[k]) {
                        dft_result.at<cv::Vec2f>(k, 0) = cv::Vec2f(0.0f, 0.0f);
                    }
                }


                // 4. Perform inverse DFT
                cv::Mat idft_result;
                // Use DFT_REAL_OUTPUT (output is real) and DFT_SCALE (scale by 1/N) to match numpy.fft.ifft
                cv::idft(dft_result, idft_result, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

                // 5. Store the temporally filtered time series for this pixel/channel
                for (int t = 0; t < num_frames; ++t) {
                     filtered_sequence[t].at<cv::Vec3f>(r, c)[ch] = idft_result.at<float>(t, 0);
                }
            } // end channel loop
        } // end col loop
    } // end row loop

    return filtered_sequence;
}


// Applies FFT-based temporal filtering and amplification to a batch of spatially filtered YIQ frames.
std::vector<cv::Mat> temporalFilterGaussianBatch(
    const std::vector<cv::Mat>& spatially_filtered_batch,
    double fl,
    double fh,
    double samplingRate,
    double alpha,
    double chromAttenuation)
{
    if (spatially_filtered_batch.empty()) {
        throw std::invalid_argument("Input batch for temporal filtering is empty.");
    }

    // 1. Apply ideal temporal bandpass filter
    std::vector<cv::Mat> filtered_batch = idealTemporalBandpassFilter(spatially_filtered_batch, fl, fh, samplingRate);

    // 2. Apply amplification and chrominance attenuation
    std::vector<cv::Mat> amplified_batch = filtered_batch; // Work on a copy or modify in place if desired

    for (cv::Mat& frame : amplified_batch) {
        if (frame.empty()) continue; // Should not happen if filtering worked

        std::vector<cv::Mat> channels;
        cv::split(frame, channels);

        if (channels.size() == 3) {
            channels[0] *= alpha; // Y channel
            channels[1] *= alpha * chromAttenuation; // I channel
            channels[2] *= alpha * chromAttenuation; // Q channel
            cv::merge(channels, frame);
        } else {
             // Handle unexpected channel count if necessary
             std::cerr << "Warning: Unexpected channel count (" << channels.size() << ") during amplification." << std::endl;
             frame *= alpha; // Apply alpha uniformly as a fallback
        }
    }

    return amplified_batch;
}


// Reconstructs the final video by adding the filtered/amplified signal back to the original frames.
std::vector<cv::Mat> reconstructGaussianVideo(
    const std::vector<cv::Mat>& original_rgb_frames,
    const std::vector<cv::Mat>& filtered_amplified_batch)
{
    if (original_rgb_frames.size() != filtered_amplified_batch.size()) {
        throw std::invalid_argument("Original frame count and filtered batch count must match for reconstruction.");
    }
    if (original_rgb_frames.empty()) {
        return {}; // Return empty vector if input is empty
    }

    size_t num_frames = original_rgb_frames.size();
    std::vector<cv::Mat> output_video(num_frames);

    for (size_t i = 0; i < num_frames; ++i) {
        const cv::Mat& original_rgb = original_rgb_frames[i];
        const cv::Mat& filtered_yiq = filtered_amplified_batch[i];

        if (original_rgb.empty() || filtered_yiq.empty()) {
            std::cerr << "Warning: Empty frame encountered during reconstruction at index " << i << ". Skipping." << std::endl;
            // Create an empty placeholder or handle as appropriate
            output_video[i] = cv::Mat(); // Or maybe a black frame?
            continue;
        }
         if (original_rgb.size() != filtered_yiq.size()) {
             std::cerr << "Warning: Size mismatch between original frame and filtered signal at index " << i << ". Skipping reconstruction for this frame." << std::endl;
             output_video[i] = original_rgb.clone(); // Or handle differently
             continue;
         }
         if (filtered_yiq.type() != CV_32FC3) {
              std::cerr << "Warning: Filtered signal is not CV_32FC3 at index " << i << ". Skipping reconstruction." << std::endl;
              output_video[i] = original_rgb.clone();
              continue;
         }


        // 1. Convert original RGB to YIQ (float)
        cv::Mat original_yiq = rgb2yiq(original_rgb); // Assumes rgb2yiq handles conversion to float

        // 2. Add filtered signal
        cv::Mat combined_yiq = original_yiq + filtered_yiq;

        // 3. Convert combined YIQ back to RGB (float)
        cv::Mat reconstructed_rgb_float = yiq2rgb(combined_yiq);

        // 4. Clip values to [0, 255]
        cv::Mat clipped_rgb_float;
        // Use cv::max/min correctly for element-wise operation
        cv::max(reconstructed_rgb_float, cv::Scalar(0.0, 0.0, 0.0), reconstructed_rgb_float); // Lower bound
        cv::min(reconstructed_rgb_float, cv::Scalar(255.0, 255.0, 255.0), clipped_rgb_float);   // Upper bound


        // 5. Convert to uint8
        cv::Mat final_rgb_uint8;
        clipped_rgb_float.convertTo(final_rgb_uint8, CV_8UC3);

        output_video[i] = final_rgb_uint8;
    }

    return output_video;
}


} // namespace evmcpp