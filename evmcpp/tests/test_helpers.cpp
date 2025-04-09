#include "test_helpers.hpp"
#include <opencv2/imgproc.hpp> // For absdiff
#include <opencv2/highgui.hpp> // Potentially for debugging, but not strictly needed for helpers
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include <stdexcept>
#include <iostream> // For error messages in helpers

// Definition of loadMatrixFromTxt moved to test_helpers.hpp as it's a template function.

// Function to compare two float matrices element-wise with tolerance
::testing::AssertionResult CompareMatrices(const cv::Mat& mat1, const cv::Mat& mat2, double tolerance) {
    if (mat1.size() != mat2.size()) {
        return ::testing::AssertionFailure() << "Matrix dimensions mismatch: "
               << mat1.rows << "x" << mat1.cols << " vs " << mat2.rows << "x" << mat2.cols;
    }
    if (mat1.type() != mat2.type()) {
        return ::testing::AssertionFailure() << "Matrix types mismatch: "
               << mat1.type() << " vs " << mat2.type();
    }

    double overall_max_diff = -1.0;
    cv::Point overall_max_loc(-1, -1);
    int overall_max_channel = -1;

    std::vector<cv::Mat> channels1, channels2;
    cv::split(mat1, channels1);
    cv::split(mat2, channels2);

    for (int i = 0; i < mat1.channels(); ++i) {
        cv::Mat channel_diff;
        cv::absdiff(channels1[i], channels2[i], channel_diff);

        double channel_min_diff, channel_max_diff;
        cv::Point channel_min_loc, channel_max_loc;
        cv::minMaxLoc(channel_diff, &channel_min_diff, &channel_max_diff, &channel_min_loc, &channel_max_loc);

        if (channel_max_diff > overall_max_diff) {
            overall_max_diff = channel_max_diff;
            overall_max_loc = channel_max_loc;
            overall_max_channel = i;
        }
    }

    if (overall_max_diff > tolerance) {
        // Get original values at the location of max difference
        std::stringstream ss_val1, ss_val2;
        int mat_type = mat1.type(); // Use the original matrix type

        // Use overall_max_loc and overall_max_channel found consistently
        if (mat_type == CV_32FC3 || mat_type == CV_32FC1) {
             ss_val1 << channels1[overall_max_channel].at<float>(overall_max_loc);
             ss_val2 << channels2[overall_max_channel].at<float>(overall_max_loc);
        } else if (mat_type == CV_64FC3 || mat_type == CV_64FC1) {
             ss_val1 << channels1[overall_max_channel].at<double>(overall_max_loc);
             ss_val2 << channels2[overall_max_channel].at<double>(overall_max_loc);
        } else if (mat_type == CV_8UC3 || mat_type == CV_8UC1) {
             ss_val1 << static_cast<int>(channels1[overall_max_channel].at<uchar>(overall_max_loc));
             ss_val2 << static_cast<int>(channels2[overall_max_channel].at<uchar>(overall_max_loc));
        } else if (mat_type == CV_16UC3 || mat_type == CV_16UC1) {
             ss_val1 << channels1[overall_max_channel].at<ushort>(overall_max_loc);
             ss_val2 << channels2[overall_max_channel].at<ushort>(overall_max_loc);
        } else if (mat_type == CV_16SC3 || mat_type == CV_16SC1) {
             ss_val1 << channels1[overall_max_channel].at<short>(overall_max_loc);
             ss_val2 << channels2[overall_max_channel].at<short>(overall_max_loc);
        } else if (mat_type == CV_32SC3 || mat_type == CV_32SC1) {
             ss_val1 << channels1[overall_max_channel].at<int>(overall_max_loc);
             ss_val2 << channels2[overall_max_channel].at<int>(overall_max_loc);
        } else {
            ss_val1 << "[Unsupported Type]";
            ss_val2 << "[Unsupported Type]";
        }

        return ::testing::AssertionFailure() << "Matrices differ by more than tolerance (" << tolerance
               << "). Max difference: " << overall_max_diff << " at (" << overall_max_loc.y << ", " << overall_max_loc.x << ")"
               << (mat1.channels() > 1 ? " in channel " + std::to_string(overall_max_channel) : "")
               << ". Values at location: " << ss_val1.str() << " vs " << ss_val2.str();
    }

    return ::testing::AssertionSuccess();
}


// Function to apply FFT-based temporal bandpass filter and amplify (implementation in test_helpers.cpp)
// Mimics the Python filterGaussianPyramids output saved as steps 4, 5, and 6b reference data.
std::vector<cv::Mat> applyFftTemporalFilterAndAmplify(
    const std::vector<cv::Mat>& spatial_filtered_sequence, // Input for filtering AND adding back
    double fl,
    double fh,
    double samplingRate,
    double alpha,
    double chromAttenuation)
{
    if (spatial_filtered_sequence.empty()) {
        throw std::runtime_error("Input sequence for FFT filtering is empty.");
    }

    int num_frames = static_cast<int>(spatial_filtered_sequence.size());
    if (num_frames <= 1) {
         std::cerr << "Warning: FFT filtering requires more than one frame. Returning input sequence." << std::endl;
         return spatial_filtered_sequence; // Cannot filter with <= 1 frame
    }

    cv::Size frame_size = spatial_filtered_sequence[0].size();
    int frame_type = spatial_filtered_sequence[0].type();
    int channels = spatial_filtered_sequence[0].channels();

    // Validate input consistency
    for (const auto& frame : spatial_filtered_sequence) {
        if (frame.size() != frame_size || frame.type() != frame_type) {
            throw std::runtime_error("Inconsistent frame size or type in input sequence for FFT filtering.");
        }
        if (frame.depth() != CV_32F) {
             throw std::runtime_error("FFT filtering currently requires CV_32F input frames.");
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
                    if (channels == 1) {
                        pixel_timeseries_mat.at<float>(t, 0) = spatial_filtered_sequence[t].at<float>(r, c);
                    } else { // Assuming 3 channels (CV_32FC3)
                        pixel_timeseries_mat.at<float>(t, 0) = spatial_filtered_sequence[t].at<cv::Vec3f>(r, c)[ch];
                    }
                }

                // 2. Perform forward DFT
                cv::Mat dft_result;
                cv::dft(pixel_timeseries_mat, dft_result, cv::DFT_COMPLEX_OUTPUT);

                // 3. Create and apply frequency mask
                // Calculate frequency step and indices corresponding to fl and fh
                // OpenCV DFT output for N real inputs has N/2 + 1 complex values.
                // Frequencies are k * samplingRate / N for k = 0, ..., N/2.
                int N = num_frames;
                int dft_size = N / 2 + 1; // Number of complex values in dft_result (rows)

                // Calculate indices based on Python's logic (closest index)
                // double freq_step = samplingRate / N; // Not directly needed if using fftfreq logic
                std::vector<double> frequencies(N);
                for(int i = 0; i < N; ++i) {
                    frequencies[i] = static_cast<double>(i < (N + 1) / 2 ? i : i - N) / N * samplingRate;
                }

                int low_idx_fft = 0;
                double min_diff_low = std::numeric_limits<double>::max();
                for(int i = 0; i < N; ++i) {
                    double diff = std::abs(frequencies[i] - fl);
                    if (diff < min_diff_low) {
                        min_diff_low = diff;
                        low_idx_fft = i;
                    }
                }

                int high_idx_fft = 0;
                double min_diff_high = std::numeric_limits<double>::max();
                 for(int i = 0; i < N; ++i) {
                    double diff = std::abs(frequencies[i] - fh);
                    if (diff < min_diff_high) {
                        min_diff_high = diff;
                        high_idx_fft = i;
                    }
                }

                // Zero out frequencies outside the band [low_idx_fft, high_idx_fft)
                // This mimics the Python slicing fft[:low] = 0, fft[high:] = 0
                // Directly zero out rows in dft_result outside the band [low_idx_fft, high_idx_fft)
                // This mimics the Python slicing fft[:low] = 0, fft[high:] = 0
                // Note: OpenCV's DFT output for real input is CCS packed.
                // dft_result has N rows (num_frames) and 1 col, with 2 channels (complex).
                // We need to zero out based on the *frequency index* k, which corresponds to the row index.

                if (low_idx_fft <= high_idx_fft) { // Normal case
                    // Zero out elements before low_idx_fft
                    for (int k = 0; k < low_idx_fft; ++k) {
                        // Check bounds: k must be < N
                        if (k < N) {
                           dft_result.at<cv::Vec2f>(k, 0) = cv::Vec2f(0.0f, 0.0f);
                        }
                    }
                    // Zero out elements at or after high_idx_fft
                    for (int k = high_idx_fft; k < N; ++k) {
                        // Check bounds: k must be < N
                        if (k < N) {
                           dft_result.at<cv::Vec2f>(k, 0) = cv::Vec2f(0.0f, 0.0f);
                        }
                    }
                } else { // Wrap-around case (e.g., high pass filter near Nyquist)
                    // Zero out elements *between* high_idx_fft (exclusive) and low_idx_fft (inclusive)
                    for (int k = high_idx_fft; k < low_idx_fft; ++k) {
                         // Check bounds: k must be < N
                         if (k < N) {
                            dft_result.at<cv::Vec2f>(k, 0) = cv::Vec2f(0.0f, 0.0f);
                         }
                    }
                }
                // 4. Perform inverse DFT
                cv::Mat idft_result;
                // Note: idft_result is the temporally filtered signal (equivalent to Step 4 output *before* amplification/combination)
                cv::idft(dft_result, idft_result, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

                // 5. Store the *temporally filtered* time series (before amplification/combination)
                // We need this intermediate result to perform amplification and combination per frame later.
                for (int t = 0; t < num_frames; ++t) {
                     if (channels == 1) {
                        // Store intermediate filtered result in the 'filtered_sequence' vector for now
                        filtered_sequence[t].at<float>(r, c) = idft_result.at<float>(t, 0);
                    } else {
                        filtered_sequence[t].at<cv::Vec3f>(r, c)[ch] = idft_result.at<float>(t, 0);
                    }
                }
            } // end channel loop
        } // end col loop
    } // end row loop

    // --- Post-FFT Processing (Amplification) ---
    // Now iterate through the frames of the filtered sequence
    std::vector<cv::Mat> amplified_sequence(num_frames);
    for (int t = 0; t < num_frames; ++t) {
        // 6. Amplification (Step 5 equivalent)
        cv::Mat amplified_frame = filtered_sequence[t].clone(); // Start with the temporally filtered result
        std::vector<cv::Mat> channels_amp;
        cv::split(amplified_frame, channels_amp);

        channels_amp[0] *= alpha;                     // Y channel
        channels_amp[1] *= alpha * chromAttenuation;  // I channel
        channels_amp[2] *= alpha * chromAttenuation;  // Q channel

        cv::merge(channels_amp, amplified_frame);

        // Store the amplified frame
        amplified_sequence[t] = amplified_frame;
    }

    // Return the amplified sequence (equivalent to Step 5 output)
    return amplified_sequence;
}

// upsamplePyramidLevel function moved to src/evmcpp/gaussian_pyramid.cpp
