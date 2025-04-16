#include "laplacian_pyramid.hpp"   // Define functions for this file
#include "processing.hpp"          // Includes butterworth, gaussian_pyramid
#include "butterworth.hpp"         // Explicitly include for Laplacian path


#include <opencv2/core.hpp>    // Explicitly include core
#include <opencv2/imgproc.hpp> // For resize, split, merge etc.
#include <vector>
#include <cmath> // For std::sqrt, std::min
#include <stdexcept> // For error handling
#include <iostream> // For logging (replace with a proper logger later)
// Placeholder for logging functions (implement properly later)
#define LOG(message) std::cout << "[LOG] " << message << std::endl
#define LOG_MATRIX(message, matrix) std::cout << "[LOG] " << message << ": Shape=" << matrix.size() << ", Channels=" << matrix.channels() << ", Type=" << matrix.type() << std::endl // Basic matrix info

namespace evmcpp {


    std::vector<cv::Mat> generateLaplacianPyramid(
        const cv::Mat& input_image,
        int level,
        const cv::Mat& kernel)
    {
        LOG("generateLaplacianPyramid: Input image shape=[" + std::to_string(input_image.rows) + ", " + std::to_string(input_image.cols) + "], levels=" + std::to_string(level));

        if (level <= 0) {
            throw std::invalid_argument("Pyramid level must be positive.");
        }
        if (input_image.empty()) {
             throw std::invalid_argument("Input image is empty.");
        }
        // Ensure input type is float for calculations
        cv::Mat float_input;
        if (input_image.type() != CV_32FC3) {
             LOG("generateLaplacianPyramid: Warning - Input image type is not CV_32FC3. Converting.");
             input_image.convertTo(float_input, CV_32FC3); // Assuming 3 channels
        } else {
             float_input = input_image.clone();
        }

        std::vector<cv::Mat> laplacian_pyramid;
        laplacian_pyramid.reserve(level); // Pre-allocate space

        cv::Mat prev_image = float_input; // Use float version

        for (int i = 0; i < level; ++i) {
            LOG_MATRIX("generateLaplacianPyramid: Level " + std::to_string(i) + " prev_image", prev_image);

            // --- Downsampling ---
            cv::Mat downsampled_image;
            cv::pyrDown(prev_image, downsampled_image);
            if (downsampled_image.empty() && !prev_image.empty()) {
                throw std::runtime_error("pyrDown resulted in empty image at level " + std::to_string(i));
            }
            LOG_MATRIX("generateLaplacianPyramid: Level " + std::to_string(i) + " downsampled_image", downsampled_image);

            // --- Upsampling ---
            cv::Mat upsampled_image;
            cv::pyrUp(downsampled_image, upsampled_image, prev_image.size());
            if (upsampled_image.empty() && !downsampled_image.empty()) {
                throw std::runtime_error("pyrUp resulted in empty image at level " + std::to_string(i));
            }
            if (upsampled_image.size() != prev_image.size() || upsampled_image.type() != prev_image.type()) {
                cv::Mat temp;
                upsampled_image.convertTo(temp, prev_image.type());
                if (temp.size() != prev_image.size()) {
                    cv::resize(temp, upsampled_image, prev_image.size());
                } else {
                    upsampled_image = temp;
                }
                LOG("generateLaplacianPyramid: Level " + std::to_string(i) + " upsampled_image resized/retyped.");
            }
            LOG_MATRIX("generateLaplacianPyramid: Level " + std::to_string(i) + " upsampled_image", upsampled_image);

            // --- Calculate Laplacian Level ---
            cv::Mat laplacian_level;
            cv::subtract(prev_image, upsampled_image, laplacian_level);
            LOG_MATRIX("generateLaplacianPyramid: Level " + std::to_string(i) + " laplacian_level", laplacian_level);

            laplacian_pyramid.push_back(laplacian_level);
            prev_image = downsampled_image;

            if (prev_image.rows <= 1 || prev_image.cols <= 1) {
                LOG("generateLaplacianPyramid: Stopping early at level " + std::to_string(i) + " due to small image size.");
                break;
            }
        }

        // Add the final residual Gaussian level
        if (!prev_image.empty()) { // Ensure we have a residual to add
             laplacian_pyramid.push_back(prev_image);
             LOG("generateLaplacianPyramid: Added final residual. New size=" + std::to_string(laplacian_pyramid.size()));
        } else {
             LOG("generateLaplacianPyramid: Warning - Final residual image was empty. Not added.");
        }
        return laplacian_pyramid;
    }


    std::vector<std::vector<cv::Mat>> getLaplacianPyramids(
        const std::vector<cv::Mat>& input_images,
        int level,
        const cv::Mat& kernel)
    {
        LOG("getLaplacianPyramids: Processing " + std::to_string(input_images.size()) + " frames.");
        std::vector<std::vector<cv::Mat>> laplacian_pyramids_batch;
        laplacian_pyramids_batch.reserve(input_images.size());

        int frame_idx = 0;
        for (const auto& image : input_images) {
            if (image.empty()) {
                LOG("getLaplacianPyramids: Warning - Skipping empty input frame " + std::to_string(frame_idx));
                laplacian_pyramids_batch.emplace_back(); // Add empty vector
                frame_idx++;
                continue;
            }

            cv::Mat yiq_image = evmcpp::rgb2yiq(image); // Call actual rgb2yiq
            LOG_MATRIX("getLaplacianPyramids: Frame " + std::to_string(frame_idx) + " yiq_image", yiq_image);

            std::vector<cv::Mat> laplacian_pyramid = generateLaplacianPyramid(yiq_image, level, kernel);
            LOG("getLaplacianPyramids: Frame " + std::to_string(frame_idx) + " generated pyramid size=" + std::to_string(laplacian_pyramid.size()));

            laplacian_pyramids_batch.push_back(std::move(laplacian_pyramid)); // Move pyramid into batch
            frame_idx++;
        }

        LOG("getLaplacianPyramids: Finished. Batch size=" + std::to_string(laplacian_pyramids_batch.size()));
        return laplacian_pyramids_batch;
    }


    std::vector<std::vector<cv::Mat>> filterLaplacianPyramids(
        const std::vector<std::vector<cv::Mat>>& pyramids_batch,
        int level,
        double fps,
        const std::pair<double, double>& freq_range,
        double alpha,
        double lambda_cutoff,
        double attenuation)
    {
        LOG("filterLaplacianPyramids: Starting filtering process.");
        size_t num_frames = pyramids_batch.size();
        if (num_frames <= 1 || level <= 0) { // Need at least 2 frames for IIR filter
            LOG("filterLaplacianPyramids: Not enough frames or invalid level, returning input.");
            return pyramids_batch; // Return original if no frames or levels
        }

        // Define the missing type locally
        using double_vector = std::vector<double>;

        // --- Initialization ---
        // Create filtered_pyramids structure matching input, initialized to zero
        std::vector<std::vector<cv::Mat>> filtered_pyramids;
        filtered_pyramids.reserve(num_frames);
        size_t actual_levels = 0; // Determine actual levels from input
        for(const auto& frame_pyramid : pyramids_batch) {
            filtered_pyramids.emplace_back();
            if (!frame_pyramid.empty()) {
                 if (actual_levels == 0) actual_levels = frame_pyramid.size();
                 else if (actual_levels != frame_pyramid.size()) {
                      // Handle inconsistent pyramid sizes if necessary, for now assume consistent
                      LOG("filterLaplacianPyramids: Warning - Inconsistent number of levels in input pyramids.");
                 }
                 filtered_pyramids.back().reserve(frame_pyramid.size());
                 for(const auto& mat : frame_pyramid) {
                     if (!mat.empty()) {
                         filtered_pyramids.back().push_back(cv::Mat::zeros(mat.size(), mat.type()));
                     } else {
                         filtered_pyramids.back().push_back(cv::Mat());
                     }
                 }
            }
        }
        if (actual_levels == 0) {
             LOG("filterLaplacianPyramids: Input contains no valid pyramids. Returning input.");
             return pyramids_batch;
        }
        // Ensure 'level' parameter matches actual levels found, or adjust
        if (level != static_cast<int>(actual_levels)) {
             LOG("filterLaplacianPyramids: Warning - 'level' parameter (" + std::to_string(level) + ") does not match actual levels found (" + std::to_string(actual_levels) + "). Using actual levels.");
             level = static_cast<int>(actual_levels);
        }


        double delta = lambda_cutoff / (8.0 * (1.0 + alpha));
        double low_freq = freq_range.first;
        double high_freq = freq_range.second;

        // --- Calculate Butterworth Coefficients ---
        std::pair<double_vector, double_vector> butter_low;
        std::pair<double_vector, double_vector> butter_high;
        try {
             butter_low = calculateButterworthCoeffs(1, low_freq, "low", fps);
             butter_high = calculateButterworthCoeffs(1, high_freq, "low", fps);
        } catch (const std::exception& e) {
             throw std::runtime_error(std::string("Error calculating Butterworth coeffs: ") + e.what());
        }

        const auto& b_low = butter_low.first;
        const auto& a_low = butter_low.second;
        const auto& b_high = butter_high.first;
        const auto& a_high = butter_high.second;

        if (b_low.size() != 2 || a_low.size() != 2 || b_high.size() != 2 || a_high.size() != 2) {
             throw std::runtime_error("Butterworth coefficients have unexpected size for order 1 filter.");
        }
        if (std::abs(a_low[0] - 1.0) > 1e-9 || std::abs(a_high[0] - 1.0) > 1e-9) {
             throw std::runtime_error("Expected a[0] coefficient to be 1.0.");
        }


        // --- Temporal Filtering State Initialization ---
        if (pyramids_batch[0].empty() || pyramids_batch[0].size() != actual_levels) {
             throw std::runtime_error("filterLaplacianPyramids: Invalid pyramid structure in first frame.");
        }
        int expected_type = -1;
        for(const auto& mat : pyramids_batch[0]) {
            if (!mat.empty()) {
                if (expected_type == -1) expected_type = mat.type();
                else if (expected_type != mat.type()) throw std::runtime_error("Inconsistent matrix types in first frame pyramid.");
            }
        }
        if (expected_type == -1) throw std::runtime_error("First frame pyramid contains only empty matrices.");


        std::vector<cv::Mat> lowpass_state; lowpass_state.reserve(level);
        std::vector<cv::Mat> highpass_state; highpass_state.reserve(level);
        std::vector<cv::Mat> prev_input; prev_input.reserve(level);

        for(const auto& mat : pyramids_batch[0]) {
            lowpass_state.push_back(mat.clone());
            highpass_state.push_back(mat.clone());
            prev_input.push_back(mat.clone());
        }

        for(size_t lvl=0; lvl < actual_levels; ++lvl) {
             if (!pyramids_batch[0][lvl].empty()) {
                 filtered_pyramids[0][lvl] = pyramids_batch[0][lvl].clone();
             }
        }
        LOG("filterLaplacianPyramids: Frame 0 copied directly.");


        // --- Temporal Filtering Loop ---
        LOG("filterLaplacianPyramids: Starting temporal filtering loop for frames 1 to " + std::to_string(num_frames - 1));
        for (size_t i = 1; i < num_frames; ++i) {
             if (pyramids_batch[i].empty() || pyramids_batch[i].size() != actual_levels) {
                 LOG("filterLaplacianPyramids: Warning - Invalid pyramid structure in frame " + std::to_string(i) + ". Skipping frame filtering, copying previous.");
                 if (i > 0) {
                     for(size_t lvl=0; lvl < actual_levels; ++lvl) {
                          if (lvl < filtered_pyramids[i-1].size() && !filtered_pyramids[i-1][lvl].empty()) { // Use &&
                              if (lvl < filtered_pyramids[i].size()) {
                                   filtered_pyramids[i][lvl] = filtered_pyramids[i-1][lvl].clone();
                              }
                          }
                     }
                 }
                 continue;
             }

            const auto& current_input_pyramid = pyramids_batch[i];
            auto& current_output_pyramid = filtered_pyramids[i];

            for (int lvl = 0; lvl < level; ++lvl) { // Use 'level' (which now matches actual_levels)
                 const cv::Mat& current_input_lvl = current_input_pyramid[lvl];
                 const cv::Mat& prev_input_lvl = prev_input[lvl];
                 cv::Mat& lowpass_state_lvl = lowpass_state[lvl];
                 cv::Mat& highpass_state_lvl = highpass_state[lvl];
                 cv::Mat& current_output_lvl = current_output_pyramid[lvl];

                 if (current_input_lvl.empty() || prev_input_lvl.empty() || lowpass_state_lvl.empty() || highpass_state_lvl.empty()) {
                     LOG("filterLaplacianPyramids: Warning - Empty matrix encountered at frame " + std::to_string(i) + ", level " + std::to_string(lvl) + ". Skipping level filtering.");
                     if (!current_input_lvl.empty()) {
                         current_output_lvl = cv::Mat::zeros(current_input_lvl.size(), current_input_lvl.type());
                     } else {
                          current_output_lvl = cv::Mat();
                     }
                     continue;
                 }
                 if (current_input_lvl.size() != prev_input_lvl.size() || current_input_lvl.type() != prev_input_lvl.type() ||
                     current_input_lvl.size() != lowpass_state_lvl.size() || current_input_lvl.type() != lowpass_state_lvl.type() ||
                     current_input_lvl.size() != highpass_state_lvl.size() || current_input_lvl.type() != highpass_state_lvl.type()) {
                     LOG("filterLaplacianPyramids: Warning - Matrix mismatch at frame " + std::to_string(i) + ", level " + std::to_string(lvl) + ". Skipping level filtering.");
                     current_output_lvl = cv::Mat::zeros(current_input_lvl.size(), current_input_lvl.type());
                     continue;
                 }

                cv::Mat lowpass_output = (b_low[0] * current_input_lvl + b_low[1] * prev_input_lvl - a_low[1] * lowpass_state_lvl);
                cv::Mat highpass_output = (b_high[0] * current_input_lvl + b_high[1] * prev_input_lvl - a_high[1] * highpass_state_lvl);

                cv::Mat bandpass_result;
                cv::subtract(highpass_output, lowpass_output, bandpass_result);

                // --- Spatial Attenuation ---
                if (lvl >= 1 && lvl < (level - 1)) { // Use &&
                    int height = bandpass_result.rows;
                    int width = bandpass_result.cols;
                    if (height > 0 && width > 0) { // Use &&
                        double lambd = std::sqrt(static_cast<double>(height * height + width * width));
                        double new_alpha = (lambd / (8.0 * delta)) - 1.0;
                        double current_alpha = std::min(alpha, new_alpha);
                        bandpass_result *= current_alpha;

                        if (bandpass_result.channels() == 3) {
                            std::vector<cv::Mat> channels;
                            cv::split(bandpass_result, channels);
                            channels[1] *= attenuation;
                            channels[2] *= attenuation;
                            cv::merge(channels, bandpass_result);
                        } else if (bandpass_result.channels() != 1) {
                             LOG("filterLaplacianPyramids: Warning - Expected 1 or 3 channels for spatial attenuation at level " + std::to_string(lvl) + ", but got " + std::to_string(bandpass_result.channels()));
                        }
                    }
                } // End spatial attenuation block

                current_output_lvl = bandpass_result;
                lowpass_state_lvl = lowpass_output;
                highpass_state_lvl = highpass_output;

            } // End level loop

            for(int lvl=0; lvl < level; ++lvl) {
                 if (!current_input_pyramid[lvl].empty()) {
                     prev_input[lvl] = current_input_pyramid[lvl].clone();
                 } else {
                     prev_input[lvl] = cv::Mat();
                 }
            }
        } // End frame loop

        LOG("filterLaplacianPyramids: Finished filtering process.");
        return filtered_pyramids;
    }


    cv::Mat reconstructLaplacianImage(
        const cv::Mat& original_rgb_image,
        const std::vector<cv::Mat>& filtered_pyramid,
        const cv::Mat& kernel)
    {
        if (original_rgb_image.empty()) {
            throw std::invalid_argument("Original RGB image is empty.");
        }
        if (filtered_pyramid.empty()) {
             LOG("reconstructLaplacianImage: Warning - Filtered pyramid is empty. Returning original image.");
             cv::Mat output_img;
             if (original_rgb_image.type() == CV_8UC3) return original_rgb_image.clone();
             original_rgb_image.convertTo(output_img, CV_8UC3);
             return output_img;
        }

        LOG("reconstructLaplacianImage: Starting reconstruction.");

        // 1. Convert original image to YIQ (float)
        cv::Mat reconstructed_yiq = evmcpp::rgb2yiq(original_rgb_image);
        if (reconstructed_yiq.type() != CV_32FC3) {
            throw std::runtime_error("Expected rgb2yiq to return CV_32FC3.");
        }

        // 2. Upsample and add filtered pyramid levels
        size_t num_levels = filtered_pyramid.size();
        if (num_levels < 2) {
            LOG("reconstructLaplacianImage: Warning - Pyramid has less than 2 levels, skipping reconstruction additions. Returning base YIQ converted to RGB.");
            cv::Mat reconstructed_rgb_float = evmcpp::yiq2rgb(reconstructed_yiq);
            cv::Mat reconstructed_rgb_uint8;
            reconstructed_rgb_float.convertTo(reconstructed_rgb_uint8, CV_8UC3);
            return reconstructed_rgb_uint8;
        } else {
            for (size_t level = 1; level < num_levels; ++level) {
                cv::Mat current_level_filtered = filtered_pyramid[level];
                if (current_level_filtered.empty()) {
                    LOG("reconstructLaplacianImage: Warning - Skipping empty filtered level " + std::to_string(level));
                    continue;
                }
                if (current_level_filtered.type() != CV_32FC3) {
                    cv::Mat temp;
                    if (current_level_filtered.channels() == 3 && current_level_filtered.depth() != CV_32F) {
                        LOG("reconstructLaplacianImage: Warning - Converting filtered level " + std::to_string(level) + " to CV_32FC3.");
                        current_level_filtered.convertTo(temp, CV_32FC3);
                        current_level_filtered = temp;
                    } else {
                        throw std::runtime_error("Filtered pyramid level " + std::to_string(level) + " has unexpected type/channels.");
                    }
                }

                cv::Mat upsampled_level = current_level_filtered.clone();
                for (size_t up_iter = 0; up_iter < level; ++up_iter) {
                    size_t target_level_idx = level - up_iter - 1;
                    if (target_level_idx >= filtered_pyramid.size()) {
                        throw std::runtime_error("Logic error: Index out of bounds during upsampling target size calculation.");
                    }
                    if (target_level_idx >= filtered_pyramid.size() || filtered_pyramid[target_level_idx].empty()) {
                        LOG("reconstructLaplacianImage: Warning - Target level " + std::to_string(target_level_idx) + " is invalid or empty. Skipping upsampling for level " + std::to_string(level) + ".");
                        upsampled_level = cv::Mat();
                        break;
                    }
                    cv::Size target_size = filtered_pyramid[target_level_idx].size();
                    if (target_size.width <= 0 || target_size.height <= 0) {
                        LOG("reconstructLaplacianImage: Warning - Target size for pyrUp is invalid at level " + std::to_string(level) + ", up_iter " + std::to_string(up_iter) + ". Skipping further upsampling for this level.");
                        upsampled_level = cv::Mat();
                        break;
                    }
                    // Use the custom evmcpp::pyrUp to match insert+convolve logic
                    upsampled_level = evmcpp::pyrUp(upsampled_level, kernel, target_size); // Use correct variable name 'kernel'
                    if (upsampled_level.empty()) {
                        LOG("reconstructLaplacianImage: Warning - pyrUp resulted in empty image at level " + std::to_string(level) + ", up_iter " + std::to_string(up_iter) + ". Skipping further upsampling for this level.");
                        break;
                    }
                }

                if (!upsampled_level.empty()) {
                    if (upsampled_level.size() != reconstructed_yiq.size() || upsampled_level.type() != reconstructed_yiq.type()) {
                        LOG("reconstructLaplacianImage: Warning - Size/type mismatch before adding level " + std::to_string(level) + ". Attempting conversion/resize.");
                        cv::Mat temp;
                        if (upsampled_level.type() != reconstructed_yiq.type()) {
                            upsampled_level.convertTo(temp, reconstructed_yiq.type());
                        } else {
                            temp = upsampled_level;
                        }
                        if (temp.size() != reconstructed_yiq.size()) {
                            cv::resize(temp, upsampled_level, reconstructed_yiq.size());
                        } else {
                            upsampled_level = temp;
                        }
                    }
                    cv::add(reconstructed_yiq, upsampled_level, reconstructed_yiq);
                }
            }
        }

        cv::Mat reconstructed_rgb_float = evmcpp::yiq2rgb(reconstructed_yiq);
        cv::Mat reconstructed_rgb_uint8;
        reconstructed_rgb_float.convertTo(reconstructed_rgb_uint8, CV_8UC3);
        LOG("reconstructLaplacianImage: Reconstruction complete.");
        return reconstructed_rgb_uint8;
    }

} // namespace evmcpp
