#include "processing.hpp"
#include "gaussian_pyramid.hpp" // For batch processing functions
#include "laplacian_pyramid.hpp" // Keep if needed for laplacian path

#include <opencv2/imgproc.hpp> // For color conversion, pyrDown, pyrUp, resize
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp> // For VideoCapture, VideoWriter
#include <stdexcept> // For error handling
#include <iostream> // For logging/debugging
#include <vector>
#include <string>
#include <chrono> // For timing (optional)

// Placeholder for logging (can be removed or replaced later)
#define LOG_PROC(message) std::cout << "[PROC LOG] " << message << std::endl

namespace evmcpp {

    // RGB to YIQ conversion matrix (float precision)
    const cv::Matx33f RGB2YIQ_MATRIX = {
        0.299f,       0.587f,       0.114f,
        0.59590059f, -0.27455667f, -0.32134392f,
        0.21153661f, -0.52273617f,  0.31119955f
    };

    // YIQ to RGB matrix (float precision)
    const cv::Matx33f YIQ2RGB_MATRIX = RGB2YIQ_MATRIX.inv();


    cv::Mat rgb2yiq(const cv::Mat& rgb_image) {
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


    cv::Mat yiq2rgb(const cv::Mat& yiq_image) {
        if (yiq_image.empty()) {
            throw std::invalid_argument("Input YIQ image is empty.");
        }
        if (yiq_image.channels() != 3) {
            throw std::invalid_argument("Input image must have 3 channels (YIQ).");
        }
        if (yiq_image.type() != CV_32FC3) {
             throw std::invalid_argument("Input YIQ image must be type CV_32FC3 for yiq2rgb.");
        }

        cv::Mat rgb_image;
        cv::transform(yiq_image, rgb_image, YIQ2RGB_MATRIX);
        return rgb_image; // Still CV_32FC3
    }


    // --- Custom Pyramid Functions (Mirroring Python's filter2D approach) ---

    // Version with explicit kernel
    cv::Mat pyrDown(const cv::Mat& image, const cv::Mat& kernel) {
        if (image.empty()) {
            throw std::invalid_argument("pyrDown: Input image is empty.");
        }
        if (kernel.empty()) {
            throw std::invalid_argument("pyrDown: Input kernel is empty.");
        }
        // Ensure input is float32
        cv::Mat float_image;
        if (image.type() != CV_32FC3) {
             image.convertTo(float_image, CV_32FC3);
        } else {
             float_image = image; // Avoid clone if already correct type
        }

        // 1. Filter using cv::filter2D
        cv::Mat filtered_image;
        cv::filter2D(float_image, filtered_image, -1, kernel, cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101);

        // 2. Downsample by taking every second pixel
        int out_rows = filtered_image.rows / 2;
        int out_cols = filtered_image.cols / 2;
        if (out_rows <= 0 || out_cols <= 0) {
             return cv::Mat(); // Return empty matrix if too small
        }

        cv::Mat downsampled_image(out_rows, out_cols, filtered_image.type());

        // Optimized subsampling using resize (INTER_NEAREST is equivalent to taking every other pixel)
        cv::resize(filtered_image, downsampled_image, downsampled_image.size(), 0, 0, cv::INTER_NEAREST);

        // Manual subsampling loop (alternative, less efficient):
        // for (int r = 0; r < out_rows; ++r) {
        //     for (int c = 0; c < out_cols; ++c) {
        //         downsampled_image.at<cv::Vec3f>(r, c) = filtered_image.at<cv::Vec3f>(r * 2, c * 2);
        //     }
        // }

        return downsampled_image;
    }

    // Overloaded version using default Gaussian kernel
    cv::Mat pyrDown(const cv::Mat& image) {
        return pyrDown(image, gaussian_kernel);
    }

    // Version with explicit kernel
    cv::Mat pyrUp(const cv::Mat& image, const cv::Mat& kernel, const cv::Size& dst_shape) {
         if (image.empty()) {
            throw std::invalid_argument("pyrUp: Input image is empty.");
        }
        if (kernel.empty()) {
            throw std::invalid_argument("pyrUp: Input kernel is empty.");
        }
         if (dst_shape.width <= 0 || dst_shape.height <= 0) {
             throw std::invalid_argument("pyrUp: Destination shape must be positive.");
         }
         // Ensure input is float32
        cv::Mat float_image;
        if (image.type() != CV_32FC3) {
             image.convertTo(float_image, CV_32FC3);
        } else {
             float_image = image; // Avoid clone if already correct type
        }

        // 1. Create intermediate upsampled image with zeros (mimicking np.insert)
        // Calculate size based on destination shape (roughly double the input)
        // OpenCV's pyrUp does this internally, but Python logic inserts zeros first.
        // Let's try mimicking the Python logic more closely for potential subtle differences.
        cv::Mat upsampled_zeros = cv::Mat::zeros(dst_shape, float_image.type());
        // Place original pixels at even indices
        for (int r = 0; r < float_image.rows; ++r) {
            for (int c = 0; c < float_image.cols; ++c) {
                 int target_r = r * 2;
                 int target_c = c * 2;
                 // Check bounds for the destination size
                 if (target_r < dst_shape.height && target_c < dst_shape.width) {
                    upsampled_zeros.at<cv::Vec3f>(target_r, target_c) = float_image.at<cv::Vec3f>(r, c);
                 }
            }
        }


        // 2. Filter the zero-padded image using cv::filter2D with 4 * kernel
        cv::Mat filtered_image;
        // Use BORDER_REFLECT_101 to match Python's default
        cv::filter2D(upsampled_zeros, filtered_image, -1, 4.0 * kernel, cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101);

        // 3. Ensure final size matches dst_shape (filter2D output size matches input)
        if (filtered_image.size() != dst_shape) {
             cv::resize(filtered_image, filtered_image, dst_shape, 0, 0, cv::INTER_LINEAR);
        }

        return filtered_image;
    }

    // Overloaded version using default Gaussian kernel
    cv::Mat pyrUp(const cv::Mat& image, const cv::Size& dst_shape) {
        return pyrUp(image, gaussian_kernel, dst_shape);
    }


    // --- Video Processing Functions ---

    // Placeholder - needs implementation if Laplacian processing is required
    void processVideoLaplacian(const std::string& inputFilename, const std::string& outputFilename,
                               int levels, double alpha, double lambda_c, double fl, double fh,
                               double chromAttenuation) {
        LOG_PROC("processVideoLaplacian called (Not Implemented)");
        throw std::runtime_error("processVideoLaplacian not implemented");
    }

    // Implementation for Gaussian Video Processing using Batch FFT approach
    void processVideoGaussianBatch(const std::string& inputFilename, const std::string& outputFilename,
                                 int levels, double alpha, double fl, double fh,
                                 double chromAttenuation)
    {
        LOG_PROC("Starting Gaussian batch video processing for: " + inputFilename);
        auto start_time = std::chrono::high_resolution_clock::now();

        // 1. Open Input Video
        cv::VideoCapture cap(inputFilename);
        if (!cap.isOpened()) {
            throw std::runtime_error("Error opening video file: " + inputFilename);
        }

        // 2. Get Video Properties
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        cv::Size original_frame_size(frame_width, frame_height);
        double fps = cap.get(cv::CAP_PROP_FPS);
        // int frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT)); // May be inaccurate

        if (fps <= 0) {
             // Try to estimate FPS if property is not available or invalid
             LOG_PROC("Warning: Could not read FPS property accurately. Estimating based on first few frames or assuming 30.");
             // Basic estimation or default needed here if CAP_PROP_FPS fails.
             // For now, let's assume 30 if it's invalid.
             fps = 30.0; // Default fallback
             // A better approach might involve timing frame reads, but that's complex.
        }
        double samplingRate = fps; // Use video FPS as sampling rate

        LOG_PROC("Video properties: " + std::to_string(frame_width) + "x" + std::to_string(frame_height) +
                 ", FPS: " + std::to_string(fps));

        // 3. Read all frames and perform initial conversions/filtering
        std::vector<cv::Mat> original_rgb_frames;
        std::vector<cv::Mat> spatially_filtered_batch; // Batch of blurred YIQ frames
        cv::Mat frameRgbUint8;

        LOG_PROC("Reading frames and performing spatial filtering...");
        int frame_num = 0;
        while (true) {
            cap >> frameRgbUint8;
            if (frameRgbUint8.empty()) {
                break; // End of video
            }

            // Store original RGB frame (needed for reconstruction)
            original_rgb_frames.push_back(frameRgbUint8.clone());

            // a. Convert RGB to YIQ (Float)
            cv::Mat frameYiqFloat = rgb2yiq(frameRgbUint8);

            // b. Spatial Filtering (Downsample then Upsample) - Expects RGB input
            // Assuming gaussian_kernel is accessible (defined in processing.hpp)
            cv::Mat blurredYiq = spatiallyFilterGaussian(frameRgbUint8, levels, gaussian_kernel);
            if (blurredYiq.empty()) {
                 LOG_PROC("Warning: spatiallyFilterGaussian returned empty frame for frame " + std::to_string(frame_num) + ". Skipping.");
                 // Need to handle this case - maybe skip adding to batch or add an empty Mat?
                 // Adding an empty Mat might cause issues later. Let's skip for now.
                 // Alternatively, store original YIQ? For now, just log and continue.
                 continue; // Skip to next frame if spatial filtering fails
            }
            spatially_filtered_batch.push_back(blurredYiq);
            // Store the original RGB frame corresponding to the successfully filtered frame
            // We need this later for reconstruction. Let's create a parallel vector.
            // This logic needs refinement - better to filter originals_rgb_frames later.
            // Let's adjust the logic: store all originals, filter later.
            // Reverting this part - keep original_rgb_frames as is.

            frame_num++;
            if (frame_num % 100 == 0) {
                LOG_PROC("Read and spatially filtered " + std::to_string(frame_num) + " frames...");
            }
        }
        cap.release(); // Release video capture object
        LOG_PROC("Finished reading and spatially filtering " + std::to_string(frame_num) + " frames.");

        if (original_rgb_frames.empty()) {
            LOG_PROC("No frames read from video. Exiting.");
            return;
        }

        // 4. Temporal Filtering (FFT-based) and Amplification
        LOG_PROC("Applying temporal filter and amplification...");
        // Ensure spatially_filtered_batch is not empty before proceeding
        if (spatially_filtered_batch.empty()) {
             LOG_PROC("No frames were successfully spatially filtered. Cannot perform temporal filtering.");
             return; // Or throw an exception
        }
        std::vector<cv::Mat> filtered_amplified_batch = temporalFilterGaussianBatch(
            spatially_filtered_batch,
            static_cast<float>(fps), // Pass fps as float
            static_cast<float>(fl), static_cast<float>(fh), // Pass frequencies as float
            static_cast<float>(alpha), static_cast<float>(chromAttenuation) // Pass factors as float
        );
        LOG_PROC("Temporal filtering and amplification complete.");

        // 5. Reconstruct Final Video
        LOG_PROC("Reconstructing final video...");
        // Check if temporal filtering produced output matching input size
        if (filtered_amplified_batch.size() != original_rgb_frames.size()) {
             // This indicates an issue, possibly frames skipped during spatial filtering
             // or temporal filtering failed.
             // We need a robust way to match original frames to filtered frames.
             // For now, assume sizes match or throw error.
             if (filtered_amplified_batch.empty() && !original_rgb_frames.empty()) {
                  throw std::runtime_error("Temporal filtering failed or produced no output.");
             } else {
                  // Log a warning, proceed with potentially mismatched sizes? Risky.
                  // Best approach: Ensure spatial filtering stores placeholders or
                  // temporal filtering handles variable batch sizes / returns map.
                  // For now, let's throw if sizes don't match after filtering.
                  throw std::runtime_error("Mismatch between number of original frames (" +
                                           std::to_string(original_rgb_frames.size()) +
                                           ") and temporally filtered frames (" +
                                           std::to_string(filtered_amplified_batch.size()) + ").");
             }
        }

        std::vector<cv::Mat> output_video;
        output_video.reserve(original_rgb_frames.size()); // Pre-allocate space

        for (size_t i = 0; i < original_rgb_frames.size(); ++i) {
             // Check if corresponding filtered frame exists and is valid
             if (i < filtered_amplified_batch.size() && !filtered_amplified_batch[i].empty()) {
                 cv::Mat reconstructed_frame = reconstructGaussianFrame(
                     original_rgb_frames[i],
                     filtered_amplified_batch[i]
                 );
                 output_video.push_back(reconstructed_frame);
             } else {
                  LOG_PROC("Warning: Missing or empty filtered frame for original frame index " + std::to_string(i) + ". Skipping reconstruction.");
                  // Add an empty frame or handle differently? Adding empty for now.
                  output_video.push_back(cv::Mat());
             }
        }
        LOG_PROC("Video reconstruction complete.");


        // 6. Initialize Output Video Writer
        cv::VideoWriter writer(outputFilename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), // Or other codec
                               fps, original_frame_size);
        if (!writer.isOpened()) {
            throw std::runtime_error("Error opening video writer for: " + outputFilename);
        }

        // 7. Write Output Frames
        LOG_PROC("Writing output video to: " + outputFilename);
        for (size_t i = 0; i < output_video.size(); ++i) {
             if (!output_video[i].empty()) {
                 writer.write(output_video[i]);
             } else {
                 LOG_PROC("Warning: Skipping empty frame during writing at index " + std::to_string(i));
             }
             if ((i + 1) % 100 == 0) {
                 LOG_PROC("Wrote " + std::to_string(i + 1) + " frames...");
             }
        }
        writer.release(); // Release video writer object
        LOG_PROC("Finished writing " + std::to_string(output_video.size()) + " frames.");

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        LOG_PROC("Total processing time: " + std::to_string(duration.count()) + " ms");
        LOG_PROC("Gaussian batch video processing complete for: " + outputFilename);
    }


} // namespace evmcpp