#include "cuda_evm.cuh"
#include "cuda_laplacian_pyramid.cuh"
#include "cuda_pyramid.cuh"
#include "cuda_color_conversion.cuh"
#include "cuda_butterworth.cuh"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp> // For VideoCapture, VideoWriter
#include <string>
#include <vector>
#include <cmath>
#include <chrono> // For timing
#include <iostream>
#include <stdexcept>

// Define logging for debug output
#define LOG_EVM(message) std::cout << "[CUDA EVM] " << message << std::endl

namespace evmcuda {

// Main function to process video using CUDA-accelerated Eulerian Video Magnification
void process_video_laplacian(
    const std::string& inputFilename, 
    const std::string& outputFilename,
    int pyramid_levels, 
    double alpha, 
    double lambda_cutoff, 
    double fl, 
    double fh, 
    double chrom_attenuation)
{
    LOG_EVM("Starting CUDA-accelerated Laplacian video processing for: " + inputFilename);
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize all necessary CUDA modules
    if (!init_evm()) {
        throw std::runtime_error("Failed to initialize EVM CUDA modules");
    }
    
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
    
    if (fps <= 0) {
        LOG_EVM("Warning: Could not read FPS property accurately. Using default value of 30.");
        fps = 30.0; // Default fallback
    }
    
    LOG_EVM("Video properties: " + std::to_string(frame_width) + "x" + std::to_string(frame_height) + 
            ", FPS: " + std::to_string(fps));
    
    // 3. Read all frames
    std::vector<cv::Mat> original_rgb_frames;
    cv::Mat frameRgbUint8;
    
    LOG_EVM("Reading frames...");
    int frame_num = 0;
    while (true) {
        cap >> frameRgbUint8;
        if (frameRgbUint8.empty()) {
            break; // End of video
        }
        
        // Store original RGB frame (needed for reconstruction)
        original_rgb_frames.push_back(frameRgbUint8.clone());
        
        frame_num++;
        if (frame_num % 100 == 0) {
            LOG_EVM("Read " + std::to_string(frame_num) + " frames...");
        }
    }
    cap.release(); // Release video capture object
    LOG_EVM("Finished reading " + std::to_string(frame_num) + " frames.");
    
    if (original_rgb_frames.empty()) {
        LOG_EVM("No frames read from video. Exiting.");
        return;
    }
    
    // 4. Generate Laplacian Pyramids
    LOG_EVM("Generating Laplacian pyramids...");
    std::vector<std::vector<cv::Mat>> laplacian_pyramids_batch = 
        get_laplacian_pyramids(original_rgb_frames, pyramid_levels);
    LOG_EVM("Generated Laplacian pyramids for " + std::to_string(laplacian_pyramids_batch.size()) + " frames.");
    
    // 5. Filter Laplacian Pyramids
    LOG_EVM("Filtering Laplacian pyramids...");
    std::pair<double, double> freq_range(fl, fh);
    std::vector<std::vector<cv::Mat>> filtered_pyramids = filter_laplacian_pyramids(
        laplacian_pyramids_batch,
        pyramid_levels,
        fps,
        freq_range,
        alpha,
        lambda_cutoff,
        chrom_attenuation
    );
    LOG_EVM("Filtered " + std::to_string(filtered_pyramids.size()) + " Laplacian pyramids.");
    
    // 6. Reconstruct Final Video
    LOG_EVM("Reconstructing final video...");
    std::vector<cv::Mat> output_video;
    output_video.reserve(original_rgb_frames.size());
    
    for (size_t i = 0; i < original_rgb_frames.size(); i++) {
        if (i < filtered_pyramids.size() && !filtered_pyramids[i].empty()) {
            cv::Mat reconstructed_frame = reconstruct_laplacian_image(
                original_rgb_frames[i],
                filtered_pyramids[i]
            );
            output_video.push_back(reconstructed_frame);
        } else {
            LOG_EVM("Warning: Missing or empty filtered pyramid for frame " + std::to_string(i) + 
                   ". Using original frame.");
            output_video.push_back(original_rgb_frames[i].clone());
        }
        
        if ((i + 1) % 100 == 0) {
            LOG_EVM("Reconstructed " + std::to_string(i + 1) + " frames...");
        }
    }
    LOG_EVM("Video reconstruction complete.");
    
    // 7. Initialize Output Video Writer
    cv::VideoWriter writer(outputFilename,
                         cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), // MJPG codec
                         fps,
                         original_frame_size);
    
    if (!writer.isOpened()) {
        throw std::runtime_error("Error opening video writer for: " + outputFilename);
    }
    
    // 8. Write Output Frames
    LOG_EVM("Writing output video to: " + outputFilename);
    for (size_t i = 0; i < output_video.size(); i++) {
        if (!output_video[i].empty()) {
            writer.write(output_video[i]);
        } else {
            LOG_EVM("Warning: Skipping empty frame during writing at index " + std::to_string(i));
        }
        
        if ((i + 1) % 100 == 0) {
            LOG_EVM("Wrote " + std::to_string(i + 1) + " frames...");
        }
    }
    writer.release(); // Release video writer object
    LOG_EVM("Finished writing " + std::to_string(output_video.size()) + " frames.");
    
    // Clean up CUDA resources
    cleanup_evm();
    
    // Calculate and report total processing time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    LOG_EVM("Total processing time: " + std::to_string(duration.count()) + " ms");
    LOG_EVM("CUDA Laplacian video processing complete for: " + outputFilename);
}

// Initialize all modules needed by the EVM pipeline
bool init_evm() {
    LOG_EVM("Initializing EVM CUDA modules...");
    
    // Initialize all required modules
    bool success = true;
    
    // Initialize color conversion module
    if (!init_color_conversion()) {
        LOG_EVM("Failed to initialize color conversion module");
        success = false;
    }
    
    // Initialize pyramid module
    if (!init_pyramid()) {
        LOG_EVM("Failed to initialize pyramid module");
        success = false;
    }
    
    // Initialize laplacian pyramid module
    if (!init_laplacian_pyramid()) {
        LOG_EVM("Failed to initialize laplacian pyramid module");
        success = false;
    }
    
    // Initialize butterworth module
    if (!init_butterworth()) {
        LOG_EVM("Failed to initialize butterworth module");
        success = false;
    }
    
    if (success) {
        LOG_EVM("All EVM CUDA modules initialized successfully");
    } else {
        LOG_EVM("Failed to initialize one or more EVM CUDA modules");
    }
    
    return success;
}

// Clean up all resources used by the EVM pipeline
void cleanup_evm() {
    LOG_EVM("Cleaning up EVM CUDA resources...");
    
    // Clean up all modules in reverse order
    cleanup_butterworth();
    cleanup_laplacian_pyramid();
    cleanup_pyramid();
    cleanup_color_conversion();
    
    LOG_EVM("EVM CUDA resources cleaned up");
}

} // namespace evmcuda