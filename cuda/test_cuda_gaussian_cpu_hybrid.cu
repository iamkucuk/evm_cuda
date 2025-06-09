// NOTE: This test file is disabled due to NO_OPENCV constraint enforcement
// The spatially_filter_gaussian_wrapper() function has been removed
// Use pure CUDA functions with float* arrays instead

/*
#include "cuda_gaussian_pyramid.cuh"
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <chrono>
#include <vector>

// CPU reference implementations for comparison
#include "../cpp/include/gaussian_pyramid.hpp"
#include "../cpp/include/processing.hpp"

/**
 * @brief Hybrid implementation: CUDA Gaussian pyramid + CPU rest
 * This replaces the CPU Gaussian pyramid components with CUDA implementations
 * while keeping the overall CPU pipeline structure for comparison
 */
bool process_video_hybrid_cuda_gaussian(
    const std::string& input_filename,
    const std::string& output_filename,
    int levels,
    double alpha,
    double fl,
    double fh,
    double chrom_attenuation)
{
    std::cout << "Hybrid CUDA Gaussian + CPU Rest Pipeline Test" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "Input: " << input_filename << std::endl;
    std::cout << "Output: " << output_filename << std::endl;
    std::cout << "Parameters: levels=" << levels << ", alpha=" << alpha 
              << ", fl=" << fl << ", fh=" << fh << ", chrom_attenuation=" << chrom_attenuation << std::endl;
    
    // Open input video
    cv::VideoCapture cap(input_filename);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open input video file: " << input_filename << std::endl;
        return false;
    }
    
    // Get video properties
    const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const double fps = cap.get(cv::CAP_PROP_FPS);
    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    std::cout << "Video info: " << width << "x" << height << ", " << total_frames << " frames at " << fps << " FPS" << std::endl;
    
    if (total_frames <= 1) {
        std::cerr << "Error: Need more than 1 frame for temporal filtering" << std::endl;
        return false;
    }
    
    // Open output video writer
    cv::VideoWriter writer(output_filename, cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Error: Cannot open output video file: " << output_filename << std::endl;
        return false;
    }
    
    try {
        // Load all frames
        std::cout << "Loading " << total_frames << " frames..." << std::endl;
        std::vector<cv::Mat> frames(total_frames);
        std::vector<cv::Mat> spatially_filtered_batch(total_frames);
        
        for (int i = 0; i < total_frames; i++) {
            cap >> frames[i];
            if (frames[i].empty()) {
                std::cerr << "Error: Failed to read frame " << i << std::endl;
                return false;
            }
            
            // HYBRID: Use CUDA spatial filtering instead of CPU
            spatially_filtered_batch[i] = cuda_evm::spatially_filter_gaussian_wrapper(frames[i], levels);
            if (spatially_filtered_batch[i].empty()) {
                std::cerr << "Error: CUDA spatial filtering failed for frame " << i << std::endl;
                return false;
            }
            
            if ((i + 1) % 50 == 0) {
                std::cout << "Loaded and spatially filtered " << (i + 1) << " frames (using CUDA)" << std::endl;
            }
        }
        
        // HYBRID: Use CPU temporal filtering (for exact comparison)
        std::cout << "Applying CPU temporal filtering..." << std::endl;
        std::vector<cv::Mat> temporally_filtered_batch = evmcpp::temporalFilterGaussianBatch(
            spatially_filtered_batch, static_cast<float>(fps), 
            static_cast<float>(fl), static_cast<float>(fh), 
            static_cast<float>(alpha), static_cast<float>(chrom_attenuation));
        
        if (temporally_filtered_batch.empty()) {
            std::cerr << "Error: CPU temporal filtering failed" << std::endl;
            return false;
        }
        
        // HYBRID: Use CPU frame reconstruction (for exact comparison)
        std::cout << "Reconstructing and writing output frames (using CPU)..." << std::endl;
        for (int i = 0; i < total_frames; i++) {
            cv::Mat reconstructed = evmcpp::reconstructGaussianFrame(frames[i], temporally_filtered_batch[i]);
            if (reconstructed.empty()) {
                std::cerr << "Error: CPU reconstruction failed for frame " << i << std::endl;
                return false;
            }
            
            writer << reconstructed;
            
            if ((i + 1) % 50 == 0) {
                std::cout << "Processed " << (i + 1) << " frames" << std::endl;
            }
        }
        
        std::cout << "Hybrid CUDA Gaussian + CPU rest processing completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during processing: " << e.what() << std::endl;
        return false;
    }
    
    cap.release();
    writer.release();
    
    return true;
}

int main() {
    std::cout << "CUDA Gaussian Pyramid Hybrid Test" << std::endl;
    std::cout << "================================" << std::endl;
    
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "❌ No CUDA devices found!" << std::endl;
        return -1;
    }
    
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    // Test parameters matching face.mp4 processing
    const std::string input_video = "/home/furkan/projects/school/mmi713/evm/data/face.mp4";
    const std::string hybrid_output = "cuda_gaussian_cpu_hybrid_output.avi";
    
    // Parameters for face magnification
    const int levels = 2;
    const double alpha = 50.0;
    const double fl = 0.5;
    const double fh = 2.0;
    const double chrom_attenuation = 0.1;
    
    std::cout << "\nProcessing face.mp4 with hybrid pipeline:" << std::endl;
    std::cout << "  - CUDA Gaussian spatial filtering" << std::endl;
    std::cout << "  - CPU temporal filtering" << std::endl;
    std::cout << "  - CPU frame reconstruction" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool success = process_video_hybrid_cuda_gaussian(
        input_video, hybrid_output, levels, alpha, fl, fh, chrom_attenuation);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto processing_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    if (success) {
        std::cout << "\n✅ SUCCESS: Hybrid processing completed in " << processing_time << " seconds" << std::endl;
        std::cout << "Output video: " << hybrid_output << std::endl;
        std::cout << "\nTo validate accuracy, compare with CPU reference using:" << std::endl;
        std::cout << "./compare_videos_frame_by_frame " << hybrid_output << " ../cpp/build/cpu_reference_gaussian.avi" << std::endl;
        std::cout << "Or use the analyze_frame_psnr script for detailed analysis." << std::endl;
    } else {
        std::cout << "\n❌ FAILED: Hybrid processing failed" << std::endl;
        return -1;
    }
    
    return 0;
}
*/

int main() {
    std::cout << "❌ TEST DISABLED: This test requires OpenCV wrapper functions that have been removed" << std::endl;
    std::cout << "   Due to NO_OPENCV constraint enforcement, use pure CUDA functions instead" << std::endl;
    std::cout << "   For video processing, use process_video_gaussian_gpu() directly" << std::endl;
    return -1;
}