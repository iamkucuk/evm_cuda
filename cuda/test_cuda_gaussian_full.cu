#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>

// CUDA Gaussian implementation
#include "include/cuda_gaussian_pyramid.cuh"

// Process video using FULL CUDA pipeline (like original OpenCV version)
bool processVideoFullCUDAGaussian(
    const std::string& input_filename,
    const std::string& output_filename,
    int levels, double alpha, double fl, double fh, double chrom_attenuation)
{
    std::cout << "Full CUDA Gaussian EVM Processing Started..." << std::endl;
    std::cout << "Input: " << input_filename << std::endl;
    std::cout << "Output: " << output_filename << std::endl;
    std::cout << "Parameters: levels=" << levels << ", alpha=" << alpha 
              << ", fl=" << fl << ", fh=" << fh << ", chrom_attenuation=" << chrom_attenuation << std::endl;
    
    // Open input video (OpenCV allowed for video I/O only)
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
    const int channels = 3;
    
    std::cout << "Video info: " << width << "x" << height << ", " << total_frames << " frames at " << fps << " FPS" << std::endl;
    
    if (total_frames <= 1) {
        std::cerr << "Error: Need more than 1 frame for temporal filtering" << std::endl;
        return false;
    }
    
    // Open output video writer (OpenCV allowed for video I/O only)
    cv::VideoWriter writer(output_filename, cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Error: Cannot open output video file: " << output_filename << std::endl;
        return false;
    }
    
    // GPU memory allocation (exactly like original OpenCV version)
    const size_t frame_size = width * height * channels * sizeof(float);
    const size_t total_size = total_frames * frame_size;
    
    float *d_original_frames = nullptr;
    float *d_spatially_filtered = nullptr;
    float *d_temporally_filtered = nullptr;
    float *d_final_output = nullptr;
    
    cudaError_t err = cudaSuccess;
    
    try {
        // Allocate GPU memory for entire video processing
        err = cudaMalloc(&d_original_frames, total_size);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate original frames buffer");
        
        err = cudaMalloc(&d_spatially_filtered, total_size);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate spatially filtered buffer");
        
        err = cudaMalloc(&d_temporally_filtered, total_size);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate temporally filtered buffer");
        
        err = cudaMalloc(&d_final_output, total_size);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate final output buffer");
        
        std::cout << "GPU memory allocated: " << (4 * total_size) / (1024*1024) << " MB" << std::endl;
        
        // Load all frames and convert to raw float arrays (like original)
        std::cout << "Loading " << total_frames << " frames..." << std::endl;
        for (int i = 0; i < total_frames; i++) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "Error: Failed to read frame " << i << std::endl;
                return false;
            }
            
            // Convert to float and copy to GPU memory (like original)
            cv::Mat frame_float;
            frame.convertTo(frame_float, CV_32FC3);
            
            float* frame_offset = d_original_frames + i * width * height * channels;
            err = cudaMemcpy(frame_offset, frame_float.ptr<float>(), frame_size, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "Failed to copy frame " << i << " to GPU: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("GPU memory copy failed");
            }
            
            if ((i + 1) % 50 == 0) {
                std::cout << "Loaded " << (i + 1) << " frames" << std::endl;
            }
        }
        cap.release();
        
        // Step 1: Apply spatial filtering to all frames using pure CUDA (like original)
        std::cout << "Applying spatial filtering on GPU..." << std::endl;
        for (int i = 0; i < total_frames; i++) {
            float* input_frame = d_original_frames + i * width * height * channels;
            float* output_frame = d_spatially_filtered + i * width * height * channels;
            
            err = cuda_evm::spatially_filter_gaussian_gpu(input_frame, output_frame, width, height, channels, levels);
            if (err != cudaSuccess) {
                std::cerr << "Spatial filtering failed for frame " << i << ": " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("Spatial filtering failed");
            }
            
            if ((i + 1) % 50 == 0) {
                std::cout << "Spatially filtered " << (i + 1) << " frames" << std::endl;
            }
        }
        
        // Step 2: Apply temporal filtering using pure CUDA (like original)
        std::cout << "Applying temporal filtering on GPU..." << std::endl;
        err = cuda_evm::temporal_filter_gaussian_batch_gpu(
            d_spatially_filtered, d_temporally_filtered, width, height, channels, total_frames,
            static_cast<float>(fl), static_cast<float>(fh), static_cast<float>(fps));
        
        if (err != cudaSuccess) {
            std::cerr << "Temporal filtering failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("Temporal filtering failed");
        }
        
        // Step 3: Reconstruct frames using pure CUDA (like original)
        std::cout << "Reconstructing frames on GPU..." << std::endl;
        for (int i = 0; i < total_frames; i++) {
            float* original_frame = d_original_frames + i * width * height * channels;
            float* filtered_frame = d_temporally_filtered + i * width * height * channels;
            float* output_frame = d_final_output + i * width * height * channels;
            
            err = cuda_evm::reconstruct_gaussian_frame_gpu(original_frame, filtered_frame, output_frame, 
                                                         width, height, channels, 
                                                         static_cast<float>(alpha), static_cast<float>(chrom_attenuation));
            if (err != cudaSuccess) {
                std::cerr << "Reconstruction failed for frame " << i << ": " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("Reconstruction failed");
            }
            
            if ((i + 1) % 50 == 0) {
                std::cout << "Reconstructed " << (i + 1) << " frames" << std::endl;
            }
        }
        
        // Download results and write to video (OpenCV allowed for video I/O only)
        std::cout << "Writing output video..." << std::endl;
        for (int i = 0; i < total_frames; i++) {
            // Download frame from GPU (like original)
            cv::Mat output_float(height, width, CV_32FC3);
            float* frame_offset = d_final_output + i * width * height * channels;
            err = cudaMemcpy(output_float.ptr<float>(), frame_offset, frame_size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                std::cerr << "Failed to download frame " << i << " from GPU: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("GPU memory download failed");
            }
            
            // Convert to uint8 and clip to valid range (like original)
            cv::Mat output_uint8;
            output_float.convertTo(output_uint8, CV_8UC3);
            cv::threshold(output_uint8, output_uint8, 255, 255, cv::THRESH_TRUNC);
            cv::threshold(output_uint8, output_uint8, 0, 0, cv::THRESH_TOZERO);
            
            writer << output_uint8;
            
            if ((i + 1) % 50 == 0) {
                std::cout << "Written " << (i + 1) << " frames" << std::endl;
            }
        }
        
        std::cout << "CUDA Gaussian EVM processing completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during processing: " << e.what() << std::endl;
        
        // Cleanup on error
        if (d_original_frames) cudaFree(d_original_frames);
        if (d_spatially_filtered) cudaFree(d_spatially_filtered);
        if (d_temporally_filtered) cudaFree(d_temporally_filtered);
        if (d_final_output) cudaFree(d_final_output);
        cap.release();
        writer.release();
        return false;
    }
    
    // Cleanup
    cudaFree(d_original_frames);
    cudaFree(d_spatially_filtered);
    cudaFree(d_temporally_filtered);
    cudaFree(d_final_output);
    cap.release();
    writer.release();
    
    return true;
}

int main() {
    std::cout << "Full CUDA Gaussian Pipeline Test" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Check CUDA availability
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "CUDA is not available or no devices found" << std::endl;
        return 1;
    }
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    // Configuration - SAME AS CPU REFERENCE
    std::string input_video_path = "/home/furkan/projects/school/mmi713/evm/data/face.mp4";
    std::string output_video_path = "cuda_gaussian_full_output.avi";
    int levels = 2;
    double alpha = 50.0;
    double fl = 0.5;
    double fh = 2.0;
    double chrom_attenuation = 0.1;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Input: " << input_video_path << std::endl;
    std::cout << "  Output: " << output_video_path << std::endl;
    std::cout << "  Levels: " << levels << std::endl;
    std::cout << "  Alpha: " << alpha << std::endl;
    std::cout << "  Frequencies: " << fl << " - " << fh << " Hz" << std::endl;
    std::cout << "  Chrom attenuation: " << chrom_attenuation << std::endl;
    
    bool success = processVideoFullCUDAGaussian(
        input_video_path, output_video_path, levels, alpha, fl, fh, chrom_attenuation);
    
    if (success) {
        std::cout << "\n✅ SUCCESS: Full CUDA Gaussian processing completed!" << std::endl;
        std::cout << "Output saved to: " << output_video_path << std::endl;
        std::cout << "\nTo analyze quality, run:" << std::endl;
        std::cout << "python3 analyze_frame_psnr.py" << std::endl;
    } else {
        std::cout << "\n❌ FAILED: Full CUDA processing failed" << std::endl;
    }
    
    return success ? 0 : 1;
}