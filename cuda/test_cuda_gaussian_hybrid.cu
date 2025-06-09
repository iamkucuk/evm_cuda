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

// CPU implementations for non-Gaussian parts
#include "../cpp/include/gaussian_pyramid.hpp"
#include "../cpp/include/processing.hpp"
#include "../cpp/include/color_conversion.hpp"

using namespace evmcpp;

// Helper functions
void convertMatToFloat(const cv::Mat& input, std::vector<float>& output) {
    if (input.type() != CV_32FC3) {
        cv::Mat float_mat;
        input.convertTo(float_mat, CV_32FC3);
        output.assign((float*)float_mat.datastart, (float*)float_mat.dataend);
    } else {
        output.assign((float*)input.datastart, (float*)input.dataend);
    }
}

void convertFloatToMat(const std::vector<float>& input, cv::Mat& output, int width, int height) {
    output = cv::Mat(height, width, CV_32FC3, (void*)input.data()).clone();
}

// Process video using CUDA Gaussian spatial filtering + CPU temporal filtering
bool processVideoHybridCUDAGaussian(
    const std::string& input_filename,
    const std::string& output_filename,
    int levels, double alpha, double fl, double fh, double chrom_attenuation)
{
    std::cout << "Hybrid CUDA Gaussian + CPU Temporal Pipeline Test" << std::endl;
    std::cout << "=================================================" << std::endl;
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
    const int channels = 3;
    
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
    
    // GPU memory allocation
    const size_t frame_size = width * height * channels * sizeof(float);
    const size_t total_size = total_frames * frame_size;
    
    float *d_original_frames = nullptr;
    float *d_spatially_filtered = nullptr;
    
    cudaError_t err = cudaSuccess;
    
    try {
        // Allocate GPU memory
        err = cudaMalloc(&d_original_frames, total_size);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate original frames buffer");
        
        err = cudaMalloc(&d_spatially_filtered, total_size);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate spatially filtered buffer");
        
        std::cout << "GPU memory allocated: " << (2 * total_size) / (1024*1024) << " MB" << std::endl;
        
        // Load all frames to GPU
        std::cout << "Loading frames to GPU..." << std::endl;
        for (int i = 0; i < total_frames; ++i) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                throw std::runtime_error("Failed to read frame " + std::to_string(i));
            }
            
            // Convert to float and copy to GPU
            cv::Mat frame_float;
            frame.convertTo(frame_float, CV_32FC3);
            
            err = cudaMemcpy(d_original_frames + i * width * height * channels,
                           frame_float.ptr<float>(), frame_size, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to copy frame " + std::to_string(i) + " to GPU");
            }
            
            if (i % 50 == 0) {
                std::cout << "  Loaded frame " << i << "/" << total_frames << std::endl;
            }
        }
        cap.release();
        
        // Step 1: CUDA Gaussian spatial filtering for all frames
        std::cout << "CUDA Gaussian spatial filtering..." << std::endl;
        for (int i = 0; i < total_frames; ++i) {
            err = cuda_evm::spatially_filter_gaussian_gpu(
                d_original_frames + i * width * height * channels,
                d_spatially_filtered + i * width * height * channels,
                width, height, channels, levels);
            if (err != cudaSuccess) {
                throw std::runtime_error("Spatial filtering failed for frame " + std::to_string(i));
            }
            
            if (i % 50 == 0) {
                std::cout << "  Processed frame " << i << "/" << total_frames << std::endl;
            }
        }
        
        // Step 2: Download spatially filtered frames for CPU temporal filtering
        std::cout << "Downloading spatially filtered frames for CPU temporal processing..." << std::endl;
        std::vector<cv::Mat> spatially_filtered_frames(total_frames);
        for (int i = 0; i < total_frames; ++i) {
            cv::Mat frame_float(height, width, CV_32FC3);
            err = cudaMemcpy(frame_float.ptr<float>(),
                           d_spatially_filtered + i * width * height * channels,
                           frame_size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to download frame " + std::to_string(i));
            }
            spatially_filtered_frames[i] = frame_float;
            
            if (i % 50 == 0) {
                std::cout << "  Downloaded frame " << i << "/" << total_frames << std::endl;
            }
        }
        
        // Step 3: CPU temporal filtering (using existing CPU Gaussian batch function)
        std::cout << "CPU temporal filtering..." << std::endl;
        std::vector<cv::Mat> temporally_filtered_frames = evmcpp::temporalFilterGaussianBatch(
            spatially_filtered_frames, static_cast<float>(fps), 
            static_cast<float>(fl), static_cast<float>(fh), 
            static_cast<float>(alpha), static_cast<float>(chrom_attenuation));
        
        // Step 4: CPU frame reconstruction
        std::cout << "CPU frame reconstruction..." << std::endl;
        std::vector<cv::Mat> original_frames(total_frames);
        for (int i = 0; i < total_frames; ++i) {
            cv::Mat frame_float(height, width, CV_32FC3);
            err = cudaMemcpy(frame_float.ptr<float>(),
                           d_original_frames + i * width * height * channels,
                           frame_size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to download original frame " + std::to_string(i));
            }
            original_frames[i] = frame_float;
        }
        
        // Reconstruct using CPU function
        std::vector<cv::Mat> output_frames(total_frames);
        for (int i = 0; i < total_frames; ++i) {
            output_frames[i] = evmcpp::reconstructGaussianFrame(original_frames[i], temporally_filtered_frames[i]);
            
            if (i % 50 == 0) {
                std::cout << "  Reconstructed frame " << i << "/" << total_frames << std::endl;
            }
        }
        
        // Step 5: Write output video
        std::cout << "Writing output video..." << std::endl;
        for (int i = 0; i < total_frames; ++i) {
            // Convert to uint8 and write
            cv::Mat output_uint8;
            output_frames[i].convertTo(output_uint8, CV_8UC3);
            writer << output_uint8;
            
            if (i % 50 == 0) {
                std::cout << "  Written frame " << i << "/" << total_frames << std::endl;
            }
        }
        
        std::cout << "CUDA Gaussian hybrid processing completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during processing: " << e.what() << std::endl;
        
        // Cleanup on error
        if (d_original_frames) cudaFree(d_original_frames);
        if (d_spatially_filtered) cudaFree(d_spatially_filtered);
        cap.release();
        writer.release();
        return false;
    }
    
    // Cleanup
    cudaFree(d_original_frames);
    cudaFree(d_spatially_filtered);
    cap.release();
    writer.release();
    
    return true;
}

int main() {
    std::cout << "CUDA Gaussian Spatial + CPU Temporal Pipeline Test" << std::endl;
    std::cout << "=================================================" << std::endl;
    
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
    std::string output_video_path = "cuda_gaussian_hybrid_output.avi";
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
    
    bool success = processVideoHybridCUDAGaussian(
        input_video_path, output_video_path, levels, alpha, fl, fh, chrom_attenuation);
    
    if (success) {
        std::cout << "\n✅ SUCCESS: Hybrid CUDA Gaussian processing completed!" << std::endl;
        std::cout << "Output saved to: " << output_video_path << std::endl;
        std::cout << "\nTo analyze quality, run:" << std::endl;
        std::cout << "python3 analyze_frame_psnr.py" << std::endl;
        std::cout << "Or use the analyze_frame_psnr script for detailed analysis." << std::endl;
    } else {
        std::cout << "\n❌ FAILED: Hybrid processing failed" << std::endl;
    }
    
    return success ? 0 : 1;
}