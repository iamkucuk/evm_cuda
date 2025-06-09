#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>

// CUDA Butterworth implementation (working approach)
#include "include/cuda_butterworth.cuh"

// CPU implementations for everything else
#include "../cpp/include/laplacian_pyramid.hpp"
#include "../cpp/include/gaussian_pyramid.hpp"
#include "../cpp/include/processing.hpp"
#include "../cpp/include/butterworth.hpp"
#include "../cpp/include/color_conversion.hpp"

using namespace evmcpp;

// Helper functions from working test
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

// Apply CUDA Butterworth temporal filtering using the WORKING approach
std::vector<std::vector<cv::Mat>> filterPyramidsWithCUDAButterworthFixed(
    const std::vector<std::vector<cv::Mat>>& pyramids,
    int pyramid_levels, 
    double fps,
    double fl, double fh,
    double alpha, double lambda_cutoff, double chrom_attenuation) {
    
    std::cout << "Applying CUDA Butterworth temporal filtering (FIXED approach)..." << std::endl;
    
    if (pyramids.empty()) return {};
    
    size_t num_frames = pyramids.size();
    std::vector<std::vector<cv::Mat>> filtered_pyramids;
    
    // Initialize output structure
    filtered_pyramids.reserve(num_frames);
    for(const auto& frame_pyramid : pyramids) {
        filtered_pyramids.emplace_back();
        if (!frame_pyramid.empty()) {
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
    
    // Calculate normalized frequencies for CUDA Butterworth constructor
    double fs_half = fps / 2.0;
    double Wn_low = fl / fs_half;
    double Wn_high = fh / fs_half;
    
    std::cout << "  Normalized frequencies: Wn_low=" << Wn_low << ", Wn_high=" << Wn_high << std::endl;
    
    // Create CUDA Butterworth filters (WORKING approach: two separate lowpass filters)
    cuda_evm::CudaButterworth lowpass1(0.0, Wn_low);   // Low cutoff filter
    cuda_evm::CudaButterworth lowpass2(0.0, Wn_high);  // High cutoff filter
    
    // Process each pyramid level separately (WORKING approach)
    for (int level = 0; level < pyramid_levels; level++) {
        std::cout << "  Processing pyramid level " << level << std::endl;
        
        if (pyramids[0].size() <= (size_t)level) {
            std::cout << "    Warning: Level " << level << " doesn't exist, skipping" << std::endl;
            continue;
        }
        
        cv::Mat first_level = pyramids[0][level];
        int width = first_level.cols;
        int height = first_level.rows;
        int channels = first_level.channels();
        int total_pixels = width * height * channels;
        
        std::cout << "    Level " << level << " size: " << width << "x" << height 
                  << " channels: " << channels << std::endl;
        
        // Allocate CUDA memory for this level (WORKING approach)
        float *d_input, *d_prev_input1, *d_prev_output1, *d_output1;
        float *d_prev_input2, *d_prev_output2, *d_output2;
        size_t data_size = total_pixels * sizeof(float);
        
        cudaMalloc(&d_input, data_size);
        cudaMalloc(&d_prev_input1, data_size);
        cudaMalloc(&d_prev_output1, data_size);
        cudaMalloc(&d_output1, data_size);
        cudaMalloc(&d_prev_input2, data_size);
        cudaMalloc(&d_prev_output2, data_size);
        cudaMalloc(&d_output2, data_size);
        
        // Initialize filter states with first frame data (WORKING approach)
        cv::Mat first_frame = pyramids[0][level];
        std::vector<float> first_frame_data;
        convertMatToFloat(first_frame, first_frame_data);
        
        cudaMemcpy(d_prev_input1, first_frame_data.data(), data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_prev_output1, first_frame_data.data(), data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_prev_input2, first_frame_data.data(), data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_prev_output2, first_frame_data.data(), data_size, cudaMemcpyHostToDevice);
        
        // First frame bypass (direct copy like CPU) - WORKING approach
        filtered_pyramids[0][level] = pyramids[0][level].clone();
        
        // Process frames 1+ through filter (skip frame 0) - WORKING approach
        for (int frame = 1; frame < (int)num_frames; frame++) {
            if (pyramids[frame].size() <= (size_t)level) {
                continue;
            }
            
            cv::Mat current_level = pyramids[frame][level];
            if (current_level.empty()) continue;
            
            // Convert to float vector and copy to device
            std::vector<float> input_data;
            convertMatToFloat(current_level, input_data);
            cudaMemcpy(d_input, input_data.data(), data_size, cudaMemcpyHostToDevice);
            
            // Apply both filters to the same input (WORKING approach)
            lowpass1.filter(d_input, d_prev_input1, d_prev_output1, d_output1, width, height, channels);
            lowpass2.filter(d_input, d_prev_input2, d_prev_output2, d_output2, width, height, channels);
            
            // Get both outputs and subtract them to create bandpass effect
            std::vector<float> lowpass_data(total_pixels);
            std::vector<float> highpass_data(total_pixels);
            cudaMemcpy(lowpass_data.data(), d_output1, data_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(highpass_data.data(), d_output2, data_size, cudaMemcpyDeviceToHost);
            
            // Create bandpass result: highpass - lowpass (WORKING approach)
            std::vector<float> filtered_data(total_pixels);
            for (int p = 0; p < total_pixels; ++p) {
                filtered_data[p] = highpass_data[p] - lowpass_data[p];
            }
            
            // Convert back to Mat
            cv::Mat filtered_mat;
            convertFloatToMat(filtered_data, filtered_mat, width, height);
            
            // Apply amplification (WORKING approach with chromatic attenuation)
            double attenuation = 1.0;
            if (level == pyramid_levels - 1) {
                attenuation = chrom_attenuation;
            }
            
            // Apply chromatic attenuation for I and Q channels
            std::vector<cv::Mat> channels_yiq;
            cv::split(filtered_mat, channels_yiq);
            if (channels_yiq.size() >= 3) {
                channels_yiq[1] *= attenuation; // I channel
                channels_yiq[2] *= attenuation; // Q channel
            }
            cv::merge(channels_yiq, filtered_mat);
            
            // Apply alpha scaling
            double lambda = static_cast<double>(width) / pow(2, pyramid_levels - level - 1);
            double scale_factor = (lambda > lambda_cutoff) ? alpha : 0.0;
            filtered_mat *= scale_factor;
            
            filtered_pyramids[frame][level] = filtered_mat;
            
            if (frame % 50 == 0 || frame == 1) {
                std::cout << "    Processed frame " << frame << "/" << num_frames 
                          << " level " << level << " (fixed approach)" << std::endl;
            }
        }
        
        // Clean up CUDA memory for this level
        cudaFree(d_input);
        cudaFree(d_prev_input1);
        cudaFree(d_prev_output1);
        cudaFree(d_output1);
        cudaFree(d_prev_input2);
        cudaFree(d_prev_output2);
        cudaFree(d_output2);
    }
    
    std::cout << "Finished CUDA Butterworth temporal filtering (FIXED approach)." << std::endl;
    return filtered_pyramids;
}

// Load video frames
bool loadVideoFrames(const std::string& path, std::vector<cv::Mat>& frames, double& fps) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file: " << path << std::endl;
        return false;
    }

    fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) {
        std::cerr << "Warning: Could not get valid FPS from video. Using default 30.0." << std::endl;
        fps = 30.0;
    }

    frames.clear();
    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) continue;
        
        // Convert BGR to RGB and then to float
        cv::Mat rgb_frame, float_frame;
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
        rgb_frame.convertTo(float_frame, CV_32F);
        frames.push_back(float_frame);
    }
    cap.release();

    if (frames.empty()) {
        std::cerr << "Error: No frames loaded from video: " << path << std::endl;
        return false;
    }

    std::cout << "Loaded " << frames.size() << " frames at " << fps << " FPS from " << path << std::endl;
    return true;
}

// Save video frames to file
bool saveVideoFrames(const std::string& path, const std::vector<cv::Mat>& frames, double fps) {
    if (frames.empty()) {
        std::cerr << "Error: No frames to save." << std::endl;
        return false;
    }

    cv::Size frame_size = frames[0].size();
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cv::VideoWriter writer(path, fourcc, fps, frame_size);

    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open video writer for path: " << path << std::endl;
        return false;
    }

    std::cout << "Saving " << frames.size() << " frames to " << path << "..." << std::endl;
    for (size_t i = 0; i < frames.size(); ++i) {
        cv::Mat uint8_frame, bgr_frame;
        
        // Convert float to uint8 and clamp values
        frames[i].convertTo(uint8_frame, CV_8U);
        
        // Convert RGB back to BGR for OpenCV VideoWriter
        cv::cvtColor(uint8_frame, bgr_frame, cv::COLOR_RGB2BGR);
        writer.write(bgr_frame);
    }
    writer.release();
    std::cout << "Video saved successfully." << std::endl;
    return true;
}

int main() {
    std::cout << "CUDA Butterworth + CPU Rest Pipeline Test (FIXED)" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    // Check CUDA availability
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "CUDA is not available or no devices found" << std::endl;
        return 1;
    }
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    // Configuration
    std::string input_video_path = "/home/furkan/projects/school/mmi713/evm/data/face.mp4";
    std::string output_video_path = "cuda_butter_cpu_rest_fixed_output.avi";
    int pyramid_levels = 4;
    double alpha = 50.0;
    double lambda_cutoff = 16.0;
    double fl = 0.8333;
    double fh = 1.0;
    double chrom_attenuation = 1.0;
    
    try {
        auto pipeline_start = std::chrono::high_resolution_clock::now();
        
        // 1. Load video frames
        std::vector<cv::Mat> original_frames;
        double fps = 0;
        if (!loadVideoFrames(input_video_path, original_frames, fps)) {
            return 1;
        }
        
        // Validate frequency range
        if (fh >= fps / 2.0 || fl >= fps / 2.0) {
            std::cerr << "Error: Frequency cutoffs must be less than Nyquist frequency (" 
                      << fps / 2.0 << " Hz)" << std::endl;
            return 1;
        }
        
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Input: " << input_video_path << std::endl;
        std::cout << "  Output: " << output_video_path << std::endl;
        std::cout << "  Pyramid levels: " << pyramid_levels << std::endl;
        std::cout << "  Alpha: " << alpha << std::endl;
        std::cout << "  Frequencies: " << fl << " - " << fh << " Hz" << std::endl;
        std::cout << "  FPS: " << fps << std::endl;
        
        // 2. Convert RGB to YIQ using CPU
        std::cout << "Converting RGB to YIQ (CPU)..." << std::endl;
        std::vector<cv::Mat> yiq_frames;
        yiq_frames.reserve(original_frames.size());
        
        for (size_t i = 0; i < original_frames.size(); i++) {
            cv::Mat yiq_frame = evmcpu::rgb_to_yiq(original_frames[i]);
            yiq_frames.push_back(yiq_frame);
            
            if (i % 50 == 0) {
                std::cout << "  Converted frame " << i << "/" << original_frames.size() << " to YIQ" << std::endl;
            }
        }
        
        // 3. Build Laplacian pyramids using CPU
        std::cout << "Building Laplacian pyramids (CPU)..." << std::endl;
        cv::Mat kernel = cv::getGaussianKernel(5, -1, CV_32F) * cv::getGaussianKernel(5, -1, CV_32F).t();
        std::vector<std::vector<cv::Mat>> laplacian_pyramids = getLaplacianPyramids(yiq_frames, pyramid_levels, kernel);
        
        // 4. Apply CUDA Butterworth temporal filtering (FIXED approach)
        std::vector<std::vector<cv::Mat>> filtered_pyramids = filterPyramidsWithCUDAButterworthFixed(
            laplacian_pyramids, pyramid_levels, fps, fl, fh, alpha, lambda_cutoff, chrom_attenuation);
        
        // 5. Reconstruct pyramids using CPU
        std::cout << "Reconstructing pyramids (CPU)..." << std::endl;
        std::vector<cv::Mat> output_yiq_frames;
        output_yiq_frames.reserve(yiq_frames.size());
        
        for (size_t i = 0; i < yiq_frames.size(); i++) {
            cv::Mat reconstructed = reconstructLaplacianImage(yiq_frames[i], filtered_pyramids[i], kernel);
            
            // Ensure the reconstructed frame has the correct type (CV_32FC3)
            if (reconstructed.type() != CV_32FC3) {
                std::cout << "  Warning: reconstructed frame " << i << " has type " << reconstructed.type() 
                          << " instead of CV_32FC3 (" << CV_32FC3 << "), converting..." << std::endl;
                cv::Mat temp;
                reconstructed.convertTo(temp, CV_32FC3);
                reconstructed = temp;
            }
            
            output_yiq_frames.push_back(reconstructed);
            
            if (i % 50 == 0) {
                std::cout << "  Reconstructed frame " << i << "/" << yiq_frames.size() << std::endl;
            }
        }
        
        // 6. Convert YIQ back to RGB using CPU
        std::cout << "Converting YIQ to RGB (CPU)..." << std::endl;
        std::vector<cv::Mat> output_frames;
        output_frames.reserve(output_yiq_frames.size());
        
        for (size_t i = 0; i < output_yiq_frames.size(); i++) {
            cv::Mat rgb_frame;
            evmcpu::yiq_to_rgb(output_yiq_frames[i], rgb_frame);
            output_frames.push_back(rgb_frame);
            
            if (i % 50 == 0) {
                std::cout << "  Converted frame " << i << "/" << output_yiq_frames.size() << " to RGB" << std::endl;
            }
        }
        
        // 7. Save output video
        if (!saveVideoFrames(output_video_path, output_frames, fps)) {
            return 1;
        }
        
        auto pipeline_end = std::chrono::high_resolution_clock::now();
        auto pipeline_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pipeline_end - pipeline_start);
        std::cout << "Total pipeline time: " << pipeline_duration.count() << " ms" << std::endl;
        
        std::cout << "CUDA Butterworth + CPU rest pipeline (FIXED) complete! Output saved to: " << output_video_path << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during processing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}