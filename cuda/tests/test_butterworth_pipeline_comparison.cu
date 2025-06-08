#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Include CUDA Butterworth implementation
#include "../include/cuda_butterworth.cuh"

// Include CPU implementations we need
#include "../../cpp/include/processing.hpp"
#include "../../cpp/include/laplacian_pyramid.hpp"
#include "../../cpp/include/color_conversion.hpp"

// Helper function to convert cv::Mat to CUDA-compatible format
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

// Hybrid filtering function that uses CUDA Butterworth instead of CPU Butterworth
std::vector<std::vector<cv::Mat>> filterLaplacianPyramidsWithCudaButterworth(
    const std::vector<std::vector<cv::Mat>>& pyramids_batch,
    int level,
    double fps,
    const std::pair<double, double>& freq_range,
    double alpha,
    double lambda_cutoff,
    double attenuation)
{
    std::cout << "Starting CUDA Butterworth filtering process..." << std::endl;
    
    size_t num_frames = pyramids_batch.size();
    if (num_frames <= 1 || level <= 0) {
        std::cout << "Not enough frames or invalid level, returning input." << std::endl;
        return pyramids_batch;
    }

    // Initialize output structure
    std::vector<std::vector<cv::Mat>> filtered_pyramids;
    filtered_pyramids.reserve(num_frames);
    size_t actual_levels = 0;
    
    for(const auto& frame_pyramid : pyramids_batch) {
        filtered_pyramids.emplace_back();
        if (!frame_pyramid.empty()) {
            if (actual_levels == 0) actual_levels = frame_pyramid.size();
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
        std::cout << "Input contains no valid pyramids. Returning input." << std::endl;
        return pyramids_batch;
    }
    
    if (level != static_cast<int>(actual_levels)) {
        std::cout << "Warning: 'level' parameter does not match actual levels found. Using actual levels." << std::endl;
        level = static_cast<int>(actual_levels);
    }

    double delta = lambda_cutoff / (8.0 * (1.0 + alpha));
    double low_freq = freq_range.first;
    double high_freq = freq_range.second;

    // Create CUDA Butterworth filters
    std::cout << "Creating CUDA Butterworth filters with fl=" << low_freq << ", fh=" << high_freq << " Hz" << std::endl;
    
    // Calculate normalized frequencies for CUDA Butterworth constructor
    double fs_half = fps / 2.0;
    double Wn_low = low_freq / fs_half;
    double Wn_high = high_freq / fs_half;
    
    cuda_evm::CudaButterworth lowpass1(0.0, Wn_low);   // Low cutoff filter
    cuda_evm::CudaButterworth lowpass2(0.0, Wn_high);  // High cutoff filter

    // Process each pyramid level
    for (int curr_level = 0; curr_level < level; ++curr_level) {
        std::cout << "Processing level " << curr_level << "..." << std::endl;
        
        if (pyramids_batch[0].size() <= curr_level) continue;
        
        cv::Mat sample_mat = pyramids_batch[0][curr_level];
        if (sample_mat.empty()) continue;
        
        int width = sample_mat.cols;
        int height = sample_mat.rows; 
        int channels = sample_mat.channels();
        int total_pixels = width * height * channels;
        
        // Allocate CUDA memory for this level
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
        
        // Initialize state memories to zero
        cudaMemset(d_prev_input1, 0, data_size);
        cudaMemset(d_prev_output1, 0, data_size);
        cudaMemset(d_prev_input2, 0, data_size);
        cudaMemset(d_prev_output2, 0, data_size);
        
        // Process each frame for this level
        for (size_t frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
            if (pyramids_batch[frame_idx].size() <= curr_level) continue;
            
            cv::Mat current_level = pyramids_batch[frame_idx][curr_level];
            if (current_level.empty()) continue;
            
            // Convert to float vector and copy to device
            std::vector<float> input_data;
            convertMatToFloat(current_level, input_data);
            cudaMemcpy(d_input, input_data.data(), data_size, cudaMemcpyHostToDevice);
            
            // Apply both filters to the same input (like CPU implementation)
            lowpass1.filter(d_input, d_prev_input1, d_prev_output1, d_output1, width, height, channels);
            lowpass2.filter(d_input, d_prev_input2, d_prev_output2, d_output2, width, height, channels);
            
            // Get both outputs and subtract them to create bandpass effect
            std::vector<float> lowpass_data(total_pixels);
            std::vector<float> highpass_data(total_pixels);
            cudaMemcpy(lowpass_data.data(), d_output1, data_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(highpass_data.data(), d_output2, data_size, cudaMemcpyDeviceToHost);
            
            // Create bandpass result: highpass - lowpass (like CPU)
            std::vector<float> filtered_data(total_pixels);
            for (int p = 0; p < total_pixels; ++p) {
                filtered_data[p] = highpass_data[p] - lowpass_data[p];
            }
            
            // Apply chromatic attenuation and temporal scaling similar to CPU implementation
            cv::Mat filtered_mat;
            convertFloatToMat(filtered_data, filtered_mat, width, height);
            
            // Apply scaling factors similar to CPU implementation
            double curr_alpha = alpha;
            
            // Apply chromatic attenuation for I and Q channels
            std::vector<cv::Mat> channels_yiq;
            cv::split(filtered_mat, channels_yiq);
            if (channels_yiq.size() >= 3) {
                channels_yiq[1] *= attenuation; // I channel
                channels_yiq[2] *= attenuation; // Q channel
            }
            cv::merge(channels_yiq, filtered_mat);
            
            // Apply alpha scaling
            filtered_mat *= curr_alpha;
            
            filtered_pyramids[frame_idx][curr_level] = filtered_mat;
            
            if (frame_idx % 50 == 0) {
                std::cout << "  Processed frame " << frame_idx << "/" << num_frames << " for level " << curr_level << std::endl;
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
    
    std::cout << "Finished CUDA Butterworth filtering." << std::endl;
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
        cv::Mat rgb_frame;
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
        frames.push_back(rgb_frame);
    }
    cap.release();

    if (frames.empty()) {
        std::cerr << "Error: No frames loaded from video: " << path << std::endl;
        return false;
    }

    std::cout << "Loaded " << frames.size() << " frames at " << fps << " FPS from " << path << std::endl;
    return true;
}

// Save video frames  
bool saveVideoFrames(const std::string& path, const std::vector<cv::Mat>& frames, double fps, const cv::Size& frame_size) {
    if (frames.empty()) {
        std::cerr << "Error: No frames to save." << std::endl;
        return false;
    }

    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cv::VideoWriter writer(path, fourcc, fps, frame_size);

    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open video writer for path: " << path << std::endl;
        return false;
    }

    std::cout << "Saving " << frames.size() << " frames to " << path << "..." << std::endl;
    for (size_t i = 0; i < frames.size(); ++i) {
        if (frames[i].empty() || frames[i].size() != frame_size) {
            std::cerr << "Warning: Skipping invalid frame " << i << " during save." << std::endl;
            continue;
        }
        cv::Mat bgr_frame;
        cv::cvtColor(frames[i], bgr_frame, cv::COLOR_RGB2BGR);
        writer.write(bgr_frame);
    }
    writer.release();
    std::cout << "Video saved successfully." << std::endl;
    return true;
}

int main() {
    std::cout << "=== CUDA Butterworth Pipeline Test ===" << std::endl;
    
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA initialization failed!" << std::endl;
        return -1;
    }
    
    // Configuration - matching CPU pipeline
    std::string input_video = "../../data/face.mp4";
    std::string output_video = "cuda_butterworth_laplacian_output.avi";
    int pyramid_levels = 4;
    double alpha = 50.0;
    double lambda_cutoff = 16.0;
    double fl = 0.8333;
    double fh = 1.0;
    double chrom_attenuation = 1.0;
    
    std::cout << "Input: " << input_video << std::endl;
    std::cout << "Output: " << output_video << std::endl;
    std::cout << "Pyramid Levels: " << pyramid_levels << std::endl;
    std::cout << "Alpha: " << alpha << std::endl;
    std::cout << "Freq Range: " << fl << " - " << fh << " Hz" << std::endl;
    
    try {
        // 1. Load Video
        std::vector<cv::Mat> original_frames;
        double fps = 0;
        if (!loadVideoFrames(input_video, original_frames, fps)) {
            return 1;
        }
        
        // Check frequency range
        if (fh >= fps / 2.0 || fl >= fps / 2.0) {
            std::cerr << "Error: Frequency cutoffs exceed Nyquist frequency" << std::endl;
            return 1;
        }
        
        // 2. Build Laplacian Pyramids (using CPU code)
        std::cout << "Building Laplacian pyramids..." << std::endl;
        cv::Mat kernel = cv::getGaussianKernel(5, -1, CV_32F);
        kernel = kernel * kernel.t();
        
        std::vector<std::vector<cv::Mat>> laplacian_pyramids = evmcpp::getLaplacianPyramids(original_frames, pyramid_levels, kernel);
        std::cout << "Finished building Laplacian pyramids." << std::endl;
        
        // 3. Filter with CUDA Butterworth
        std::cout << "Filtering with CUDA Butterworth..." << std::endl;
        std::vector<std::vector<cv::Mat>> filtered_pyramids = filterLaplacianPyramidsWithCudaButterworth(
            laplacian_pyramids,
            pyramid_levels,
            fps,
            {fl, fh},
            alpha,
            lambda_cutoff,
            chrom_attenuation
        );
        std::cout << "Finished CUDA Butterworth filtering." << std::endl;
        
        // 4. Reconstruct frames (using CPU code)
        std::cout << "Reconstructing frames..." << std::endl;
        std::vector<cv::Mat> output_frames;
        output_frames.reserve(original_frames.size());
        
        for (size_t i = 0; i < original_frames.size(); ++i) {
            if (i >= filtered_pyramids.size() || original_frames[i].empty() || filtered_pyramids[i].empty()) {
                output_frames.push_back(original_frames[i].clone());
                continue;
            }
            
            output_frames.push_back(evmcpp::reconstructLaplacianImage(original_frames[i], filtered_pyramids[i], kernel));
            
            if (i > 0 && i % 50 == 0) {
                std::cout << "  Reconstructed frame " << i << "/" << original_frames.size() << std::endl;
            }
        }
        std::cout << "Finished reconstruction." << std::endl;
        
        // 5. Save output
        if (!output_frames.empty() && !output_frames[0].empty()) {
            cv::Size frame_size = output_frames[0].size();
            if (!saveVideoFrames(output_video, output_frames, fps, frame_size)) {
                return 1;
            }
        } else {
            std::cerr << "Error: No valid output frames generated." << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error during processing: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "CUDA Butterworth pipeline completed successfully!" << std::endl;
    std::cout << "Output saved to: " << output_video << std::endl;
    
    return 0;
}