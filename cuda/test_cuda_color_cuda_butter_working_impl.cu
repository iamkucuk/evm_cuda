#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>

// CUDA implementations
#include "include/cuda_color_conversion.cuh"
#include "include/cuda_format_conversion.cuh"
#include "include/cuda_butterworth.cuh"  // Use working implementation

// CPU implementations for pyramid operations
#include "../cpp/include/laplacian_pyramid.hpp"
#include "../cpp/include/gaussian_pyramid.hpp"
#include "../cpp/include/processing.hpp"
#include "../cpp/include/color_conversion.hpp"

using namespace evmcpp;

// Helper function to save video frames
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
        
        // Convert RGB to BGR for OpenCV VideoWriter
        cv::cvtColor(uint8_frame, bgr_frame, cv::COLOR_RGB2BGR);
        
        writer.write(bgr_frame);
        
        if (i % 50 == 0) {
            std::cout << "  Saved frame " << i << "/" << frames.size() << std::endl;
        }
    }
    
    return true;
}

// Helper function to convert cv::Mat to device float3 array
void mat_to_device_float3(const cv::Mat& mat, float3* d_array) {
    std::vector<float3> h_array(mat.rows * mat.cols);
    const float* data = mat.ptr<float>();
    
    for (int i = 0; i < mat.rows * mat.cols; i++) {
        h_array[i].x = data[i * 3 + 0];
        h_array[i].y = data[i * 3 + 1]; 
        h_array[i].z = data[i * 3 + 2];
    }
    
    cudaMemcpy(d_array, h_array.data(), h_array.size() * sizeof(float3), cudaMemcpyHostToDevice);
}

// Helper function to convert device float3 array to cv::Mat
void device_float3_to_mat(const float3* d_array, cv::Mat& mat) {
    std::vector<float3> h_array(mat.rows * mat.cols);
    cudaMemcpy(h_array.data(), d_array, h_array.size() * sizeof(float3), cudaMemcpyDeviceToHost);
    
    float* data = mat.ptr<float>();
    for (int i = 0; i < mat.rows * mat.cols; i++) {
        data[i * 3 + 0] = h_array[i].x;
        data[i * 3 + 1] = h_array[i].y;
        data[i * 3 + 2] = h_array[i].z;
    }
}

// Helper function to convert device flat float array to cv::Mat
void device_float_to_mat(const float* d_array, cv::Mat& mat) {
    std::vector<float> h_array(mat.rows * mat.cols * mat.channels());
    cudaMemcpy(h_array.data(), d_array, h_array.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    float* data = mat.ptr<float>();
    for (int i = 0; i < mat.rows * mat.cols * mat.channels(); i++) {
        data[i] = h_array[i];
    }
}

// Helper function to convert cv::Mat to device flat float array
void mat_to_device_float(const cv::Mat& mat, float* d_array) {
    std::vector<float> h_array(mat.rows * mat.cols * mat.channels());
    const float* data = mat.ptr<float>();
    
    for (int i = 0; i < mat.rows * mat.cols * mat.channels(); i++) {
        h_array[i] = data[i];
    }
    
    cudaMemcpy(d_array, h_array.data(), h_array.size() * sizeof(float), cudaMemcpyHostToDevice);
}

// Convert Mat to flat float vector
void convertMatToFloat(const cv::Mat& mat, std::vector<float>& data) {
    data.clear();
    data.reserve(mat.rows * mat.cols * mat.channels());
    
    const float* ptr = mat.ptr<float>();
    for (int i = 0; i < mat.rows * mat.cols * mat.channels(); i++) {
        data.push_back(ptr[i]);
    }
}

// Convert flat float vector to Mat
void convertFloatToMat(const std::vector<float>& data, cv::Mat& mat, int width, int height) {
    mat = cv::Mat(height, width, CV_32FC3);
    float* ptr = mat.ptr<float>();
    
    for (size_t i = 0; i < data.size(); i++) {
        ptr[i] = data[i];
    }
}

// Convert vector of RGB frames to YIQ using CUDA
std::vector<cv::Mat> convertRGBtoYIQwithCUDA(const std::vector<cv::Mat>& rgb_frames) {
    std::cout << "Converting RGB to YIQ using CUDA..." << std::endl;
    
    if (rgb_frames.empty()) return {};
    
    int width = rgb_frames[0].cols;
    int height = rgb_frames[0].rows;
    int num_pixels = width * height;
    
    // Allocate CUDA memory
    float3* d_rgb_frame;
    float3* d_yiq_frame;
    float* d_yiq_flat;
    cudaMalloc(&d_rgb_frame, num_pixels * sizeof(float3));
    cudaMalloc(&d_yiq_frame, num_pixels * sizeof(float3));
    cudaMalloc(&d_yiq_flat, num_pixels * 3 * sizeof(float));
    
    std::vector<cv::Mat> yiq_frames;
    yiq_frames.reserve(rgb_frames.size());
    
    for (size_t i = 0; i < rgb_frames.size(); i++) {
        const cv::Mat& rgb_frame = rgb_frames[i];
        
        // Upload RGB to GPU
        mat_to_device_float3(rgb_frame, d_rgb_frame);
        
        // Convert RGB → YIQ using CUDA
        evmcuda::rgb_to_yiq(d_rgb_frame, d_yiq_frame, width, height);
        
        // Convert float3 to flat float on GPU
        evmcuda::convert_float3_to_flat(d_yiq_frame, d_yiq_flat, width, height);
        
        // Download YIQ result back to CPU
        cv::Mat yiq_frame(height, width, CV_32FC3);
        device_float_to_mat(d_yiq_flat, yiq_frame);
        
        yiq_frames.push_back(yiq_frame);
        
        if (i % 50 == 0) {
            std::cout << "  Converted frame " << i << "/" << rgb_frames.size() << " to YIQ" << std::endl;
        }
    }
    
    // Cleanup
    cudaFree(d_rgb_frame);
    cudaFree(d_yiq_frame);
    cudaFree(d_yiq_flat);
    
    std::cout << "Finished CUDA RGB to YIQ conversion." << std::endl;
    return yiq_frames;
}

// Apply WORKING CUDA Butterworth temporal filtering (dual lowpass approach)
std::vector<std::vector<cv::Mat>> filterPyramidsWithWorkingCUDAButterworth(
    const std::vector<std::vector<cv::Mat>>& pyramids,
    int pyramid_levels, 
    double fps,
    double fl, double fh,
    double alpha, double lambda_cutoff, double chrom_attenuation) {
    
    std::cout << "Applying WORKING CUDA Butterworth temporal filtering (dual lowpass)..." << std::endl;
    
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
        std::cout << "  Processing pyramid level " << level << " (WORKING dual lowpass)" << std::endl;
        
        if (pyramids[0].size() <= (size_t)level) {
            std::cout << "    Warning: Level " << level << " doesn't exist, skipping" << std::endl;
            continue;
        }
        
        cv::Mat first_level = pyramids[0][level];
        int width = first_level.cols;
        int height = first_level.rows;
        int channels = first_level.channels();
        int total_pixels = width * height * channels;
        
        // Allocate GPU memory for this level (WORKING approach)
        float *d_input;
        float *d_prev_input1, *d_prev_output1, *d_output1;
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
        
        // First frame bypass (WORKING approach)
        filtered_pyramids[0][level] = pyramids[0][level].clone();
        
        // Process frames 1 onwards through dual lowpass filters (WORKING approach)
        for (size_t frame = 1; frame < num_frames; frame++) {
            if (pyramids[frame].size() <= (size_t)level) {
                continue;
            }
            
            cv::Mat current_frame = pyramids[frame][level];
            std::vector<float> current_data;
            convertMatToFloat(current_frame, current_data);
            
            // Upload current frame data
            cudaMemcpy(d_input, current_data.data(), data_size, cudaMemcpyHostToDevice);
            
            // Apply first lowpass filter (for low frequency cutoff) 
            lowpass1.filter(d_input, d_prev_input1, d_prev_output1, d_output1, width, height, channels);
            
            // Apply second lowpass filter (for high frequency cutoff)
            lowpass2.filter(d_input, d_prev_input2, d_prev_output2, d_output2, width, height, channels);
            
            // Download both filter outputs
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
                cv::merge(channels_yiq, filtered_mat);
            }
            
            // Compute amplification factor
            double lambda = static_cast<double>(width) / pow(2, pyramid_levels - level - 1);
            double scale_factor = (lambda > lambda_cutoff) ? alpha : 0.0;
            
            filtered_mat *= scale_factor;
            filtered_pyramids[frame][level] = filtered_mat;
            
            if (frame % 50 == 0 || frame == 1) {
                std::cout << "    Processed frame " << frame << "/" << num_frames 
                          << " level " << level << " (working dual lowpass)" << std::endl;
            }
        }
        
        // Cleanup GPU memory for this level
        cudaFree(d_input);
        cudaFree(d_prev_input1);
        cudaFree(d_prev_output1);
        cudaFree(d_output1);
        cudaFree(d_prev_input2);
        cudaFree(d_prev_output2);
        cudaFree(d_output2);
    }
    
    std::cout << "Finished WORKING CUDA Butterworth temporal filtering." << std::endl;
    return filtered_pyramids;
}

int main() {
    std::cout << "CUDA Color + WORKING CUDA Butterworth Pipeline Test" << std::endl;
    std::cout << "===================================================" << std::endl;
    
    // Print CUDA device info
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // Configuration
    std::string input_video = "/home/furkan/projects/school/mmi713/evm/data/face.mp4";
    std::string output_video = "cuda_color_cuda_butter_working_impl_output.avi";
    int pyramid_levels = 4;
    double alpha = 50.0;
    double fl = 0.8333, fh = 1.0;
    double fps = 30.0;
    double lambda_cutoff = 16.0;
    double chrom_attenuation = 0.1;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Input: " << input_video << std::endl;
    std::cout << "  Output: " << output_video << std::endl;
    std::cout << "  Pyramid levels: " << pyramid_levels << std::endl;
    std::cout << "  Alpha: " << alpha << std::endl;
    std::cout << "  Frequencies: " << fl << " - " << fh << " Hz" << std::endl;
    std::cout << "  FPS: " << fps << std::endl;
    
    // Load video
    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video " << input_video << std::endl;
        return -1;
    }
    
    std::vector<cv::Mat> rgb_frames;
    cv::Mat frame;
    while (cap.read(frame) && rgb_frames.size() < 301) {
        // Convert BGR to RGB and then to float
        cv::Mat rgb_frame, float_frame;
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
        rgb_frame.convertTo(float_frame, CV_32F);
        rgb_frames.push_back(float_frame);
    }
    
    std::cout << "Loaded " << rgb_frames.size() << " frames at " << fps << " FPS from " << input_video << std::endl;
    
    // Step 1: Convert RGB to YIQ (CUDA)
    auto yiq_frames = convertRGBtoYIQwithCUDA(rgb_frames);
    
    // Step 2: Build Laplacian pyramids (CPU)
    std::cout << "Building Laplacian pyramids (CPU)..." << std::endl;
    auto pyramids = getLaplacianPyramids(yiq_frames, pyramid_levels, cv::getGaussianKernel(5, 0.83, CV_32F));
    
    // Step 3: Apply WORKING CUDA Butterworth temporal filtering
    auto filtered_pyramids = filterPyramidsWithWorkingCUDAButterworth(
        pyramids, pyramid_levels, fps, fl, fh, alpha, lambda_cutoff, chrom_attenuation);
    
    // Step 4: Reconstruct pyramids and convert back to RGB (CPU)
    std::cout << "Reconstructing pyramids (CPU)..." << std::endl;
    cv::Mat kernel = cv::getGaussianKernel(5, -1, CV_32F) * cv::getGaussianKernel(5, -1, CV_32F).t();
    std::vector<cv::Mat> output_yiq_frames;
    output_yiq_frames.reserve(yiq_frames.size());
    
    for (size_t i = 0; i < yiq_frames.size(); i++) {
        cv::Mat reconstructed = reconstructLaplacianImage(yiq_frames[i], filtered_pyramids[i], kernel);
        
        // Ensure the reconstructed frame has the correct type (CV_32FC3)
        if (reconstructed.type() != CV_32FC3) {
            cv::Mat temp;
            reconstructed.convertTo(temp, CV_32FC3);
            reconstructed = temp;
        }
        
        output_yiq_frames.push_back(reconstructed);
        
        if (i % 50 == 0) {
            std::cout << "  Reconstructed frame " << i << "/" << yiq_frames.size() << std::endl;
        }
    }
    
    // Convert YIQ back to RGB
    std::cout << "Converting YIQ to RGB (CPU)..." << std::endl;
    std::vector<cv::Mat> final_rgb_frames;
    for (size_t i = 0; i < output_yiq_frames.size(); i++) {
        cv::Mat rgb_frame;
        evmcpu::yiq_to_rgb(output_yiq_frames[i], rgb_frame);
        
        // Clamp values and convert to uint8
        cv::Mat clamped_frame;
        rgb_frame.convertTo(clamped_frame, CV_8U);
        final_rgb_frames.push_back(clamped_frame);
        
        if (i % 50 == 0) {
            std::cout << "  Converted frame " << i << "/" << output_yiq_frames.size() << " to RGB" << std::endl;
        }
    }
    
    // Write output video
    if (!saveVideoFrames(output_video, final_rgb_frames, fps)) {
        std::cerr << "Error: Failed to save output video." << std::endl;
        return 1;
    }
    
    std::cout << "Output video saved as: " << output_video << std::endl;
    
    std::cout << "Test completed successfully!" << std::endl;
    
    return 0;
}