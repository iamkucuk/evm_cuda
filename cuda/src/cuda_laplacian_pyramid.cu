#include "cuda_laplacian_pyramid.cuh"
#include "cuda_pyramid.cuh"
#include "cuda_color_conversion.cuh"
#include "cuda_butterworth.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Define logging macros for debugging
#define LOG(message) std::cout << "[CUDA LOG] " << message << std::endl
#define LOG_MATRIX(message, matrix) std::cout << "[CUDA LOG] " << message << ": Shape=" << matrix.size() << ", Channels=" << matrix.channels() << ", Type=" << matrix.type() << std::endl

namespace evmcuda {

// Helper functions for OpenCV Mat to/from vector conversions
std::vector<float> matToVector(const cv::Mat& mat) {
    std::vector<float> data(mat.rows * mat.cols * mat.channels());
    
    if (mat.isContinuous()) {
        std::memcpy(data.data(), mat.data, data.size() * sizeof(float));
    } else {
        size_t idx = 0;
        for (int i = 0; i < mat.rows; ++i) {
            const float* row_ptr = mat.ptr<float>(i);
            for (int j = 0; j < mat.cols * mat.channels(); ++j) {
                data[idx++] = row_ptr[j];
            }
        }
    }
    
    return data;
}

cv::Mat vectorToMat(const std::vector<float>& data, int width, int height, int channels) {
    cv::Mat mat(height, width, CV_32FC(channels));
    
    if (data.size() != static_cast<size_t>(width * height * channels)) {
        std::cerr << "Data size mismatch: expected " << (width * height * channels)
                  << ", got " << data.size() << std::endl;
        return mat; // Return empty mat
    }
    
    // Copy data to mat
    std::memcpy(mat.data, data.data(), data.size() * sizeof(float));
    return mat;
}

// Kernel to subtract two images (for Laplacian level construction)
__global__ void subtract_kernel(
    const float* __restrict__ d_src,
    const float* __restrict__ d_upsampled,
    float* __restrict__ d_laplacian,
    int width,
    int height,
    int channels,
    int stride)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = (y * stride + x) * channels;
    
    // Subtract upsampled from source for each channel
    for (int c = 0; c < channels; c++) {
        d_laplacian[idx + c] = d_src[idx + c] - d_upsampled[idx + c];
    }
}

// Helper function to check CUDA errors
inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        char errorMsg[256];
        snprintf(errorMsg, sizeof(errorMsg), "%s: %s", msg, cudaGetErrorString(err));
        throw std::runtime_error(errorMsg);
    }
}

// Device function to create a Laplacian level
void construct_laplacian_level_gpu(
    const float* d_src,
    int src_width,
    int src_height,
    int channels,
    const float* d_upsampled,
    float* d_laplacian,
    cudaStream_t stream)
{
    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((src_width + blockSize.x - 1) / blockSize.x, 
                 (src_height + blockSize.y - 1) / blockSize.y);
    
    // Launch the subtraction kernel
    subtract_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_src, d_upsampled, d_laplacian, src_width, src_height, channels, src_width);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Failed to launch subtract_kernel in construct_laplacian_level_gpu");
}

// Generate a Laplacian pyramid for a single image
std::vector<cv::Mat> generate_laplacian_pyramid(
    const cv::Mat& input_image,
    int level)
{
    LOG("generate_laplacian_pyramid: Input image shape=[" + 
        std::to_string(input_image.rows) + ", " + 
        std::to_string(input_image.cols) + "], levels=" + 
        std::to_string(level));
    
    if (level <= 0) {
        throw std::invalid_argument("Pyramid level must be positive.");
    }
    if (input_image.empty()) {
        throw std::invalid_argument("Input image is empty.");
    }
    
    // Ensure input type is float for calculations
    cv::Mat float_input;
    if (input_image.type() != CV_32FC3) {
        LOG("generate_laplacian_pyramid: Warning - Input image type is not CV_32FC3. Converting.");
        input_image.convertTo(float_input, CV_32FC3);
    } else {
        float_input = input_image.clone();
    }
    
    std::vector<cv::Mat> laplacian_pyramid;
    laplacian_pyramid.reserve(level + 1); // Reserve space for levels plus residual
    
    // Initialize CUDA modules needed
    if (!init_pyramid()) {
        throw std::runtime_error("Failed to initialize CUDA pyramid module");
    }
    
    cv::Mat prev_image = float_input; // Start with float input
    
    for (int i = 0; i < level; ++i) {
        LOG_MATRIX("generate_laplacian_pyramid: Level " + std::to_string(i) + " prev_image", prev_image);
        
        // Step 1: Downsampling
        cv::Mat downsampled_image = pyr_down(prev_image);
        if (downsampled_image.empty() && !prev_image.empty()) {
            throw std::runtime_error("pyrDown resulted in empty image at level " + std::to_string(i));
        }
        LOG_MATRIX("generate_laplacian_pyramid: Level " + std::to_string(i) + " downsampled_image", downsampled_image);
        
        // Step 2: Upsampling
        cv::Mat upsampled_image = pyr_up(downsampled_image, prev_image.size());
        if (upsampled_image.empty() && !downsampled_image.empty()) {
            throw std::runtime_error("pyrUp resulted in empty image at level " + std::to_string(i));
        }
        LOG_MATRIX("generate_laplacian_pyramid: Level " + std::to_string(i) + " upsampled_image", upsampled_image);
        
        // Step 3: Calculate Laplacian Level
        cv::Mat laplacian_level;
        cv::subtract(prev_image, upsampled_image, laplacian_level);
        LOG_MATRIX("generate_laplacian_pyramid: Level " + std::to_string(i) + " laplacian_level", laplacian_level);
        
        laplacian_pyramid.push_back(laplacian_level);
        prev_image = downsampled_image;
        
        if (prev_image.rows <= 1 || prev_image.cols <= 1) {
            LOG("generate_laplacian_pyramid: Stopping early at level " + std::to_string(i) + " due to small image size.");
            break;
        }
    }
    
    // Add the final residual Gaussian level
    if (!prev_image.empty()) {
        laplacian_pyramid.push_back(prev_image);
        LOG("generate_laplacian_pyramid: Added final residual. New size=" + std::to_string(laplacian_pyramid.size()));
    } else {
        LOG("generate_laplacian_pyramid: Warning - Final residual image was empty. Not added.");
    }
    
    return laplacian_pyramid;
}

// Generate Laplacian pyramids for a sequence of images
std::vector<std::vector<cv::Mat>> get_laplacian_pyramids(
    const std::vector<cv::Mat>& input_images,
    int level)
{
    LOG("get_laplacian_pyramids: Processing " + std::to_string(input_images.size()) + " frames.");
    std::vector<std::vector<cv::Mat>> laplacian_pyramids_batch;
    laplacian_pyramids_batch.reserve(input_images.size());
    
    // Initialize CUDA modules needed
    if (!init_color_conversion()) {
        throw std::runtime_error("Failed to initialize CUDA color conversion module");
    }
    
    int frame_idx = 0;
    for (const auto& image : input_images) {
        if (image.empty()) {
            LOG("get_laplacian_pyramids: Warning - Skipping empty input frame " + std::to_string(frame_idx));
            laplacian_pyramids_batch.emplace_back(); // Add empty vector
            frame_idx++;
            continue;
        }
        
        // Convert RGB to YIQ
        cv::Mat yiq_image;
        if (image.depth() == CV_32F) {
            // Use CUDA implementation directly
            cv::Mat floatImage = image.clone();
            // Use the CUDA implementation through the wrapper function
            std::vector<float> rgb_data = matToVector(floatImage);
            std::vector<float> yiq_data(rgb_data.size());
            rgb_to_yiq_wrapper(rgb_data.data(), yiq_data.data(), image.cols, image.rows);
            yiq_image = vectorToMat(yiq_data, image.cols, image.rows, 3);
        } else {
            // Convert to float first
            cv::Mat float_image;
            image.convertTo(float_image, CV_32FC3);
            // Use the CUDA implementation through the wrapper function
            std::vector<float> rgb_data = matToVector(float_image);
            std::vector<float> yiq_data(rgb_data.size());
            rgb_to_yiq_wrapper(rgb_data.data(), yiq_data.data(), float_image.cols, float_image.rows);
            yiq_image = vectorToMat(yiq_data, float_image.cols, float_image.rows, 3);
        }
        LOG_MATRIX("get_laplacian_pyramids: Frame " + std::to_string(frame_idx) + " yiq_image", yiq_image);
        
        // Generate Laplacian pyramid
        std::vector<cv::Mat> laplacian_pyramid = generate_laplacian_pyramid(yiq_image, level);
        LOG("get_laplacian_pyramids: Frame " + std::to_string(frame_idx) + " generated pyramid size=" + 
            std::to_string(laplacian_pyramid.size()));
        
        laplacian_pyramids_batch.push_back(std::move(laplacian_pyramid));
        frame_idx++;
    }
    
    LOG("get_laplacian_pyramids: Finished. Batch size=" + std::to_string(laplacian_pyramids_batch.size()));
    
    return laplacian_pyramids_batch;
}

// Apply spatial attenuation based on lambda_cutoff and current level
double computeAlphaForLevel(int width, int height, double alpha, double lambda_cutoff) {
    // Estimate lambda for the current pyramid level
    double lambda = estimate_lambda(width, height);
    
    // Compute delta based on lambda_cutoff
    double delta = lambda_cutoff / 8.0;
    
    // Compute adjusted alpha
    return compute_alpha(lambda, delta, alpha);
}

// Kernel to apply spatial attenuation
__global__ void apply_spatial_attenuation_kernel(
    float* __restrict__ d_bandpass_result,
    int width,
    int height,
    int channels,
    int stride,
    float current_alpha,
    float chrom_attenuation)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const int idx = (y * stride + x) * channels;
    
    // Apply alpha scaling to the Y component (luminance)
    d_bandpass_result[idx] *= current_alpha;
    
    // Apply additional attenuation to chrominance channels (I and Q)
    if (channels > 1) {
        d_bandpass_result[idx + 1] *= current_alpha * chrom_attenuation;
    }
    if (channels > 2) {
        d_bandpass_result[idx + 2] *= current_alpha * chrom_attenuation;
    }
}

// Filter Laplacian pyramids (apply temporal bandpass and amplification)
std::vector<std::vector<cv::Mat>> filter_laplacian_pyramids(
    const std::vector<std::vector<cv::Mat>>& pyramids_batch,
    int level,
    double fps,
    const std::pair<double, double>& freq_range,
    double alpha,
    double lambda_cutoff,
    double chrom_attenuation)
{
    LOG("filter_laplacian_pyramids: Processing " + std::to_string(pyramids_batch.size()) + " frames");
    
    if (pyramids_batch.empty()) {
        throw std::invalid_argument("Pyramid batch is empty.");
    }
    
    if (freq_range.first <= 0.0 || freq_range.second <= 0.0 || freq_range.first >= freq_range.second) {
        throw std::invalid_argument("Invalid frequency range. Must be 0 < fl < fh.");
    }
    
    // Initialize modules
    if (!init_butterworth()) {
        throw std::runtime_error("Failed to initialize CUDA Butterworth module");
    }
    
    // Create butterworth filter
    Butterworth butter_filter(freq_range.first / (fps/2), freq_range.second / (fps/2));
    
    std::vector<std::vector<cv::Mat>> filtered_pyramids_batch;
    filtered_pyramids_batch.reserve(pyramids_batch.size());
    
    // For each level of the pyramid:
    for (int i = 0; i < level; ++i) {
        if (i >= pyramids_batch[0].size()) {
            LOG("filter_laplacian_pyramids: Skipping level " + std::to_string(i) + " as it's out of bounds.");
            continue;
        }
        
        LOG("filter_laplacian_pyramids: Processing level " + std::to_string(i));
        
        // Get dimensions from first frame
        int width = pyramids_batch[0][i].cols;
        int height = pyramids_batch[0][i].rows;
        int channels = pyramids_batch[0][i].channels();
        size_t frameSize = width * height * channels;
        
        // Calculate alpha for this level based on lambda
        double levelAlpha = computeAlphaForLevel(width, height, alpha, lambda_cutoff);
        LOG("filter_laplacian_pyramids: Level " + std::to_string(i) + " original alpha=" + std::to_string(alpha) + 
            ", calculated alpha=" + std::to_string(levelAlpha));
        
        // Allocate device memory for this level's batch processing
        float* d_input = nullptr;
        float* d_prev_input = nullptr;
        float* d_prev_output = nullptr;
        float* d_output = nullptr;
        float* d_filtered = nullptr;
        
        cudaError_t err;
        err = cudaMalloc(&d_input, frameSize * sizeof(float));
        checkCudaError(err, "Failed to allocate device memory for filter input");
        
        err = cudaMalloc(&d_prev_input, frameSize * sizeof(float));
        checkCudaError(err, "Failed to allocate device memory for filter prev_input");
        
        err = cudaMalloc(&d_prev_output, frameSize * sizeof(float));
        checkCudaError(err, "Failed to allocate device memory for filter prev_output");
        
        err = cudaMalloc(&d_output, frameSize * sizeof(float));
        checkCudaError(err, "Failed to allocate device memory for filter output");
        
        err = cudaMalloc(&d_filtered, frameSize * sizeof(float));
        checkCudaError(err, "Failed to allocate device memory for filtered results");
        
        // Apply Butterworth filter frame by frame
        for (size_t frame_idx = 0; frame_idx < pyramids_batch.size(); ++frame_idx) {
            // Check if this frame has enough pyramid levels
            if (i >= pyramids_batch[frame_idx].size()) {
                LOG("filter_laplacian_pyramids: Frame " + std::to_string(frame_idx) + 
                    " does not have level " + std::to_string(i));
                continue;
            }
            
            const cv::Mat& curr_level = pyramids_batch[frame_idx][i];
            
            // Ensure we are dealing with the expected dimensions
            if (curr_level.cols != width || curr_level.rows != height || curr_level.channels() != channels) {
                LOG("filter_laplacian_pyramids: Inconsistent dimensions at frame " + std::to_string(frame_idx) + 
                    ", level " + std::to_string(i));
                continue;
            }
            
            // Copy input data to device
            err = cudaMemcpy(d_input, curr_level.data, frameSize * sizeof(float), cudaMemcpyHostToDevice);
            checkCudaError(err, "Failed to copy input data to device");
            
            // Apply Butterworth filter
            butter_filter.filter(d_input, width, height, channels, d_prev_input, d_prev_output, d_output);
            
            // Apply spatial attenuation based on level
            dim3 blockSize(16, 16);
            dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
            
            float curr_alpha = static_cast<float>(levelAlpha);
            float chrom_atten = static_cast<float>(chrom_attenuation);
            
            apply_spatial_attenuation_kernel<<<gridSize, blockSize>>>(
                d_output, width, height, channels, width, curr_alpha, chrom_atten);
            
            err = cudaGetLastError();
            checkCudaError(err, "Failed to launch apply_spatial_attenuation_kernel");
            
            // Copy filtered and attenuated result to host
            cv::Mat filtered_level(height, width, CV_32FC(channels));
            err = cudaMemcpy(filtered_level.data, d_output, frameSize * sizeof(float), cudaMemcpyDeviceToHost);
            checkCudaError(err, "Failed to copy filtered output to host");
            
            // Update filter states for next frame
            err = cudaMemcpy(d_prev_input, d_input, frameSize * sizeof(float), cudaMemcpyDeviceToDevice);
            checkCudaError(err, "Failed to update prev_input state");
            
            err = cudaMemcpy(d_prev_output, d_output, frameSize * sizeof(float), cudaMemcpyDeviceToDevice);
            checkCudaError(err, "Failed to update prev_output state");
            
            // Add filtered level to this frame's pyramid
            if (frame_idx >= filtered_pyramids_batch.size()) {
                // Initialize with empty pyramid if not already exists
                std::vector<cv::Mat> new_pyramid;
                new_pyramid.reserve(level);
                
                // Add empty levels up to current level
                for (int j = 0; j < i; ++j) {
                    new_pyramid.push_back(cv::Mat());
                }
                
                // Add filtered level
                new_pyramid.push_back(filtered_level);
                
                filtered_pyramids_batch.push_back(new_pyramid);
            } else {
                // Make sure we have enough levels
                while (filtered_pyramids_batch[frame_idx].size() <= static_cast<size_t>(i)) {
                    filtered_pyramids_batch[frame_idx].push_back(cv::Mat());
                }
                
                // Set filtered level
                filtered_pyramids_batch[frame_idx][i] = filtered_level;
            }
        }
        
        // Free device memory for this level
        cudaFree(d_input);
        cudaFree(d_prev_input);
        cudaFree(d_prev_output);
        cudaFree(d_output);
        cudaFree(d_filtered);
    }
    
    // Add residual Gaussian level from original pyramids
    for (size_t frame_idx = 0; frame_idx < pyramids_batch.size(); ++frame_idx) {
        if (frame_idx >= filtered_pyramids_batch.size()) {
            // Create new pyramid if needed
            std::vector<cv::Mat> new_pyramid;
            if (!pyramids_batch[frame_idx].empty()) {
                // Add residual level
                new_pyramid.push_back(pyramids_batch[frame_idx].back().clone());
            }
            filtered_pyramids_batch.push_back(new_pyramid);
        } else if (!pyramids_batch[frame_idx].empty() && 
                  pyramids_batch[frame_idx].size() > filtered_pyramids_batch[frame_idx].size()) {
            // Add residual level
            filtered_pyramids_batch[frame_idx].push_back(pyramids_batch[frame_idx].back().clone());
        }
    }
    
    return filtered_pyramids_batch;
}

// Reconstruct image from Laplacian pyramid
cv::Mat reconstruct_laplacian_image(
    const cv::Mat& original_rgb_image,
    const std::vector<cv::Mat>& filtered_pyramid)
{
    LOG("reconstruct_laplacian_image: Starting reconstruction from pyramid size=" + 
        std::to_string(filtered_pyramid.size()));
    
    if (filtered_pyramid.empty()) {
        LOG("reconstruct_laplacian_image: Warning - Empty pyramid. Returning original image.");
        return original_rgb_image.clone();
    }
    
    // Convert RGB to YIQ for reconstruction
    cv::Mat rgb_float;
    original_rgb_image.convertTo(rgb_float, CV_32FC3);
    std::vector<float> rgb_data = matToVector(rgb_float);
    std::vector<float> yiq_data(rgb_data.size());
    rgb_to_yiq_wrapper(rgb_data.data(), yiq_data.data(), original_rgb_image.cols, original_rgb_image.rows);
    cv::Mat reconstructed_yiq = vectorToMat(yiq_data, original_rgb_image.cols, original_rgb_image.rows, 3);
    
    LOG_MATRIX("reconstruct_laplacian_image: original_yiq", reconstructed_yiq);
    
    // Start with the smallest level (residual Gaussian)
    cv::Mat reconstructed = filtered_pyramid.back().clone();
    LOG_MATRIX("reconstruct_laplacian_image: starting with residual level", reconstructed);
    
    // Reconstruct the image by upsampling and adding levels
    for (int i = filtered_pyramid.size() - 2; i >= 0; --i) {
        if (filtered_pyramid[i].empty()) {
            LOG("reconstruct_laplacian_image: Warning - Skipping empty level " + std::to_string(i));
            continue;
        }
        
        LOG_MATRIX("reconstruct_laplacian_image: level " + std::to_string(i), filtered_pyramid[i]);
        
        // Upscale the current result to match the next level size
        cv::Mat upsampled = pyr_up(reconstructed, filtered_pyramid[i].size());
        LOG_MATRIX("reconstruct_laplacian_image: upsampled", upsampled);
        
        // Add the Laplacian level
        cv::add(upsampled, filtered_pyramid[i], reconstructed);
        LOG_MATRIX("reconstruct_laplacian_image: after adding level " + std::to_string(i), reconstructed);
    }
    
    LOG_MATRIX("reconstruct_laplacian_image: final YIQ result", reconstructed);
    
    // Replace the Y channel in the original YIQ image with the reconstructed Y channel
    if (reconstructed.channels() == 3 && reconstructed_yiq.channels() == 3) {
        for (int y = 0; y < reconstructed.rows; ++y) {
            for (int x = 0; x < reconstructed.cols; ++x) {
                cv::Vec3f& pixel = reconstructed_yiq.at<cv::Vec3f>(y, x);
                const cv::Vec3f& rec_pixel = reconstructed.at<cv::Vec3f>(y, x);
                
                // Replace Y channel only
                pixel[0] = rec_pixel[0];
            }
        }
        LOG("reconstruct_laplacian_image: Replaced Y channel in original YIQ image with reconstructed Y channel");
    } else {
        // Use fully reconstructed image if channels don't match
        reconstructed_yiq = reconstructed;
        LOG("reconstruct_laplacian_image: Using fully reconstructed image");
    }
    
    // Convert back to RGB
    std::vector<float> reconstructed_yiq_data = matToVector(reconstructed_yiq);
    std::vector<float> reconstructed_rgb_data(reconstructed_yiq_data.size());
    yiq_to_rgb_wrapper(reconstructed_yiq_data.data(), reconstructed_rgb_data.data(), reconstructed_yiq.cols, reconstructed_yiq.rows);
    cv::Mat reconstructed_rgb_float = vectorToMat(reconstructed_rgb_data, reconstructed_yiq.cols, reconstructed_yiq.rows, 3);
    
    LOG_MATRIX("reconstruct_laplacian_image: final RGB float result", reconstructed_rgb_float);
    
    // Clip values to [0,255] range
    cv::Mat clipped;
    cv::max(cv::min(reconstructed_rgb_float, 255.0f), 0.0f, clipped);
    
    // Convert to 8-bit for output
    cv::Mat result_uint8;
    clipped.convertTo(result_uint8, CV_8UC3);
    
    LOG_MATRIX("reconstruct_laplacian_image: final 8-bit RGB result", result_uint8);
    
    return result_uint8;
}

// Initialize Laplacian pyramid module
bool init_laplacian_pyramid() {
    LOG("init_laplacian_pyramid: Initializing Laplacian pyramid module");
    
    // Ensure dependent modules are initialized
    if (!init_pyramid()) {
        LOG("init_laplacian_pyramid: Failed to initialize pyramid module");
        return false;
    }
    
    if (!init_color_conversion()) {
        LOG("init_laplacian_pyramid: Failed to initialize color conversion module");
        return false;
    }
    
    if (!init_butterworth()) {
        LOG("init_laplacian_pyramid: Failed to initialize Butterworth module");
        return false;
    }
    
    LOG("init_laplacian_pyramid: Initialization complete");
    return true;
}

// Cleanup Laplacian pyramid module
void cleanup_laplacian_pyramid() {
    LOG("cleanup_laplacian_pyramid: Cleaning up Laplacian pyramid module");
    
    // Clean up dependent modules
    cleanup_pyramid();
    cleanup_color_conversion();
    cleanup_butterworth();
    
    LOG("cleanup_laplacian_pyramid: Cleanup complete");
}

} // namespace evmcuda