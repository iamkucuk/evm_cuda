#include "cuda_gaussian_pyramid.cuh"
#include "cuda_color_conversion.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>

// CPU reference implementations for comparison
#include "../cpp/include/gaussian_pyramid.hpp"
#include "../cpp/include/processing.hpp"

void test_spatial_filtering() {
    std::cout << "\n=== Testing CUDA Spatial Filtering ===\n";
    
    // Load test image
    cv::Mat test_image = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::randu(test_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    
    const int level = 2;
    
    // CPU reference
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cv::Mat cpu_result = evmcpp::spatiallyFilterGaussian(test_image, level, evmcpp::gaussian_kernel);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count();
    
    // CUDA implementation
    auto start_cuda = std::chrono::high_resolution_clock::now();
    cv::Mat cuda_result = cuda_evm::spatially_filter_gaussian_wrapper(test_image, level);
    auto end_cuda = std::chrono::high_resolution_clock::now();
    auto cuda_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_cuda - start_cuda).count();
    
    if (cpu_result.empty() || cuda_result.empty()) {
        std::cout << "❌ FAILED: One of the implementations returned empty result" << std::endl;
        return;
    }
    
    // Compare results
    cv::Mat diff;
    cv::absdiff(cpu_result, cuda_result, diff);
    cv::Scalar mean_diff = cv::mean(diff);
    double max_diff;
    cv::minMaxLoc(diff, nullptr, &max_diff);
    
    // Calculate PSNR
    double mse = cv::sum(diff.mul(diff))[0] / (diff.rows * diff.cols * diff.channels());
    double psnr = 20 * log10(255.0 / sqrt(mse));
    
    std::cout << "Results:" << std::endl;
    std::cout << "  CPU Time: " << cpu_time << " ms" << std::endl;
    std::cout << "  CUDA Time: " << cuda_time << " ms" << std::endl;
    std::cout << "  Mean absolute difference: " << mean_diff[0] << std::endl;
    std::cout << "  Max absolute difference: " << max_diff << std::endl;
    std::cout << "  PSNR: " << psnr << " dB" << std::endl;
    
    if (psnr > 30.0) {
        std::cout << "✅ PASSED: Spatial filtering achieves >30 dB PSNR" << std::endl;
    } else {
        std::cout << "❌ FAILED: Spatial filtering PSNR too low (" << psnr << " dB)" << std::endl;
    }
}

void test_temporal_filtering() {
    std::cout << "\n=== Testing CUDA Temporal Filtering ===\n";
    
    // Create test batch of small frames
    const int width = 50;
    const int height = 50;
    const int num_frames = 64; // Power of 2 for FFT
    const float fps = 30.0f;
    const float fl = 0.5f;
    const float fh = 2.0f;
    const float alpha = 10.0f;
    const float chrom_attenuation = 0.1f;
    
    std::vector<cv::Mat> test_batch(num_frames);
    for (int i = 0; i < num_frames; i++) {
        test_batch[i] = cv::Mat::zeros(height, width, CV_32FC3);
        cv::randu(test_batch[i], cv::Scalar(-10, -10, -10), cv::Scalar(10, 10, 10));
    }
    
    // CPU reference
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::vector<cv::Mat> cpu_result = evmcpp::temporalFilterGaussianBatch(
        test_batch, fps, fl, fh, alpha, chrom_attenuation);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count();
    
    // CUDA implementation
    auto start_cuda = std::chrono::high_resolution_clock::now();
    std::vector<cv::Mat> cuda_result = cuda_evm::temporal_filter_gaussian_batch_wrapper(
        test_batch, fps, fl, fh, alpha, chrom_attenuation);
    auto end_cuda = std::chrono::high_resolution_clock::now();
    auto cuda_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_cuda - start_cuda).count();
    
    if (cpu_result.empty() || cuda_result.empty()) {
        std::cout << "❌ FAILED: One of the implementations returned empty result" << std::endl;
        return;
    }
    
    if (cpu_result.size() != cuda_result.size()) {
        std::cout << "❌ FAILED: Result sizes don't match" << std::endl;
        return;
    }
    
    // Compare results across all frames
    double total_psnr = 0.0;
    double total_mean_diff = 0.0;
    double total_max_diff = 0.0;
    
    for (size_t i = 0; i < cpu_result.size(); i++) {
        cv::Mat diff;
        cv::absdiff(cpu_result[i], cuda_result[i], diff);
        cv::Scalar mean_diff = cv::mean(diff);
        double max_diff;
        cv::minMaxLoc(diff, nullptr, &max_diff);
        
        double mse = cv::sum(diff.mul(diff))[0] / (diff.rows * diff.cols * diff.channels());
        double psnr = (mse > 0) ? 20 * log10(255.0 / sqrt(mse)) : 100.0;
        
        total_psnr += psnr;
        total_mean_diff += mean_diff[0];
        total_max_diff = std::max(total_max_diff, max_diff);
    }
    
    double avg_psnr = total_psnr / cpu_result.size();
    double avg_mean_diff = total_mean_diff / cpu_result.size();
    
    std::cout << "Results:" << std::endl;
    std::cout << "  CPU Time: " << cpu_time << " ms" << std::endl;
    std::cout << "  CUDA Time: " << cuda_time << " ms" << std::endl;
    std::cout << "  Average PSNR: " << avg_psnr << " dB" << std::endl;
    std::cout << "  Average mean difference: " << avg_mean_diff << std::endl;
    std::cout << "  Max difference across all frames: " << total_max_diff << std::endl;
    
    if (avg_psnr > 20.0) {
        std::cout << "✅ PASSED: Temporal filtering achieves >20 dB average PSNR" << std::endl;
    } else {
        std::cout << "❌ FAILED: Temporal filtering PSNR too low (" << avg_psnr << " dB)" << std::endl;
    }
}

void test_frame_reconstruction() {
    std::cout << "\n=== Testing CUDA Frame Reconstruction ===\n";
    
    // Create test data
    cv::Mat original_rgb = cv::Mat::zeros(50, 50, CV_8UC3);
    cv::randu(original_rgb, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    
    cv::Mat filtered_yiq = cv::Mat::zeros(50, 50, CV_32FC3);
    cv::randu(filtered_yiq, cv::Scalar(-5, -5, -5), cv::Scalar(5, 5, 5));
    
    // CPU reference
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cv::Mat cpu_result = evmcpp::reconstructGaussianFrame(original_rgb, filtered_yiq);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count();
    
    // CUDA implementation
    auto start_cuda = std::chrono::high_resolution_clock::now();
    cv::Mat cuda_result = cuda_evm::reconstruct_gaussian_frame_wrapper(original_rgb, filtered_yiq);
    auto end_cuda = std::chrono::high_resolution_clock::now();
    auto cuda_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_cuda - start_cuda).count();
    
    if (cpu_result.empty() || cuda_result.empty()) {
        std::cout << "❌ FAILED: One of the implementations returned empty result" << std::endl;
        return;
    }
    
    // Compare results
    cv::Mat diff;
    cv::absdiff(cpu_result, cuda_result, diff);
    cv::Scalar mean_diff = cv::mean(diff);
    double max_diff;
    cv::minMaxLoc(diff, nullptr, &max_diff);
    
    // Calculate PSNR
    double mse = cv::sum(diff.mul(diff))[0] / (diff.rows * diff.cols * diff.channels());
    double psnr = 20 * log10(255.0 / sqrt(mse));
    
    std::cout << "Results:" << std::endl;
    std::cout << "  CPU Time: " << cpu_time << " ms" << std::endl;
    std::cout << "  CUDA Time: " << cuda_time << " ms" << std::endl;
    std::cout << "  Mean absolute difference: " << mean_diff[0] << std::endl;
    std::cout << "  Max absolute difference: " << max_diff << std::endl;
    std::cout << "  PSNR: " << psnr << " dB" << std::endl;
    
    if (psnr > 30.0) {
        std::cout << "✅ PASSED: Frame reconstruction achieves >30 dB PSNR" << std::endl;
    } else {
        std::cout << "❌ FAILED: Frame reconstruction PSNR too low (" << psnr << " dB)" << std::endl;
    }
}

void test_end_to_end_processing() {
    std::cout << "\n=== Testing End-to-End CUDA Gaussian Processing ===\n";
    
    const std::string input_video = "/home/furkan/projects/school/mmi713/evm/data/face.mp4";
    const std::string cuda_output = "cuda_gaussian_test_output.avi";
    
    // Test parameters
    const int levels = 1;
    const double alpha = 10.0;
    const double fl = 0.5;
    const double fh = 2.0;
    const double chrom_attenuation = 0.1;
    
    std::cout << "Processing video with CUDA Gaussian pipeline..." << std::endl;
    std::cout << "Input: " << input_video << std::endl;
    std::cout << "Output: " << cuda_output << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool success = cuda_evm::process_video_gaussian_gpu(
        input_video, cuda_output, levels, alpha, fl, fh, chrom_attenuation);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto processing_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    if (success) {
        std::cout << "✅ PASSED: End-to-end processing completed in " << processing_time << " seconds" << std::endl;
        std::cout << "Output video saved as: " << cuda_output << std::endl;
    } else {
        std::cout << "❌ FAILED: End-to-end processing failed" << std::endl;
    }
}

int main() {
    std::cout << "CUDA Gaussian Pyramid Implementation Test Suite" << std::endl;
    std::cout << "==============================================\n" << std::endl;
    
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "❌ No CUDA devices found!" << std::endl;
        return -1;
    }
    
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    // Run component tests
    test_spatial_filtering();
    test_temporal_filtering();
    test_frame_reconstruction();
    test_end_to_end_processing();
    
    std::cout << "\n=== Test Suite Complete ===" << std::endl;
    
    return 0;
}