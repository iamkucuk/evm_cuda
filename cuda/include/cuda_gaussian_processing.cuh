#ifndef CUDA_GAUSSIAN_PROCESSING_CUH
#define CUDA_GAUSSIAN_PROCESSING_CUH

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <vector>

namespace evmcuda {

// Initialize Gaussian processing module
bool init_gaussian_processing();

// Clean up Gaussian processing module
void cleanup_gaussian_processing();

// CUDA Gaussian spatial filtering (equivalent to CPU spatiallyFilterGaussian)
// Performs down/up sampling for specified levels to create spatial lowpass filtering
cv::Mat spatially_filter_gaussian_gpu(const cv::Mat& input_rgb, int levels);

// CUDA FFT-based temporal filtering (equivalent to CPU temporalFilterGaussianBatch)
// Applies bandpass filtering using FFT to a batch of spatially filtered frames
std::vector<cv::Mat> temporal_filter_gaussian_batch_gpu(
    const std::vector<cv::Mat>& spatially_filtered_batch,
    float fps,
    float fl,
    float fh, 
    float alpha,
    float chrom_attenuation
);

// CUDA Gaussian frame reconstruction (equivalent to CPU reconstructGaussianFrame)
// Combines original frame with filtered signal to produce amplified output
cv::Mat reconstruct_gaussian_frame_gpu(
    const cv::Mat& original_rgb,
    const cv::Mat& filtered_yiq_signal
);

// Complete Gaussian mode processing pipeline
void process_video_gaussian(
    const std::string& input_filename,
    const std::string& output_filename,
    int levels,
    double alpha,
    double fl,
    double fh,
    double chrom_attenuation
);

// Low-level CUDA kernels for Gaussian processing

// Spatial filtering kernel: applies down/up sampling
__global__ void spatial_filter_kernel(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    int width,
    int height,
    int channels,
    int levels
);

// FFT setup and execution for temporal filtering
void setup_fft_plans(int num_frames);
void cleanup_fft_plans();

// FFT-based temporal filtering kernel
__global__ void fft_temporal_filter_kernel(
    const float* __restrict__ d_input_batch,
    float* __restrict__ d_output_batch,
    const float* __restrict__ d_frequency_mask,
    int width,
    int height,
    int channels,
    int num_frames,
    float alpha,
    float chrom_attenuation
);

// Frame reconstruction kernel
__global__ void reconstruct_frame_kernel(
    const float* __restrict__ d_original_yiq,
    const float* __restrict__ d_filtered_signal,
    float* __restrict__ d_output_rgb,
    int width,
    int height,
    int channels
);

} // namespace evmcuda

#endif // CUDA_GAUSSIAN_PROCESSING_CUH