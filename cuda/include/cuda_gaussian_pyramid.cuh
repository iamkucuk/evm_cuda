#ifndef CUDA_GAUSSIAN_PYRAMID_CUH
#define CUDA_GAUSSIAN_PYRAMID_CUH

#include <cuda_runtime.h>
#include "cuda_temporal_filter.cuh"
#include "cuda_processing.cuh"
#include "cuda_color_conversion.cuh"
#include <cufft.h>

namespace cuda_evm {

/**
 * @brief CUDA implementation of spatially filter Gaussian
 * Equivalent to CPU spatiallyFilterGaussian function
 * @param d_input_rgb Input RGB image data on device (float)
 * @param d_output_yiq Output YIQ filtered data on device (float)
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels (3 for RGB/YIQ)
 * @param level Number of pyramid levels for down/up sampling
 * @return cudaError_t error code
 */
cudaError_t spatially_filter_gaussian_gpu(
    const float* d_input_rgb,
    float* d_output_yiq,
    int width,
    int height,
    int channels,
    int level
);

/**
 * @brief GPU-resident batch spatial filtering for all frames
 * Processes all frames without CPU transfers (optimal GPU residency)
 * @param d_input_rgb_batch All input RGB frames on device [num_frames][H][W][C]
 * @param d_output_yiq_batch All output YIQ frames on device [num_frames][H][W][C]
 * @param width Image width
 * @param height Image height  
 * @param channels Number of channels (3)
 * @param num_frames Number of frames to process
 * @param level Number of pyramid levels
 * @return cudaError_t error code
 */
cudaError_t spatially_filter_gaussian_batch_gpu(
    const float* d_input_rgb_batch,
    float* d_output_yiq_batch,
    int width,
    int height,
    int channels,
    int num_frames,
    int level
);

/**
 * @brief OpenCV-compatible CUDA pyrDown implementation
 * @param d_src Source image on device
 * @param d_dst Destination image on device (half size)
 * @param src_width Source width
 * @param src_height Source height
 * @param dst_width Destination width
 * @param dst_height Destination height
 * @param channels Number of channels
 * @param d_temp_buffer Temporary buffer for convolution
 * @return cudaError_t error code
 */
cudaError_t cuda_pyrDown(
    const float* d_src, float* d_dst,
    int src_width, int src_height, int dst_width, int dst_height, int channels,
    float* d_temp_buffer);

/**
 * @brief OpenCV-compatible CUDA pyrUp implementation
 * @param d_src Source image on device
 * @param d_dst Destination image on device (double size)
 * @param src_width Source width
 * @param src_height Source height
 * @param dst_width Destination width
 * @param dst_height Destination height
 * @param channels Number of channels
 * @param d_temp_buffer Temporary buffer for convolution
 * @return cudaError_t error code
 */
cudaError_t cuda_pyrUp(
    const float* d_src, float* d_dst,
    int src_width, int src_height, int dst_width, int dst_height, int channels,
    float* d_temp_buffer);


// Note: NO_OPENCV constraint - all functions work with raw float* arrays
// Video processing functions removed - use external wrapper with OpenCV for video I/O

} // namespace cuda_evm

#endif // CUDA_GAUSSIAN_PYRAMID_CUH