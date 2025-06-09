#ifndef CUDA_GAUSSIAN_PYRAMID_CUH
#define CUDA_GAUSSIAN_PYRAMID_CUH

#include <cuda_runtime.h>
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
 * @brief CUDA implementation of temporal Gaussian batch filtering using FFT
 * Equivalent to CPU temporalFilterGaussianBatch function
 * @param d_all_frames All spatially filtered frames (frame-major layout)
 * @param d_filtered_frames Output filtered frames
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels (3 for YIQ)
 * @param num_frames Number of frames in batch
 * @param fl Low cutoff frequency (Hz)
 * @param fh High cutoff frequency (Hz)
 * @param fps Frames per second
 * @return cudaError_t error code
 */
cudaError_t temporal_filter_gaussian_batch_gpu(
    const float* d_all_frames,
    float* d_filtered_frames,
    int width,
    int height,
    int channels,
    int num_frames,
    float fl,
    float fh,
    float fps
);

/**
 * @brief CUDA implementation of Gaussian frame reconstruction
 * Equivalent to CPU reconstructGaussianFrame function
 * @param d_original_rgb Original RGB frame on device
 * @param d_filtered_yiq Filtered YIQ signal on device
 * @param d_output_rgb Reconstructed RGB frame on device
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels (3)
 * @param alpha Amplification factor
 * @param chrom_attenuation Chrominance attenuation factor
 * @return cudaError_t error code
 */
cudaError_t reconstruct_gaussian_frame_gpu(
    const float* d_original_rgb,
    const float* d_filtered_yiq,
    float* d_output_rgb,
    int width,
    int height,
    int channels,
    float alpha,
    float chrom_attenuation
);

// Note: NO_OPENCV constraint - all functions work with raw float* arrays
// Video processing functions removed - use external wrapper with OpenCV for video I/O

} // namespace cuda_evm

#endif // CUDA_GAUSSIAN_PYRAMID_CUH