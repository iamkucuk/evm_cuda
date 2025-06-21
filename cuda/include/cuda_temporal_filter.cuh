#ifndef CUDA_TEMPORAL_FILTER_CUH
#define CUDA_TEMPORAL_FILTER_CUH

#include <cuda_runtime.h>

namespace cuda_evm {

/**
 * @brief Apply temporal filtering using CUDA (Butterworth bandpass filter)
 * 
 * @param d_input_frames Input frames in pixel-major layout [spatial_location][time_series]
 * @param d_output_frames Output filtered frames in pixel-major layout
 * @param width Frame width
 * @param height Frame height  
 * @param channels Number of channels (3 for YIQ)
 * @param num_frames Number of frames in time series
 * @param fl Low cutoff frequency
 * @param fh High cutoff frequency
 * @param fps Video frame rate
 * @param alpha Amplification factor
 * @param chrom_attenuation Chrominance attenuation factor
 * @return cudaError_t CUDA error code
 */
cudaError_t temporal_filter_gaussian_batch_gpu(
    const float* d_input_frames,
    float* d_output_frames,
    int width, int height, int channels, int num_frames,
    float fl, float fh, float fps, float alpha, float chrom_attenuation
);

/**
 * @brief Apply IIR temporal filtering to Laplacian pyramid levels using CUDA
 * 
 * @param d_pyramid_levels Array of device pointers to pyramid levels
 * @param d_filtered_levels Array of device pointers for filtered output levels
 * @param level_widths Width of each pyramid level
 * @param level_heights Height of each pyramid level
 * @param level_sizes Total elements per level (width * height * channels)
 * @param num_levels Number of pyramid levels
 * @param num_frames Number of frames in time series
 * @param channels Number of channels (3 for YIQ)
 * @param fl Low cutoff frequency
 * @param fh High cutoff frequency
 * @param fps Video frame rate
 * @param alpha Amplification factor
 * @param lambda_cutoff Lambda cutoff for spatial attenuation
 * @param chrom_attenuation Chrominance attenuation factor
 * @return cudaError_t CUDA error code
 */
cudaError_t temporal_filter_laplacian_pyramids_gpu(
    float** d_pyramid_levels,
    float** d_filtered_levels,
    const int* level_widths,
    const int* level_heights, 
    const size_t* level_sizes,
    int num_levels,
    int num_frames,
    int channels,
    float fl, float fh, float fps,
    float alpha, float lambda_cutoff, float chrom_attenuation
);

/**
 * @brief Apply IIR temporal filtering to single frame of Laplacian pyramid levels
 * 
 * @param d_pyramid_levels Input pyramid levels for current frame
 * @param d_filtered_levels Output filtered levels for current frame
 * @param d_prev_input_levels Previous frame input for IIR state
 * @param d_lowpass_state_levels Low-pass filter state memory
 * @param d_highpass_state_levels High-pass filter state memory
 * @param level_widths Width of each pyramid level
 * @param level_heights Height of each pyramid level
 * @param level_sizes Total elements per level (width * height * channels)
 * @param num_levels Number of pyramid levels
 * @param channels Number of channels (3 for YIQ)
 * @param fl Low cutoff frequency
 * @param fh High cutoff frequency
 * @param fps Video frame rate
 * @param alpha Amplification factor
 * @param lambda_cutoff Lambda cutoff for spatial attenuation
 * @param chrom_attenuation Chrominance attenuation factor
 * @param frame_idx Current frame index (0-based)
 * @return cudaError_t CUDA error code
 */
cudaError_t temporal_filter_laplacian_frame_gpu(
    float** d_pyramid_levels,
    float** d_filtered_levels,
    float** d_prev_input_levels,
    float** d_lowpass_state_levels,
    float** d_highpass_state_levels,
    const int* level_widths,
    const int* level_heights, 
    const size_t* level_sizes,
    int num_levels,
    int channels,
    float fl, float fh, float fps,
    float alpha, float lambda_cutoff, float chrom_attenuation,
    int frame_idx
);

} // namespace cuda_evm

#endif // CUDA_TEMPORAL_FILTER_CUH