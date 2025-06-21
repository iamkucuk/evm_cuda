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

} // namespace cuda_evm

#endif // CUDA_TEMPORAL_FILTER_CUH