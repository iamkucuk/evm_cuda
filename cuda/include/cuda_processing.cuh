#ifndef CUDA_PROCESSING_CUH
#define CUDA_PROCESSING_CUH

#include <cuda_runtime.h>

namespace cuda_evm {

/**
 * @brief Reconstruct EVM frame by combining original and filtered signals
 * 
 * @param d_original_rgb Original frame in RGB format [0,1] range
 * @param d_filtered_yiq_signal Temporally filtered YIQ signal [0,1] range  
 * @param d_output_rgb Output reconstructed frame in RGB format [0,1] range
 * @param width Frame width
 * @param height Frame height
 * @param channels Number of channels (must be 3)
 * @param alpha Amplification factor (currently unused, amplification done in temporal filtering)
 * @param chrom_attenuation Chrominance attenuation (currently unused)
 * @return cudaError_t CUDA error code
 */
cudaError_t reconstruct_gaussian_frame_gpu(
    const float* d_original_rgb,
    const float* d_filtered_yiq_signal,
    float* d_output_rgb,
    int width, int height, int channels,
    float alpha, float chrom_attenuation
);

/**
 * @brief GPU-resident batch reconstruction for all frames
 * Processes all frames without CPU transfers (optimal GPU residency)
 * @param d_original_rgb_batch All original RGB frames [num_frames][H][W][C]
 * @param d_filtered_yiq_batch All temporally filtered YIQ frames [num_frames][H][W][C]  
 * @param d_output_rgb_batch All output RGB frames [num_frames][H][W][C]
 * @param width Frame width
 * @param height Frame height
 * @param channels Number of channels (must be 3)
 * @param num_frames Number of frames to process
 * @param alpha Amplification factor
 * @param chrom_attenuation Chrominance attenuation
 * @return cudaError_t CUDA error code
 */
cudaError_t reconstruct_gaussian_batch_gpu(
    const float* d_original_rgb_batch,
    const float* d_filtered_yiq_batch,
    float* d_output_rgb_batch,
    int width, int height, int channels, int num_frames,
    float alpha, float chrom_attenuation
);

} // namespace cuda_evm

#endif // CUDA_PROCESSING_CUH