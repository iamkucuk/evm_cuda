#ifndef EVMCUDA_TEMPORAL_FILTER_CUH
#define EVMCUDA_TEMPORAL_FILTER_CUH

#include <cuda_runtime.h>
#include <cstddef> // For size_t
#include <vector>

namespace evmcuda {

/**
 * @brief Applies ideal temporal bandpass filtering (FFT) and amplification
 *        to a batch of images stored on the GPU.
 * Operates on raw device pointers. Assumes input is 3-channel float (YIQ).
 * Output is written back to the input buffer (in-place).
 *
 * @param d_imageBatch Pointer to the batch of image data on the GPU.
 *                     Data is organized frame by frame (e.g., frame0_row0, frame0_row1, ..., frame1_row0, ...).
 *                     Expected format is 3-channel float (YIQ). Data will be modified in-place.
 * @param numFrames Number of frames in the batch.
 * @param width Image width.
 * @param height Image height.
 * @param pitch Image pitch (bytes per row).
 * @param fps Video frame rate (for frequency calculation).
 * @param fl Low cutoff frequency (Hz).
 * @param fh High cutoff frequency (Hz).
 * @param alpha Amplification factor.
 * @param chromAttenuation Attenuation factor for chrominance channels (IQ).
 * @param stream CUDA stream for asynchronous execution (optional, defaults to 0).
 */
void temporalFilterGaussianBatch_gpu(float* d_imageBatch, // In-place modification
                                     int numFrames, int width, int height, size_t pitch,
                                     float fps, float fl, float fh,
                                     float alpha, float chromAttenuation,
                                     cudaStream_t stream = 0);


/**
 * @brief Applies one step of IIR temporal bandpass filtering (Butterworth, Low-pass - High-pass)
 *        and spatially attenuated amplification to a single Laplacian pyramid level on the GPU.
 * Operates on raw device pointers. Assumes input is 3-channel float (YIQ).
 * Updates state buffers in place for the next frame.
 *
 * @param d_inputLevelData Pointer to the input data for the current frame (n).
 * @param d_outputLevelData Pointer to the output buffer for the filtered/amplified data of the current frame (n).
 * @param d_state_xl_1 Pointer to device memory storing the previous frame's input for the low-pass stage (xl[n-1]). Updated to inputLevelData[n].
 * @param d_state_yl_1 Pointer to device memory storing the previous frame's low-pass output (yl[n-1]). Updated to low-pass output[n].
 * @param d_state_xh_1 Pointer to device memory storing the previous frame's input for the high-pass stage (xh[n-1]). Updated to inputLevelData[n].
 * @param d_state_yh_1 Pointer to device memory storing the previous frame's high-pass output (yh[n-1]). Updated to high-pass output[n].
 * @param width Width of the pyramid level.
 * @param height Height of the pyramid level.
 * @param pitch Pitch (bytes per row) of the pyramid level data.
 * @param b0_l, b1_l, a1_l Low-pass Butterworth filter coefficients.
 * @param b0_h, b1_h, a1_h High-pass Butterworth filter coefficients.
 * @param level The current pyramid level index (e.g., 0 for the highest resolution).
 * @param fps The video frame rate (needed for attenuation calculation).
 * @param alpha The base amplification factor.
 * @param lambda_cutoff The spatial cutoff wavelength for attenuation.
 * @param chromAttenuation Factor to attenuate chrominance channels.
 * @param stream CUDA stream for asynchronous execution (optional, defaults to 0).
 */
void filterLaplacianLevelFrame_gpu(const float* d_inputLevelData,
                                   float* d_outputLevelData,
                                   float* d_state_xl_1, // Read & Write
                                   float* d_state_yl_1, // Read & Write
                                   float* d_state_xh_1, // Read & Write
                                   float* d_state_yh_1, // Read & Write
                                   int width, int height, size_t pitch,
                                   float b0_l, float b1_l, float a1_l,
                                   float b0_h, float b1_h, float a1_h,
                                   int level, float fps,
                                   float alpha, float lambda_cutoff, float chromAttenuation,
                                   cudaStream_t stream = 0);


} // namespace evmcuda

#endif // EVMCUDA_TEMPORAL_FILTER_CUH