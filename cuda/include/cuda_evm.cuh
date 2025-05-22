#ifndef CUDA_EVM_CUH
#define CUDA_EVM_CUH

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <string>
#include <utility> // For std::pair

namespace evmcuda {

/**
 * @brief Laplacian mode processing using CUDA-accelerated Eulerian Video Magnification
 * 
 * This mode uses multi-scale Laplacian pyramid decomposition and IIR Butterworth
 * temporal filtering. Best for motion amplification applications.
 * 
 * @param inputFilename Path to the input video file
 * @param outputFilename Path to save the output video
 * @param pyramid_levels Number of pyramid levels to use
 * @param alpha Magnification factor for motion/color amplification
 * @param lambda_cutoff Spatial wavelength cutoff for attenuation
 * @param fl Low frequency cutoff for temporal bandpass filter
 * @param fh High frequency cutoff for temporal bandpass filter
 * @param chrom_attenuation Attenuation factor for chrominance (I,Q) channels
 */
void process_video_laplacian(
    const std::string& inputFilename, 
    const std::string& outputFilename,
    int pyramid_levels, 
    double alpha, 
    double lambda_cutoff, 
    double fl, 
    double fh, 
    double chrom_attenuation
);

/**
 * @brief Gaussian mode processing using CUDA-accelerated Eulerian Video Magnification
 * 
 * This mode uses simple spatial lowpass filtering and FFT-based temporal filtering.
 * Best for color/intensity amplification applications (e.g., pulse detection).
 * 
 * @param inputFilename Path to the input video file
 * @param outputFilename Path to save the output video
 * @param levels Number of downsampling/upsampling levels for spatial filtering
 * @param alpha Magnification factor for motion/color amplification
 * @param fl Low frequency cutoff for temporal bandpass filter
 * @param fh High frequency cutoff for temporal bandpass filter
 * @param chrom_attenuation Attenuation factor for chrominance (I,Q) channels
 */
void process_video_gaussian(
    const std::string& inputFilename,
    const std::string& outputFilename,
    int levels,
    double alpha,
    double fl,
    double fh,
    double chrom_attenuation
);

/**
 * @brief Initialize all resources needed by the EVM pipeline
 * 
 * @return true if initialization succeeded, false otherwise
 */
bool init_evm();

/**
 * @brief Clean up all resources used by the EVM pipeline
 */
void cleanup_evm();

} // namespace evmcuda

#endif // CUDA_EVM_CUH