#ifndef CUDA_LAPLACIAN_PYRAMID_CUH
#define CUDA_LAPLACIAN_PYRAMID_CUH

#include <cuda_runtime.h>
#include <vector>

namespace cuda_evm {

/**
 * @brief GPU data structure for storing multi-level Laplacian pyramid
 * Each level has different dimensions, requiring careful memory management
 */
struct LaplacianPyramidGPU {
    std::vector<float*> d_levels;      // Device pointers for each pyramid level
    std::vector<int> widths;           // Width of each level
    std::vector<int> heights;          // Height of each level  
    std::vector<size_t> level_sizes;   // Total elements per level (width*height*channels)
    int num_levels;                    // Number of pyramid levels
    int channels;                      // Number of channels (3 for YIQ)
    
    // Constructor
    LaplacianPyramidGPU() : num_levels(0), channels(3) {}
    
    // Destructor - free GPU memory
    ~LaplacianPyramidGPU() {
        for (auto ptr : d_levels) {
            if (ptr) cudaFree(ptr);
        }
    }
    
    // Allocate GPU memory for pyramid levels
    cudaError_t allocate(int base_width, int base_height, int levels, int num_channels = 3);
    
    // Free GPU memory
    void deallocate();
};

/**
 * @brief Generate Laplacian pyramid for a single frame on GPU
 * 
 * @param d_input_yiq Input YIQ frame (CV_32FC3 format)
 * @param width Frame width
 * @param height Frame height
 * @param pyramid_levels Number of pyramid levels to generate
 * @param pyramid Output pyramid structure (will be allocated)
 * @return cudaError_t CUDA error code
 */
cudaError_t generateLaplacianPyramid_gpu(
    const float* d_input_yiq,
    int width, int height,
    int pyramid_levels,
    LaplacianPyramidGPU& pyramid
);

/**
 * @brief Generate Laplacian pyramids for batch of frames
 * 
 * @param d_input_frames Input YIQ frames in frame-major layout [frame][height][width][channels]
 * @param width Frame width
 * @param height Frame height
 * @param num_frames Number of frames in batch
 * @param pyramid_levels Number of pyramid levels
 * @param pyramids Output vector of pyramid structures
 * @return cudaError_t CUDA error code
 */
cudaError_t getLaplacianPyramids_gpu(
    const float* d_input_frames,
    int width, int height, int num_frames,
    int pyramid_levels,
    std::vector<LaplacianPyramidGPU>& pyramids
);

/**
 * @brief Apply temporal filtering to Laplacian pyramids with spatial attenuation
 * 
 * This implements IIR Butterworth filtering per pyramid level with spatial wavelength cutoff.
 * More complex than Gaussian temporal filtering due to per-level processing.
 * 
 * @param pyramids Input/output vector of pyramid structures
 * @param num_frames Number of frames in sequence
 * @param pyramid_levels Number of pyramid levels
 * @param fps Video frame rate
 * @param fl Low cutoff frequency (Hz)
 * @param fh High cutoff frequency (Hz)
 * @param alpha Magnification factor
 * @param delta Calculated delta value for spatial attenuation (lambda_cutoff / (8 * (1 + alpha)))
 * @param chrom_attenuation Chrominance attenuation factor
 * @return cudaError_t CUDA error code
 */
cudaError_t filterLaplacianPyramids_gpu(
    std::vector<LaplacianPyramidGPU>& pyramids,
    int num_frames, int pyramid_levels,
    float fps, float fl, float fh,
    float alpha, float delta, float chrom_attenuation
);

/**
 * @brief Reconstruct magnified image from original and filtered Laplacian pyramid
 * 
 * Performs pyramid collapse: upsample each level and add to reconstruct final image.
 * This is significantly more complex than Gaussian reconstruction.
 * 
 * @param d_original_yiq Original YIQ frame
 * @param filtered_pyramid Temporally filtered Laplacian pyramid
 * @param d_output_yiq Output reconstructed YIQ frame
 * @param width Frame width
 * @param height Frame height
 * @return cudaError_t CUDA error code
 */
cudaError_t reconstructLaplacianImage_gpu(
    const float* d_original_yiq,
    const LaplacianPyramidGPU& filtered_pyramid,
    float* d_output_yiq,
    int width, int height
);

/**
 * @brief Helper function to calculate pyramid level dimensions
 * 
 * @param base_width Original frame width
 * @param base_height Original frame height  
 * @param level Pyramid level (0 = base level)
 * @param level_width Output: width at this level
 * @param level_height Output: height at this level
 */
void calculatePyramidLevelDimensions(
    int base_width, int base_height, int level,
    int& level_width, int& level_height
);

/**
 * @brief Helper function to calculate spatial attenuation factor for pyramid level
 * 
 * @param level Pyramid level
 * @param lambda_cutoff Spatial wavelength cutoff
 * @param base_width Original frame width
 * @param base_height Original frame height
 * @return Attenuation factor for this pyramid level
 */
float calculateSpatialAttenuation(
    int level, float lambda_cutoff,
    int base_width, int base_height
);

} // namespace cuda_evm

#endif // CUDA_LAPLACIAN_PYRAMID_CUH