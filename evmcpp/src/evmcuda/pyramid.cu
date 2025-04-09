#include "evmcuda/pyramid.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <cmath> // For floor

// Define PRINT_ERROR if not globally available via headers
#ifndef PRINT_ERROR
#include <iostream>
#define PRINT_ERROR(msg) std::cerr << "ERROR: " << msg << std::endl
#endif

namespace evmcuda {

// --- CUDA Kernel for 2D Convolution (used by pyrDown/pyrUp) ---

// Simple implementation assuming 5x5 kernel and BORDER_REFLECT_101
// Constants for shared memory convolution
#define TILE_WIDTH 16
#define KERNEL_RADIUS 2 // For 5x5 kernel
#define HALO_WIDTH (KERNEL_RADIUS * 2)
#define SHMEM_TILE_WIDTH (TILE_WIDTH + HALO_WIDTH) // e.g., 16 + 4 = 20

__device__ inline int reflect101(int coord, int max_coord) {
    // Clamp mode might be simpler/closer to filter2D? Let's stick with reflect101 for now.
    if (coord < 0) return -coord - 1;
    if (coord >= max_coord) return 2 * max_coord - coord - 1;
    return coord;
}

__global__ void conv2DKernelSharedMem(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int width, int height, size_t inputPitch, size_t outputPitch,
                                      const float* __restrict__ kernel) // Assumes 5x5 kernel
{
    // Shared memory tile for 3 channels
    __shared__ float tile[3][SHMEM_TILE_WIDTH][SHMEM_TILE_WIDTH];

    // Thread indices within the block (e.g., 0-19 for 20x20 block)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Base global coordinates (top-left corner of the *output* tile this block computes)
    const int base_gx = blockIdx.x * TILE_WIDTH;
    const int base_gy = blockIdx.y * TILE_WIDTH;

    // Global read coordinates for the element this thread loads into shared memory
    const int rx = base_gx + tx - KERNEL_RADIUS;
    const int ry = base_gy + ty - KERNEL_RADIUS;

    // Apply border handling (reflect101)
    const int rfl_rx = reflect101(rx, width);
    const int rfl_ry = reflect101(ry, height);

    // Load data into shared memory (all threads in the 20x20 block participate)
    const size_t inputPitchF = inputPitch / sizeof(float);
    for (int c = 0; c < 3; ++c) {
        tile[c][ty][tx] = input[rfl_ry * inputPitchF + rfl_rx * 3 + c];
    }

    // Synchronize threads to ensure shared memory is fully loaded
    __syncthreads();

    // --- Perform convolution using shared memory ---
    // Only threads corresponding to the output tile (0-15) compute and write output
    if (tx < TILE_WIDTH && ty < TILE_WIDTH) {
        // Global output coordinates
        const int gx = base_gx + tx;
        const int gy = base_gy + ty;

        // Check output bounds (redundant if grid calculation is correct, but safer)
        if (gx < width && gy < height) {
            const size_t outputPitchF = outputPitch / sizeof(float);

            for (int c = 0; c < 3; ++c) {
                double sum = 0.0; // Use double for accumulator - moved inside channel loop
                // Iterate through kernel
                for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ++ky) {
                    for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; ++kx) {
                        // Read from shared memory tile (indices are relative to tile)
                        // Center of kernel corresponds to tile[c][ty+KERNEL_RADIUS][tx+KERNEL_RADIUS]
                        float pixelVal = tile[c][ty + KERNEL_RADIUS + ky][tx + KERNEL_RADIUS + kx];

                        // Calculate kernel index
                        int kernelIdx = (ky + KERNEL_RADIUS) * 5 + (kx + KERNEL_RADIUS);
                        sum += (double)pixelVal * (double)kernel[kernelIdx]; // Cast operands to double for accumulation
                    }
                }
                // Write output
                output[gy * outputPitchF + gx * 3 + c] = (float)sum; // Cast back to float for output
            }
        }
    }
}

// Forward declarations for CUDA kernels defined later in the file
__global__ void subsampleKernel(const float* input, float* output, int outW, int outH, size_t inP, size_t outP);
__global__ void insertKernel(const float* input, float* outputZeros, int inW, int inH, int outW, int outH, size_t inP, size_t outP);


// --- pyrDown ---

void pyrDown_gpu(const float* d_input, float* d_output,
                 int width, int height, size_t inputPitch, size_t outputPitch,
                 const float* d_kernel, cudaStream_t stream)
{
    if (!d_input || !d_output || !d_kernel || width <= 0 || height <= 0 || inputPitch == 0 || outputPitch == 0) {
        throw std::invalid_argument("[pyrDown_gpu] Invalid arguments.");
    }

    // 1. Allocate temporary buffer for filtered image (same size as input)
    float* d_filtered = nullptr;
    size_t filteredPitch = 0;
    cudaError_t err = cudaMallocPitch((void**)&d_filtered, &filteredPitch, width * sizeof(float) * 3, height);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[pyrDown_gpu] cudaMallocPitch failed for temp buffer: ") + cudaGetErrorString(err));
    }

    // 2. Launch Shared Memory Convolution Kernel
    // Block size matches shared memory tile dimensions
    const dim3 blockSizeConv(SHMEM_TILE_WIDTH, SHMEM_TILE_WIDTH);
    // Grid size based on output dimensions and TILE_WIDTH (not SHMEM_TILE_WIDTH)
    const dim3 gridSizeConv((width + TILE_WIDTH - 1) / TILE_WIDTH,
                            (height + TILE_WIDTH - 1) / TILE_WIDTH);

    conv2DKernelSharedMem<<<gridSizeConv, blockSizeConv, 0, stream>>>(
        d_input, d_filtered, width, height, inputPitch, filteredPitch, d_kernel
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_filtered);
        throw std::runtime_error(std::string("[pyrDown_gpu] conv2DKernelSharedMem launch failed: ") + cudaGetErrorString(err));
    }
    // Add explicit sync after convolution
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFree(d_filtered);
        throw std::runtime_error(std::string("[pyrDown_gpu] cudaStreamSynchronize after conv2DKernel failed: ") + cudaGetErrorString(err));
    }

    // 3. Launch Downsampling Kernel (simple subsampling)
    int outWidth = width / 2;
    int outHeight = height / 2;
    if (outWidth <= 0 || outHeight <= 0) {
         cudaFree(d_filtered);
         // Return or throw? Let's throw as output is invalid size.
         throw std::runtime_error("[pyrDown_gpu] Output dimensions are zero or negative after downsampling.");
    }

    const dim3 blockSizeDown(16, 16);
    const dim3 gridSizeDown((outWidth + blockSizeDown.x - 1) / blockSizeDown.x,
                            (outHeight + blockSizeDown.y - 1) / blockSizeDown.y);

    // Call the __global__ subsampleKernel
    subsampleKernel<<<gridSizeDown, blockSizeDown, 0, stream>>>(
        d_filtered, d_output, outWidth, outHeight, filteredPitch, outputPitch
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_filtered);
        throw std::runtime_error(std::string("[pyrDown_gpu] subsampleKernel launch failed: ") + cudaGetErrorString(err));
    }
    // Add explicit sync after subsampling (before freeing d_filtered)
    err = cudaStreamSynchronize(stream);
     if (err != cudaSuccess) {
        cudaFree(d_filtered); // Attempt cleanup even if sync fails
        throw std::runtime_error(std::string("[pyrDown_gpu] cudaStreamSynchronize after subsampleKernel failed: ") + cudaGetErrorString(err));
    }

    // 4. Free temporary buffer
    err = cudaFree(d_filtered);
    if (err != cudaSuccess) {
        // Log warning but continue, memory leak is better than crashing here maybe?
        PRINT_ERROR("[pyrDown_gpu] cudaFree failed for temp buffer: " + std::string(cudaGetErrorString(err)));
    }
}


// --- pyrUp ---

void pyrUp_gpu(const float* d_input, float* d_output,
               int width, int height, int outputWidth, int outputHeight,
               size_t inputPitch, size_t outputPitch,
               const float* d_kernel_x4, cudaStream_t stream)
{
     if (!d_input || !d_output || !d_kernel_x4 || width <= 0 || height <= 0 || outputWidth <= 0 || outputHeight <= 0 || inputPitch == 0 || outputPitch == 0) {
        throw std::invalid_argument("[pyrUp_gpu] Invalid arguments.");
    }

    // 1. Allocate temporary buffer for upsampled image with zeros
    float* d_upsampled_zeros = nullptr;
    size_t upsampledPitch = 0;
    cudaError_t err = cudaMallocPitch((void**)&d_upsampled_zeros, &upsampledPitch, outputWidth * sizeof(float) * 3, outputHeight);
     if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[pyrUp_gpu] cudaMallocPitch failed for upsampled buffer: ") + cudaGetErrorString(err));
    }
    // Initialize with zeros
    err = cudaMemset2DAsync(d_upsampled_zeros, upsampledPitch, 0, outputWidth * sizeof(float) * 3, outputHeight, stream);
     if (err != cudaSuccess) {
        cudaFree(d_upsampled_zeros);
        throw std::runtime_error(std::string("[pyrUp_gpu] cudaMemset2DAsync failed: ") + cudaGetErrorString(err));
    }
    // Synchronize stream to ensure memset completes before insertKernel
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFree(d_upsampled_zeros);
        throw std::runtime_error(std::string("[pyrUp_gpu] cudaStreamSynchronize after memset failed: ") + cudaGetErrorString(err));
    }
    // Add explicit sync after insertKernel
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFree(d_upsampled_zeros);
        throw std::runtime_error(std::string("[pyrUp_gpu] cudaStreamSynchronize after insertKernel failed: ") + cudaGetErrorString(err));
    }

    // 2. Launch Kernel to insert input pixels into zero-padded buffer
    const dim3 blockSizeInsert(16, 16);
    // Grid size based on *input* dimensions
    const dim3 gridSizeInsert((width + blockSizeInsert.x - 1) / blockSizeInsert.x,
                              (height + blockSizeInsert.y - 1) / blockSizeInsert.y);

    // Call the __global__ insertKernel
    insertKernel<<<gridSizeInsert, blockSizeInsert, 0, stream>>>(
        d_input, d_upsampled_zeros, width, height, outputWidth, outputHeight, inputPitch, upsampledPitch
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_upsampled_zeros);
        throw std::runtime_error(std::string("[pyrUp_gpu] insertKernel launch failed: ") + cudaGetErrorString(err));
    }
    // Add explicit sync after convolution (before freeing d_upsampled_zeros)
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFree(d_upsampled_zeros); // Attempt cleanup
        throw std::runtime_error(std::string("[pyrUp_gpu] cudaStreamSynchronize after conv2DKernel failed: ") + cudaGetErrorString(err));
    }

    // 3. Launch Convolution Kernel on the zero-padded upsampled image
    const dim3 blockSizeConv(16, 16);
    const dim3 gridSizeConv((outputWidth + blockSizeConv.x - 1) / blockSizeConv.x,
                            (outputHeight + blockSizeConv.y - 1) / blockSizeConv.y);

    // Block size matches shared memory tile dimensions
    // const dim3 blockSizeConv(SHMEM_TILE_WIDTH, SHMEM_TILE_WIDTH); // Redeclaration removed
     // Grid size based on output dimensions and TILE_WIDTH
    // const dim3 gridSizeConv((outputWidth + TILE_WIDTH - 1) / TILE_WIDTH, // Redeclaration removed
    //                         (outputHeight + TILE_WIDTH - 1) / TILE_WIDTH);

    conv2DKernelSharedMem<<<gridSizeConv, blockSizeConv, 0, stream>>>(
        d_upsampled_zeros, d_output, outputWidth, outputHeight, upsampledPitch, outputPitch, d_kernel_x4
    );
     err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_upsampled_zeros);
        throw std::runtime_error(std::string("[pyrUp_gpu] conv2DKernelSharedMem launch failed: ") + cudaGetErrorString(err));
    }

    // 4. Free temporary buffer
    err = cudaFree(d_upsampled_zeros);
     if (err != cudaSuccess) {
        PRINT_ERROR("[pyrUp_gpu] cudaFree failed for temp buffer: " + std::string(cudaGetErrorString(err)));
    }
}


// --- __global__ Kernel Definitions ---

// Simple kernel to copy every second pixel (Subsampling for pyrDown)
__global__ void subsampleKernel(const float* input, float* output, int outW, int outH, size_t inP, size_t outP) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= outW || y >= outH) return;

    size_t inPitchF = inP / sizeof(float);
    size_t outPitchF = outP / sizeof(float);

    const float* inPtr = input + (y * 2 * inPitchF) + (x * 2 * 3); // Read from y*2, x*2
    float* outPtr = output + (y * outPitchF) + (x * 3);

    outPtr[0] = inPtr[0]; // C1
    outPtr[1] = inPtr[1]; // C2
    outPtr[2] = inPtr[2]; // C3
}

// Kernel to insert pixels into a zero-padded larger buffer (Upsampling for pyrUp)
__global__ void insertKernel(const float* input, float* outputZeros, int inW, int inH, int outW, int outH, size_t inP, size_t outP) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= inW || y >= inH) return;

    int target_r = y * 2;
    int target_c = x * 2;

    if (target_r < outH && target_c < outW) {
        size_t inPitchF = inP / sizeof(float);
        size_t outPitchF = outP / sizeof(float);
        const float* inPtr = input + (y * inPitchF) + (x * 3);
        float* outPtr = outputZeros + (target_r * outPitchF) + (target_c * 3);
        outPtr[0] = inPtr[0];
        outPtr[1] = inPtr[1];
        outPtr[2] = inPtr[2];
    }
}


// --- C++ Wrapper for Shared Memory Convolution Kernel ---

void conv2DKernelSharedMem_gpu(const float* d_input, float* d_output,
                               int width, int height, size_t inputPitch, size_t outputPitch,
                               const float* d_kernel, cudaStream_t stream)
{
     if (!d_input || !d_output || !d_kernel || width <= 0 || height <= 0 || inputPitch == 0 || outputPitch == 0) {
        throw std::invalid_argument("[conv2DKernelSharedMem_gpu] Invalid arguments.");
    }

    // Block size matches shared memory tile dimensions
    const dim3 blockSizeConv(SHMEM_TILE_WIDTH, SHMEM_TILE_WIDTH);
     // Grid size based on output dimensions and TILE_WIDTH
    const dim3 gridSizeConv((width + TILE_WIDTH - 1) / TILE_WIDTH,
                            (height + TILE_WIDTH - 1) / TILE_WIDTH);

    conv2DKernelSharedMem<<<gridSizeConv, blockSizeConv, 0, stream>>>(
        d_input, d_output, width, height, inputPitch, outputPitch, d_kernel
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[conv2DKernelSharedMem_gpu] CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
    // Caller is responsible for synchronization if needed
}


} // namespace evmcuda