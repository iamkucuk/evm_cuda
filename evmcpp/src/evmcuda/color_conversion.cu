// Move cuda_runtime.h to the top
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "evmcuda/color_conversion.cuh" // Standard include path relative to include directories

// Includes needed for stub function bodies (e.g., stdexcept)
#include <stdexcept>
#include <string>
#include <iostream> // For PRINT_ERROR/PRINT_WARNING

// Define PRINT_ERROR and PRINT_WARNING macros/functions if not globally available
#ifndef PRINT_ERROR
#define PRINT_ERROR(msg) std::cerr << "ERROR: " << msg << std::endl
#endif
#ifndef PRINT_WARNING
#define PRINT_WARNING(msg) std::cout << "WARNING: " << msg << std::endl
#endif

// Define the RGB->YIQ matrix constants on the device (Global Scope)
// Using __constant__ memory for read-only, frequently accessed data
static const __constant__ float d_RGB2YIQ_MATRIX[9] = {
    0.299f,       0.587f,       0.114f,
    0.59590059f, -0.27455667f, -0.32134392f,
    0.21153661f, -0.52273617f,  0.31119955f
};

// Define the YIQ->RGB matrix constants on the device (Global Scope)
// Calculated as inv(RGB2YIQ) - use pre-calculated values for consistency/performance
// Values from numpy.linalg.inv(yiq_from_rgb) in constants.py
static const __constant__ float d_YIQ2RGB_MATRIX[9] = {
    1.0f,        0.9559863f,   0.6208248f,
    1.0f,       -0.2720128f,  -0.6472042f,
    1.0f,       -1.1067402f,   1.7042304f
}; // Updated with values from NumPy linalg.inv


namespace evmcuda {

// CUDA Kernel for RGB (uint8) to YIQ Conversion
__global__ void rgb2yiqKernelUint8(const unsigned char* __restrict__ inputRgb,
                                  float* __restrict__ outputYiq,
                                  int width, int height, size_t inputStep, size_t outputStep)
{
    // Calculate global thread coordinates
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (x >= width || y >= height) {
        return;
    }

    // Calculate pointers to the current pixel (handling step/pitch)
    const unsigned char* rgbPixelPtr = inputRgb + (y * inputStep) + (x * 3); // 3 channels
    float* yiqPixelPtr = outputYiq + (y * (outputStep / sizeof(float))) + (x * 3); // 3 channels

    // Read RGB values and convert to float [0,255]
    const float r = static_cast<float>(rgbPixelPtr[0]);
    const float g = static_cast<float>(rgbPixelPtr[1]);
    const float b = static_cast<float>(rgbPixelPtr[2]);

    // Perform matrix multiplication (RGB2YIQ * [R G B]')
    // Y = M[0]*R + M[1]*G + M[2]*B
    // I = M[3]*R + M[4]*G + M[5]*B
    // Q = M[6]*R + M[7]*G + M[8]*B
    const float y_val = d_RGB2YIQ_MATRIX[0] * r + d_RGB2YIQ_MATRIX[1] * g + d_RGB2YIQ_MATRIX[2] * b;
    const float i_val = d_RGB2YIQ_MATRIX[3] * r + d_RGB2YIQ_MATRIX[4] * g + d_RGB2YIQ_MATRIX[5] * b;
    const float q_val = d_RGB2YIQ_MATRIX[6] * r + d_RGB2YIQ_MATRIX[7] * g + d_RGB2YIQ_MATRIX[8] * b;

    // Write YIQ values
    yiqPixelPtr[0] = y_val;
    yiqPixelPtr[1] = i_val;
    yiqPixelPtr[2] = q_val;
}

// --- CUDA Kernel for RGB (float) to YIQ Conversion ---
__global__ void rgb2yiqKernel(const float* __restrict__ inputRgb,
                             float* __restrict__ outputYiq,
                             int width, int height, size_t inputStep, size_t outputStep)
{
    // Calculate global thread coordinates
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (x >= width || y >= height) {
        return;
    }

    // Calculate pointers to the current pixel (handling step/pitch)
    const float* rgbPixelPtr = inputRgb + (y * (inputStep / sizeof(float))) + (x * 3); // 3 channels
    float* yiqPixelPtr = outputYiq + (y * (outputStep / sizeof(float))) + (x * 3); // 3 channels

    // Read RGB values
    const float r = rgbPixelPtr[0];
    const float g = rgbPixelPtr[1];
    const float b = rgbPixelPtr[2];

    // Perform matrix multiplication (RGB2YIQ * [R G B]')
    // Y = M[0]*R + M[1]*G + M[2]*B
    // I = M[3]*R + M[4]*G + M[5]*B
    // Q = M[6]*R + M[7]*G + M[8]*B
    const float y_val = d_RGB2YIQ_MATRIX[0] * r + d_RGB2YIQ_MATRIX[1] * g + d_RGB2YIQ_MATRIX[2] * b;
    const float i_val = d_RGB2YIQ_MATRIX[3] * r + d_RGB2YIQ_MATRIX[4] * g + d_RGB2YIQ_MATRIX[5] * b;
    const float q_val = d_RGB2YIQ_MATRIX[6] * r + d_RGB2YIQ_MATRIX[7] * g + d_RGB2YIQ_MATRIX[8] * b;

    // Write YIQ values
    yiqPixelPtr[0] = y_val;
    yiqPixelPtr[1] = i_val;
    yiqPixelPtr[2] = q_val;
}

// --- CUDA Kernel for YIQ to RGB Conversion ---
__global__ void yiq2rgbKernel(const float* __restrict__ inputYiq,
                             float* __restrict__ outputRgb,
                             int width, int height, size_t inputStep, size_t outputStep)
{
    // Calculate global thread coordinates
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (x >= width || y >= height) {
        return;
    }

    // Calculate pointers to the current pixel (handling step/pitch)
    const float* yiqPixelPtr = inputYiq + (y * (inputStep / sizeof(float))) + (x * 3);
    float* rgbPixelPtr = outputRgb + (y * (outputStep / sizeof(float))) + (x * 3);

    // Read YIQ values
    const float y_val = yiqPixelPtr[0];
    const float i_val = yiqPixelPtr[1];
    const float q_val = yiqPixelPtr[2];

    // Perform matrix multiplication (YIQ2RGB * [Y I Q]')
    // R = M[0]*Y + M[1]*I + M[2]*Q
    // G = M[3]*Y + M[4]*I + M[5]*Q
    // B = M[6]*Y + M[7]*I + M[8]*Q
    const float r = d_YIQ2RGB_MATRIX[0] * y_val + d_YIQ2RGB_MATRIX[1] * i_val + d_YIQ2RGB_MATRIX[2] * q_val;
    const float g = d_YIQ2RGB_MATRIX[3] * y_val + d_YIQ2RGB_MATRIX[4] * i_val + d_YIQ2RGB_MATRIX[5] * q_val;
    const float b = d_YIQ2RGB_MATRIX[6] * y_val + d_YIQ2RGB_MATRIX[7] * i_val + d_YIQ2RGB_MATRIX[8] * q_val;

    // Write RGB values
    rgbPixelPtr[0] = r;
    rgbPixelPtr[1] = g;
    rgbPixelPtr[2] = b;
}

// --- C++ Wrapper Functions ---

void rgb2yiq_gpu(const unsigned char* d_inputRgb, float* d_outputYiq,
                 int width, int height, size_t inputPitch, size_t outputPitch,
                 cudaStream_t stream)
{
    // Basic validation
    if (!d_inputRgb || !d_outputYiq || width <= 0 || height <= 0 || inputPitch == 0 || outputPitch == 0) {
         throw std::invalid_argument("[rgb2yiq_gpu] Invalid arguments (null pointers, non-positive dimensions/pitch).");
    }

    // Define kernel launch parameters
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                       (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel using the passed cudaStream_t handle
    rgb2yiqKernelUint8<<<gridSize, blockSize, 0, stream>>>(
        d_inputRgb,
        d_outputYiq,
        width,
        height,
        inputPitch,
        outputPitch
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[rgb2yiq_gpu] CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

void rgb2yiq_gpu(const float* d_inputRgb, float* d_outputYiq,
                 int width, int height, size_t inputPitch, size_t outputPitch,
                 cudaStream_t stream)
{
    // Basic validation
    if (!d_inputRgb || !d_outputYiq || width <= 0 || height <= 0 || inputPitch == 0 || outputPitch == 0) {
         throw std::invalid_argument("[rgb2yiq_gpu] Invalid arguments (null pointers, non-positive dimensions/pitch).");
    }

    // Define kernel launch parameters
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                       (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel using the passed cudaStream_t handle
    rgb2yiqKernel<<<gridSize, blockSize, 0, stream>>>(
        d_inputRgb,
        d_outputYiq,
        width,
        height,
        inputPitch,
        outputPitch
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[rgb2yiq_gpu] CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

void yiq2rgb_gpu(const float* d_inputYiq, float* d_outputRgb,
                 int width, int height, size_t inputPitch, size_t outputPitch,
                 cudaStream_t stream)
{
    // Basic validation
    if (!d_inputYiq || !d_outputRgb || width <= 0 || height <= 0 || inputPitch == 0 || outputPitch == 0) {
         throw std::invalid_argument("[yiq2rgb_gpu] Invalid arguments.");
    }

    // Define kernel launch parameters
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                       (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    yiq2rgbKernel<<<gridSize, blockSize, 0, stream>>>(
        d_inputYiq,
        d_outputRgb,
        width,
        height,
        inputPitch,
        outputPitch
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[yiq2rgb_gpu] CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace evmcuda