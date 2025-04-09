#include "evmcuda/reconstruction.cuh"
#include "evmcuda/color_conversion.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace {

// YIQ to RGB conversion matrix (stored in constant memory)
// Using the more precise version found in color_conversion.cu
__constant__ float d_yiq2rgb[9] = {
    1.0f,        0.9559863f,   0.6208248f,
    1.0f,       -0.2720128f,  -0.6472042f,
    1.0f,       -1.1067402f,   1.7042304f
};

// Kernel for adding filtered signal to original frame
__global__ void addFilteredSignalKernel(
    const float* original_yiq,
    size_t original_pitch,
    const float* filtered_yiq,
    size_t filtered_pitch,
    float* result_yiq,
    size_t result_pitch,
    int width,
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Convert pitch from bytes to float elements (3 channels)
    const size_t orig_stride = original_pitch / sizeof(float);
    const size_t filt_stride = filtered_pitch / sizeof(float);
    const size_t res_stride = result_pitch / sizeof(float);
    
    // Process all 3 channels (Y,I,Q)
    for (int c = 0; c < 3; ++c) {
        const int idx = y * orig_stride + x * 3 + c;
        result_yiq[idx] = original_yiq[idx] + filtered_yiq[y * filt_stride + x * 3 + c];
    }
}

// Combined kernel for YIQ->RGB conversion, clipping, and uint8 casting
__global__ void convertYiqToRgbClipAndCastKernel(
    const float* yiq_frame,
    size_t yiq_pitch,
    unsigned char* rgb_frame,
    size_t rgb_pitch,
    int width,
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Get input YIQ pixel
    const size_t yiq_stride = yiq_pitch / sizeof(float);
    float yiq[3];
    for (int c = 0; c < 3; ++c) {
        yiq[c] = yiq_frame[y * yiq_stride + x * 3 + c];
    }
    
    // Convert to RGB using the constant matrix
    float rgb[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            rgb[i] += d_yiq2rgb[i * 3 + j] * yiq[j];
        }
        
        // Clip to [0, 255]
        rgb[i] = fminf(fmaxf(rgb[i], 0.0f), 255.0f);
    }
    
    // Store as uint8
    const size_t rgb_stride = rgb_pitch;  // Pitch is already in bytes for uint8
    for (int c = 0; c < 3; ++c) {
        rgb_frame[y * rgb_stride + x * 3 + c] = static_cast<unsigned char>(rgb[c]);
    }
}

} // anonymous namespace

namespace evmcuda {

cudaError_t addFilteredSignal(
    const float* original_yiq,
    size_t original_pitch,
    const float* filtered_yiq,
    size_t filtered_pitch,
    float* result_yiq,
    size_t result_pitch,
    int width,
    int height,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
              
    addFilteredSignalKernel<<<grid, block, 0, stream>>>(
        original_yiq, original_pitch,
        filtered_yiq, filtered_pitch,
        result_yiq, result_pitch,
        width, height
    );
    
    return cudaGetLastError();
}

cudaError_t convertYiqToRgbClipAndCast(
    const float* yiq_frame,
    size_t yiq_pitch,
    unsigned char* rgb_frame,
    size_t rgb_pitch,
    int width,
    int height,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
              
    convertYiqToRgbClipAndCastKernel<<<grid, block, 0, stream>>>(
        yiq_frame, yiq_pitch,
        rgb_frame, rgb_pitch,
        width, height
    );
    
    return cudaGetLastError();
}

cudaError_t reconstructGaussianFrame(
    const cv::Mat& original_rgb,
    const float* d_filtered_yiq,
    size_t filtered_pitch,
    cv::Mat& output_rgb,
    cudaStream_t stream)
{
    if (original_rgb.empty() || original_rgb.type() != CV_8UC3) {
        return cudaErrorInvalidValue;
    }

    const int width = original_rgb.cols;
    const int height = original_rgb.rows;

    // Ensure output matrix has correct size and type
    if (output_rgb.size() != original_rgb.size() || output_rgb.type() != CV_8UC3) {
        output_rgb.create(original_rgb.size(), CV_8UC3);
    }

    // Define grid and block dimensions before any potential goto
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // Allocate device memory
    unsigned char *d_original_rgb = nullptr;
    float *d_original_yiq = nullptr;
    float *d_combined_yiq = nullptr;

    size_t rgb_pitch = 0;
    size_t yiq_pitch = 0;
    cudaError_t error = cudaSuccess;

    // Allocate memory and initialize pointers
    error = cudaMallocPitch(&d_original_rgb, &rgb_pitch, width * 3, height);
    if (error != cudaSuccess) goto cleanup;

    error = cudaMallocPitch(&d_original_yiq, &yiq_pitch, width * 3 * sizeof(float), height);
    if (error != cudaSuccess) goto cleanup;

    error = cudaMallocPitch(&d_combined_yiq, &yiq_pitch, width * 3 * sizeof(float), height);
    if (error != cudaSuccess) goto cleanup;

    // Copy input RGB (Host) to d_original_rgb (Device)
    error = cudaMemcpy2DAsync(d_original_rgb, rgb_pitch,
                             original_rgb.data, original_rgb.step,
                             width * 3, height,
                             cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) goto cleanup;

    // Convert uint8 RGB directly to YIQ using the new overload
    rgb2yiq_gpu(d_original_rgb, d_original_yiq,
                width, height, rgb_pitch, yiq_pitch,
                stream);
    error = cudaGetLastError();
    if (error != cudaSuccess) goto cleanup;

    // Add filtered signal (Device)
    error = addFilteredSignal(d_original_yiq, yiq_pitch,
                             d_filtered_yiq, filtered_pitch,
                             d_combined_yiq, yiq_pitch,
                             width, height, stream);
    if (error != cudaSuccess) goto cleanup;

    // Convert combined YIQ back to RGB uint8 with clipping (Device)
    error = convertYiqToRgbClipAndCast(d_combined_yiq, yiq_pitch,
                                      d_original_rgb, rgb_pitch,
                                      width, height, stream);
    if (error != cudaSuccess) goto cleanup;

    // Copy final RGB result (Device) back to output_rgb (Host)
    error = cudaMemcpy2DAsync(output_rgb.data, output_rgb.step,
                             d_original_rgb, rgb_pitch,
                             width * 3, height,
                             cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) goto cleanup;

    // Synchronize stream if asynchronous copies/kernels were used
    error = cudaStreamSynchronize(stream);

cleanup:
    if (d_original_rgb) cudaFree(d_original_rgb);
    if (d_original_yiq) cudaFree(d_original_yiq);
    if (d_combined_yiq) cudaFree(d_combined_yiq);

    return error;
}

} // namespace evmcuda