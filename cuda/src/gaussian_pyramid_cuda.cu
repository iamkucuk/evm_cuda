// CUDA implementation of spatial Gaussian blur for EVM
#include "gaussian_pyramid_cuda.hpp"
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for 2D Gaussian blur (single channel, float32)
__global__ void gaussianBlurKernel(
    const float* input, float* output,
    int height, int width, int channels,
    const float* kernel, int kernel_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int half = kernel_size / 2;
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int ky = -half; ky <= half; ++ky) {
            for (int kx = -half; kx <= half; ++kx) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                float kval = kernel[(ky + half) * kernel_size + (kx + half)];
                sum += input[(iy * width + ix) * channels + c] * kval;
            }
        }
        output[(y * width + x) * channels + c] = sum;
    }
}

// Host wrapper for CUDA Gaussian blur
bool cudaSpatiallyFilterGaussian(
    const float* input, float* output, int height, int width, int channels,
    const float* kernel, int kernel_size
) {
    if (!input || !output || !kernel || channels != 3) return false;
    size_t img_bytes = height * width * channels * sizeof(float);
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_kernel = nullptr;
    cudaMalloc(&d_input, img_bytes);
    cudaMalloc(&d_output, img_bytes);
    cudaMalloc(&d_kernel, kernel_bytes);
    cudaMemcpy(d_input, input, img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    gaussianBlurKernel<<<grid, block>>>(d_input, d_output, height, width, channels, d_kernel, kernel_size);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, img_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    return true;
}

// CUDA kernel for pyrDown: Gaussian blur + subsample
__global__ void pyrDownKernel(
    const float* input, float* output,
    int in_height, int in_width, int channels,
    const float* kernel, int kernel_size
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_height = in_height / 2;
    int out_width = in_width / 2;
    if (out_x >= out_width || out_y >= out_height) return;
    int half = kernel_size / 2;
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int ky = -half; ky <= half; ++ky) {
            for (int kx = -half; kx <= half; ++kx) {
                int in_x = min(max(2 * out_x + kx, 0), in_width - 1);
                int in_y = min(max(2 * out_y + ky, 0), in_height - 1);
                float kval = kernel[(ky + half) * kernel_size + (kx + half)];
                sum += input[(in_y * in_width + in_x) * channels + c] * kval;
            }
        }
        output[(out_y * out_width + out_x) * channels + c] = sum;
    }
}

bool cudaPyrDown(
    const float* input, float* output, int height, int width, int channels,
    const float* kernel, int kernel_size
) {
    if (!input || !output || !kernel || channels != 3) return false;
    int out_height = height / 2;
    int out_width = width / 2;
    size_t in_bytes = height * width * channels * sizeof(float);
    size_t out_bytes = out_height * out_width * channels * sizeof(float);
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_kernel = nullptr;
    cudaMalloc(&d_input, in_bytes);
    cudaMalloc(&d_output, out_bytes);
    cudaMalloc(&d_kernel, kernel_bytes);
    cudaMemcpy(d_input, input, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((out_width + block.x - 1) / block.x, (out_height + block.y - 1) / block.y);
    pyrDownKernel<<<grid, block>>>(d_input, d_output, height, width, channels, d_kernel, kernel_size);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    return true;
}

// CUDA kernel for pyrUp: zero-insert + Gaussian blur (kernel * 4)
__global__ void pyrUpZeroInsertKernel(
    const float* input, float* upsampled,
    int in_height, int in_width, int channels,
    int out_height, int out_width
) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= out_height || x >= out_width) return;
    for (int c = 0; c < channels; ++c) {
        upsampled[(y * out_width + x) * channels + c] = 0.0f;
        if (y % 2 == 0 && x % 2 == 0) {
            int in_y = y / 2;
            int in_x = x / 2;
            if (in_y < in_height && in_x < in_width) {
                upsampled[(y * out_width + x) * channels + c] = input[(in_y * in_width + in_x) * channels + c];
            }
        }
    }
}

__global__ void pyrUpBlurKernel(
    const float* input, float* output,
    int height, int width, int channels,
    const float* kernel, int kernel_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int half = kernel_size / 2;
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int ky = -half; ky <= half; ++ky) {
            for (int kx = -half; kx <= half; ++kx) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                float kval = kernel[(ky + half) * kernel_size + (kx + half)] * 4.0f;
                sum += input[(iy * width + ix) * channels + c] * kval;
            }
        }
        output[(y * width + x) * channels + c] = sum;
    }
}

bool cudaPyrUp(
    const float* input, float* output, int height, int width, int channels,
    const float* kernel, int kernel_size
) {
    if (!input || !output || !kernel || channels != 3) return false;
    int out_height = height * 2;
    int out_width = width * 2;
    size_t in_bytes = height * width * channels * sizeof(float);
    size_t up_bytes = out_height * out_width * channels * sizeof(float);
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);

    float* d_input = nullptr;
    float* d_upsampled = nullptr;
    float* d_output = nullptr;
    float* d_kernel = nullptr;
    cudaMalloc(&d_input, in_bytes);
    cudaMalloc(&d_upsampled, up_bytes);
    cudaMalloc(&d_output, up_bytes);
    cudaMalloc(&d_kernel, kernel_bytes);
    cudaMemcpy(d_input, input, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((out_width + block.x - 1) / block.x, (out_height + block.y - 1) / block.y);
    pyrUpZeroInsertKernel<<<grid, block>>>(d_input, d_upsampled, height, width, channels, out_height, out_width);
    cudaDeviceSynchronize();
    pyrUpBlurKernel<<<grid, block>>>(d_upsampled, d_output, out_height, out_width, channels, d_kernel, kernel_size);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, up_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_upsampled);
    cudaFree(d_output);
    cudaFree(d_kernel);
    return true;
}
