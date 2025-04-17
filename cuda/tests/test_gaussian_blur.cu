#include "gaussian_pyramid_cuda.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>

// Helper: Create a normalized 2D Gaussian kernel (host-side)
std::vector<float> createGaussianKernel(int size, float sigma) {
    assert(size % 2 == 1);
    int half = size / 2;
    std::vector<float> kernel(size * size);
    float sum = 0.0f;
    for (int y = -half; y <= half; ++y) {
        for (int x = -half; x <= half; ++x) {
            float val = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel[(y + half) * size + (x + half)] = val;
            sum += val;
        }
    }
    for (float& v : kernel) v /= sum;
    return kernel;
}

// Helper: Convert cv::Mat (CV_32FC3) to float array (H x W x 3)
void matToArray(const float* mat_data, float* arr, int height, int width, int channels) {
    std::copy(mat_data, mat_data + height * width * channels, arr);
}

// Helper: Compute max absolute difference between two float arrays
float maxAbsDiff(const float* a, const float* b, int n) {
    float maxd = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = std::abs(a[i] - b[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

// Simple CPU pyrDown: Gaussian blur + subsample
void cpuPyrDown(const float* input, float* output, int height, int width, int channels, const float* kernel, int kernel_size) {
    int out_height = height / 2;
    int out_width = width / 2;
    int half = kernel_size / 2;
    std::vector<float> blurred(height * width * channels, 0.0f);
    // Blur
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int ky = -half; ky <= half; ++ky) {
                    for (int kx = -half; kx <= half; ++kx) {
                        int ix = std::min(std::max(x + kx, 0), width - 1);
                        int iy = std::min(std::max(y + ky, 0), height - 1);
                        float kval = kernel[(ky + half) * kernel_size + (kx + half)];
                        sum += input[(iy * width + ix) * channels + c] * kval;
                    }
                }
                blurred[(y * width + x) * channels + c] = sum;
            }
        }
    }
    // Subsample
    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            for (int c = 0; c < channels; ++c) {
                output[(y * out_width + x) * channels + c] = blurred[(2 * y * width + 2 * x) * channels + c];
            }
        }
    }
}

// Simple CPU pyrUp: zero-insert + blur (kernel * 4)
void cpuPyrUp(const float* input, float* output, int height, int width, int channels, const float* kernel, int kernel_size) {
    int out_height = height * 2;
    int out_width = width * 2;
    int half = kernel_size / 2;
    std::vector<float> upsampled(out_height * out_width * channels, 0.0f);
    // Zero-insert
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                upsampled[(2 * y * out_width + 2 * x) * channels + c] = input[(y * width + x) * channels + c];
            }
        }
    }
    // Blur with kernel * 4
    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int ky = -half; ky <= half; ++ky) {
                    for (int kx = -half; kx <= half; ++kx) {
                        int ix = std::min(std::max(x + kx, 0), out_width - 1);
                        int iy = std::min(std::max(y + ky, 0), out_height - 1);
                        float kval = kernel[(ky + half) * kernel_size + (kx + half)] * 4.0f;
                        sum += upsampled[(iy * out_width + ix) * channels + c] * kval;
                    }
                }
                output[(y * out_width + x) * channels + c] = sum;
            }
        }
    }
}

// Simple test: Compare CUDA and CPU (OpenCV) Gaussian blur for a random image
int main() {
    int height = 128, width = 128, channels = 3;
    int kernel_size = 5;
    float sigma = 1.0f;
    std::vector<float> img(height * width * channels);
    for (float& v : img) v = static_cast<float>(rand()) / RAND_MAX;

    auto kernel = createGaussianKernel(kernel_size, sigma);
    std::vector<float> out_cuda(height * width * channels, 0.0f);

    // CUDA blur
    bool ok = cudaSpatiallyFilterGaussian(img.data(), out_cuda.data(), height, width, channels, kernel.data(), kernel_size);
    if (!ok) {
        std::cerr << "CUDA blur failed!\n";
        return 1;
    }

    // CPU blur (naive, for reference)
    std::vector<float> out_cpu(height * width * channels, 0.0f);
    int half = kernel_size / 2;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int ky = -half; ky <= half; ++ky) {
                    for (int kx = -half; kx <= half; ++kx) {
                        int ix = std::min(std::max(x + kx, 0), width - 1);
                        int iy = std::min(std::max(y + ky, 0), height - 1);
                        float kval = kernel[(ky + half) * kernel_size + (kx + half)];
                        sum += img[(iy * width + ix) * channels + c] * kval;
                    }
                }
                out_cpu[(y * width + x) * channels + c] = sum;
            }
        }
    }

    float maxdiff = maxAbsDiff(out_cuda.data(), out_cpu.data(), height * width * channels);
    std::cout << "Max abs diff (CUDA vs CPU): " << maxdiff << std::endl;
    if (maxdiff > 1e-4f) {
        std::cerr << "Test FAILED: difference too large!\n";
        return 2;
    }
    std::cout << "Test PASSED!\n";

    // --- Test pyrDown ---
    int down_height = height / 2, down_width = width / 2;
    std::vector<float> down_cpu(down_height * down_width * channels, 0.0f);
    std::vector<float> down_cuda(down_height * down_width * channels, 0.0f);
    cpuPyrDown(img.data(), down_cpu.data(), height, width, channels, kernel.data(), kernel_size);
    bool ok_down = cudaPyrDown(img.data(), down_cuda.data(), height, width, channels, kernel.data(), kernel_size);
    if (!ok_down) {
        std::cerr << "CUDA pyrDown failed!\n";
        return 3;
    }
    float maxdiff_down = maxAbsDiff(down_cpu.data(), down_cuda.data(), down_height * down_width * channels);
    std::cout << "Max abs diff (pyrDown CUDA vs CPU): " << maxdiff_down << std::endl;
    if (maxdiff_down > 1e-4f) {
        std::cerr << "pyrDown Test FAILED: difference too large!\n";
        return 4;
    }
    std::cout << "pyrDown Test PASSED!\n";

    // --- Test pyrUp ---
    int up_height = down_height * 2, up_width = down_width * 2;
    std::vector<float> up_cpu(up_height * up_width * channels, 0.0f);
    std::vector<float> up_cuda(up_height * up_width * channels, 0.0f);
    cpuPyrUp(down_cpu.data(), up_cpu.data(), down_height, down_width, channels, kernel.data(), kernel_size);
    bool ok_up = cudaPyrUp(down_cpu.data(), up_cuda.data(), down_height, down_width, channels, kernel.data(), kernel_size);
    if (!ok_up) {
        std::cerr << "CUDA pyrUp failed!\n";
        return 5;
    }
    float maxdiff_up = maxAbsDiff(up_cpu.data(), up_cuda.data(), up_height * up_width * channels);
    std::cout << "Max abs diff (pyrUp CUDA vs CPU): " << maxdiff_up << std::endl;
    if (maxdiff_up > 1e-4f) {
        std::cerr << "pyrUp Test FAILED: difference too large!\n";
        return 6;
    }
    std::cout << "pyrUp Test PASSED!\n";
    return 0;
}
