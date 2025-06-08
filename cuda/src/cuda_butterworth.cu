#include "../include/cuda_butterworth.cuh"
#include <iostream>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cuda_evm {

// Helper functions from CPU implementation
using complex_t = std::complex<double>;
using complex_vector = std::vector<complex_t>;
using double_vector = std::vector<double>;

double_vector poly_mult(const double_vector& p1, const double_vector& p2) {
    if (p1.empty() || p2.empty()) return {};
    size_t n1 = p1.size();
    size_t n2 = p2.size();
    double_vector result(n1 + n2 - 1, 0.0);
    for (size_t i = 0; i < n1; ++i) {
        for (size_t j = 0; j < n2; ++j) {
            result[i + j] += p1[i] * p2[j];
        }
    }
    return result;
}

double_vector roots_to_poly(const complex_vector& roots) {
    double_vector poly = {1.0};
    for (const auto& root : roots) {
        if (std::abs(root.imag()) < 1e-10) {
            poly = poly_mult(poly, {1.0, -root.real()});
        } else {
            if (root.imag() > 0) {
                double real_part = root.real();
                double mag_sq = std::norm(root);
                poly = poly_mult(poly, {1.0, -2.0 * real_part, mag_sq});
            } else if (root.imag() < -1e-10) {
                double real_part = root.real();
                double mag_sq = std::norm(root);
                poly = poly_mult(poly, {1.0, -2.0 * real_part, mag_sq});
            }
        }
    }
    return poly;
}

// Host-side coefficient calculation (reuses CPU logic)
std::pair<std::vector<double>, std::vector<double>> calculateButterworthCoeffs(
    int order,
    double cutoff_freq,
    const std::string& btype,
    double fs)
{
    if (order <= 0) throw std::invalid_argument("Filter order must be positive.");
    if (fs <= 0) throw std::invalid_argument("Sampling frequency (fs) must be positive.");
    if (cutoff_freq <= 0 || cutoff_freq >= fs / 2.0) {
        throw std::invalid_argument("Cutoff frequency must be between 0 and fs/2.");
    }

    // Pre-warp frequency
    double omega_c = (2.0 * fs) * std::tan(M_PI * cutoff_freq / fs);

    // Analog Low-Pass Prototype Poles
    complex_vector analog_poles_proto;
    for (int k = 0; k < order; ++k) {
        double angle = M_PI * (2.0 * k + order + 1.0) / (2.0 * order);
        analog_poles_proto.push_back(complex_t(std::cos(angle), std::sin(angle)));
    }

    // Frequency Transformation
    complex_vector analog_poles;
    complex_vector analog_zeros;

    if (btype == "low") {
        std::transform(analog_poles_proto.begin(), analog_poles_proto.end(),
                       std::back_inserter(analog_poles),
                       [omega_c](const complex_t& p){ return p * omega_c; });
    } else if (btype == "high") {
        std::transform(analog_poles_proto.begin(), analog_poles_proto.end(),
                       std::back_inserter(analog_poles),
                       [omega_c](const complex_t& p){ return omega_c / p; });
        for(int i=0; i<order; ++i) analog_zeros.push_back(complex_t(0.0, 0.0));
    } else {
        throw std::invalid_argument("Filter type '" + btype + "' not implemented yet.");
    }

    // Digital Conversion (Bilinear Transform)
    complex_vector digital_poles;
    complex_vector digital_zeros;
    double fs2 = 2.0 * fs;

    std::transform(analog_poles.begin(), analog_poles.end(),
                   std::back_inserter(digital_poles),
                   [fs2](const complex_t& p){ return (fs2 + p) / (fs2 - p); });

    std::transform(analog_zeros.begin(), analog_zeros.end(),
                   std::back_inserter(digital_zeros),
                   [fs2](const complex_t& z){ return (fs2 + z) / (fs2 - z); });

    if (btype == "low") {
        for(int i=0; i<order; ++i) digital_zeros.push_back(complex_t(-1.0, 0.0));
    }

    // Calculate Gain
    complex_t freq_response_at_norm_point(1.0, 0.0);
    complex_t norm_point = (btype == "low") ? complex_t(1.0, 0.0) : complex_t(-1.0, 0.0);

    for(const auto& z : digital_zeros) {
        freq_response_at_norm_point *= (norm_point - z);
    }
    for(const auto& p : digital_poles) {
        freq_response_at_norm_point /= (norm_point - p);
    }

    double gain = std::abs(1.0 / freq_response_at_norm_point);

    // Calculate Coefficients
    double_vector b = roots_to_poly(digital_zeros);
    double_vector a = roots_to_poly(digital_poles);

    // Apply gain to numerator coefficients
    std::transform(b.begin(), b.end(), b.begin(), [gain](double val){ return val * gain; });

    // Normalize denominator
    if (!a.empty() && std::abs(a[0] - 1.0) > 1e-9) {
        double a0 = a[0];
        std::transform(a.begin(), a.end(), a.begin(), [a0](double val){ return val / a0; });
        std::transform(b.begin(), b.end(), b.begin(), [a0](double val){ return val / a0; });
    }

    return {b, a};
}

std::pair<std::vector<double>, std::vector<double>> calculateButterworthCoeffs(
    int order,
    const std::pair<double, double>& cutoff_freqs,
    const std::string& btype,
    double fs)
{
    throw std::runtime_error("Bandpass/Bandstop Butterworth not implemented yet.");
}

// CUDA kernel for IIR filtering
__global__ void butterworthFilterKernel(
    const float* input, 
    const float* prev_input,
    const float* prev_output,
    float* output,
    float* new_prev_input,
    float* new_prev_output,
    const float* b_coeffs,
    const float* a_coeffs,
    int width, 
    int height, 
    int channels,
    int order
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx >= width || idy >= height || idz >= channels) return;
    
    int pixel_idx = (idy * width + idx) * channels + idz;
    
    // Apply 1st order IIR filter: output = b[0]*input + b[1]*prev_input - a[1]*prev_output
    float result = b_coeffs[0] * input[pixel_idx] + 
                   b_coeffs[1] * prev_input[pixel_idx] - 
                   a_coeffs[1] * prev_output[pixel_idx];
    
    output[pixel_idx] = result;
    new_prev_input[pixel_idx] = input[pixel_idx];
    new_prev_output[pixel_idx] = result;
}

// Utility function for error checking
cudaError_t checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " - " << cudaGetErrorString(error) << std::endl;
    }
    return error;
}

// CudaButterworth class implementation
CudaButterworth::CudaButterworth(double Wn_low, double Wn_high) 
    : order_(1), d_b_coeffs_(nullptr), d_a_coeffs_(nullptr) {
    
    double fs_normalized = 2.0;
    double cutoff_low = Wn_low * (fs_normalized / 2.0);
    double cutoff_high = Wn_high * (fs_normalized / 2.0);
    
    try {
        auto coeffs_high = calculateButterworthCoeffs(order_, cutoff_high, "low", fs_normalized);
        b_coeffs_ = coeffs_high.first;
        a_coeffs_ = coeffs_high.second;
        
        if (b_coeffs_.size() != order_ + 1 || a_coeffs_.size() != order_ + 1) {
            throw std::runtime_error("Butterworth coefficient calculation returned unexpected size for order " + std::to_string(order_));
        }
        if (std::abs(a_coeffs_[0] - 1.0) > 1e-9) {
            throw std::runtime_error("Denominator coefficient a[0] must be 1.0 after normalization.");
        }
        
        allocateDeviceMemory();
        copyCoeffsToDevice();
        
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error initializing CUDA Butterworth filter coefficients: ") + e.what());
    }
}

CudaButterworth::~CudaButterworth() {
    freeDeviceMemory();
}

void CudaButterworth::allocateDeviceMemory() {
    size_t coeff_size = (order_ + 1) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_b_coeffs_, coeff_size));
    CUDA_CHECK(cudaMalloc(&d_a_coeffs_, coeff_size));
}

void CudaButterworth::copyCoeffsToDevice() {
    std::vector<float> b_float(b_coeffs_.begin(), b_coeffs_.end());
    std::vector<float> a_float(a_coeffs_.begin(), a_coeffs_.end());
    
    size_t coeff_size = (order_ + 1) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_b_coeffs_, b_float.data(), coeff_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_coeffs_, a_float.data(), coeff_size, cudaMemcpyHostToDevice));
}

void CudaButterworth::freeDeviceMemory() {
    if (d_b_coeffs_) {
        cudaFree(d_b_coeffs_);
        d_b_coeffs_ = nullptr;
    }
    if (d_a_coeffs_) {
        cudaFree(d_a_coeffs_);
        d_a_coeffs_ = nullptr;
    }
}

void CudaButterworth::filter(const float* d_input, float* d_prev_input_state, 
                            float* d_prev_output_state, float* d_output, 
                            int width, int height, int channels, cudaStream_t stream) {
    
    // Calculate grid and block dimensions
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y,
                  (channels + blockSize.z - 1) / blockSize.z);
    
    // Launch kernel
    butterworthFilterKernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_prev_input_state, d_prev_output_state,
        d_output, d_prev_input_state, d_prev_output_state,
        d_b_coeffs_, d_a_coeffs_,
        width, height, channels, order_
    );
    
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_evm