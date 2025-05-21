#include "cuda_butterworth.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <complex>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>

// Placeholder for logging
#define LOG_BUTTER(message) std::cout << "[CUDA BUTTER LOG] " << message << std::endl

// Define PI if not available
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace evmcuda {

using complex_t = std::complex<double>;
using complex_vector = std::vector<complex_t>;

// Constant memory for filter coefficients
__constant__ float d_butterworth_b[2]; // Numerator coefficients
__constant__ float d_butterworth_a[2]; // Denominator coefficients

// Helper function to multiply polynomials represented by coefficient vectors
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

// Helper function to convert poles/zeros to polynomial coefficients
double_vector roots_to_poly(const complex_vector& roots) {
    double_vector poly = {1.0}; // Start with z^0 coefficient
    for (const auto& root : roots) {
        // Treat as real root if imaginary part is very small
        if (std::abs(root.imag()) < 1e-10) {
            poly = poly_mult(poly, {1.0, -root.real()});
        } else {
            // Handle complex conjugate pairs
            if (root.imag() > 0) {
                double real_part = root.real();
                double mag_sq = std::norm(root); // magnitude squared
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

// Kernel to apply the butterworth filter
__global__ void butterworth_filter_kernel(
    const float* __restrict__ d_input,
    const float* __restrict__ d_prev_input,
    const float* __restrict__ d_prev_output,
    float* __restrict__ d_output,
    int width,
    int height,
    int channels,
    int stride)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Return if out of bounds
    if (x >= width || y >= height) return;
    
    const int idx = (y * stride + x) * channels;
    
    // Apply the first-order IIR filter equation for each channel:
    // output = b[0]*input + b[1]*prev_input - a[1]*prev_output
    // Note: a[0] is assumed to be 1
    for (int c = 0; c < channels; c++) {
        d_output[idx + c] = d_butterworth_b[0] * d_input[idx + c] + 
                           d_butterworth_b[1] * d_prev_input[idx + c] - 
                           d_butterworth_a[1] * d_prev_output[idx + c];
    }
}

// Implementation of the butterworth calculation function
std::pair<double_vector, double_vector> calculate_butterworth_coeffs(
    int order,
    double cutoff_freq,
    const std::string& btype,
    double fs)
{
    LOG_BUTTER("Calculating Butterworth: order=" + std::to_string(order) +
               ", cutoff=" + std::to_string(cutoff_freq) + ", type=" + btype + ", Fs=" + std::to_string(fs));

    if (order <= 0) throw std::invalid_argument("Filter order must be positive.");
    if (fs <= 0) throw std::invalid_argument("Sampling frequency (fs) must be positive.");
    if (cutoff_freq <= 0 || cutoff_freq >= fs / 2.0) {
        throw std::invalid_argument("Cutoff frequency must be between 0 and fs/2.");
    }

    // Pre-warp frequency
    double omega_c = (2.0 * fs) * std::tan(M_PI * cutoff_freq / fs);
    LOG_BUTTER("Pre-warped analog cutoff omega_c: " + std::to_string(omega_c));

    // Analog Low-Pass Prototype Poles (cutoff = 1 rad/s)
    complex_vector analog_poles_proto;
    for (int k = 0; k < order; ++k) {
        double angle = M_PI * (2.0 * k + order + 1.0) / (2.0 * order);
        analog_poles_proto.push_back(complex_t(std::cos(angle), std::sin(angle)));
    }

    // Frequency Transformation
    complex_vector analog_poles;
    complex_vector analog_zeros;
    double gain = 1.0;

    if (btype == "low") {
        // Scale prototype poles by omega_c
        std::transform(analog_poles_proto.begin(), analog_poles_proto.end(),
                      std::back_inserter(analog_poles),
                      [omega_c](const complex_t& p){ return p * omega_c; });
    } else if (btype == "high") {
        // Transform s -> omega_c / s
        std::transform(analog_poles_proto.begin(), analog_poles_proto.end(),
                      std::back_inserter(analog_poles),
                      [omega_c](const complex_t& p){ return omega_c / p; });
        // High-pass introduces 'order' zeros at s=0 (origin)
        for(int i=0; i<order; ++i) analog_zeros.push_back(complex_t(0.0, 0.0));
    } else {
        throw std::invalid_argument("Filter type '" + btype + "' not implemented yet.");
    }

    // Digital Conversion (Bilinear Transform)
    // z = (2*fs + s) / (2*fs - s)  =>  s = 2*fs * (z - 1) / (z + 1)
    complex_vector digital_poles;
    complex_vector digital_zeros;
    double fs2 = 2.0 * fs;

    std::transform(analog_poles.begin(), analog_poles.end(),
                  std::back_inserter(digital_poles),
                  [fs2](const complex_t& p){ return (fs2 + p) / (fs2 - p); });

    std::transform(analog_zeros.begin(), analog_zeros.end(),
                  std::back_inserter(digital_zeros),
                  [fs2](const complex_t& z){ return (fs2 + z) / (fs2 - z); });

    // Butterworth low-pass analog zeros at infinity map to z = -1
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

    gain = std::abs(1.0 / freq_response_at_norm_point);
    LOG_BUTTER("Calculated gain k: " + std::to_string(gain));

    // Calculate Coefficients
    double_vector b = roots_to_poly(digital_zeros);
    double_vector a = roots_to_poly(digital_poles);

    // Apply gain to numerator coefficients
    std::transform(b.begin(), b.end(), b.begin(), [gain](double val){ return val * gain; });

    // Normalize denominator so a[0] = 1
    if (!a.empty() && std::abs(a[0] - 1.0) > 1e-9) {
        LOG_BUTTER("Warning: Normalizing denominator coefficients.");
        double a0 = a[0];
        std::transform(a.begin(), a.end(), a.begin(), [a0](double val){ return val / a0; });
        // Also scale b by a0 to keep transfer function equivalent
        std::transform(b.begin(), b.end(), b.begin(), [a0](double val){ return val / a0; });
    }

    LOG_BUTTER("Calculated coefficients: b.size=" + std::to_string(b.size()) + ", a.size=" + std::to_string(a.size()));
    return {b, a};
}

// Overload for band filters (placeholder implementation)
std::pair<double_vector, double_vector> calculate_butterworth_coeffs(
    int order,
    const std::pair<double, double>& cutoff_freqs,
    const std::string& btype,
    double fs)
{
    LOG_BUTTER("Warning: Band filter Butterworth implementation not yet complete.");
    throw std::runtime_error("Bandpass/Bandstop Butterworth not implemented yet.");
}

// Butterworth class implementation
Butterworth::Butterworth(double Wn_low, double Wn_high) {
    // Calculate coefficients for the bandpass filter (approximated as lowpass)
    // Normalized frequencies Wn are relative to Nyquist (Fs/2)
    double fs_normalized = 2.0; // Nyquist = 1.0, so Fs = 2.0
    double cutoff_high = Wn_high * (fs_normalized / 2.0);
    
    LOG_BUTTER("Initializing Butterworth class for low-pass at Wn=" + std::to_string(Wn_high));
    
    // Use order 1 filter based on CPU implementation
    order_ = 1;
    try {
        // Calculate coefficients for the higher cutoff frequency (main low-pass stage)
        auto coeffs_high = calculate_butterworth_coeffs(order_, cutoff_high, "low", fs_normalized);
        b_coeffs_ = coeffs_high.first;
        a_coeffs_ = coeffs_high.second;
        
        // Validation of coefficient sizes
        if (b_coeffs_.size() != order_ + 1 || a_coeffs_.size() != order_ + 1) {
            throw std::runtime_error("Butterworth coefficient calculation returned unexpected size for order " + 
                                    std::to_string(order_));
        }
        if (std::abs(a_coeffs_[0] - 1.0) > 1e-9) {
            throw std::runtime_error("Denominator coefficient a[0] must be 1.0 after normalization.");
        }
        
        // Copy coefficients to device constant memory
        float h_b[2] = { static_cast<float>(b_coeffs_[0]), static_cast<float>(b_coeffs_[1]) };
        float h_a[2] = { static_cast<float>(a_coeffs_[0]), static_cast<float>(a_coeffs_[1]) };
        
        cudaError_t err = cudaMemcpyToSymbol(d_butterworth_b, h_b, 2 * sizeof(float));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy b coefficients to constant memory: " + 
                                    std::string(cudaGetErrorString(err)));
        }
        
        err = cudaMemcpyToSymbol(d_butterworth_a, h_a, 2 * sizeof(float));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy a coefficients to constant memory: " + 
                                    std::string(cudaGetErrorString(err)));
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error initializing Butterworth filter coefficients: ") + e.what());
    }
    
    LOG_BUTTER("Butterworth class initialized with b=" + std::to_string(b_coeffs_[0]) + "," + 
               std::to_string(b_coeffs_[1]) + " a=" + std::to_string(a_coeffs_[0]) + "," + 
               std::to_string(a_coeffs_[1]));
}

// Filter method implementation
void Butterworth::filter(
    const float* d_input,
    int width,
    int height,
    int channels,
    float* d_prev_input_state,
    float* d_prev_output_state,
    float* d_output,
    cudaStream_t stream)
{
    if (b_coeffs_.size() != order_ + 1 || a_coeffs_.size() != order_ + 1) {
        throw std::runtime_error("Butterworth filter coefficients are not correctly initialized.");
    }
    
    if (width <= 0 || height <= 0 || channels <= 0) {
        throw std::invalid_argument("Invalid dimensions for Butterworth::filter.");
    }
    
    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                 (height + blockSize.y - 1) / blockSize.y);
    
    // Launch the butterworth filter kernel
    butterworth_filter_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_prev_input_state, d_prev_output_state, d_output,
        width, height, channels, width
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to launch butterworth_filter_kernel: " + 
                                std::string(cudaGetErrorString(err)));
    }
}

// Module initialization
bool init_butterworth() {
    // No specific initialization needed for now
    return true;
}

// Module cleanup
void cleanup_butterworth() {
    // No specific cleanup needed for now
}

} // namespace evmcuda