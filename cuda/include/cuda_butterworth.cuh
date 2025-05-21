#ifndef CUDA_BUTTERWORTH_CUH
#define CUDA_BUTTERWORTH_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <utility> // For std::pair
#include <opencv2/core.hpp>

namespace evmcuda {

// Type alias for vector of doubles
using double_vector = std::vector<double>;

/**
 * @brief Calculates the coefficients for a digital Butterworth filter using CUDA.
 * @param order The order of the filter (e.g., 1 based on CPU implementation).
 * @param cutoff_freq The cutoff frequency for low/high pass filter.
 * @param btype The type of filter: "low", "high" (bandpass/bandstop to be implemented).
 * @param fs The sampling frequency (e.g., video FPS).
 * @return A pair of vectors: {numerator coefficients (b), denominator coefficients (a)}.
 */
std::pair<double_vector, double_vector> calculate_butterworth_coeffs(
    int order,
    double cutoff_freq,
    const std::string& btype,
    double fs
);

/**
 * @brief Overload for bandpass/bandstop filters which require two frequencies
 */
std::pair<double_vector, double_vector> calculate_butterworth_coeffs(
    int order,
    const std::pair<double, double>& cutoff_freqs,
    const std::string& btype, // Should be "bandpass" or "bandstop"
    double fs
);

/**
 * @brief CUDA implementation of the Butterworth filter
 * Encapsulates filter coefficients and state for applying the filter
 */
class Butterworth {
public:
    /**
     * @brief Constructor for bandpass filter with frequency range [Wn_low, Wn_high]
     * @param Wn_low Low cutoff normalized frequency (f_cutoff / (fs/2))
     * @param Wn_high High cutoff normalized frequency (f_cutoff / (fs/2))
     */
    Butterworth(double Wn_low, double Wn_high);
    
    /**
     * @brief Applies the filter to an input signal (matrix of pixels)
     * @param d_input Input signal in device memory
     * @param width Width of the input signal matrix
     * @param height Height of the input signal matrix
     * @param channels Number of channels in the input signal
     * @param d_prev_input_state Previous input state in device memory
     * @param d_prev_output_state Previous output state in device memory
     * @param d_output Output signal buffer in device memory
     * @param stream CUDA stream to use for the operation
     */
    void filter(
        const float* d_input,
        int width,
        int height,
        int channels,
        float* d_prev_input_state,
        float* d_prev_output_state,
        float* d_output,
        cudaStream_t stream = nullptr
    );
    
    /**
     * @brief Get numerator coefficients
     * @return Vector of numerator coefficients
     */
    const double_vector& get_b_coeffs() const { return b_coeffs_; }
    
    /**
     * @brief Get denominator coefficients
     * @return Vector of denominator coefficients
     */
    const double_vector& get_a_coeffs() const { return a_coeffs_; }

private:
    int order_ = 1; // Based on CPU implementation
    double_vector b_coeffs_; // Numerator coefficients
    double_vector a_coeffs_; // Denominator coefficients (a[0] assumed 1)
};

/**
 * @brief Initialize resources needed by the Butterworth filter module
 * @return true if initialization succeeded, false otherwise
 */
bool init_butterworth();

/**
 * @brief Clean up resources used by the Butterworth filter module
 */
void cleanup_butterworth();

} // namespace evmcuda

#endif // CUDA_BUTTERWORTH_CUH