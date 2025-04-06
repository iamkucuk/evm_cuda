#include "evmcpp/butterworth.hpp"

#include <vector>
#include <string>
#include <utility> // For std::pair
#include <stdexcept> // For exceptions
#include <iostream> // For placeholder logging
#include <cmath> // For M_PI etc. (might be needed for implementation)

#include <complex> // For complex numbers
#include <vector>
#include <cmath>   // For M_PI, tan, exp, cos, sin, pow, abs, real, imag
#include <numeric> // For std::accumulate
#include <algorithm> // For std::transform

// Placeholder for logging
#define LOG_BUTTER(message) std::cout << "[BUTTER LOG] " << message << std::endl

// Define PI if not available (cmath might not define M_PI in strict C++)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace evmcpp {

    using complex_t = std::complex<double>;
    using complex_vector = std::vector<complex_t>;
    // using double_vector = std::vector<double>; // Removed, now defined in header
    using double_vector = std::vector<double>;

    // Helper function to multiply polynomials represented by coefficient vectors
    // Example: poly_mult({1, 2}, {1, 3}) -> {1, 5, 6} representing (x+2)(x+3)=x^2+5x+6
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
    // Converts roots (poles or zeros) into polynomial coefficients B(z) or A(z)
    // Example: roots_to_poly({-2, -3}) -> {1, 5, 6} representing (z+2)(z+3)=z^2+5z+6
    double_vector roots_to_poly(const complex_vector& roots) {
        double_vector poly = {1.0}; // Start with z^0 coefficient
        for (const auto& root : roots) {
            // Multiply by (z - root) = (1*z - root) -> coefficients {1, -root}
            // Handle complex conjugate pairs carefully to ensure real coefficients
            if (std::abs(root.imag()) < 1e-10) { // Treat as real root
                poly = poly_mult(poly, {1.0, -root.real()});
            } else {
                // If complex, its conjugate must also be present (or should be added)
                // Multiply by (z - root)(z - conj(root)) = z^2 - 2*real(root)*z + |root|^2
                // Coefficients: {1, -2*real(root), |root|^2}
                // This assumes roots come in conjugate pairs for real filters.
                // We only process one of the pair (e.g., the one with positive imag part)
                if (root.imag() > 0) { // Process only one of the conjugate pair
                     double real_part = root.real();
                     double mag_sq = std::norm(root); // norm is magnitude squared
                     poly = poly_mult(poly, {1.0, -2.0 * real_part, mag_sq});
                } else if (root.imag() < -1e-10) {
                    // If only negative imaginary part is present, something is wrong, but handle it
                     double real_part = root.real();
                     double mag_sq = std::norm(root);
                     poly = poly_mult(poly, {1.0, -2.0 * real_part, mag_sq});
                }
                 // If imag is exactly zero, it was handled by the real root case.
            }
        }
        return poly;
    }


    // --- Main Butterworth Calculation ---
    std::pair<double_vector, double_vector> calculateButterworthCoeffs(
        int order,
        double cutoff_freq, // For low/high pass
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

        // --- 1. Pre-warp frequency ---
        double omega_c = (2.0 * fs) * std::tan(M_PI * cutoff_freq / fs);
        LOG_BUTTER("Pre-warped analog cutoff omega_c: " + std::to_string(omega_c));

        // --- 2. Analog Low-Pass Prototype Poles (cutoff = 1 rad/s) ---
        complex_vector analog_poles_proto;
        for (int k = 0; k < order; ++k) {
            double angle = M_PI * (2.0 * k + order + 1.0) / (2.0 * order);
            // Poles are on the unit circle in the left-half plane
            analog_poles_proto.push_back(complex_t(std::cos(angle), std::sin(angle)));
        }

        // --- 3. Frequency Transformation ---
        complex_vector analog_poles;
        complex_vector analog_zeros; // Zeros might be introduced for HP/BP/BS
        double gain = 1.0;

        if (btype == "low") {
            // Scale prototype poles by omega_c
            std::transform(analog_poles_proto.begin(), analog_poles_proto.end(),
                           std::back_inserter(analog_poles),
                           [omega_c](const complex_t& p){ return p * omega_c; });
            // Low-pass Butterworth has 'order' zeros at infinity, gain needs calculation later
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
        // TODO: Implement band-pass and band-stop transformations if needed

        // --- 4. Digital Conversion (Bilinear Transform) ---
        // z = (2*fs + s) / (2*fs - s)  =>  s = 2*fs * (z - 1) / (z + 1)
        // Poles: p_digital = (2*fs + p_analog) / (2*fs - p_analog)
        // Zeros: z_digital = (2*fs + z_analog) / (2*fs - z_analog)
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
        // High-pass analog zeros at s=0 map to z = 1
        // (Already handled by transforming analog_zeros containing 0.0)


        // --- 5. Calculate Gain ---
        // Gain 'k' is chosen to normalize the frequency response.
        // For low-pass, gain at DC (z=1) should be 1.
        // For high-pass, gain at Nyquist (z=-1) should be 1.
        // H(z) = k * Product(z - zero_k) / Product(z - pole_k)
        complex_t freq_response_at_norm_point(1.0, 0.0);
        complex_t norm_point = (btype == "low") ? complex_t(1.0, 0.0) : complex_t(-1.0, 0.0); // z=1 for LP, z=-1 for HP

        for(const auto& z : digital_zeros) {
            freq_response_at_norm_point *= (norm_point - z);
        }
        for(const auto& p : digital_poles) {
            freq_response_at_norm_point /= (norm_point - p);
        }

        gain = std::abs(1.0 / freq_response_at_norm_point);
        LOG_BUTTER("Calculated gain k: " + std::to_string(gain));


        // --- 6. Calculate Coefficients ---
        double_vector b = roots_to_poly(digital_zeros); // Numerator B(z)
        double_vector a = roots_to_poly(digital_poles); // Denominator A(z)

        // Apply gain to numerator coefficients
        std::transform(b.begin(), b.end(), b.begin(), [gain](double val){ return val * gain; });

        // Normalize denominator so a[0] = 1 (should already be the case if roots_to_poly works correctly)
        if (!a.empty() && std::abs(a[0] - 1.0) > 1e-9) {
             LOG_BUTTER("Warning: Normalizing denominator coefficients.");
             double a0 = a[0];
             std::transform(a.begin(), a.end(), a.begin(), [a0](double val){ return val / a0; });
             // Also need to scale b by a0 to keep transfer function equivalent
             std::transform(b.begin(), b.end(), b.begin(), [a0](double val){ return val / a0; });
        }

        LOG_BUTTER("Calculated coefficients: b.size=" + std::to_string(b.size()) + ", a.size=" + std::to_string(a.size()));
        return {b, a};
    }

     // Overload for band filters (placeholder)
     std::pair<double_vector, double_vector> calculateButterworthCoeffs(
        int order,
        const std::pair<double, double>& cutoff_freqs,
        const std::string& btype,
        double fs)
    {
        LOG_BUTTER("Warning: Band filter Butterworth implementation not yet complete.");
        // TODO: Implement bandpass/bandstop transformations
        throw std::runtime_error("Bandpass/Bandstop Butterworth not implemented yet.");
        // Dummy return to satisfy compiler if exception is removed
        // return {{1.0}, {1.0}};
    }

// --- Butterworth Class Implementation ---

// Constructor
Butterworth::Butterworth(double Wn_low, double Wn_high) {
    // Calculate coefficients for the bandpass filter defined by Wn_low and Wn_high
    // Note: calculateButterworthCoeffs expects cutoff frequencies relative to Fs,
    // while Wn is relative to Nyquist frequency (Fs/2). We need to adjust.
    // Assuming fs = 1.0 for normalized frequencies Wn.
    double fs_normalized = 2.0; // Nyquist corresponds to 1.0, so Fs is 2.0
    double cutoff_low = Wn_low * (fs_normalized / 2.0);
    double cutoff_high = Wn_high * (fs_normalized / 2.0);

    // We need a bandpass filter. The current calculateButterworthCoeffs only handles low/high.
    // For now, let's approximate with two separate low-pass filters as was done implicitly before.
    // This matches the state variables used in GaussianPyramid (lowpass1_, lowpass2_)
    // We'll calculate coefficients for a low-pass filter at Wn_high.
    // The GaussianPyramid::processFrame seems to handle the bandpass logic using these states.

    LOG_BUTTER("Initializing Butterworth class for low-pass at Wn=" + std::to_string(Wn_high));

    // Use the existing function for low-pass
    // Need to decide on the order. Previous code used order 1 implicitly.
    order_ = 1;
    try {
        // Calculate coeffs for the higher cutoff frequency (acting as the main low-pass stage)
        auto coeffs_high = calculateButterworthCoeffs(order_, cutoff_high, "low", fs_normalized);
        b_coeffs_ = coeffs_high.first;
        a_coeffs_ = coeffs_high.second;

        // Basic validation of coefficient sizes for order 1
        if (b_coeffs_.size() != order_ + 1 || a_coeffs_.size() != order_ + 1) {
             throw std::runtime_error("Butterworth coefficient calculation returned unexpected size for order " + std::to_string(order_));
        }
         if (std::abs(a_coeffs_[0] - 1.0) > 1e-9) {
             throw std::runtime_error("Denominator coefficient a[0] must be 1.0 after normalization.");
         }

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error initializing Butterworth filter coefficients: ") + e.what());
    }
     LOG_BUTTER("Butterworth class initialized with b=" + std::to_string(b_coeffs_[0]) + "," + std::to_string(b_coeffs_[1]) +
                " a=" + std::to_string(a_coeffs_[0]) + "," + std::to_string(a_coeffs_[1]));
}

// Filter method
cv::Mat Butterworth::filter(const cv::Mat& input, cv::Mat& prev_input_state, cv::Mat& prev_output_state) {
    if (b_coeffs_.size() != order_ + 1 || a_coeffs_.size() != order_ + 1) {
        throw std::runtime_error("Butterworth filter coefficients are not correctly initialized.");
    }
    if (input.empty()) {
        throw std::invalid_argument("Input matrix to Butterworth::filter is empty.");
    }
     if (input.size() != prev_input_state.size() || input.type() != prev_input_state.type() ||
         input.size() != prev_output_state.size() || input.type() != prev_output_state.type()) {
          throw std::invalid_argument("Input and state matrices must have the same size and type in Butterworth::filter.");
     }

    // Apply the 1st order IIR filter equation:
    // output = b[0]*input + b[1]*prev_input - a[1]*prev_output
    // Note: a[0] is assumed to be 1
    cv::Mat output = b_coeffs_[0] * input + b_coeffs_[1] * prev_input_state - a_coeffs_[1] * prev_output_state;

    // Update states for the next iteration
    prev_input_state = input.clone(); // Update previous input state
    prev_output_state = output.clone(); // Update previous output state

    return output;
}


} // namespace evmcpp