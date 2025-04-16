#ifndef EVMCPP_BUTTERWORTH_HPP
#define EVMCPP_BUTTERWORTH_HPP

#include <vector>
#include <string>
#include <utility> // For std::pair
#include <vector>  // For std::vector
#include <opencv2/core/mat.hpp> // For cv::Mat needed by the class

namespace evmcpp {

    // Type alias for vector of doubles
    using double_vector = std::vector<double>;

    /**
     * @brief Calculates the coefficients for a digital Butterworth filter.
     * @param order The order of the filter (e.g., 1 based on Python code).
     * @param cutoff_freq The cutoff frequency (or frequencies for bandpass/bandstop).
     *                    For low/high pass, this is a single value.
     * @param btype The type of filter: "low", "high", "bandpass", "bandstop".
     * @param fs The sampling frequency (e.g., video FPS).
     * @return A pair of vectors: {numerator coefficients (b), denominator coefficients (a)}.
     * @note This needs a robust implementation matching scipy.signal.butter behavior.
     *       The implementation will involve analog prototype design, bilinear transform, etc.
     */
    std::pair<std::vector<double>, std::vector<double>> calculateButterworthCoeffs(
        int order,
        double cutoff_freq, // Or maybe std::vector<double> for band filters
        const std::string& btype,
        double fs
    );

    // Overload for bandpass/bandstop filters which require two frequencies
     std::pair<std::vector<double>, std::vector<double>> calculateButterworthCoeffs(
        int order,
        const std::pair<double, double>& cutoff_freqs,
        const std::string& btype, // Should be "bandpass" or "bandstop"
        double fs
    );

// --- Butterworth Filter Class ---
// Encapsulates filter coefficients and state for applying the filter
class Butterworth {
public:
    // Constructor: Takes normalized cutoff frequencies (Wn = f_cutoff / (fs/2))
    // For bandpass, Wn would be a pair {low_cutoff / (fs/2), high_cutoff / (fs/2)}
    // This constructor handles low-pass based on GaussianPyramid usage.
    Butterworth(double Wn_low, double Wn_high); // Wn = cutoff / (fs/2)

    // Applies the filter to an input signal (single frame channel)
    // Updates internal state (prev_input, prev_output) implicitly if needed,
    // or takes state explicitly as done in GaussianPyramid.
    // Let's match the GaussianPyramid usage which passes state explicitly.
    cv::Mat filter(const cv::Mat& input, cv::Mat& prev_input_state, cv::Mat& prev_output_state);

private:
    int order_ = 1; // Based on previous implementation attempts
    double_vector b_coeffs_; // Numerator coefficients
    double_vector a_coeffs_; // Denominator coefficients (a[0] assumed 1)

    // Internal state (if not passed explicitly) - currently unused based on filter signature
    // cv::Mat prev_input_;
    // cv::Mat prev_output_;
};

} // namespace evmcpp

#endif // EVMCPP_BUTTERWORTH_HPP