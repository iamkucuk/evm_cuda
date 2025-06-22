#ifndef CUDA_BUTTERWORTH_CUH
#define CUDA_BUTTERWORTH_CUH

#include <vector>
#include <string>
#include <utility>
#include <cuda_runtime.h>

namespace cuda_evm {

// Host-side functions for coefficient calculation (matches CPU interface)
std::pair<std::vector<double>, std::vector<double>> calculateButterworthCoeffs(
    int order,
    double cutoff_freq,
    const std::string& btype,
    double fs
);

std::pair<std::vector<double>, std::vector<double>> calculateButterworthCoeffs(
    int order,
    const std::pair<double, double>& cutoff_freqs,
    const std::string& btype,
    double fs
);

// CUDA Butterworth filter class
class CudaButterworth {
public:
    // Constructor: Takes normalized cutoff frequencies
    CudaButterworth(double Wn_low, double Wn_high);
    
    // Destructor
    ~CudaButterworth();
    
    // Apply filter to device memory
    // input, prev_input_state, prev_output_state, output are device pointers
    void filter(const float* d_input, float* d_prev_input_state, float* d_prev_output_state, 
                float* d_output, int width, int height, int channels, cudaStream_t stream = 0);
    
    // Get filter coefficients for validation
    const std::vector<double>& getBCoeffs() const { return b_coeffs_; }
    const std::vector<double>& getACoeffs() const { return a_coeffs_; }

private:
    int order_;
    std::vector<double> b_coeffs_;
    std::vector<double> a_coeffs_;
    
    // Device memory for coefficients
    float* d_b_coeffs_;
    float* d_a_coeffs_;
    
    void allocateDeviceMemory();
    void copyCoeffsToDevice();
    void freeDeviceMemory();
};

// CUDA kernel declarations
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
);

// Utility functions
cudaError_t checkCudaError(cudaError_t error, const char* file, int line);
#define CUDA_CHECK(error) checkCudaError(error, __FILE__, __LINE__)

} // namespace cuda_evm

#endif // CUDA_BUTTERWORTH_CUH