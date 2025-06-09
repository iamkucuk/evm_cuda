#ifndef CUDA_FORMAT_CONVERSION_CUH
#define CUDA_FORMAT_CONVERSION_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace evmcuda {

// Direct CUDA conversion: float3 array to flat float array
cudaError_t convert_float3_to_flat(const float3* input, float* output, int width, int height);

// Direct CUDA conversion: flat float array to float3 array  
cudaError_t convert_flat_to_float3(const float* input, float3* output, int width, int height);

} // namespace evmcuda

#endif // CUDA_FORMAT_CONVERSION_CUH