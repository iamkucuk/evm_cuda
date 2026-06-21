// evm_check.cuh — CUDA error-checking macros for the EVM port.
//
// Every CUDA runtime call in this project is wrapped in CUDA_CHECK so a
// silent error can never propagate into a tolerance failure. The macros are
// hard-fail (abort) because in a correctness-oracle workflow there is no
// useful "partial result" to return.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr,                                               \
                         "CUDA error %s at %s:%d: %s\n",                       \
                         cudaGetErrorName(_err), __FILE__, __LINE__,           \
                         cudaGetErrorString(_err));                            \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define CUFFT_CHECK(call)                                                      \
    do {                                                                       \
        cufftResult _err = (call);                                             \
        if (_err != CUFFT_SUCCESS) {                                           \
            std::fprintf(stderr,                                               \
                         "cuFFT error %d at %s:%d\n",                          \
                         static_cast<int>(_err), __FILE__, __LINE__);          \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
