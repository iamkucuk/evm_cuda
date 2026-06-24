// evm_check.cuh — CUDA error-checking macros for the EVM port.
//
// Every CUDA runtime call in this project is wrapped in CUDA_CHECK so a
// silent error can never propagate into a tolerance failure. The macros
// throw std::runtime_error on failure. When called from the pybind11
// bindings (the normal path), pybind11 converts the C++ exception into a
// Python exception that callers can catch in try/except. This is strictly
// better than std::abort() — the host process survives, pytest reports the
// failure as a normal test error instead of a process crash, and notebook
// cells can print the error and continue.

#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            throw std::runtime_error(                                          \
                std::string("CUDA error ") +                                   \
                cudaGetErrorName(_err) + " at " + __FILE__ + ":" +             \
                std::to_string(__LINE__) + ": " +                              \
                cudaGetErrorString(_err));                                      \
        }                                                                      \
    } while (0)

#define CUFFT_CHECK(call)                                                      \
    do {                                                                       \
        cufftResult _err = (call);                                             \
        if (_err != CUFFT_SUCCESS) {                                           \
            throw std::runtime_error(                                          \
                std::string("cuFFT error ") +                                  \
                std::to_string(static_cast<int>(_err)) + " at " +              \
                __FILE__ + ":" + std::to_string(__LINE__));                    \
        }                                                                      \
    } while (0)
