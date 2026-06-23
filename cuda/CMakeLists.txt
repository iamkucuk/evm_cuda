# CMakeLists.txt — builds _evm_cuda.so, the pybind11 extension wrapping the
# EVM CUDA kernels. Designed to be invoked from setup.py (`pip install -e .`)
# OR standalone from a build directory on TRUBA.
#
# The build is portable across TRUBA's GPU fleet: we compile for several
# arches simultaneously so the resulting .so runs on P100/V100/A100/H100/H200
# without per-node rebuilds. `-use_fast_math=false` keeps bit-accuracy.

cmake_minimum_required(VERSION 3.18)
project(evm_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- CUDA setup ------------------------------------------------------------

# Portable arch coverage for TRUBA ARF-ACC: P100 (sm_60) -> V100 (sm_70) ->
# A100 (sm_80) -> Ada/H100-class (sm_89/sm_90). Drop sm_60 if your nvcc
# doesn't support it; the others will still cover all current cards.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 80 89 90)
endif()

# Keep precision: NO fast math. nvcc's -use_fast_math is OFF by default, so we
# deliberately do NOT add it — the tests demand IEEE-accurate arithmetic.
# Add -Xptxas -v so every kernel build prints register/shared-mem usage.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

find_package(CUDAToolkit REQUIRED)

# --- Python + pybind11 + numpy --------------------------------------------

# When invoked via setup.py (scikit-build), Python/NumPy are already found
# by the parent. When invoked standalone (e.g. on TRUBA in a manual build),
# fall back to find_package.
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# pybind11: prefer system install, fall back to FetchContent.
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via find_package; fetching from GitHub.")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# --- the extension ---------------------------------------------------------

pybind11_add_module(_evm_cuda
    bindings.cpp
    kernels/color_cvt.cu
    kernels/spatial.cu
    kernels/transpose.cu
    kernels/iir_bandpass.cu
    kernels/butter_bandpass.cu
    kernels/ideal_bandpass.cu
    kernels/lpyr.cu
    kernels/blur_dn.cu
    kernels/amplify_render.cu
    kernels/fp16_cvt.cu
)

target_include_directories(_evm_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(_evm_cuda PRIVATE
    CUDA::cudart
    CUDA::cufft
)

# Place the built .so next to the cuda/evm_cuda/ Python package so it's
# importable as `from evm_cuda import _evm_cuda` after `pip install -e .`.
set_target_properties(_evm_cuda PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/evm_cuda"
)
