# CMakeLists.txt — builds _evm_cuda, the pybind11 extension wrapping the
# EVM CUDA kernels. Driven by scikit-build-core during `pip install .`, or
# directly with `cmake -S cuda -B cuda/build`.
#
# CUDA is OPTIONAL. On a machine with no CUDA compiler this file configures
# cleanly and defines no target, so the install still produces a working
# CPU-only package. Set EVM_CUDA_REQUIRE=1 to turn a missing nvcc into a hard
# error instead.
#
# `-use_fast_math` is never added: nvcc has it off by default and the tests
# demand IEEE-accurate arithmetic. Keep it that way.

# 3.24 is the first CMake release that accepts CMAKE_CUDA_ARCHITECTURES=native,
# which is the default architecture selection below.
cmake_minimum_required(VERSION 3.24)

# CXX only. The CUDA language is enabled further down, and only if a CUDA
# compiler actually exists — declaring it here would hard-fail on any machine
# without nvcc, which is exactly what step 1.6 removes.
project(evm_cuda LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- is there a CUDA compiler? ---------------------------------------------

# Fail-loud escape hatch: EVM_CUDA_REQUIRE=1, as a cache option or an
# environment variable, makes a missing CUDA compiler an error rather than a
# silently CPU-only build. An explicit -DEVM_CUDA_REQUIRE=... wins over the
# environment.
set(_evm_cuda_require_default OFF)
if(DEFINED ENV{EVM_CUDA_REQUIRE})
    set(_evm_cuda_require_default "$ENV{EVM_CUDA_REQUIRE}")
endif()
option(EVM_CUDA_REQUIRE
       "Fail configuration when no CUDA compiler is found, instead of skipping the GPU extension"
       ${_evm_cuda_require_default})

include(CheckLanguage)
check_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)
    if(EVM_CUDA_REQUIRE)
        message(FATAL_ERROR
            "EVM_CUDA_REQUIRE is set, but no CUDA compiler was found.\n"
            "Install the CUDA Toolkit and put nvcc on PATH (or point CUDACXX at "
            "it), or unset EVM_CUDA_REQUIRE to build the CPU-only package.")
    endif()
    message(STATUS "")
    message(STATUS "============================================================")
    message(STATUS "  NO CUDA COMPILER FOUND")
    message(STATUS "  The _evm_cuda GPU extension will NOT be built.")
    message(STATUS "  The package still installs and the CPU pipelines still")
    message(STATUS "  work; every GPU entry point raises a named error.")
    message(STATUS "  To build it: install the CUDA Toolkit, put nvcc on PATH")
    message(STATUS "  (or set CUDACXX), and reinstall.")
    message(STATUS "  To make this situation an error: EVM_CUDA_REQUIRE=1")
    message(STATUS "============================================================")
    message(STATUS "")
    return()
endif()

# --- GPU architecture selection --------------------------------------------

# This block MUST run before enable_language(CUDA). Enabling the CUDA language
# sets CMAKE_CUDA_ARCHITECTURES to the compiler's own default (5.2 for CUDA 12)
# whenever the variable is still empty, so any value set afterwards is ignored.
# That was the bug: the list below used to sit after project(... LANGUAGES CUDA)
# behind an `if(NOT DEFINED ...)` guard that could never fire, so the build
# silently targeted compute capability 5.2 — where __hfma2, used at
# kernels/spatial.cu:918, does not exist. See docs/dev/packaging-notes.md.
#
#   EVM_CUDA_ARCHS=native   (default) build only for the GPUs in this machine;
#                           the right choice for a source install, ~7 s on a 3090
#   EVM_CUDA_ARCHS=all      60;70;80;89;90 — P100 / V100 / A100 / Ada / H100,
#                           the portable set for release wheels and benchmarks
#
# An explicit -DCMAKE_CUDA_ARCHITECTURES=... always wins over both.
set(_evm_cuda_archs_default "native")
if(DEFINED ENV{EVM_CUDA_ARCHS})
    set(_evm_cuda_archs_default "$ENV{EVM_CUDA_ARCHS}")
endif()
set(EVM_CUDA_ARCHS "${_evm_cuda_archs_default}"
    CACHE STRING "GPU architecture set to build for: native or all")
set_property(CACHE EVM_CUDA_ARCHS PROPERTY STRINGS native all)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(EVM_CUDA_ARCHS STREQUAL "native")
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
    elseif(EVM_CUDA_ARCHS STREQUAL "all")
        set(CMAKE_CUDA_ARCHITECTURES 60 70 80 89 90 CACHE STRING "CUDA architectures")
    else()
        message(FATAL_ERROR
            "EVM_CUDA_ARCHS must be 'native' or 'all', got '${EVM_CUDA_ARCHS}'. "
            "For an arbitrary set pass -DCMAKE_CUDA_ARCHITECTURES=... instead.")
    endif()
endif()

enable_language(CUDA)
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Add -Xptxas -v so every kernel build prints register/shared-mem usage.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

find_package(CUDAToolkit REQUIRED)

# --- Python + pybind11 -----------------------------------------------------

# Development.Module, not the full Development component: it provides the
# headers and the module link rules without demanding libpython, which is what
# scikit-build-core and every modern wheel build need (manylinux images ship no
# libpython to link against).
#
# NumPy is deliberately NOT requested. bindings.cpp includes only
# <pybind11/numpy.h>, which reaches NumPy through the ordinary Python C API at
# runtime; nothing in cuda/ uses the NumPy C API, so neither its headers nor an
# install-time NumPy are needed.
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

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
)

target_link_libraries(_evm_cuda PRIVATE
    CUDA::cudart
    CUDA::cufft
)

# The Python wrapper package lives at src/evm/cuda/ and does `from . import
# _evm_cuda`, so the module has to land beside it. Two paths need that:
# a plain `cmake --build` (drops it straight into the source tree) and a wheel
# build (installs it at the same relative location inside the wheel).
set_target_properties(_evm_cuda PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../src/evm/cuda"
)
install(TARGETS _evm_cuda LIBRARY DESTINATION evm/cuda)
