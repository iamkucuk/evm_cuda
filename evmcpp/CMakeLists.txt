# CMakeLists.txt (located in evmcpp/)
cmake_minimum_required(VERSION 3.10) # Specify minimum CMake version

project(EVMCPP VERSION 0.1.0 LANGUAGES CXX CUDA) # Add CUDA language

# Set C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard if needed (optional, often defaults work)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Set CUDA architecture flags (adjust based on target GPU)
# Example: set(CMAKE_CUDA_ARCHITECTURES 75) # Turing
# Or let CMake detect, or specify multiple: 61;70;75;80;86 etc.
# For broader compatibility, let's start without setting it explicitly,
# allowing CMake/nvcc to use defaults or detect.
# If compilation issues arise, uncomment and set appropriately.
set(CMAKE_CUDA_ARCHITECTURES 75) # Set target GPU architecture (e.g., 75 for Turing)

# --- Find Required Packages ---
# Find OpenCV (Core, Image Processing, Video I/O needed)
# Users might need to set OpenCV_DIR environment variable if not found automatically
# Find CUDA Toolkit (Modern CMake approach)
# This also finds related libraries like cufft
find_package(CUDAToolkit REQUIRED)
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found. Please ensure it's installed and findable.")
else()
     message(STATUS "Found CUDA Toolkit ${CUDAToolkit_VERSION}")
     # Check if cufft was found (usually included with Toolkit)
     if(TARGET CUDA::cufft)
         message(STATUS "Found cuFFT")
     else()
         message(WARNING "cuFFT target not found by find_package(CUDAToolkit). Linking might fail.")
     endif()
endif()

# Find OpenCV (Core, Image Processing, Video I/O, and CUDA modules needed)
# Add opencv_cudafilters, opencv_cudaarithm, opencv_cudaimgproc
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio) # Removed CUDA modules
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please ensure it's installed and findable (e.g., set OpenCV_DIR).")
else()
    message(STATUS "Found OpenCV ${OpenCV_VERSION}")
    # Add OpenCV include directories globally for the project
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# --- Project Structure ---
# Add our own include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- Build Targets ---
# --- CPU Implementation Library ---
add_library(evmcpu STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/evmcpu/color_conversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/evmcpu/pyramid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/evmcpu/temporal_filter.cpp
)
target_link_libraries(evmcpu PRIVATE ${OpenCV_LIBS})
target_include_directories(evmcpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Option 1: Build a static library for the core EVM logic (Recommended)
add_library(evm_core STATIC
    ../src/evmcpp/laplacian_pyramid.cpp # Corrected relative path
    ../src/evmcpp/processing.cpp     # Corrected relative path
    ../src/evmcpp/butterworth.cpp    # Corrected relative path
    ../src/evmcpp/gaussian_pyramid.cpp # Added Gaussian pyramid source
)
# Link the library against OpenCV
# Link the library against OpenCV (including CUDA modules)
target_link_libraries(evm_core PRIVATE ${OpenCV_LIBS})
# Link against CUDA runtime library if needed (often handled by CMake CUDA integration)
# target_link_libraries(evm_core PRIVATE CUDA::cudart)
# Ensure users of the library can find its headers
target_include_directories(evm_core PUBLIC ../include) # CPU headers

# --- CUDA Library Target ---
# ENABLE_LANGUAGE(CUDA) call removed
# Note: Files added here will be compiled with the CUDA compiler (nvcc)
add_library(evm_cuda STATIC
    ../src/evmcuda/color_conversion.cu
    # ../src/evmcuda/pyramid.cu # Removed - Abandoned custom kernel
    ../src/evmcuda/temporal_filter.cu
    ../src/evmcuda/reconstruction.cu # Added reconstruction source
    # Add other .cu files here later
)
# Explicitly enable CUDA language for this target if not inherited
# target_compile_features(evm_cuda PUBLIC cxx_std_17 cuda_std_11) # Example if needed
# Link CUDA library against OpenCV CUDA modules and CUDA runtime
# Link CUDA library against CUDA runtime and potentially core OpenCV if GpuMat is used in headers
target_link_libraries(evm_cuda PRIVATE CUDA::cudart CUDA::cufft ${OpenCV_LIBS}) # Link against cufft and core OpenCV
# Ensure users of this library can find CUDA headers
target_include_directories(evm_cuda PUBLIC ${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS}) # Add OpenCV includes explicitly

# Link evm_core against evm_cuda and evmcpu
target_link_libraries(evm_core PRIVATE evm_cuda evmcpu)

# Option 2: Directly build an executable (Simpler for starting)
# If using Option 1, link the executable against the library
add_executable(evm_app ../src/main.cpp) # Corrected relative path
target_link_libraries(evm_app PRIVATE evm_core) # Link against our library
# If NOT using Option 1 (library), link executable directly to OpenCV and list all sources:
# add_executable(evm_app
#     src/main.cpp
#     src/laplacian_pyramid.cpp
#     src/processing.cpp
#     src/butterworth.cpp
# )
# target_link_libraries(evm_app PRIVATE ${OpenCV_LIBS})


# --- Testing ---
# Enable testing with CTest
enable_testing()
# Add the tests subdirectory (will contain its own CMakeLists.txt)
add_subdirectory(tests) # Uncommented to include tests

message(STATUS "EVMCPP project configured. Build inside the 'build' directory.")