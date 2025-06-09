cmake_minimum_required(VERSION 3.18)
project(cuda_evm LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50")

# Add library for CUDA implementations
add_library(cuda_evm
    src/cuda_butterworth.cu
    src/cuda_color_conversion.cu
    src/cuda_format_conversion.cu
)

# Set CUDA properties
set_target_properties(cuda_evm PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "50;60;70;75"
)

# Link libraries
target_link_libraries(cuda_evm ${OpenCV_LIBS})

# Add video comparison tool
add_executable(compare_videos_frame_by_frame
    tests/compare_videos_frame_by_frame.cu
)

# Link comparison tool with OpenCV
target_link_libraries(compare_videos_frame_by_frame ${OpenCV_LIBS})

# Set CUDA properties for comparison tool
set_target_properties(compare_videos_frame_by_frame PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "50;60;70;75"
)

# Add CUDA Butterworth + CPU rest test (FIXED approach)
add_executable(test_cuda_butter_cpu_rest_fixed
    test_cuda_butter_cpu_rest_fixed.cu
    ../cpp/src/laplacian_pyramid.cpp
    ../cpp/src/processing.cpp
    ../cpp/src/color_conversion.cpp
    ../cpp/src/pyramid.cpp
    ../cpp/src/gaussian_pyramid.cpp
    ../cpp/src/butterworth.cpp
    ../cpp/src/temporal_filter.cpp
)

# Link CUDA Butterworth + CPU rest test (FIXED) with libraries
target_link_libraries(test_cuda_butter_cpu_rest_fixed cuda_evm ${OpenCV_LIBS})

# Include CPU headers for the CUDA Butterworth + CPU rest test (FIXED)
target_include_directories(test_cuda_butter_cpu_rest_fixed PRIVATE ../cpp/include)

# Set CUDA properties for CUDA Butterworth + CPU rest test (FIXED)
set_target_properties(test_cuda_butter_cpu_rest_fixed PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "50;60;70;75"
)

# Add CUDA color + WORKING CUDA Butterworth test
add_executable(test_cuda_color_cuda_butter_working_impl
    test_cuda_color_cuda_butter_working_impl.cu
    ../cpp/src/laplacian_pyramid.cpp
    ../cpp/src/processing.cpp
    ../cpp/src/color_conversion.cpp
    ../cpp/src/pyramid.cpp
    ../cpp/src/gaussian_pyramid.cpp
)

# Link CUDA color + WORKING CUDA Butterworth test with libraries
target_link_libraries(test_cuda_color_cuda_butter_working_impl cuda_evm ${OpenCV_LIBS})

# Include CPU headers for the CUDA color + WORKING CUDA Butterworth test
target_include_directories(test_cuda_color_cuda_butter_working_impl PRIVATE ../cpp/include)

# Set CUDA properties for CUDA color + WORKING CUDA Butterworth test
set_target_properties(test_cuda_color_cuda_butter_working_impl PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "50;60;70;75"
)