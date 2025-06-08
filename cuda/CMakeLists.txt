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
)

# Set CUDA properties
set_target_properties(cuda_evm PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "50;60;70;75"
)

# Link libraries
target_link_libraries(cuda_evm ${OpenCV_LIBS})

# Add test executable
add_executable(test_cuda_butterworth_simple
    tests/test_cuda_butterworth_simple.cu
)

# Link test with library
target_link_libraries(test_cuda_butterworth_simple cuda_evm ${OpenCV_LIBS})

# Set CUDA properties for test
set_target_properties(test_cuda_butterworth_simple PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "50;60;70;75"
)

# Add pipeline comparison test
add_executable(test_butterworth_pipeline_comparison
    tests/test_butterworth_pipeline_comparison.cu
    ../cpp/src/laplacian_pyramid.cpp
    ../cpp/src/processing.cpp
    ../cpp/src/color_conversion.cpp
    ../cpp/src/butterworth.cpp
    ../cpp/src/pyramid.cpp
    ../cpp/src/gaussian_pyramid.cpp
    ../cpp/src/temporal_filter.cpp
)

# Link pipeline test with libraries
target_link_libraries(test_butterworth_pipeline_comparison cuda_evm ${OpenCV_LIBS})

# Include CPU headers for the pipeline test
target_include_directories(test_butterworth_pipeline_comparison PRIVATE ../cpp/include)

# Set CUDA properties for pipeline test
set_target_properties(test_butterworth_pipeline_comparison PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "50;60;70;75"
)

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