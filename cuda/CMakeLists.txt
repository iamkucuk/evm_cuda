cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(evm_cuda LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75 86)  # Adjust based on your GPU architecture
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# OpenCV is needed for some components
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create CUDA libraries
# Color conversion library
add_library(cuda_color_conversion STATIC
    src/cuda_color_conversion.cu
    include/cuda_color_conversion.cuh
)
set_target_properties(cuda_color_conversion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(cuda_color_conversion ${CUDA_LIBRARIES} ${OpenCV_LIBS})

# Pyramid operations library
add_library(cuda_pyramid STATIC
    src/cuda_pyramid.cu
    include/cuda_pyramid.cuh
)
set_target_properties(cuda_pyramid PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(cuda_pyramid ${CUDA_LIBRARIES} ${OpenCV_LIBS})

# Butterworth filter library
add_library(cuda_butterworth STATIC
    src/cuda_butterworth.cu
    include/cuda_butterworth.cuh
)
set_target_properties(cuda_butterworth PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(cuda_butterworth ${CUDA_LIBRARIES} ${OpenCV_LIBS})

# Laplacian pyramid library
add_library(cuda_laplacian_pyramid STATIC
    src/cuda_laplacian_pyramid.cu
    include/cuda_laplacian_pyramid.cuh
)
set_target_properties(cuda_laplacian_pyramid PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(cuda_laplacian_pyramid 
    cuda_pyramid 
    cuda_color_conversion
    cuda_butterworth
    ${CUDA_LIBRARIES} 
    ${OpenCV_LIBS}
)

# EVM core library
add_library(cuda_evm STATIC
    src/cuda_evm.cu
    include/cuda_evm.cuh
)
set_target_properties(cuda_evm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(cuda_evm 
    cuda_laplacian_pyramid
    cuda_pyramid 
    cuda_color_conversion
    cuda_butterworth
    ${CUDA_LIBRARIES} 
    ${OpenCV_LIBS}
)

# Add the main executable
add_executable(evm_cuda src/main.cpp)
target_link_libraries(evm_cuda 
    cuda_evm
    cuda_laplacian_pyramid
    cuda_pyramid 
    cuda_color_conversion
    cuda_butterworth
    ${CUDA_LIBRARIES} 
    ${OpenCV_LIBS}
)

# Tests directory
add_subdirectory(tests)

# Install
install(TARGETS evm_cuda DESTINATION bin)