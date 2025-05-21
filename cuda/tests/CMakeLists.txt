cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Find test framework (Google Test, Catch2, or custom)
# find_package(GTest REQUIRED)  # Uncomment if using Google Test

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Add test executables
# Test color conversion
add_executable(test_cuda_color_conversion test_cuda_color_conversion.cu)
target_link_libraries(test_cuda_color_conversion cuda_color_conversion)
set_target_properties(test_cuda_color_conversion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Test pyramid operations
add_executable(test_cuda_pyramid test_cuda_pyramid.cu)
target_link_libraries(test_cuda_pyramid cuda_pyramid cuda_color_conversion)
set_target_properties(test_cuda_pyramid PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Test Butterworth filter
add_executable(test_cuda_butterworth test_cuda_butterworth.cu)
target_link_libraries(test_cuda_butterworth cuda_butterworth)
set_target_properties(test_cuda_butterworth PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Test Laplacian pyramid operations
add_executable(test_cuda_laplacian_pyramid test_cuda_laplacian_pyramid.cu)
target_link_libraries(test_cuda_laplacian_pyramid cuda_laplacian_pyramid cuda_pyramid cuda_color_conversion)
set_target_properties(test_cuda_laplacian_pyramid PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Test temporal filtering operations
add_executable(test_cuda_temporal_filter test_cuda_temporal_filter.cu)
target_link_libraries(test_cuda_temporal_filter cuda_laplacian_pyramid cuda_butterworth)
set_target_properties(test_cuda_temporal_filter PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Test end-to-end EVM pipeline
add_executable(test_cuda_evm_pipeline test_cuda_evm_pipeline.cu)
target_link_libraries(test_cuda_evm_pipeline cuda_evm cuda_laplacian_pyramid cuda_butterworth cuda_pyramid cuda_color_conversion)
set_target_properties(test_cuda_evm_pipeline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Copy test data from CPU implementation
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/tests/data/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)