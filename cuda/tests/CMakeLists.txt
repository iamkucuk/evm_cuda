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

# Test temporal filtering comparison with CPU approach
add_executable(test_cpu_temporal_filter_comparison test_cpu_temporal_filter_comparison.cu)
target_link_libraries(test_cpu_temporal_filter_comparison cuda_butterworth)
set_target_properties(test_cpu_temporal_filter_comparison PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Test end-to-end EVM pipeline
add_executable(test_cuda_evm_pipeline test_cuda_evm_pipeline.cu)
target_link_libraries(test_cuda_evm_pipeline cuda_evm cuda_laplacian_pyramid cuda_butterworth cuda_pyramid cuda_color_conversion ${OpenCV_LIBS})
set_target_properties(test_cuda_evm_pipeline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Test CUDA vs CPU color conversion
add_executable(test_cuda_vs_cpu_color test_cuda_vs_cpu_color.cu)
target_link_libraries(test_cuda_vs_cpu_color cuda_color_conversion ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/build/libevm_core.a ${OpenCV_LIBS})
set_target_properties(test_cuda_vs_cpu_color PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(test_cuda_vs_cpu_color PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/include)

# Test CUDA vs CPU pyramid operations
add_executable(test_cuda_vs_cpu_pyramid test_cuda_vs_cpu_pyramid.cu)
target_link_libraries(test_cuda_vs_cpu_pyramid cuda_pyramid cuda_color_conversion ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/build/libevm_core.a ${OpenCV_LIBS})
set_target_properties(test_cuda_vs_cpu_pyramid PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(test_cuda_vs_cpu_pyramid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/include)

# Test CUDA vs CPU temporal filtering
add_executable(test_cuda_vs_cpu_temporal test_cuda_vs_cpu_temporal.cu)
target_link_libraries(test_cuda_vs_cpu_temporal cuda_butterworth ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/build/libevm_core.a ${OpenCV_LIBS})
set_target_properties(test_cuda_vs_cpu_temporal PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(test_cuda_vs_cpu_temporal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/include)
target_include_directories(test_cuda_vs_cpu_temporal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/tests)

# Comprehensive validation test (runs all component tests)
add_executable(test_comprehensive_validation test_comprehensive_validation.cu)
target_link_libraries(test_comprehensive_validation ${OpenCV_LIBS})
set_target_properties(test_comprehensive_validation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Frame-by-frame video comparison tool
add_executable(compare_videos_frame_by_frame compare_videos_frame_by_frame.cu)
target_link_libraries(compare_videos_frame_by_frame ${OpenCV_LIBS})
set_target_properties(compare_videos_frame_by_frame PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# cuFFT step-by-step validation test
add_executable(test_cufft_step_by_step_validation test_cufft_step_by_step_validation.cu)
target_link_libraries(test_cufft_step_by_step_validation cuda_color_conversion cuda_pyramid ${OpenCV_LIBS} cufft)
set_target_properties(test_cufft_step_by_step_validation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Copy test data from CPU implementation
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/tests/data/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)