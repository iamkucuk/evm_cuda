cmake_minimum_required(VERSION 3.10)

project(evm_cuda_pipeline VERSION 1.0 LANGUAGES CXX CUDA)

# Set C++ standard (same as reference)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV (required for all modes)
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86")

# Gather all source and header files (same pattern as reference)
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/include/*.cuh")

# Remove main.cu from library sources (same pattern as reference)
list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/main.cu")

# Add transpose_only.cu to sources
list(APPEND SOURCES "${CMAKE_SOURCE_DIR}/transpose_only.cu")

# CPU Reference Implementation for validation (not part of main library)
add_executable(cpu_reference_laplacian ${CMAKE_SOURCE_DIR}/cpu_reference_laplacian.cu)
set_target_properties(cpu_reference_laplacian PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "86"
)
target_link_libraries(cpu_reference_laplacian PRIVATE ${OpenCV_LIBS})

# Component Isolation Test for systematic debugging (not part of main library)
add_executable(component_isolation_test ${CMAKE_SOURCE_DIR}/component_isolation_test.cu)
set_target_properties(component_isolation_test PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "86"
)
target_link_libraries(component_isolation_test PRIVATE ${OpenCV_LIBS})

# Hybrid Component Test for isolating specific CUDA components
add_executable(hybrid_component_test 
    ${CMAKE_SOURCE_DIR}/hybrid_component_test.cu
    ${CMAKE_SOURCE_DIR}/cuda_cpu_interface.cu)
set_target_properties(hybrid_component_test PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "86"
)
target_link_libraries(hybrid_component_test PRIVATE evm_core ${OpenCV_LIBS})

# Main library with all core sources (same name pattern as reference: evm_core)
add_library(evm_core STATIC ${SOURCES} ${HEADERS})

# Set CUDA properties
set_target_properties(evm_core PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "86"
)

# Link libraries (same pattern as reference)
target_link_libraries(evm_core PUBLIC ${OpenCV_LIBS} cufft)

# Main unified pipeline executable (supports both Gaussian and Laplacian modes)
add_executable(evmpipeline ${CMAKE_SOURCE_DIR}/src/main.cu)

# Set CUDA properties for main executable
set_target_properties(evmpipeline PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "86"
)

# Link main executable (same pattern as reference)
target_link_libraries(evmpipeline PRIVATE evm_core ${OpenCV_LIBS})

# Install rules (same as reference)
install(TARGETS evmpipeline RUNTIME DESTINATION bin)

# Custom target for easy execution with environment variables
# Usage: INPUT=input_video.mp4 OUTPUT=output_video.avi make run
# Example: INPUT=../../data/face.mp4 OUTPUT=magnified_face.avi make run
# With mode: INPUT=../../data/face.mp4 OUTPUT=magnified_face.avi MODE=laplacian make run
# With extra args: INPUT=../../data/face.mp4 OUTPUT=magnified_face.avi ARGS="--alpha 100" make run
add_custom_target(run
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/evmpipeline 
        --input $(INPUT) 
        --output $(OUTPUT) 
        --mode $(if $(MODE),$(MODE),gaussian) 
        $(ARGS)
    DEPENDS evmpipeline
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running CUDA EVM pipeline (use INPUT=... OUTPUT=... MODE=... make run)"
)

# Custom target for quick test with default face video (Gaussian mode)
# Usage: make test_face
add_custom_target(test_face
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/evmpipeline 
        --input ${CMAKE_SOURCE_DIR}/../data/face.mp4 
        --output face_gaussian_magnified.avi
        --mode gaussian
    DEPENDS evmpipeline
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running CUDA EVM Gaussian pipeline on default face video"
)

# Custom target for quick test with Laplacian mode
# Usage: make test_face_laplacian
add_custom_target(test_face_laplacian
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/evmpipeline 
        --input ${CMAKE_SOURCE_DIR}/../data/face.mp4 
        --output face_laplacian_magnified.avi
        --mode laplacian
    DEPENDS evmpipeline
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running CUDA EVM Laplacian pipeline on default face video"
)

# Custom target for heart rate detection preset (Gaussian mode)
# Usage: INPUT=face_video.mp4 make heart_rate
add_custom_target(heart_rate
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/evmpipeline 
        --input $(INPUT) 
        --output heart_rate_gaussian_magnified.avi 
        --mode gaussian 
        --alpha 50 --fl 0.8333 --fh 1.0 --level 4
    DEPENDS evmpipeline
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running CUDA EVM Gaussian pipeline optimized for heart rate detection (set INPUT=video.mp4)"
)

# Custom target for heart rate detection with Laplacian (faster processing)
# Usage: INPUT=face_video.mp4 make heart_rate_laplacian
add_custom_target(heart_rate_laplacian
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/evmpipeline 
        --input $(INPUT) 
        --output heart_rate_laplacian_magnified.avi 
        --mode laplacian 
        --alpha 50 --fl 0.8333 --fh 1.0 --level 4
    DEPENDS evmpipeline
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running CUDA EVM Laplacian pipeline optimized for heart rate detection (set INPUT=video.mp4)"
)

# Custom target for breathing analysis preset (Gaussian mode)
# Usage: INPUT=chest_video.mp4 make breathing
add_custom_target(breathing
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/evmpipeline 
        --input $(INPUT) 
        --output breathing_gaussian_magnified.avi 
        --mode gaussian 
        --alpha 30 --fl 0.17 --fh 0.5 --level 5
    DEPENDS evmpipeline
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running CUDA EVM Gaussian pipeline optimized for breathing analysis (set INPUT=video.mp4)"
)

# Custom target for breathing analysis with Laplacian (faster processing)
# Usage: INPUT=chest_video.mp4 make breathing_laplacian
add_custom_target(breathing_laplacian
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/evmpipeline 
        --input $(INPUT) 
        --output breathing_laplacian_magnified.avi 
        --mode laplacian 
        --alpha 30 --fl 0.17 --fh 0.5 --level 5
    DEPENDS evmpipeline
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running CUDA EVM Laplacian pipeline optimized for breathing analysis (set INPUT=video.mp4)"
)

# Custom target to show available algorithms and their performance
# Usage: make info
add_custom_target(info
    COMMAND echo "CUDA Eulerian Video Magnification - Dual Algorithm Support"
    COMMAND echo "=========================================================="
    COMMAND echo ""
    COMMAND echo "Available Algorithms:"
    COMMAND echo "  gaussian   - FFT-based temporal filtering (42.89 dB PSNR)"
    COMMAND echo "  laplacian  - IIR-based temporal filtering (37.62 dB PSNR, 78.4 FPS)"
    COMMAND echo ""
    COMMAND echo "Usage Examples:"
    COMMAND echo "  ./evmpipeline --input=video.mp4 --output=result.avi --mode=gaussian"
    COMMAND echo "  ./evmpipeline --input=video.mp4 --output=result.avi --mode=laplacian"
    COMMAND echo ""
    COMMAND echo "Available Make Targets:"
    COMMAND echo "  make test_face           - Test Gaussian mode on face video"
    COMMAND echo "  make test_face_laplacian - Test Laplacian mode on face video"
    COMMAND echo "  make heart_rate          - Heart rate detection (Gaussian)"
    COMMAND echo "  make heart_rate_laplacian- Heart rate detection (Laplacian)"
    COMMAND echo "  make breathing           - Breathing analysis (Gaussian)"
    COMMAND echo "  make breathing_laplacian - Breathing analysis (Laplacian)"
    COMMAND echo "  make info                - Show this information"
    COMMENT "Displaying CUDA EVM pipeline information"
)