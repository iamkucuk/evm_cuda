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

# Main library with all core sources (same name pattern as reference: evm_core)
add_library(evm_core STATIC ${SOURCES} ${HEADERS})

# Set CUDA properties
set_target_properties(evm_core PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "86"
)

# Link libraries (same pattern as reference)
target_link_libraries(evm_core PUBLIC ${OpenCV_LIBS} cufft)

# Main executable (pipeline entrypoint) - same name as reference: evmpipeline
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
# With extra args: INPUT=../../data/face.mp4 OUTPUT=magnified_face.avi ARGS="--alpha 100" make run
add_custom_target(run
    COMMAND ${CMAKE_SOURCE_DIR}/run_evm.sh
    DEPENDS evmpipeline
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running CUDA EVM pipeline (use INPUT=... OUTPUT=... make run)"
)

# Custom target for quick test with default face video
# Usage: make test_face
add_custom_target(test_face
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/evmpipeline 
        --input ${CMAKE_SOURCE_DIR}/../data/face.mp4 
        --output face_magnified.avi
    DEPENDS evmpipeline
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running CUDA EVM pipeline on default face video"
)

# Custom target for heart rate detection preset
# Usage: INPUT=face_video.mp4 make heart_rate
add_custom_target(heart_rate
    COMMAND ${CMAKE_SOURCE_DIR}/run_evm.sh $(INPUT) heart_rate_magnified.avi --alpha 50 --fl 0.8333 --fh 1.0 --level 4
    DEPENDS evmpipeline
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running CUDA EVM pipeline optimized for heart rate detection (set INPUT=video.mp4)"
)

# Custom target for breathing analysis preset
# Usage: INPUT=chest_video.mp4 make breathing
add_custom_target(breathing
    COMMAND ${CMAKE_SOURCE_DIR}/run_evm.sh $(INPUT) breathing_magnified.avi --alpha 30 --fl 0.17 --fh 0.5 --level 5
    DEPENDS evmpipeline
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running CUDA EVM pipeline optimized for breathing analysis (set INPUT=video.mp4)"
)