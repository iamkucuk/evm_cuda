# CMakeLists.txt (located in evmcpp/)
cmake_minimum_required(VERSION 3.10) # Specify minimum CMake version

project(EVMCPP VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Find Required Packages ---
# Find OpenCV (Core, Image Processing, Video I/O needed)
# Users might need to set OpenCV_DIR environment variable if not found automatically
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please ensure it's installed and findable (e.g., set OpenCV_DIR).")
else()
    message(STATUS "Found OpenCV ${OpenCV_VERSION}")
    # Add OpenCV include directories globally for the project
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# --- Project Structure ---
# Add our own include directory
include_directories(../include) # Corrected path relative to CMakeLists.txt

# --- Build Targets ---
# Option 1: Build a static library for the core EVM logic (Recommended)
add_library(evm_core STATIC
    ../src/evmcpp/laplacian_pyramid.cpp # Corrected relative path
    ../src/evmcpp/processing.cpp     # Corrected relative path
    ../src/evmcpp/butterworth.cpp    # Corrected relative path
    ../src/evmcpp/gaussian_pyramid.cpp # Added Gaussian pyramid source
)
# Link the library against OpenCV
target_link_libraries(evm_core PRIVATE ${OpenCV_LIBS})
# Ensure users of the library can find its headers
target_include_directories(evm_core PUBLIC ../include) # Corrected path relative to CMakeLists.txt

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