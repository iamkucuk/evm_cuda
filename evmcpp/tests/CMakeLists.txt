# CMakeLists.txt (located in evmcpp/tests/)
cmake_minimum_required(VERSION 3.10)

# --- Find/Download GoogleTest ---
# Use FetchContent for modern CMake to handle dependencies
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip # Specify a stable version
)
# For older CMake versions:
# FetchContent_Declare(
#   googletest
#   GIT_REPOSITORY https://github.com/google/googletest.git
#   GIT_TAG        release-1.14.0 # Or main, but a tag is safer
# )

# Set properties before population (optional, but good practice)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  # Prevent GTest from installing and overriding parent project settings
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# --- Define Test Executable ---
add_executable(evm_tests
    test_processing.cpp
    test_laplacian.cpp
    test_butterworth.cpp
    test_gaussian.cpp # Added Gaussian pyramid tests
    test_helpers.cpp  # Added common test helper functions
    # test_laplacian.cpp # Add later
)

# Define the path to the source data directory for the tests
target_compile_definitions(evm_tests PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/data")

# --- Link Test Executable ---
# Link against GoogleTest libraries (gtest_main includes main function)
target_link_libraries(evm_tests PRIVATE gtest_main)

# Link against our core library (evm_core defined in parent CMakeLists.txt)
# We need access to the functions we want to test
target_link_libraries(evm_tests PRIVATE evm_core)

# Include directories needed for tests (parent include and OpenCV)
# Note: OpenCV includes should be inherited from evm_core linking if set up correctly
# target_include_directories(evm_tests PRIVATE ../include) # Already handled by linking evm_core
# --- Add Test to CTest ---
include(GoogleTest)
gtest_discover_tests(evm_tests)

# --- Copy Test Data ---
# Define the source data directory relative to this CMakeLists.txt
set(TEST_DATA_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/data)
# Define the destination directory relative to the build output for this target
set(TEST_DATA_DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/data)

# Create a custom target to copy the data directory contents
add_custom_target(copy_test_data ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${TEST_DATA_SRC_DIR} ${TEST_DATA_DEST_DIR}
    COMMENT "Copying test data to build directory"
)

# Make the test executable depend on the data being copied
add_dependencies(evm_tests copy_test_data)

message(STATUS "Configured tests subdirectory. Test data will be copied to ${TEST_DATA_DEST_DIR}")