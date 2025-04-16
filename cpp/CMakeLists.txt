cmake_minimum_required(VERSION 3.10)

project(evm_cpu_pipeline VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
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

# Gather all source and header files
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/include/*.hpp")

# Main library with all core sources
add_library(evm_core STATIC ${SOURCES} ${HEADERS})
target_link_libraries(evm_core PUBLIC ${OpenCV_LIBS})

# Main executable (pipeline entrypoint)
add_executable(evmpipeline ${CMAKE_SOURCE_DIR}/src/main.cpp)
target_link_libraries(evmpipeline PRIVATE evm_core ${OpenCV_LIBS})

# Enable testing
enable_testing()

# Find GoogleTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Gather all test files
file(GLOB TEST_SOURCES "${CMAKE_SOURCE_DIR}/tests/*.cpp")

# Add test executable
add_executable(evm_tests ${TEST_SOURCES})
target_link_libraries(evm_tests PRIVATE evm_core GTest::GTest GTest::Main ${OpenCV_LIBS})

# Register tests with CTest
add_test(NAME EVMUnitTests COMMAND evm_tests)

# Install rules (optional)
install(TARGETS evmpipeline RUNTIME DESTINATION bin)
