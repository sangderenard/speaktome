cmake_minimum_required(VERSION 3.15)
project(asciioscilliscope LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add eigen submodule
add_subdirectory(eigen)

# Public headers (single include/ tree)
target_include_directories(asciioscilliscope PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Source files
file(GLOB SRC_FILES src/*.cpp)

# Build library
add_library(asciioscilliscope STATIC ${SRC_FILES})
target_link_libraries(asciioscilliscope PUBLIC Eigen3::Eigen)

# Find GoogleTest for unit testing
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

enable_testing()
# Add example executable
add_executable(ascii_demo examples/ascii_demo.cpp)
target_link_libraries(ascii_demo PRIVATE asciioscilliscope)

# Add tests
file(GLOB TEST_SRC_FILES test/*.cpp)
foreach(test_src ${TEST_SRC_FILES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE asciioscilliscope GTest::GTest GTest::Main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
