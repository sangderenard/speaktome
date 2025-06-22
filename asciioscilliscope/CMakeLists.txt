cmake_minimum_required(VERSION 3.15)
project(asciioscilliscope LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen is header-only in eigen/ — no separate build needed

# Source files
file(GLOB SRC_FILES src/*.cpp)

# Build library
add_library(asciioscilliscope STATIC ${SRC_FILES})
# Expose public headers (project and Eigen header-only library)
target_include_directories(asciioscilliscope PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/eigen>
)

# Use bundled GoogleTest submodule
add_subdirectory(googletest)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include)

enable_testing()
# Add example executable
add_executable(ascii_demo examples/ascii_demo.cpp)
target_link_libraries(ascii_demo PRIVATE asciioscilliscope)

# Add tests
file(GLOB TEST_SRC_FILES test/*.cpp)
# Collect test executable names
set(TEST_NAMES)
foreach(test_src ${TEST_SRC_FILES})
    get_filename_component(test_name ${test_src} NAME_WE)
    list(APPEND TEST_NAMES ${test_name})
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE asciioscilliscope gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Alias 'build-all' target to build library, demo, and tests
add_custom_target(build-all
    DEPENDS asciioscilliscope ascii_demo ${TEST_NAMES}
    COMMENT "Build all targets"
)

# Alias 'run-tests' target to run all tests
add_custom_target(run-tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${TEST_NAMES}
    COMMENT "Run all tests"
)
