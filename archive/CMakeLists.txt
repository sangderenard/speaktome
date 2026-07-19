cmake_minimum_required(VERSION 3.15)
project(asciioscilliscope LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer system Eigen3; fallback to bundled headers if present
find_package(Eigen3)

# Source files
# Enumerate source files explicitly to allow partial builds
set(SRC_FILES
    src/ConicProjector3D.cpp
    src/asciioscilliscope/Projection2D.cpp
    src/PixelFrameBuffer.cpp
)

# Build library
add_library(asciioscilliscope STATIC ${SRC_FILES})
# Expose public headers (project and Eigen header-only library)
target_include_directories(asciioscilliscope PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/asciioscilliscope>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/asciioscilliscope/kernels>
)
if(Eigen3_FOUND)
    target_link_libraries(asciioscilliscope PUBLIC Eigen3::Eigen)
else()
    target_include_directories(asciioscilliscope PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/eigen>
    )
endif()

# Prefer system GoogleTest. Fall back to submodule if not found.
find_package(GTest)
if(GTest_FOUND)
    message(STATUS "Using system GTest")
    set(GTEST_BOTH_LIBRARIES GTest::gtest GTest::gtest_main)
else()
    message(STATUS "System GTest not found; using bundled submodule")
    add_subdirectory(googletest)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include)
    set(GTEST_BOTH_LIBRARIES gtest gtest_main)
endif()

enable_testing()
# Add example executable
# Optional demo executable. Commented out to keep CI lightweight.
# add_executable(ascii_demo examples/ascii_demo.cpp)
# target_link_libraries(ascii_demo PRIVATE asciioscilliscope)

# Add tests
file(GLOB TEST_SRC_FILES test/*.cpp)
# Collect test executable names
set(TEST_NAMES)
foreach(test_src ${TEST_SRC_FILES})
    get_filename_component(test_name ${test_src} NAME_WE)
    list(APPEND TEST_NAMES ${test_name})
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE asciioscilliscope ${GTEST_BOTH_LIBRARIES})
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Alias 'build-all' target to build library, demo, and tests
add_custom_target(build-all
    DEPENDS asciioscilliscope ${TEST_NAMES}
    COMMENT "Build all targets"
)

# Alias 'run-tests' target to run all tests
add_custom_target(run-tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${TEST_NAMES}
    COMMENT "Run all tests"
)
