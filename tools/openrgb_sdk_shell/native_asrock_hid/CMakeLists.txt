cmake_minimum_required(VERSION 3.20)
project(asrock_polychrome_hid LANGUAGES CXX)

add_library(asrock_polychrome_hid SHARED asrock_polychrome_hid.cpp)
target_compile_features(asrock_polychrome_hid PRIVATE cxx_std_17)
target_compile_definitions(asrock_polychrome_hid PRIVATE ASROCK_HID_EXPORTS WIN32_LEAN_AND_MEAN NOMINMAX)

if(MSVC)
  target_compile_options(asrock_polychrome_hid PRIVATE /O2 /EHsc)
endif()
