cmake_minimum_required(VERSION 3.20)
project(openrgb_audio_scope LANGUAGES CXX)

add_library(openrgb_audio_scope SHARED openrgb_audio_scope.cpp)
target_compile_features(openrgb_audio_scope PRIVATE cxx_std_17)
target_compile_definitions(openrgb_audio_scope PRIVATE OPENRGB_AUDIO_SCOPE_EXPORTS)
target_link_libraries(openrgb_audio_scope PRIVATE ole32)

if(MSVC)
  target_compile_options(openrgb_audio_scope PRIVATE /O2 /EHsc)
endif()
