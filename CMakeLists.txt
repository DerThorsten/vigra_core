cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(vigra_core VERSION 1.0 LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vigra_cmake")

add_library(vigra_core INTERFACE)

target_include_directories(vigra_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(DIRECTORY include/ DESTINATION include)

enable_testing()
add_subdirectory(test)
