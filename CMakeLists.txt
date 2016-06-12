cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(foundation VERSION 1.0 LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_library(foundation INTERFACE)

target_include_directories(foundation INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(DIRECTORY include/ DESTINATION include)

enable_testing()
add_subdirectory(test)
