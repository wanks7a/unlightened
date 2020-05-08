cmake_minimum_required(VERSION 3.16)
project(unlightened VERSION 1.0.0)
project(unlightened LANGUAGES CUDA CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/libs)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/libs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(unlightened)
add_subdirectory(tests)
add_subdirectory(model)