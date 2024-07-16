cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pure_json)

# Example app executable
add_executable(create_json create_json.cpp)

# Include the shared header directory
include_directories(${CMAKE_SOURCE_DIR}/../../../shared/include)

# Benchmark executable
set_property(TARGET create_json PROPERTY CXX_STANDARD 17)
