cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

# Example app executable
add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
# NOTE: Need to use C++17 or later to compile (ATen & PyTorch)
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

# Include the shared header directory
include_directories(${CMAKE_SOURCE_DIR}/../../../shared/include)

# Benchmark executable
# Do this when tqdm.hpp is in the same directory
# add_executable(benchmark benchmark.cpp tqdm.hpp)
# Do this if we "include_directories"
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark "${TORCH_LIBRARIES}")
set_property(TARGET benchmark PROPERTY CXX_STANDARD 17)
