cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

# Benchmark executable
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark "${TORCH_LIBRARIES}")
set_property(TARGET benchmark PROPERTY CXX_STANDARD 17)
