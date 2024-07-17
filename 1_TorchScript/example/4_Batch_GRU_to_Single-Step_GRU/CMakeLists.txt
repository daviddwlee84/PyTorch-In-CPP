cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(complex_gru)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Benchmark executable
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark "${TORCH_LIBRARIES}")
set_property(TARGET benchmark PROPERTY CXX_STANDARD 17)
