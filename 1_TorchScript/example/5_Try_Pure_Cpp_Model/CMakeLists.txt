cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pure_cpp_model)

set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

add_executable(SingleStepLSTMRegression single_file.cpp)
target_link_libraries(SingleStepLSTMRegression "${TORCH_LIBRARIES}")

set_property(TARGET SingleStepLSTMRegression PROPERTY CXX_STANDARD 17)


add_executable(benchmark benchmark_from_single_header.cpp)
target_link_libraries(benchmark "${TORCH_LIBRARIES}")

set_property(TARGET benchmark PROPERTY CXX_STANDARD 17)
