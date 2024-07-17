cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pure_cpp_model)

set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

add_executable(SingleStepLSTMRegression single_file.cpp)
target_link_libraries(SingleStepLSTMRegression "${TORCH_LIBRARIES}")

set_property(TARGET SingleStepLSTMRegression PROPERTY CXX_STANDARD 17)
