cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(load_statedict_json)

set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

add_executable(load_json_to_tensor load_json_to_torch_tensor.cpp)
target_link_libraries(load_json_to_tensor "${TORCH_LIBRARIES}")

# Include the shared header directory
include_directories(${CMAKE_SOURCE_DIR}/../../../shared/include ${CMAKE_SOURCE_DIR}/include)

set_property(TARGET load_json_to_tensor PROPERTY CXX_STANDARD 17)

