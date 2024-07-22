cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(glog_cmake)

set(TARGET_NAME glog_cmake)

# Example app executable
add_executable(${TARGET_NAME} main.cpp)

# https://google.github.io/glog/stable/usage/#using-glog-in-a-cmake-project
find_package (glog 0.7.1 REQUIRED)
target_link_libraries (${TARGET_NAME} glog::glog)

# Benchmark executable
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
