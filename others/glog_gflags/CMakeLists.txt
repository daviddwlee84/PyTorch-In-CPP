cmake_minimum_required(VERSION 3.15)
project(glog_gflags)

# Set the target name
set(TARGET_NAME glog_gflags_cmake)

# Add the executable
add_executable(${TARGET_NAME} main.cpp)

# Include directories
include_directories(/usr/local/include)

# Link directories and libraries
link_directories(/usr/local/lib)
target_link_libraries(${TARGET_NAME} glog gflags)

# Define preprocessor macros
target_compile_definitions(${TARGET_NAME} PRIVATE GLOG_USE_GLOG_EXPORT)

# Set C++ standard
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 14)

# BUG: somehow the find_package didn't find the gflags
# 
# cmake_minimum_required(VERSION 3.15)
# project(glog_gflags)
# 
# set(TARGET_NAME glog_gflags_cmake)
# # set(GFLAGS_INCLUDE_DIR /usr/local/lib/cmake/gflags)
# 
# # Include the Conan setup (Not working)
# # include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
# # conan_basic_setup()
# 
# # Add the executable
# add_executable(${TARGET_NAME} main.cpp)
# 
# # https://google.github.io/glog/stable/usage/#using-glog-in-a-cmake-project
# # find_package(glog 0.7.1 REQUIRED)
# find_package(glog REQUIRED)
# target_link_libraries(${TARGET_NAME} glog::glog)
# 
# set(CMAKE_PREFIX_PATH /usr/local/lib/cmake/gflags)
# # https://gflags.github.io/gflags/#cmake
# # find_package(gflags 2.2.2 REQUIRED)
# find_package(gflags REQUIRED)
# # target_link_libraries(${TARGET_NAME} gflags::gflags)
# target_link_libraries(${TARGET_NAME} glog::glog gflags::gflags)
# 
# 
# set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 11)

