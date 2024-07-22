#!/bin/bash

# Default to Release if no argument is provided
BUILD_TYPE=${1:-Release}

# Source the Intel oneAPI environment variables
source /opt/intel/oneapi/setvars.sh

# Create and navigate to the build directory
mkdir -p build
cd build

# Configure the project using CMake
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ..

# Build the specified target using CMake
# https://stackoverflow.com/questions/36633074/set-the-number-of-threads-in-a-cmake-build
cmake --build . --target cpu-gpu -- -j `nproc`
