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
cmake --build . --target cpu-gpu

# Create state_dict.json (path is not correct)
# python ../model.py

# Execute the compiled program with the specified argument
GLOG_logtostderr=1 ./load_and_run ../state_dict.json
