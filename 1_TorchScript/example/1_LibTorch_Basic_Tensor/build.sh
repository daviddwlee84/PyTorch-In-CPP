#!/bin/bash

# Default to Release if no argument is provided
BUILD_TYPE=${1:-Release}

mkdir build
cd build

# Absolute Path
# cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
# cmake -DCMAKE_PREFIX_PATH=$(pwd)/../../../../libtorch ..
# Use PIP installed PyTorch (make sure we are using same version torch for tensor dump and load)
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ..

cmake --build . --config ${BUILD_TYPE}
