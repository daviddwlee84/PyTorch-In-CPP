#!/bin/bash

mkdir build
cd build

# Absolute Path
# cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
# cmake -DCMAKE_PREFIX_PATH=$(pwd)/../../../../libtorch ..
# Use PIP installed PyTorch (make sure we are using same version torch for tensor dump and load)
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..

cmake --build . --config Release
