#!/bin/bash

mkdir build
cd build

# Absolute Path
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
# Use PIP installed PyTorch
# cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..

cmake --build . --config Release
