#!/bin/bash

mkdir build
cd build

# Use PIP installed PyTorch
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..

cmake --build . --config Release
