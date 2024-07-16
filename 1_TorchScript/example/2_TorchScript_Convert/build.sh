#!/bin/bash

mkdir build
cd build

# Absolute Path
# NOTE: Need to use C++17 or later to compile (ATen & PyTorch)
# cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch -DCMAKE_CXX_STANDARD=17 ..
# cmake -DCMAKE_PREFIX_PATH=$(pwd)/../../../../libtorch -DCMAKE_CXX_STANDARD=17 ..
cmake -DCMAKE_PREFIX_PATH=/mnt/NAS/sda/ShareFolder/lidawei/library/libtorch -DCMAKE_CXX_STANDARD=17 ..
# Use PIP installed PyTorch
# cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..

cmake --build . --config Release
