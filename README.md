# PyTorch In CPP

Showcase of running Python trained PyTorch model in C++ (tested on Ubuntu)

## Prerequisite

- `cmake` version 3.18 or higher

```bash
sudo apt install build-essential libtool autoconf unzip wget cmake
```

- [software installation - How do I install the latest version of cmake from the command line? - Ask Ubuntu](https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line)
- [Download CMake](https://cmake.org/download/)

```bash
# Check if 3.18 or higher
cmake --version

# If your cmake version is not high enough
wget https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0.tar.gz
tar -xvzf cmake-3.30.0.tar.gz
cd cmake-3.30.0
# This requires a few minutes
./bootstrap
# Build in parallel
make -j$(nproc)
# Install
sudo make install
```

## 1. TorchScript

## 2. Json + MKL
