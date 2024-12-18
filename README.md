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

- GCC 9 or later

```bash
gcc --version

sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-13 g++-13
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 60 --slave /usr/bin/g++ g++ /usr/bin/g++-13
```

- [software installation - How do I use the latest GCC on Ubuntu? - Ask Ubuntu](https://askubuntu.com/questions/466651/how-do-i-use-the-latest-gcc-on-ubuntu)
- [ppa - install gcc-9 on Ubuntu 18.04? - Ask Ubuntu](https://askubuntu.com/questions/1140183/install-gcc-9-on-ubuntu-18-04)

- GDB (optional)

```bash
sudo apt update
sudo apt install gdb
```

- [Configure launch.json for C/C++ debugging in Visual Studio Code](https://code.visualstudio.com/docs/cpp/launch-json-reference)

> TODO: `clang`..?

### Dependencies

glog

- [google/glog: C++ implementation of the Google logging module](https://github.com/google/glog)
  - [glog - Conan 2.0: C and C++ Open Source Package Manager](https://conan.io/center/recipes/glog?version=0.7.1)

> ```bash
> git clone https://github.com/google/glog.git
> cd glog
> mkdir build
> cd build
> cmake ..
> make
> sudo make install
> ```
>
> `sudo apt install libgoogle-glog-dev` (somehow can solve g++ and Makefile dependencies)

gflags

- [gflags/INSTALL.md at master · gflags/gflags](https://github.com/gflags/gflags/blob/master/INSTALL.md)
- [gflags - Conan 2.0: C and C++ Open Source Package Manager](https://conan.io/center/recipes/gflags?version=2.2.2)
- [gflags/example: Example C++ project using the gflags library](https://github.com/gflags/example)

> ```bash
> git clone https://github.com/gflags/gflags.git
> ... # the same
> ```
>
> `sudo apt-get install libgflags-dev` (seems might already included in glog)

## 1. TorchScript

## 2. Json + MKL

## Resources

- [Get Started with C++ on Linux in Visual Studio Code](https://code.visualstudio.com/docs/cpp/config-linux)
  - [microsoft/vscode-cpptools: Official repository for the Microsoft C/C++ extension for VS Code.](https://github.com/microsoft/vscode-cpptools)
  - [Introductory Videos for C++ in Visual Studio Code](https://code.visualstudio.com/docs/cpp/introvideos-cpp)
  - [c_cpp_properties.json reference](https://code.visualstudio.com/docs/cpp/c-cpp-properties-schema-reference)
- [Tasks in Visual Studio Code](https://code.visualstudio.com/docs/editor/tasks)

---

- [microsoft/vcpkg: C++ Library Manager for Windows, Linux, and MacOS](https://github.com/Microsoft/vcpkg)
- [conan-io/conan: Conan - The open-source C and C++ package manager](https://github.com/conan-io/conan)
  - [Build a simple CMake project using Conan — conan 2.5.0 documentation](https://docs.conan.io/2/tutorial/consuming_packages/build_simple_cmake_project.html)

```bash
# the very first time using conan
conan profile detect

# will install items in conanfile.txt
# (haven't tested yet)
conan install . --build=missing
```
