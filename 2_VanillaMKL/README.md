# Vanilla MKL

- [nlohmann/json: JSON for Modern C++](https://github.com/nlohmann/json?tab=readme-ov-file)
- [oneapi-src/oneMKL: oneAPI Math Kernel Library (oneMKL) Interfaces](https://github.com/oneapi-src/oneMKL)

## Getting Started

### Setup MKL

- [Accelerate Fast Math with Intel® oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)
  - [Download the Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&linux-install-type=apt) (Toolkit)
  - [Get Intel® oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=linux&linux-install=apt) (Stand-Alone Version)
  - [APT](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2024-2/apt.html)

```bash
sudo apt update
sudo apt install -y gpg-agent wget

wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update

sudo apt install intel-basekit
```

```bash
. /opt/intel/oneapi/setvars.sh

oneapi-cli
# (1) Create a project
# (1) cpp
# => Toolkit > Get Started > Base: Vector Add

mkdir build
cd build
cmake ..

make cpu-gpu

./vector-add-buffers
```

- [Get Started with Intel® oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/docs/onemkl/get-started-guide/2024-1/overview.html)
  - [Developer Guide for Intel® oneAPI Math Kernel Library for Linux*](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-2/overview.html)
  - [Using Code Examples](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-2/using-code-examples.html)

1. [Selecting a Compiler — oneAPI Math Kernel Library Interfaces 0.1 documentation](https://oneapi-src.github.io/oneMKL/selecting_a_compiler.html)
   - [Compile Cross-Architecture: Intel® oneAPI DPC++/C++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html) => should already included in the toolkit

- [Debugging SYCL™ code with DPC++ and Visual Studio® Code - Codeplay Software Ltd](https://codeplay.com/portal/blogs/2023/03/01/debugging-sycl-code-with-dpc-and-visual-studio-code)
  - [Setting up C++ development with Visual Studio® Code on Ubuntu - Codeplay Software Ltd](https://codeplay.com/portal/blogs/2023/03/01/setting-up-c-development-with-visual-studio-code-on-ubuntu.html)
  - [Setting up SYCL™ development with oneAPI™, DPC++ and Visual Studio® Code on Ubuntu - Codeplay Software Ltd](https://codeplay.com/portal/blogs/2023/03/01/setting-up-sycl-development-with-oneapi-dpc-and-visual-studio-code-on-ubuntu.html)
