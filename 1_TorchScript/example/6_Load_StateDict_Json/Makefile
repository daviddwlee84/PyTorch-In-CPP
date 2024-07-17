# Makefile for compiling load_json_to_torch_tensor.cpp (TODO: this is currently buggy)

# Compiler
CXX = g++

# Paths
TORCH_PATH = /mnt/NAS/sda/ShareFolder/lidawei/library/libtorch

# Compiler flags
# NOTE: need c++17 to compile libtorch
CXXFLAGS = -std=c++17 -I$(TORCH_PATH)/include -I$(TORCH_PATH)/include/torch/csrc/api/include -I../../../shared/include -Iinclude

# g++ -o load_state_dict load_state_dict.cpp -I/path/to/json/include -I/path/to/libtorch/include -I/path/to/libtorch/include/torch/csrc/api/include -L/path/to/libtorch/lib -ltorch -ltorch_cpu -lc10 -I/path/to/oneMKL/include -L/path/to/oneMKL/lib -loneMKL
# Torch:INTERNAL=[/mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/lib/libtorch.so][/mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/include;/mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/include/torch/csrc/api/include][v()]

# Linker flags
LDFLAGS = -L$(TORCH_PATH)/lib -ltorch -ltorch_cpu -lc10 -Wl,--no-as-needed -pthread -lgomp -ltorch_cpu -lc10

# Source files
SRC = load_json_to_torch_tensor.cpp

# Executable name
EXEC = load_json_to_tensor

# Build the executable
all: $(EXEC)

$(EXEC): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXEC) $(LDFLAGS)

# Clean the build directory
clean:
	rm -f $(EXEC)
