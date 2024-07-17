# Makefile for compiling create_json.cpp

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -Iinclude/ -I../../../shared/include/

# Source files
SRC = load_json_only.cpp

# Executable name
EXEC = load_json

# Build the executable
all: $(EXEC)

$(EXEC): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(SRC)

# Clean the build directory
clean:
	rm -f $(EXEC)
