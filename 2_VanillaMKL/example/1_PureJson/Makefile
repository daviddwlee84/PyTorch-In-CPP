# Makefile for compiling create_json.cpp

# Compiler
CXX = g++

# Compiler flags
# CXXFLAGS = -std=c++11 -Iinclude/thirdparty/
CXXFLAGS = -std=c++11 -I../../../shared/include/

# Source files
SRC = create_json.cpp

# Executable name
EXEC = create_json

# Build the executable
all: $(EXEC)

$(EXEC): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(SRC)

# Clean the build directory
clean:
	rm -f $(EXEC)
