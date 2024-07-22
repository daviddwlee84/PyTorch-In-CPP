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

# debug: $(EXEC)
# -g flag for debugging breakpoints to work
# -DDEBUG for enable #ifdef DEBUG code blocks
# $(EXEC): $(SRC)
# 	$(CXX) $(CXXFLAGS) -DDEBUG -g -o $(EXEC) $(SRC)
debug: CXXFLAGS += -DDEBUG -g
debug: clean $(EXEC)

# Clean the build directory
clean:
	rm -f $(EXEC)
