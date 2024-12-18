if(UNIX)
    # Direct CMake to use icpx rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER icpx)
else() # Windows
    # Force CMake to use icx-cl rather than the default C++ compiler/linker
    # (needed on Windows only)
    include (CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP)
    include (Platform/Windows-Clang)
endif()

cmake_minimum_required (VERSION 3.4)

project(VectorAdd CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

message(STATUS "Using the USM variant.")
set(SOURCE_FILE main.cpp)
set(TARGET_NAME equivalent_ops)


# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# Include the shared header directory
include_directories(${CMAKE_SOURCE_DIR}/../../../shared/include ${CMAKE_SOURCE_DIR}/include)

# LibTorch
find_package(Torch REQUIRED)

#
# SECTION 1
# This section defines rules to create a cpu-gpu make target
# This can safely be removed if your project is only targetting FPGAs
#

# Compile and link flags for Intel oneAPI MKL and SYCL (NOTE: this is updated from the 2_MKL_Basic_Vector_Add example by referencing another oneMKL Makefile example)
set(COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG} -DMKL_ILP64 -qopenmp -fopenmp-targets=spir64 -fiopenmp")
set(LINK_FLAGS "-fsycl -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl")

# To compile in a single command:
#    icpx -fsycl <file>.cpp -o <file>
# CMake executes:
#    [compile] icpx -fsycl -o <file>.cpp.o -c <file>.cpp
#    [link]    icpx -fsycl <file>.cpp.o -o <file>
add_executable(${TARGET_NAME} ${SOURCE_FILE})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
# Check the build type and add the DEBUG definition if Debug
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # https://cmake.org/cmake/help/latest/command/target_compile_definitions.html
    target_compile_definitions(${TARGET_NAME} PRIVATE DEBUG)
endif()

target_link_libraries(${TARGET_NAME} "${TORCH_LIBRARIES}")

add_custom_target(cpu-gpu DEPENDS ${TARGET_NAME})

#
# End of SECTION 1
#
