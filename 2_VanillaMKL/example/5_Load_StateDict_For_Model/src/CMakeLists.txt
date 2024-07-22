message(STATUS "Using the USM variant.")
set(SOURCE_FILE main.cpp SingleStepLSTMRegressionMKL.cpp weights_loader.cpp)
set(TARGET_NAME load_and_run)


# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# Include the shared header directory
include_directories(${CMAKE_SOURCE_DIR}/../../../shared/include ${CMAKE_SOURCE_DIR}/include)

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

# https://google.github.io/glog/stable/usage/#using-glog-in-a-cmake-project
# Assuming that glog was previously built using CMake or installed using a package manager, you can use the CMake command find_package to build against glog in your CMake project
find_package (glog 0.7.1 REQUIRED)
target_link_libraries (${TARGET_NAME} glog::glog)

add_custom_target(cpu-gpu DEPENDS ${TARGET_NAME})

#
# End of SECTION 1
#
