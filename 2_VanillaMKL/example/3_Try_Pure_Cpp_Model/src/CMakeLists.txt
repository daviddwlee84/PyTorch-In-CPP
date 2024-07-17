message(STATUS "Using the USM variant.")
set(SOURCE_FILE main.cpp SingleStepLSTMRegressionMKL.cpp)
set(TARGET_NAME pure_cpp)


# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# Link MKL libraries (failed)
# find_package(MKL REQUIRED)
# include_directories(${MKL_INCLUDE_DIRS})
# link_directories(${MKL_LIBRARY_DIRS})
# set(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel_lp64 mkl_sequential mkl_core)


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
add_custom_target(cpu-gpu DEPENDS ${TARGET_NAME})

#
# End of SECTION 1
#
