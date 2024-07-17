# Build Pure C++ GRU Model Using oneMKL Library

```bash
mkdir build
cd build
cmake ..
make cpu-gpu
```

```bash
# Default
(research) ➜  build git:(main) ✗ ./pure_cpp 
MKL is using 60 threads.
OpenMP is using 120 threads.
Output: 2.41577

# Try MKL_NUM_THREADS
(research) ➜  build git:(main) ✗ MKL_NUM_THREADS=1 ./pure_cpp
MKL is using 1 threads.
OpenMP is using 120 threads.
Output: 2.41577

# Try OMP_NUM_THREADS
(research) ➜  build git:(main) ✗ OMP_NUM_THREADS=1 ./pure_cpp 
MKL is using 1 threads.
OpenMP is using 1 threads.
Output: 2.41577
```

```bash
(research) ➜  build git:(main) ✗ ./pure_cpp 1000
MKL is using 60 threads.
OpenMP is using 120 threads.
Output: 0.100014
Will use average of 1000 iterations.
Benchmarking Pure oneMKL C++ model...
Inference time: 5.99911e-05 seconds
(research) ➜  build git:(main) ✗ ./pure_cpp 10000
MKL is using 60 threads.
OpenMP is using 120 threads.
Output: 0.100014
Will use average of 10000 iterations.
Benchmarking Pure oneMKL C++ model...
Inference time: 6.06239e-05 seconds

(research) ➜  build git:(main) ✗ OMP_NUM_THREADS=1 ./pure_cpp 10000
MKL is using 1 threads.
OpenMP is using 1 threads.
Output: 0.100014
Will use average of 10000 iterations.
Benchmarking Pure oneMKL C++ model...
Inference time: 6.74404e-05 seconds
(research) ➜  build git:(main) ✗ MKL_NUM_THREADS=1 ./pure_cpp 10000
MKL is using 1 threads.
OpenMP is using 120 threads.
Output: 0.100014
Will use average of 10000 iterations.
Benchmarking Pure oneMKL C++ model...
Inference time: 6.81793e-05 seconds
```

---

- [Developer Reference for Intel® oneAPI Math Kernel Library for C - Overview](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/overview-001.html)
- [BLAS - WikiPedia](https://zh.wikipedia.org/wiki/BLAS) => Basic Linear Algebra Subprograms

> BLAS Routines
>
> The BLAS routines and functions are divided into the following groups according to the operations they perform:
>
> 1. BLAS Level 1 Routines perform operations of both **addition and reduction on vectors of data**. Typical operations include scaling and dot products.
> 2. BLAS Level 2 Routines perform **matrix-vector operations**, such as matrix-vector multiplication, rank-1 and rank-2 matrix updates, and solution of triangular systems.
> 3. BLAS Level 3 Routines perform **matrix-matrix operations**, such as matrix-matrix multiplication, rank-k update, and solution of triangular systems.
