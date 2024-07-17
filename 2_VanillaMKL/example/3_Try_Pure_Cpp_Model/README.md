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
