
```bash
. /opt/intel/oneapi/setvars.sh
mkdir build
cd build
cmake ..
make cpu-gpu
./load_and_run ../state_dict.json

...

Output: -0.302049
Will use average of 100 iterations.
Benchmarking Pure oneMKL C++ model...
Inference time: 3.11092e-06 seconds
```

```bash
./build_and_run.sh
./build.sh
```

```bash
./gdb_wrapper.sh ./build/load_and_run

(gdb) break main
(gdb) run ./state_dict.json
(gdb) next
(gdb) print argc
$1 = 2
(gdb) continue
```

```bash
python python_load.py
tensor([[-0.1297]], grad_fn=<AddmmBackward0>)
```

=> BUG: Currently Python single step result is not align with C++ version
