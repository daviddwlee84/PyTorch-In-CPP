
```bash
. /opt/intel/oneapi/setvars.sh
mkdir build
cd build
cmake ..
make cpu-gpu
./load_and_run ../state_dict.json
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
