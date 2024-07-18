
```bash
. /opt/intel/oneapi/setvars.sh
mkdir build
cd build
cmake ..
make cpu-gpu
./load_and_run ../state_dict.json
```