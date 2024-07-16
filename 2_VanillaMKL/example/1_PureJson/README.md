# Pure JSON

Use Makefile

1. ~~Download `json.hpp`: `wget --directory-prefix=./include/thirdparty/nlohmann https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp` (https://stackoverflow.com/questions/11258271/wget-o-for-non-existing-save-path)~~ => We are now put this in `shared/include`
2. `make` => try it `./create_json`

---

Use CMake

```bash
mkdir build
cd build
cmake ..
make
./create_json
```
