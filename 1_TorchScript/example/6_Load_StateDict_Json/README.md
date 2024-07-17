

## Dump State Dict and Load Json

```bash
# Create state_dict.json
python model.py
# Compile and execute
./build.sh
./build/load_json_to_tensor state_dict.json

# TODO: Use Makefile (failed)
make

# LibTorch
# https://stackoverflow.com/questions/12057852/multiple-makefiles-in-one-directory
# make -f Makefile_LibTorch
```

---

- [pytorch/aten/src/ATen/test/basic.cpp at 455e85a2f181a66e7505d0e7eeb0ad0825bc4a41 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/455e85a2f181a66e7505d0e7eeb0ad0825bc4a41/aten/src/ATen/test/basic.cpp)
- [Can I initialize tensor from std::vector in libtorch? - C++ - PyTorch Forums](https://discuss.pytorch.org/t/can-i-initialize-tensor-from-std-vector-in-libtorch/33236/4)

> ATen (short for "A Tensor Library") is the core tensor library in PyTorch (LibTorch being its C++ API counterpart).
