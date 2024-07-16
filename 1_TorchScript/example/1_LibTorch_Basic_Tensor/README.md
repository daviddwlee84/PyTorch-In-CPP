# Basic LibTorch Example: Print Tensor (See if environment setup correctly)

## Getting Started

Finish the prerequisites (get `libtorch` installed or use `torch.utils.cmake_prefix_path`), just modify in `build.sh` and then run it.
Go to `build/` and will see `example-app` executable to run

(Assume we download `libtorch` at the root of this repository (same level as `.git`), if you tend to put in other places, you sould modify the path in `build.sh`)

---

## Load PyTorch Tensor with LibTorch

`./build/load-tensor`

TL;DR

Directly using `torch.save(..., 'xxx.pt')` and `torch::load(tensor, 'xxx.pt')` will get error.
We should write byte and read byte to bypass this. For more information see the bolded link.

### Trouble Shooting

- [PytorchStreamReader failed locating file constants.pkl: file not found - C++ - PyTorch Forums](https://discuss.pytorch.org/t/pytorchstreamreader-failed-locating-file-constants-pkl-file-not-found/169884)
- [PytorchStreamReader failed locating file constants.pkl - Mobile - PyTorch Forums](https://discuss.pytorch.org/t/pytorchstreamreader-failed-locating-file-constants-pkl/186146)
- [How to load python tensor in C++? - C++ - PyTorch Forums](https://discuss.pytorch.org/t/how-to-load-python-tensor-in-c/88813)
- [Load pytorch tensor created by torch.save(tensor_name, tensor_path) in c++ libtorch failed. · Issue #36568 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/36568)
- [**Load tensor from file in C++ fails · Issue #20356 · pytorch/pytorch**](https://github.com/pytorch/pytorch/issues/20356#issuecomment-567663701) => Best solution
  - [C++中使用pytorch保存的tensor | Yunfeng's Simple Blog](https://vra.github.io/2021/03/21/torch-tensor-python-to-cpp/)
