# Convert `nn.Module` into TorchScript

## Getting Started

- [Loading a TorchScript Model in C++ — PyTorch Tutorials 2.3.0+cu121 documentation](https://pytorch.org/tutorials/advanced/cpp_export.html)

1. `./build.sh`
2. Generate `torch.jit.ScriptModule` by `python convert_by_tracing.py` or by `python convert_by_annotation.py`
3. `./example-app ../traced_resnet_model.pt` or `./example-app ../script_resnet_model.pt` (in `./build/`)
4. `./benchmark` (in `./build/`)

## Benchmark

| Convert Using | Convert TorchScript | Inference in C++ (Average 1000 iterations in 60 threads) | Inference in C++ (Average 1000 iterations in single thread) |
| ------------- | ------------------- | -------------------------------------------------------- | ----------------------------------------------------------- |
| Tracing       | 0.8884697519242764  | 0.0239399                                                | 0.151959 (x6.3475 than 60 threads)                          |
| Annotation    | 0.4122754968702793  | 0.0255857                                                | 0.153334 (x5.9929 than 60 threads)                          |

```txt
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ResNet                                   [1, 1000]                 --
├─Conv2d: 1-1                            [1, 64, 112, 112]         9,408
├─BatchNorm2d: 1-2                       [1, 64, 112, 112]         128
├─ReLU: 1-3                              [1, 64, 112, 112]         --
├─MaxPool2d: 1-4                         [1, 64, 56, 56]           --
├─Sequential: 1-5                        [1, 64, 56, 56]           --
│    └─BasicBlock: 2-1                   [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-1                  [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-2             [1, 64, 56, 56]           128
│    │    └─ReLU: 3-3                    [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-4                  [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-5             [1, 64, 56, 56]           128
│    │    └─ReLU: 3-6                    [1, 64, 56, 56]           --
│    └─BasicBlock: 2-2                   [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-7                  [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-8             [1, 64, 56, 56]           128
│    │    └─ReLU: 3-9                    [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-10                 [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-11            [1, 64, 56, 56]           128
│    │    └─ReLU: 3-12                   [1, 64, 56, 56]           --
├─Sequential: 1-6                        [1, 128, 28, 28]          --
│    └─BasicBlock: 2-3                   [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-13                 [1, 128, 28, 28]          73,728
│    │    └─BatchNorm2d: 3-14            [1, 128, 28, 28]          256
│    │    └─ReLU: 3-15                   [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-16                 [1, 128, 28, 28]          147,456
│    │    └─BatchNorm2d: 3-17            [1, 128, 28, 28]          256
│    │    └─Sequential: 3-18             [1, 128, 28, 28]          8,448
│    │    └─ReLU: 3-19                   [1, 128, 28, 28]          --
│    └─BasicBlock: 2-4                   [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-20                 [1, 128, 28, 28]          147,456
│    │    └─BatchNorm2d: 3-21            [1, 128, 28, 28]          256
│    │    └─ReLU: 3-22                   [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-23                 [1, 128, 28, 28]          147,456
│    │    └─BatchNorm2d: 3-24            [1, 128, 28, 28]          256
│    │    └─ReLU: 3-25                   [1, 128, 28, 28]          --
├─Sequential: 1-7                        [1, 256, 14, 14]          --
│    └─BasicBlock: 2-5                   [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-26                 [1, 256, 14, 14]          294,912
│    │    └─BatchNorm2d: 3-27            [1, 256, 14, 14]          512
│    │    └─ReLU: 3-28                   [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-29                 [1, 256, 14, 14]          589,824
│    │    └─BatchNorm2d: 3-30            [1, 256, 14, 14]          512
│    │    └─Sequential: 3-31             [1, 256, 14, 14]          33,280
│    │    └─ReLU: 3-32                   [1, 256, 14, 14]          --
│    └─BasicBlock: 2-6                   [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-33                 [1, 256, 14, 14]          589,824
│    │    └─BatchNorm2d: 3-34            [1, 256, 14, 14]          512
│    │    └─ReLU: 3-35                   [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-36                 [1, 256, 14, 14]          589,824
│    │    └─BatchNorm2d: 3-37            [1, 256, 14, 14]          512
│    │    └─ReLU: 3-38                   [1, 256, 14, 14]          --
├─Sequential: 1-8                        [1, 512, 7, 7]            --
│    └─BasicBlock: 2-7                   [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-39                 [1, 512, 7, 7]            1,179,648
│    │    └─BatchNorm2d: 3-40            [1, 512, 7, 7]            1,024
│    │    └─ReLU: 3-41                   [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-42                 [1, 512, 7, 7]            2,359,296
│    │    └─BatchNorm2d: 3-43            [1, 512, 7, 7]            1,024
│    │    └─Sequential: 3-44             [1, 512, 7, 7]            132,096
│    │    └─ReLU: 3-45                   [1, 512, 7, 7]            --
│    └─BasicBlock: 2-8                   [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-46                 [1, 512, 7, 7]            2,359,296
│    │    └─BatchNorm2d: 3-47            [1, 512, 7, 7]            1,024
│    │    └─ReLU: 3-48                   [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-49                 [1, 512, 7, 7]            2,359,296
│    │    └─BatchNorm2d: 3-50            [1, 512, 7, 7]            1,024
│    │    └─ReLU: 3-51                   [1, 512, 7, 7]            --
├─AdaptiveAvgPool2d: 1-9                 [1, 512, 1, 1]            --
├─Linear: 1-10                           [1, 1000]                 513,000
==========================================================================================
Total params: 11,689,512
Trainable params: 11,689,512
Non-trainable params: 0
Total mult-adds (G): 1.81
==========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 39.75
Params size (MB): 46.76
Estimated Total Size (MB): 87.11
==========================================================================================
```

## Resources

Libraries

- [projectchrono/chrono: High-performance C++ library for multiphysics and multibody dynamics simulations](https://github.com/projectchrono/chrono)
- [pytorch/cpuinfo: CPU INFOrmation library (x86/x86-64/ARM/ARM64, Linux/Windows/Android/macOS/iOS)](https://github.com/pytorch/cpuinfo)
- [mraggi/tqdm-cpp: Easily display progress in C++17. Inspired by python's awesome tqdm library.](https://github.com/mraggi/tqdm-cpp)

Control single thread

- [Use single thread on Intel CPU - C++ - PyTorch Forums](https://discuss.pytorch.org/t/use-single-thread-on-intel-cpu/34233)
- [python - Pytorch C++ (Libtroch), using inter-op parallelism - Stack Overflow](https://stackoverflow.com/questions/68361267/pytorch-c-libtroch-using-inter-op-parallelism)

Example

- [examples/cpp/mnist at cpp · goldsborough/examples](https://github.com/goldsborough/examples/tree/cpp/cpp/mnist)

---

Trouble Shooting for Thread Settings

- [torch::set_num_threads does not work · Issue #19213 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/19213)
- [Pytorch seems very slow on CPU - nlp - PyTorch Forums](https://discuss.pytorch.org/t/pytorch-seems-very-slow-on-cpu/94360/7)

Related Environment Variables

- `MKL_NUM_THREADS`
- `OMP_NUM_THREADS`

Related Function

- `at::set_num_threads` (`#include “ATen/Parallel.h”`)
- `omp_get_max_threads`
- `torch::get_num_threads()`, `torch::set_num_threads()`, `torch::set_num_interop_threads()` (`#include <torch/torch.h>`)
