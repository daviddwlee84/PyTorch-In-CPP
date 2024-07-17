# Convert a Batch Model into a Single-Step Model with same weights

## Benchmark

Test single tick data single-thread inference time

### Model: GRU 148 dimension features, 128 hidden size, 3 layers

C++

```zsh
(conbond_venv) (research) ➜  4_Batch_GRU_to_Single-Step_GRU git:(main) ✗ ./build/benchmark 100 1
Default Torch Threads: 60
Updated Torch Threads: 1
Will use average of 100 iterations.
Benchmarking scripted model...
Inference time: 0.00103861 seconds
(conbond_venv) (research) ➜  4_Batch_GRU_to_Single-Step_GRU git:(main) ✗ ./build/benchmark 1000 1
Default Torch Threads: 60
Updated Torch Threads: 1
Will use average of 1000 iterations.
Benchmarking scripted model...
Inference time: 0.000897268 seconds
(conbond_venv) (research) ➜  4_Batch_GRU_to_Single-Step_GRU git:(main) ✗ ./build/benchmark 10000 1
Default Torch Threads: 60
Updated Torch Threads: 1
Will use average of 10000 iterations.
Benchmarking scripted model...
Inference time: 0.000882398 seconds
```

Python

```zsh
(conbond_venv) (research) ➜  4_Batch_GRU_to_Single-Step_GRU git:(main) ✗ python test_in_python.py  
intra-op threads: 60
inter-op threads: 120
Average single tick inference time (sec): 0.0013070359490811826
(conbond_venv) (research) ➜  4_Batch_GRU_to_Single-Step_GRU git:(main) ✗ python test_in_python.py 1
intra-op threads: 1
inter-op threads: 1
Average single tick inference time (sec): 0.0011017058747820555
```

### Model: GRU 148 dimension features, 96 hidden size, 2 layers

C++

```zsh
(conbond_venv) (research) ➜  4_Batch_GRU_to_Single-Step_GRU git:(main) ✗ ./build/benchmark 10000 1
Default Torch Threads: 60
Updated Torch Threads: 1
Will use average of 10000 iterations.
Benchmarking scripted model...
Inference time: 0.00061052 seconds
```

Python

```zsh
(conbond_venv) (research) ➜  4_Batch_GRU_to_Single-Step_GRU git:(main) ✗ python test_in_python.py 
intra-op threads: 60
inter-op threads: 120
Average single tick inference time (sec): 0.001085287950700149
(conbond_venv) (research) ➜  4_Batch_GRU_to_Single-Step_GRU git:(main) ✗ python test_in_python.py 1
intra-op threads: 1
inter-op threads: 1
Average single tick inference time (sec): 0.0008055593005847185
```


## Model

### Small Model

Batch Model

```txt
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
PaddedLSTMRegression                     [4, 5]                    --
├─BatchNorm1d: 1-1                       [20, 10]                  20
├─GRU: 1-2                               [4, 5, 64]                39,552
├─Linear: 1-3                            [4, 5, 1]                 65
==========================================================================================
Total params: 39,637
Trainable params: 39,637
Non-trainable params: 0
Total mult-adds (M): 0.79
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.01
Params size (MB): 0.16
Estimated Total Size (MB): 0.17
==========================================================================================
```

Single Step Model

```txt
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
SingleStepLSTMRegression                 [4, 1]                    --
├─BatchNorm1d: 1-1                       [4, 10]                   20
├─GRU: 1-2                               [4, 1, 64]                39,552
├─Linear: 1-3                            [4, 1]                    65
==========================================================================================
Total params: 39,637
Trainable params: 39,637
Non-trainable params: 0
Total mult-adds (M): 0.16
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.16
Estimated Total Size (MB): 0.16
==========================================================================================
```

### Large Model

Batch Model

```txt
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
PaddedLSTMRegression                     [1, 4000]                 --
├─BatchNorm1d: 1-1                       [4000, 148]               296
├─GRU: 1-2                               [1, 4000, 128]            304,896
├─Linear: 1-3                            [1, 4000, 1]              129
==========================================================================================
Total params: 305,321
Trainable params: 305,321
Non-trainable params: 0
Total mult-adds (G): 1.22
==========================================================================================
Input size (MB): 2.37
Forward/backward pass size (MB): 8.86
Params size (MB): 1.22
Estimated Total Size (MB): 12.45
==========================================================================================
```

Single Step Model

```txt
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
SingleStepLSTMRegression                 [1, 1]                    --
├─BatchNorm1d: 1-1                       [1, 148]                  296
├─GRU: 1-2                               [1, 1, 128]               304,896
├─Linear: 1-3                            [1, 1]                    129
==========================================================================================
Total params: 305,321
Trainable params: 305,321
Non-trainable params: 0
Total mult-adds (M): 0.31
==========================================================================================
Input size (MB): 2.37
Forward/backward pass size (MB): 0.00
Params size (MB): 1.22
Estimated Total Size (MB): 3.59
==========================================================================================
```

---

Trouble Shooting

```txt
-- Added CUDA NVCC flags for: -gencode;arch=compute_75,code=sm_75
CMake Warning at /mnt/NAS/sda/ShareFolder/lidawei/ConvertibleBond_LimitOrderBook/conbond_venv/lib/python3.8/site-packages/torch/share/cmake/Torch/TorchConfig.cmake:22 (message):
  static library kineto_LIBRARY-NOTFOUND not found.
Call Stack (most recent call first):
  /mnt/NAS/sda/ShareFolder/lidawei/ConvertibleBond_LimitOrderBook/conbond_venv/lib/python3.8/site-packages/torch/share/cmake/Torch/TorchConfig.cmake:127 (append_torchlib_if_found)
  CMakeLists.txt:4 (find_package)

/usr/bin/ld: cannot find -lLIBNVTOOLSEXT-NOTFOUND: No such file or directory
```

---

- https://chatgpt.com/share/559b6998-72f8-4803-98c9-c7cebcc49848
