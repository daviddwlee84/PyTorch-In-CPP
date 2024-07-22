
```bash
. /opt/intel/oneapi/setvars.sh
mkdir build
cd build
cmake ..
make cpu-gpu
./load_and_run ../state_dict.json

...

x: [1, 1, 1, 1, 1]
(size: 5)
h: [1, 1, 1, 1, 1]
(size: 5)
Batch Norm: [0.999995, 0.999995, 0.999995, 0.999995, 0.999995]
(size: 5)
GRU x: [0.999995, 0.999995, 0.999995, 0.999995, 0.999995]
(size: 5)
GRU h: [-0.0618359, -0.315901, 0.274827, -0.0130877, 0.150302, 0.459158, 0.0839139, -0.274271, 0.225491, -0.197983]
(size: 10)
Linear x: [-0.302049]
(size: 1)
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
x: tensor([[[1., 1., 1., 1., 1.]]])
h: tensor([[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],

        [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]])
Batch Norm: tensor([[[1.0000, 1.0000, 1.0000, 1.0000, 1.0000]]],
       grad_fn=<UnsqueezeBackward0>)
GRU x: tensor([[[ 0.3525,  0.7142,  0.2723,  0.4367, -0.0280,  0.9293,  0.7255,
           0.1220,  0.9522,  0.4537]]], grad_fn=<TransposeBackward1>)
GRU h: tensor([[[ 0.4954,  0.2043,  0.4286,  0.6808,  0.3849,  0.7889,  0.6873,
           0.6005,  0.8363,  0.3157]],

        [[ 0.3525,  0.7142,  0.2723,  0.4367, -0.0280,  0.9293,  0.7255,
           0.1220,  0.9522,  0.4537]]], grad_fn=<StackBackward0>)
Linear x: tensor([[1.1921]], grad_fn=<AddmmBackward0>)
tensor([[1.1921]], grad_fn=<AddmmBackward0>)
```

=> BUG: Currently Python single step result is not aligned with C++ version (and also not aligned with from scratch Python version)

- [gru-from-scratch/gru.py at main · gursi26/gru-from-scratch](https://github.com/gursi26/gru-from-scratch/blob/main/gru.py)
- [d2l-pytorch/Ch10_Recurrent_Neural_Networks/Gated_Recurrent_Units.ipynb at master · dsgiitr/d2l-pytorch](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch10_Recurrent_Neural_Networks/Gated_Recurrent_Units.ipynb)
- [d2l-pytorch/Ch10_Recurrent_Neural_Networks/Long_Short_Term_Memory.ipynb at master · dsgiitr/d2l-pytorch](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch10_Recurrent_Neural_Networks/Long_Short_Term_Memory.ipynb)

[Adjusting Output - Google Logging Library](https://google.github.io/glog/stable/flags/#using-command-line-parameters-and-environment-variables)

```bash
GLOG_logtostderr=1 ./build/load_and_run state_dict.json 1
```
