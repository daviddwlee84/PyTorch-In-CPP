# Convert `nn.GRU` into TorchScript

## Getting Started

1. `./build.sh`
2. Generate `torch.jit.ScriptModule` by `python convert_by_annotation.py`
4. `./build/benchmark`

## Benchmark

Entire Batch

```
Default Torch Threads: 60
Will use average of 100 iterations.
Benchmarking scripted model...
Inference time: 4.09082 seconds
```
