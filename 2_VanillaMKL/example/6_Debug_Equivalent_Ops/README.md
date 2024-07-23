
```bash
./build.sh

. /opt/intel/oneapi/setvars.sh
./build/equivalent_ops
```

There is serious issue that invoking `torch::matmul` will cause MKL error `Intel oneMKL ERROR: Parameter 6 was incorrect on entry to SGEMV .`

- [Build without MKL is not possible when MKL is installed · Issue #32407 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/32407)

Maybe PyTorch use its own MKL version which is conflict with our manual one.
