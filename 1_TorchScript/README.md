# Torch Script

## Setup Environment

- [Installing C++ Distributions of PyTorch — PyTorch main documentation](https://pytorch.org/cppdocs/installing.html) - `LibTorch`

```bash
# About 162MB
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
# About 717MB, will get the `libtorch`
unzip libtorch-shared-with-deps-latest.zip
```

> If PyTorch was installed via `conda` or `pip`, `CMAKE_PREFIX_PATH` can be queried using `torch.utils.cmake_prefix_path` variable.

```python
import torch
print(torch.utils.cmake_prefix_path)
```

## Resources

- [Loading a TorchScript Model in C++ — PyTorch Tutorials 2.3.0+cu121 documentation](https://pytorch.org/tutorials/advanced/cpp_export.html)
- [TorchScript — PyTorch 2.3 documentation](https://pytorch.org/docs/stable/jit.html)
- [LibTorch Project - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=YiZhang.LibTorch001)
- [soumik12345/libtorch-examples: Basic Deep Learning examples using LibTorch C++ frontend](https://github.com/soumik12345/libtorch-examples)
