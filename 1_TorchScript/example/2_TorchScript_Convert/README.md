
- [Loading a TorchScript Model in C++ — PyTorch Tutorials 2.3.0+cu121 documentation](https://pytorch.org/tutorials/advanced/cpp_export.html)

1. `./build.sh`
2. Generate `torch.jit.ScriptModule` by `python convert_by_tracing.py` or by `python convert_by_annotation.py`
3. `./example-app ../traced_resnet_model.pt` or `./example-app ../script_resnet_model.pt`
