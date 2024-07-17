

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

```txt
g++ -std=c++17 -I/mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/include -I/mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/include/torch/csrc/api/include -I../../../shared/include -Iinclude load_json_to_torch_tensor.cpp -o load_json_to_tensor -L/mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/lib -ltorch -ltorch_cpu -lc10 -Wl,--no-as-needed -pthread -lgomp -ltorch_cpu -lc10
/usr/bin/ld: /tmp/cc8F5PwP.o: in function `c10::Device::validate()':
load_json_to_torch_tensor.cpp:(.text._ZN3c106Device8validateEv[_ZN3c106Device8validateEv]+0x88): undefined reference to `c10::detail::torchInternalAssertFail(char const*, char const*, unsigned int, char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
/usr/bin/ld: load_json_to_torch_tensor.cpp:(.text._ZN3c106Device8validateEv[_ZN3c106Device8validateEv]+0x125): undefined reference to `c10::detail::torchInternalAssertFail(char const*, char const*, unsigned int, char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
/usr/bin/ld: /tmp/cc8F5PwP.o: in function `c10::intrusive_ptr_target::~intrusive_ptr_target()':
load_json_to_torch_tensor.cpp:(.text._ZN3c1020intrusive_ptr_targetD2Ev[_ZN3c1020intrusive_ptr_targetD5Ev]+0x166): undefined reference to `c10::detail::torchInternalAssertFail(char const*, char const*, unsigned int, char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
/usr/bin/ld: /tmp/cc8F5PwP.o: in function `caffe2::TypeMeta::fromScalarType(c10::ScalarType)':
load_json_to_torch_tensor.cpp:(.text._ZN6caffe28TypeMeta14fromScalarTypeEN3c1010ScalarTypeE[_ZN6caffe28TypeMeta14fromScalarTypeEN3c1010ScalarTypeE]+0x84): undefined reference to `c10::detail::torchInternalAssertFail(char const*, char const*, unsigned int, char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
/usr/bin/ld: /tmp/cc8F5PwP.o: in function `c10::fromIntArrayRefSlow(c10::ArrayRef<long>)':
load_json_to_torch_tensor.cpp:(.text._ZN3c1019fromIntArrayRefSlowENS_8ArrayRefIlEE[_ZN3c1019fromIntArrayRefSlowENS_8ArrayRefIlEE]+0xc0): undefined reference to `c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
/usr/bin/ld: /tmp/cc8F5PwP.o: in function `c10::IValue::isIntrusivePtr() const':
load_json_to_torch_tensor.cpp:(.text._ZNK3c106IValue14isIntrusivePtrEv[_ZNK3c106IValue14isIntrusivePtrEv]+0x84): undefined reference to `c10::detail::torchInternalAssertFail(char const*, char const*, unsigned int, char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
/usr/bin/ld: /tmp/cc8F5PwP.o: in function `c10::IValue::toComplexDouble() const':
load_json_to_torch_tensor.cpp:(.text._ZNK3c106IValue15toComplexDoubleEv[_ZNK3c106IValue15toComplexDoubleEv]+0xa1): undefined reference to `c10::detail::torchInternalAssertFail(char const*, char const*, unsigned int, char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
/usr/bin/ld: /tmp/cc8F5PwP.o: in function `c10::IValue::toSymInt() const &':
load_json_to_torch_tensor.cpp:(.text._ZNKR3c106IValue8toSymIntEv[_ZNKR3c106IValue8toSymIntEv]+0xc8): undefined reference to `c10::detail::torchInternalAssertFail(char const*, char const*, unsigned int, char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
/usr/bin/ld: /tmp/cc8F5PwP.o: in function `c10::IValue::toSymFloat() const &':
load_json_to_torch_tensor.cpp:(.text._ZNKR3c106IValue10toSymFloatEv[_ZNKR3c106IValue10toSymFloatEv]+0xc8): undefined reference to `c10::detail::torchInternalAssertFail(char const*, char const*, unsigned int, char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
/usr/bin/ld: /tmp/cc8F5PwP.o: in function `torch::detail::TensorDataContainer::fill_tensor(at::Tensor&) const':
load_json_to_torch_tensor.cpp:(.text._ZNK5torch6detail19TensorDataContainer11fill_tensorERN2at6TensorE[_ZNK5torch6detail19TensorDataContainer11fill_tensorERN2at6TensorE]+0xbe): undefined reference to `c10::detail::torchInternalAssertFail(char const*, char const*, unsigned int, char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
/usr/bin/ld: /tmp/cc8F5PwP.o:load_json_to_torch_tensor.cpp:(.text._ZNK5torch6detail19TensorDataContainer11fill_tensorERN2at6TensorE[_ZNK5torch6detail19TensorDataContainer11fill_tensorERN2at6TensorE]+0x1fd): more undefined references to `c10::detail::torchInternalAssertFail(char const*, char const*, unsigned int, char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)' follow
collect2: error: ld returned 1 exit status
make: *** [Makefile:29: load_json_to_tensor] Error 1
```
