#include <torch/torch.h>
#include <iostream>
#include <fstream>

void print_tensor(const torch::Tensor &tensor, const std::string &name)
// void print_tensor(const std::vector<torch::Tensor> &tensor, const std::string &name)
{
    std::cout << name << ": " << tensor << std::endl;
}

// https://github.com/pytorch/pytorch/issues/20356#issuecomment-1061667333
std::vector<char> get_the_bytes(std::string filename)
{
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

int main()
{
    // Load the saved tensors from files (BUG: failed)
    // torch::Tensor example_input;
    // torch::Tensor example_lengths;
    // std::vector<torch::Tensor> example_input;
    // std::vector<torch::Tensor> example_lengths;
    // torch::Tensor example_input = torch::ones({5, 10, 10});
    // torch::Tensor example_lengths = torch::ones({5});
    // torch::load(example_input, "example_input.pt");
    // torch::load(example_lengths, "example_lengths.pt");

    // Open the files and load the tensors
    /* BUG: same error
    std::ifstream input_file("example_input.pt", std::ios::binary);
    std::ifstream lengths_file("example_lengths.pt", std::ios::binary);
    torch::load(example_input, input_file);
    torch::load(example_lengths, lengths_file);
    */

    std::vector<char> input_fp = get_the_bytes("example_input.pt");
    torch::IValue raw_example_input = torch::pickle_load(input_fp);
    torch::Tensor example_input = raw_example_input.toTensor();
    std::vector<char> lengths_fp = get_the_bytes("example_lengths.pt");
    torch::IValue raw_example_lengths = torch::pickle_load(lengths_fp);
    torch::Tensor example_lengths = raw_example_lengths.toTensor();

    print_tensor(example_input, "Input");
    print_tensor(example_lengths, "Lengths");

    return 0;
}

/*
BUG => Solved by dump as byte and load as byte

terminate called after throwing an instance of 'c10::Error'
  what():  PytorchStreamReader failed locating file constants.pkl: file not found
Exception raised from valid at ../caffe2/serialize/inline_container.cc:236 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7ff1c223da36 in /mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/lib/libc10.so)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::string const&) + 0x64 (0x7ff1c21eb6aa in /mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/lib/libc10.so)
frame #2: caffe2::serialize::PyTorchStreamReader::valid(char const*, char const*) + 0x8e (0x7ff1b25ca80e in /mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/lib/libtorch_cpu.so)
frame #3: caffe2::serialize::PyTorchStreamReader::getRecordID(std::string const&) + 0x46 (0x7ff1b25cb706 in /mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/lib/libtorch_cpu.so)
frame #4: caffe2::serialize::PyTorchStreamReader::getRecord(std::string const&) + 0x5c (0x7ff1b25cb7fc in /mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/lib/libtorch_cpu.so)
frame #5: torch::jit::readArchiveAndTensors(std::string const&, std::string const&, std::string const&, std::optional<std::function<c10::StrongTypePtr (c10::QualifiedName const&)> >, std::optional<std::function<c10::intrusive_ptr<c10::ivalue::Object, c10::detail::intrusive_target_default_null_type<c10::ivalue::Object> > (c10::StrongTypePtr const&, c10::IValue)> >, std::optional<c10::Device>, caffe2::serialize::PyTorchStreamReader&, c10::Type::SingletonOrSharedTypePtr<c10::Type> (*)(std::string const&), std::shared_ptr<torch::jit::DeserializationStorageContext>) + 0xa5 (0x7ff1b38ad735 in /mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/lib/libtorch_cpu.so)
frame #6: <unknown function> + 0x5893a26 (0x7ff1b38a3a26 in /mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/lib/libtorch_cpu.so)
frame #7: <unknown function> + 0x589636c (0x7ff1b38a636c in /mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/lib/libtorch_cpu.so)
frame #8: torch::jit::import_ir_module(std::shared_ptr<torch::jit::CompilationUnit>, std::string const&, std::optional<c10::Device>, std::unordered_map<std::string, std::string, std::hash<std::string>, std::equal_to<std::string>, std::allocator<std::pair<std::string const, std::string> > >&, bool, bool) + 0x3d6 (0x7ff1b38ab5c6 in /mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/lib/libtorch_cpu.so)
frame #9: torch::jit::import_ir_module(std::shared_ptr<torch::jit::CompilationUnit>, std::string const&, std::optional<c10::Device>, bool) + 0x7f (0x7ff1b38ab7bf in /mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/lib/libtorch_cpu.so)
frame #10: torch::jit::load(std::string const&, std::optional<c10::Device>, bool) + 0xac (0x7ff1b38ab89c in /mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/lib/libtorch_cpu.so)
frame #11: torch::serialize::InputArchive::load_from(std::string const&, std::optional<c10::Device>) + 0x28 (0x7ff1b3fa8a88 in /mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/lib/libtorch_cpu.so)
frame #12: <unknown function> + 0xca96 (0x55c372daaa96 in ./build/load-tensor)
frame #13: <unknown function> + 0x4ab9 (0x55c372da2ab9 in ./build/load-tensor)
frame #14: <unknown function> + 0x29d90 (0x7ff1adbb2d90 in /lib/x86_64-linux-gnu/libc.so.6)
frame #15: __libc_start_main + 0x80 (0x7ff1adbb2e40 in /lib/x86_64-linux-gnu/libc.so.6)
frame #16: <unknown function> + 0x4885 (0x55c372da2885 in ./build/load-tensor)

[1]    507561 IOT instruction (core dumped)  ./build/load-tensor
*/
