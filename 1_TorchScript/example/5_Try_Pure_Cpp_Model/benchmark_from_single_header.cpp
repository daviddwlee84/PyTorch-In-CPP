#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "model_header_with_logic.h"

#define FEATURE_DIM 148
#define NUM_LAYERS 3
#define HIDDEN_SIZE 128

void benchmark_single_input(int iterations)
{
    int batch_size = 1;
    int sequence_length = 1;

    SingleStepLSTMRegression model(FEATURE_DIM, HIDDEN_SIZE, NUM_LAYERS);

    // Example input
    auto input = torch::randn({batch_size, sequence_length, FEATURE_DIM});      // Batch size 32, sequence length 1, feature dimension
    auto h = torch::zeros({model->num_layers, batch_size, model->hidden_size}); // Initial hidden state

    // Set the model to evaluation mode
    // terminate called after throwing an instance of 'c10::Error'
    //     what():  Expected more than 1 value per channel when training, got input size [1, 148]
    // Exception raised from batch_norm at /mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/include/torch/csrc/api/include/torch/nn/functional/batchnorm.h:32 (most recent call first):
    // frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7f75644cca36 in /mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/lib/libc10.so)
    // frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::string const&) + 0x64 (0x7f756447a6aa in /mnt/NAS/sda/ShareFolder/lidawei/library/libtorch/lib/libc10.so)
    // frame #2: <unknown function> + 0x11b51 (0x55b46bc64b51 in ./build/benchmark)
    // frame #3: <unknown function> + 0x1c4d3 (0x55b46bc6f4d3 in ./build/benchmark)
    // frame #4: <unknown function> + 0x1789b (0x55b46bc6a89b in ./build/benchmark)
    // frame #5: <unknown function> + 0x12971 (0x55b46bc65971 in ./build/benchmark)
    // frame #6: <unknown function> + 0x91d3 (0x55b46bc5c1d3 in ./build/benchmark)
    // frame #7: <unknown function> + 0x96a8 (0x55b46bc5c6a8 in ./build/benchmark)
    // frame #8: <unknown function> + 0x29d90 (0x7f754fe41d90 in /lib/x86_64-linux-gnu/libc.so.6)
    // frame #9: __libc_start_main + 0x80 (0x7f754fe41e40 in /lib/x86_64-linux-gnu/libc.so.6)
    // frame #10: <unknown function> + 0x8f45 (0x55b46bc5bf45 in ./build/benchmark)
    // [1]    3022424 IOT instruction (core dumped)  ./build/benchmark 10000 1
    model->eval();

    auto output = model->forward(input, h);
    std::cout << std::get<0>(output) << std::endl; // Print output tensor

    // Measure inference time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        auto output = model->forward(input, h);
        h = std::get<1>(output);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Inference time: " << diff.count() / iterations << " seconds" << std::endl;
}

int main(int argc, const char *argv[])
{
    int iterations = 100; // Default value
    int threads = -1;     // Default value for no limit

    if (argc > 3)
    {
        std::cerr << "Usage: " << argv[0] << " [iterations] [threads]" << std::endl;
        return 1;
    }

    if (argc >= 2)
    {
        iterations = std::atoi(argv[1]);
        if (iterations <= 0)
        {
            std::cerr << "Iterations must be a positive integer." << std::endl;
            return 1;
        }
    }

    std::cout << "Default Torch Threads: " << torch::get_num_threads() << std::endl;

    if (argc == 3)
    {
        threads = std::atoi(argv[2]);
        if (threads > 0)
        {
            // Set the number of threads for intra-op parallelism
            torch::set_num_threads(threads);
            // Set the number of threads for inter-op parallelism
            torch::set_num_interop_threads(threads);

            std::cout << "Updated Torch Threads: " << torch::get_num_threads() << std::endl;
        }
    }

    std::cout << "Will use average of " << iterations << " iterations." << std::endl;

    // Benchmark scripted model
    std::cout << "Benchmarking Pure LibTorch C++ model..." << std::endl;
    benchmark_single_input(iterations);

    return 0;
}
