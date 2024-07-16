// set threads and Tensor
#include <torch/torch.h>
// torch::jit (include a part of torch/torch.h)
#include <torch/script.h>
#include <iostream>
#include <chrono>

void benchmark(const std::string &model_path, torch::Tensor input, torch::Tensor lengths, int iterations)
{
    // Load the TorchScript model
    auto module = torch::jit::load(model_path);

    // Warm-up run
    module.forward({input, lengths});

    // Measure inference time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        module.forward({input, lengths});
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Inference time: " << diff.count() / iterations << " seconds" << std::endl;
}

std::vector<char> get_the_bytes(std::string filename)
{
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
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

    // Example input tensor
    std::vector<char> input_fp = get_the_bytes("example_input.pt");
    torch::IValue raw_example_input = torch::pickle_load(input_fp);
    torch::Tensor example_input = raw_example_input.toTensor();
    // std::cout << "Example input:\n" << example_input << std::endl;
    std::vector<char> lengths_fp = get_the_bytes("example_lengths.pt");
    torch::IValue raw_example_lengths = torch::pickle_load(lengths_fp);
    torch::Tensor example_lengths = raw_example_lengths.toTensor();
    // std::cout << "Example lengths:\n" << example_lengths << std::endl;

    // Benchmark scripted model
    std::cout << "Benchmarking scripted model..." << std::endl;
    benchmark("scripted_gru.pt", example_input, example_lengths, iterations);

    return 0;
}
