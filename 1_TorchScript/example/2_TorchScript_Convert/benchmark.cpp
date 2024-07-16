// set threads and Tensor
#include <torch/torch.h>
// torch::jit (include a part of torch/torch.h)
#include <torch/script.h>
#include <iostream>
#include <chrono>

void benchmark(const std::string &model_path, torch::Tensor input, int iterations)
{
    // Load the TorchScript model
    auto module = torch::jit::load(model_path);

    // Warm-up run
    module.forward({input});

    // Measure inference time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        module.forward({input});
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

    // Example input tensor
    torch::Tensor input = torch::rand({1, 3, 224, 224});

    // Set the number of threads to 1 for single-threaded execution
    torch::set_num_threads(1);
    // torch::set_num_interop_threads(1);

    // Benchmark traced model
    std::cout << "Benchmarking traced model..." << std::endl;
    benchmark("../traced_resnet_model.pt", input, iterations);

    // Benchmark scripted model
    std::cout << "Benchmarking scripted model..." << std::endl;
    benchmark("../scripted_resnet_model.pt", input, iterations);

    return 0;
}
