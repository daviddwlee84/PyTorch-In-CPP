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

    if (argc > 2)
    {
        std::cerr << "Usage: " << argv[0] << " [iterations]" << std::endl;
        return 1;
    }

    if (argc == 2)
    {
        iterations = std::atoi(argv[1]);
        if (iterations <= 0)
        {
            std::cerr << "Iterations must be a positive integer." << std::endl;
            return 1;
        }
    }

    std::cout << "Will use average of " << iterations << " iterations." << std::endl;

    // Example input tensor
    torch::Tensor input = torch::rand({1, 3, 224, 224});

    // Benchmark traced model
    std::cout << "Benchmarking traced model..." << std::endl;
    benchmark("../traced_resnet_model.pt", input, iterations);

    // Benchmark scripted model
    std::cout << "Benchmarking scripted model..." << std::endl;
    benchmark("../scripted_resnet_model.pt", input, iterations);

    return 0;
}
