#include <iostream>
#include <vector>
#include <cstdlib>
#include "SingleStepLSTMRegressionMKL.h"
#include <mkl.h>
#include <omp.h>
#include <chrono>

int main(int argc, const char *argv[])
{
    int64_t feature_dim = 148; // Example feature dimension
    int64_t hidden_size = 128; // Example hidden size
    int64_t num_layers = 3;   // Example number of layers
    SingleStepLSTMRegressionMKL model(feature_dim, hidden_size, num_layers);

    int iterations = 100; // Default value

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

    // Print the number of threads being used
    int mkl_threads = mkl_get_max_threads();
    std::cout << "MKL is using " << mkl_threads << " threads." << std::endl;

    // Print the number of CPUs being used (OpenMP)
    int omp_threads = omp_get_max_threads();
    std::cout << "OpenMP is using " << omp_threads << " threads." << std::endl;

    // Example input
    std::vector<float> input(feature_dim, 1.0); // Batch size 1, sequence length 1, feature dimension
    std::vector<float> h(hidden_size, 0.0);     // Initial hidden state

    auto output = model.forward(input, h);

    std::cout << "Output: " << std::get<0>(output)[0] << std::endl; // Print output tensor

    if (argc <= 1)
    {
        return 0;
    }

    std::cout << "Will use average of " << iterations << " iterations." << std::endl;
    std::cout << "Benchmarking Pure oneMKL C++ model..." << std::endl;

    // Measure inference time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        auto output = model.forward(input, h);
        h = std::get<1>(output);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Inference time: " << diff.count() / iterations << " seconds" << std::endl;

    return 0;
}
