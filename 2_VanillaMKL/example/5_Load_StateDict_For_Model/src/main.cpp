#include <iostream>
#include <vector>
#include <cstdlib>
#include "SingleStepLSTMRegressionMKL.h"
#include "../include/weights_loader.h"
#include <mkl.h>
#include <omp.h>
#include <chrono>
#include <nlohmann/json.hpp> // Include the JSON library header

using json = nlohmann::json;

int main(int argc, const char *argv[])
{
    if (argc < 2 or argc > 3)
    {
        std::cerr << "usage: " << argv[0] << " <path-to-state_dict_json> [iterations=100]\n";
        return -1;
    }

    json state_dict;
    try
    {
        // Load weights from JSON file
        std::string file_path = argv[1];
        state_dict = load_weights(file_path);
    }
    catch (...)
    {
        // https://stackoverflow.com/questions/315948/c-catching-all-exceptions
        std::cerr << "error loading the json\n";
        std::exception_ptr p = std::current_exception();
        std::clog << (p ? p.__cxa_exception_type()->name() : "null") << std::endl;
        return -1;
    }

    int64_t feature_dim = state_dict["batch_norm.weight"].size(); // Get feature dimension from state_dict
    int64_t hidden_size = 10;                                     // Example hidden size
    int64_t num_layers = 2;                                       // Example number of layers
    SingleStepLSTMRegressionMKL model(feature_dim, hidden_size, num_layers);

    int iterations = 100; // Default value

    if (argc == 3)
    {
        iterations = std::atoi(argv[2]);
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

    // Initialize model weights from JSON state_dict
    model.load_state_dict(state_dict.dump());

    auto output = model.forward(input, h);

    std::cout << "Output: " << std::get<0>(output)[0] << std::endl; // Print output tensor

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
