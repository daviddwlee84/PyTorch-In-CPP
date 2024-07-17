#include <iostream>
#include <vector>
#include <cstdlib>
#include "SingleStepLSTMRegressionMKL.h"
#include <mkl.h>
#include <omp.h>

int main()
{
    int64_t feature_dim = 10; // Example feature dimension
    int64_t hidden_size = 64; // Example hidden size
    int64_t num_layers = 2;   // Example number of layers
    SingleStepLSTMRegressionMKL model(feature_dim, hidden_size, num_layers);

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
    return 0;
}
