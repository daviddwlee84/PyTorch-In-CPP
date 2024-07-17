// set threads and Tensor
#include <torch/torch.h>
// torch::jit (include a part of torch/torch.h)
#include <torch/script.h>
#include <iostream>
#include <chrono>
// C++11
#include <tuple>

#define FEATURE_DIM 148
#define NUM_LAYERS 3
#define HIDDEN_SIZE 128

void benchmark_single_input(const std::string &model_path, int iterations)
{
    // Load the TorchScript model
    auto module = torch::jit::load(model_path);

    int batch_size = 1;
    int sequence_length = 1;

    // Prepare input tensor (example)
    torch::Tensor input_tensor = torch::randn({batch_size, sequence_length, FEATURE_DIM}); // Batch size 1, sequence length 1, feature dimension 10
    torch::Tensor hidden_state = torch::zeros({NUM_LAYERS, sequence_length, HIDDEN_SIZE}); // Example initialization of hidden state

    // Move tensors to device if GPU is available
    // input_tensor = input_tensor.to(torch::kCUDA);
    // hidden_state = hidden_state.to(torch::kCUDA);
    // module.to(torch::kCUDA);

    // Prepare inputs as a tuple (example input, example hidden state)
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    inputs.push_back(hidden_state);

    // Warm-up run
    // C++17 (failed)
    // auto [output_tensor, hidden_state] = module.forward(inputs);
    // C++11 (failed)
    // torch::Tensor output_tensor;
    // std::tie(output_tensor, hidden_state) = module.forward(inputs);
    // failed
    // std::tuple<torch::Tensor, torch::Tensor> output = module.forward(inputs).toTuple();
    // torch::Tensor output_tensor = std::get<0>(output);
    // hidden_state = std::get<1>(output);
    auto output = module.forward(inputs).toTuple();
    torch::Tensor output_tensor = output->elements()[0].toTensor();
    hidden_state = output->elements()[1].toTensor();
    // For debug
    // std::cout << output << "\n"
    //           << hidden_state << std::endl;

    // Measure inference time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        auto output = module.forward(inputs).toTuple();
        torch::Tensor output_tensor = output->elements()[0].toTensor();
        hidden_state = output->elements()[1].toTensor();

        // Update inputs with the new hidden state
        inputs[1] = hidden_state;
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
    std::cout << "Benchmarking scripted model..." << std::endl;
    benchmark_single_input("scripted_single_step_model.pt", iterations);

    return 0;
}
