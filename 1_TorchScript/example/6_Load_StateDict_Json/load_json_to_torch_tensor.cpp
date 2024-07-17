#include <iostream>
#include <vector>
#include <string>
#include <torch/torch.h>
#include "weights_loader.h"

// Function to debug and print std::vector<float>
void debug_vector(const std::vector<float> &vec, const std::string &name = "")
{
    if (!name.empty())
    {
        std::cout << name << ": ";
    }
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        std::cout << vec[i];
        if (i < vec.size() - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

// Function to debug and print std::vector<std::vector<float>>
void debug_matrix(const std::vector<std::vector<float>> &mat, const std::string &name = "")
{
    if (!name.empty())
    {
        std::cout << name << ": ";
    }
    std::cout << "[\n";
    for (size_t i = 0; i < mat.size(); ++i)
    {
        std::cout << "  [";
        for (size_t j = 0; j < mat[i].size(); ++j)
        {
            std::cout << mat[i][j];
            if (j < mat[i].size() - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i < mat.size() - 1)
        {
            std::cout << ",\n";
        }
    }
    std::cout << "\n]" << std::endl;
}

// Function to load a vector from JSON
std::vector<float> load_vector(const json &j)
{
    return j.get<std::vector<float>>();
}

// Function to load a matrix from JSON
std::vector<std::vector<float>> load_matrix(const json &j)
{
    return j.get<std::vector<std::vector<float>>>();
}

int main(int argc, const char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "usage: load_json <path-to-state_dict_json>\n";
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

    std::cout << state_dict << std::endl;

    // Load batch_norm weights (Vector)
    std::vector<float> batch_norm_weight = load_vector(state_dict["batch_norm.weight"]);
    debug_vector(batch_norm_weight, "batch_norm.weight");

    // Load LSTM weights into oneMKL matrices (Matrix)
    std::vector<std::vector<float>> lstm_weight_ih_l0 = load_matrix(state_dict["lstm.weight_ih_l0"]);
    debug_matrix(lstm_weight_ih_l0, "lstm.weight_ih_l0");

    // Load batch_norm weights and biases into LibTorch tensors
    torch::Tensor batch_norm_weight_tensor = torch::tensor(batch_norm_weight.data());
    std::cout << batch_norm_weight_tensor << std::endl;

    // Load LSTM weights into LibTorch tensors
    torch::Tensor lstm_weight_ih_l0_tensor = torch::tensor(lstm_weight_ih_l0[0].data());
    std::cout << lstm_weight_ih_l0_tensor << std::endl;

    // Try from_blob
    torch::Tensor batch_norm_weight_tensor_from_blob = torch::from_blob(batch_norm_weight.data(), {static_cast<long>(batch_norm_weight.size())}, torch::kFloat32);
    std::cout << batch_norm_weight_tensor_from_blob << std::endl;

    return 0;
}
