#include <iostream>
#include <vector>
#include <string>
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

/*
void load_to_mkl(const nlohmann::json &weights, int num_layers = 2)
{
    std::vector<float> batch_norm_mean;
    std::vector<float> batch_norm_var;
    std::vector<float> batch_norm_gamma;
    std::vector<float> batch_norm_beta;
    std::vector<std::vector<float>> lstm_weights;
    std::vector<std::vector<float>> lstm_biases;
    std::vector<float> linear_weights;
    std::vector<float> linear_biases;

    // Load batch norm weights
    batch_norm_gamma = weights["batch_norm.weight"].get<std::vector<float>>();
    batch_norm_beta = weights["batch_norm.bias"].get<std::vector<float>>();
    batch_norm_mean = weights["batch_norm.running_mean"].get<std::vector<float>>();
    batch_norm_var = weights["batch_norm.running_var"].get<std::vector<float>>();

    // Load LSTM weights and biases
    for (int layer = 0; layer < num_layers; ++layer)
    {
        lstm_weights[layer].insert(lstm_weights[layer].end(), weights["lstm.weight_ih_l" + std::to_string(layer)].get<std::vector<float>>().begin(), weights["lstm.weight_ih_l" + std::to_string(layer)].get<std::vector<float>>().end());
        lstm_weights[layer].insert(lstm_weights[layer].end(), weights["lstm.weight_hh_l" + std::to_string(layer)].get<std::vector<float>>().begin(), weights["lstm.weight_hh_l" + std::to_string(layer)].get<std::vector<float>>().end());

        lstm_biases[layer].insert(lstm_biases[layer].end(), weights["lstm.bias_ih_l" + std::to_string(layer)].get<std::vector<float>>().begin(), weights["lstm.bias_ih_l" + std::to_string(layer)].get<std::vector<float>>().end());
        lstm_biases[layer].insert(lstm_biases[layer].end(), weights["lstm.bias_hh_l" + std::to_string(layer)].get<std::vector<float>>().begin(), weights["lstm.bias_hh_l" + std::to_string(layer)].get<std::vector<float>>().end());
    }

    // Load linear weights and biases
    linear_weights = weights["linear.weight"].get<std::vector<float>>();
    linear_biases = weights["linear.bias"].get<std::vector<float>>();

    debug_vector(batch_norm_gamma, "batch_norm.weight");
    debug_vector(batch_norm_beta, "batch_norm.bias");
    debug_vector(batch_norm_mean, "batch_norm.running_mean");
    debug_vector(batch_norm_var, "batch_norm.running_var");
    debug_matrix(lstm_weights, "lstm.weight (lstm.weight_ih_l, lstm.weight_hh_l)");
    debug_matrix(lstm_biases, "lstm.biases (lstm.bias_ih_l, lstm.bias_hh_l)");
    debug_vector(linear_weights, "linear.weight");
    debug_vector(linear_biases, "linear.bias");
}
*/

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

    // BUG
    // terminate called after throwing an instance of 'nlohmann::json_abi_v3_11_3::detail::type_error'
    //     what():  [json.exception.type_error.302] type must be number, but is array
    // load_to_mkl(state_dict);

    // Load batch_norm weights (Vector)
    std::vector<float> batch_norm_weight = load_vector(state_dict["batch_norm.weight"]);
    debug_vector(batch_norm_weight, "batch_norm.weight");

    // Load LSTM weights into oneMKL matrices (Matrix)
    std::vector<std::vector<float>> lstm_weight_ih_l0 = load_matrix(state_dict["lstm.weight_ih_l0"]);
    debug_matrix(lstm_weight_ih_l0, "lstm.weight_ih_l0");
    return 0;
}
