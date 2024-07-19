#include "../include/SingleStepLSTMRegressionMKL.h"
#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
#include "../include/weights_loader.h"

using json = nlohmann::json;

SingleStepLSTMRegressionMKL::SingleStepLSTMRegressionMKL(int64_t feature_dim, int64_t hidden_size, int64_t num_layers)
    : feature_dim(feature_dim), hidden_size(hidden_size), num_layers(num_layers)
{
    batch_norm_mean.resize(feature_dim, 0.0f);
    batch_norm_var.resize(feature_dim, 1.0f);
    batch_norm_gamma.resize(feature_dim, 1.0f);
    batch_norm_beta.resize(feature_dim, 0.0f);

    lstm_weights.resize(num_layers, std::vector<float>(3 * hidden_size * feature_dim, 0.1f));
    lstm_biases.resize(num_layers, std::vector<float>(3 * hidden_size, 0.1f));
    linear_weights.resize(hidden_size, 0.1f);
    linear_biases.resize(1, 0.1f);
}

void SingleStepLSTMRegressionMKL::batch_norm(std::vector<float> &x)
{
    int64_t N = x.size() / feature_dim;
    for (int64_t i = 0; i < N; ++i)
    {
        for (int64_t j = 0; j < feature_dim; ++j)
        {
            int64_t idx = i * feature_dim + j;
            x[idx] = batch_norm_gamma[j] * (x[idx] - batch_norm_mean[j]) / sqrt(batch_norm_var[j] + 1e-5) + batch_norm_beta[j];
        }
    }
}

void SingleStepLSTMRegressionMKL::gru_cell(const std::vector<float> &x, const std::vector<float> &h, std::vector<float> &new_h, int layer)
{
    std::vector<float> z(hidden_size);
    std::vector<float> r(hidden_size);
    std::vector<float> n(hidden_size);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, hidden_size, 1, feature_dim, 1.0, lstm_weights[layer].data(), feature_dim, x.data(), 1, 0.0, z.data(), 1);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, hidden_size, 1, feature_dim, 1.0, lstm_weights[layer].data() + hidden_size * feature_dim, feature_dim, x.data(), 1, 0.0, r.data(), 1);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, hidden_size, 1, feature_dim, 1.0, lstm_weights[layer].data() + 2 * hidden_size * feature_dim, feature_dim, x.data(), 1, 0.0, n.data(), 1);

    for (int64_t i = 0; i < hidden_size; ++i)
    {
        z[i] += lstm_biases[layer][i];
        r[i] += lstm_biases[layer][i + hidden_size];
        n[i] += lstm_biases[layer][i + 2 * hidden_size];

        z[i] = 1.0 / (1.0 + exp(-z[i]));
        r[i] = 1.0 / (1.0 + exp(-r[i]));
        n[i] = tanh(n[i] + r[i] * h[i]);

        new_h[i] = (1 - z[i]) * n[i] + z[i] * h[i];
    }
}

std::tuple<std::vector<float>, std::vector<float>> SingleStepLSTMRegressionMKL::forward(const std::vector<float> &x, const std::vector<float> &h)
{
    std::vector<float> x_copy = x;
    batch_norm(x_copy);

    std::vector<float> new_h(hidden_size);
    std::vector<float> current_h = h;
    for (int layer = 0; layer < num_layers; ++layer)
    {
        gru_cell(x_copy, current_h, new_h, layer);
        current_h = new_h;
    }

    std::vector<float> output(1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 1, hidden_size, 1.0, linear_weights.data(), hidden_size, new_h.data(), 1, 0.0, output.data(), 1);
    output[0] += linear_biases[0];

    return std::make_tuple(output, new_h);
}

void SingleStepLSTMRegressionMKL::load_state_dict(const std::string &json_str)
{
    json root = json::parse(json_str);
    std::cout << json_str << std::endl;
    std::cout << root << std::endl;

    batch_norm_gamma = load_vector(root["batch_norm.weight"]);
    batch_norm_beta = load_vector(root["batch_norm.bias"]);
    batch_norm_mean = load_vector(root["batch_norm.running_mean"]);
    batch_norm_var = load_vector(root["batch_norm.running_var"]);

    debug_vector(batch_norm_gamma, "batch_norm.weight");

    for (int layer = 0; layer < num_layers; ++layer)
    {
        std::vector<std::vector<float>> ih = load_matrix(root["lstm.weight_ih_l" + std::to_string(layer)]);
        std::vector<std::vector<float>> hh = load_matrix(root["lstm.weight_hh_l" + std::to_string(layer)]);
        debug_matrix(ih, "lstm.weight_ih_l" + std::to_string(layer));
        // lstm_weights[layer].assign(ih.begin(), ih.end());
        // https://cplusplus.com/reference/vector/vector/insert/
        // lstm_weights[layer].insert(lstm_weights[layer].end(), hh.begin(), hh.end());

        std::vector<float> bias_ih = load_vector(root["lstm.bias_ih_l" + std::to_string(layer)]);
        std::vector<float> bias_hh = load_vector(root["lstm.bias_hh_l" + std::to_string(layer)]);
        // lstm_biases[layer].assign(bias_ih.begin(), bias_ih.end());
        // lstm_biases[layer].insert(lstm_biases[layer].end(), bias_hh.begin(), bias_hh.end());
    }

    linear_weights = load_vector(root["linear.weight"][0]);
    linear_biases = load_vector(root["linear.bias"]);
    debug_vector(linear_biases, "linear.bias");
}