#include "../include/SingleStepLSTMRegressionMKL.h"
#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
#include "../include/weights_loader.h"
#include <glog/logging.h>

using json = nlohmann::json;

SingleStepLSTMRegressionMKL::SingleStepLSTMRegressionMKL(int64_t feature_dim, int64_t hidden_size, int64_t num_layers)
    : feature_dim(feature_dim), hidden_size(hidden_size), num_layers(num_layers)
{
    batch_norm_mean.resize(feature_dim, 0.0f);
    batch_norm_var.resize(feature_dim, 1.0f);
    batch_norm_gamma.resize(feature_dim, 1.0f);
    batch_norm_beta.resize(feature_dim, 0.0f);

    lstm_weights_ih.resize(num_layers);
    lstm_weights_hh.resize(num_layers);

    lstm_weights_ih[0].resize(3 * hidden_size * feature_dim, 0.1f); // First layer
    for (int i = 1; i < num_layers; ++i)
    {
        lstm_weights_ih[i].resize(3 * hidden_size * hidden_size, 0.1f); // Subsequent layers
    }

    lstm_weights_hh[0].resize(3 * hidden_size * hidden_size, 0.1f);
    for (int i = 1; i < num_layers; ++i)
    {
        lstm_weights_hh[i].resize(3 * hidden_size * hidden_size, 0.1f);
    }

    lstm_biases_ih.resize(num_layers, std::vector<float>(3 * hidden_size, 0.1f));
    lstm_biases_hh.resize(num_layers, std::vector<float>(3 * hidden_size, 0.1f));
    linear_weights.resize(hidden_size, 0.1f);
    linear_biases.resize(1, 0.1f);
}

// https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#batchnorm1d
void SingleStepLSTMRegressionMKL::batch_norm(std::vector<float> &x)
{
    int64_t N = x.size() / feature_dim;
    const double eps = 1e-5;
    // Batch
    for (int64_t i = 0; i < N; ++i)
    {
        // Feature
        for (int64_t j = 0; j < feature_dim; ++j)
        {
            int64_t idx = i * feature_dim + j;
            x[idx] = batch_norm_gamma[j] * (x[idx] - batch_norm_mean[j]) / std::sqrt(batch_norm_var[j] + eps) + batch_norm_beta[j];
        }
    }
}

// https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html#torch.nn.GRUCell
void SingleStepLSTMRegressionMKL::gru_cell(const std::vector<float> &x, const std::vector<float> &h, std::vector<float> &new_h, int layer)
{
    // r: reset gate; z: update gate; n: new hidden
    std::vector<float> r(hidden_size), z(hidden_size), n(hidden_size);
    std::vector<float> x_gates(3 * hidden_size), h_gates(3 * hidden_size);

    int64_t N = (layer == 0) ? feature_dim : hidden_size;

    // Compute input gates (x_gates)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 3 * hidden_size, N, 1.0, lstm_weights_ih[layer].data(), N, x.data(), 1, 0.0, x_gates.data(), 1);
    // Add biases for input gates
    for (int i = 0; i < 3 * hidden_size; ++i)
    {
        x_gates[i] += lstm_biases_ih[layer][i];
    }
#ifdef DEBUG
    debug_vector(x_gates, "x_gates");
#endif // DEBUG

    // Compute hidden gates (h_gates)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 3 * hidden_size, hidden_size, 1.0, lstm_weights_hh[layer].data(), hidden_size, h.data(), 1, 0.0, h_gates.data(), 1);
    // Add biases for hidden gates
    for (int i = 0; i < 3 * hidden_size; ++i)
    {
        h_gates[i] += lstm_biases_hh[layer][i];
    }
#ifdef DEBUG
    debug_vector(h_gates, "h_gates");
#endif // DEBUG

    // Split gates into reset, update, and new gates
    for (int i = 0; i < hidden_size; ++i)
    {
        r[i] = 1.0f / (1.0f + std::exp(-(x_gates[i] + h_gates[i])));
        z[i] = 1.0f / (1.0f + std::exp(-(x_gates[hidden_size + i] + h_gates[hidden_size + i])));
        n[i] = std::tanh(x_gates[2 * hidden_size + i] + r[i] * h_gates[2 * hidden_size + i]);
    }

#ifdef DEBUG
    debug_vector(r, "Reset Gate");
    debug_vector(z, "Update Gate");
    debug_vector(n, "New Hidden");
#endif // DEBUG

    // Compute the new hidden state
    for (int i = 0; i < hidden_size; ++i)
    {
        new_h[i] = (1.0f - z[i]) * n[i] + z[i] * h[i];
    }

#ifdef DEBUG
    debug_vector(new_h, "Output");
#endif // DEBUG
}

std::tuple<std::vector<float>, std::vector<std::vector<float>>> SingleStepLSTMRegressionMKL::forward(const std::vector<float> &x, const std::vector<std::vector<float>> &h)
{
#ifdef DEBUG
    debug_vector(x, "x");
    debug_flatten_matrix(h, hidden_size, "h");
#endif // DEBUG

    std::vector<float> x_copy = x;
    // BUG: somehow get a little gap between Python result
    batch_norm(x_copy);

#ifdef DEBUG
    debug_vector(x_copy, "Batch Norm");
#endif // DEBUG

    std::vector<std::vector<float>> new_h(num_layers, std::vector<float>(hidden_size));
    std::vector<float> layer_input = x_copy;
    for (int layer = 0; layer < num_layers; ++layer)
    {
        std::vector<float> layer_new_h(hidden_size);
        std::vector<float> current_h = h[layer];
        gru_cell(layer_input, current_h, layer_new_h, layer);
        new_h[layer] = layer_new_h;
        layer_input = layer_new_h;
    }

#ifdef DEBUG
    debug_vector(layer_input, "GRU x");
    debug_flatten_matrix(new_h, hidden_size, "GRU h");
#endif // DEBUG

    std::vector<float> output(1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 1, hidden_size, 1.0, linear_weights.data(), hidden_size, layer_input.data(), 1, 0.0, output.data(), 1);
    output[0] += linear_biases[0];

#ifdef DEBUG
    debug_vector(output, "Linear x");
#endif // DEBUG

    return std::make_tuple(output, new_h);
}

void SingleStepLSTMRegressionMKL::load_state_dict(const std::string &json_str)
{
    json root = json::parse(json_str);
#ifdef DEBUG
    std::cout << json_str << std::endl;
    std::cout << root << std::endl;
#endif // DEBUG

    batch_norm_gamma = load_vector(root["batch_norm.weight"]);
    batch_norm_beta = load_vector(root["batch_norm.bias"]);
    batch_norm_mean = load_vector(root["batch_norm.running_mean"]);
    batch_norm_var = load_vector(root["batch_norm.running_var"]);
    // https://en.cppreference.com/w/cpp/error/assert
    assert(batch_norm_gamma.size() == feature_dim);
    assert(batch_norm_beta.size() == feature_dim);
    assert(batch_norm_mean.size() == feature_dim);
    assert(batch_norm_var.size() == feature_dim);

#ifdef DEBUG
    debug_vector(batch_norm_gamma, "batch_norm.weight");
#endif // DEBUG

    for (int layer = 0; layer < num_layers; ++layer)
    {
        std::vector<float> ih = load_matrix_to_vector(root["lstm.weight_ih_l" + std::to_string(layer)]);
        std::vector<float> hh = load_matrix_to_vector(root["lstm.weight_hh_l" + std::to_string(layer)]);
        // https://google.github.io/glog/stable/logging/#runtime-checks
        // assert(ih.size() == (3 * hidden_size * ((layer == 0) ? feature_dim : hidden_size)));
        // assert(hh.size() == (3 * hidden_size * hidden_size));
        CHECK(ih.size() == (3 * hidden_size * ((layer == 0) ? feature_dim : hidden_size))) << "Size lstm.weight_ih_l" << std::to_string(layer) << " not matched.";
        CHECK(hh.size() == (3 * hidden_size * hidden_size)) << "Size lstm.weight_hh_l" << std::to_string(layer) << " not matched.";
#ifdef DEBUG
        debug_flatten_matrix(ih, (layer == 0) ? feature_dim : hidden_size, "Flattened Matrix lstm.weight_ih_l" + std::to_string(layer));
        debug_flatten_matrix(hh, hidden_size, "Flattened Matrix lstm.weight_hh_l" + std::to_string(layer));
#endif // DEBUG
       // https://stackoverflow.com/questions/2119177/stl-vector-assign-vs-insert
        lstm_weights_ih[layer].assign(ih.begin(), ih.end());
        // https://cplusplus.com/reference/vector/vector/insert/
        // https://www.digitalocean.com/community/tutorials/vector-insert-in-c-plus-plus
        lstm_weights_ih[layer].insert(lstm_weights_ih[layer].end(), ih.begin(), ih.end());

        lstm_weights_hh[layer].assign(hh.begin(), hh.end());
        lstm_weights_hh[layer].insert(lstm_weights_hh[layer].end(), hh.begin(), hh.end());

        std::vector<float> bias_ih = load_vector(root["lstm.bias_ih_l" + std::to_string(layer)]);
        std::vector<float> bias_hh = load_vector(root["lstm.bias_hh_l" + std::to_string(layer)]);
        // assert(bias_ih.size() == (3 * hidden_size));
        // assert(bias_hh.size() == (3 * hidden_size));
        CHECK(bias_ih.size() == (3 * hidden_size)) << "Size lstm.bias_ih_l" << std::to_string(layer) << " not matched.";
        CHECK(bias_hh.size() == (3 * hidden_size)) << "Size lstm.bias_hh_l" << std::to_string(layer) << " not matched.";
        lstm_biases_ih[layer].assign(bias_ih.begin(), bias_ih.end());
        lstm_biases_ih[layer].insert(lstm_biases_ih[layer].end(), bias_ih.begin(), bias_ih.end());
        lstm_biases_hh[layer].assign(bias_hh.begin(), bias_hh.end());
        lstm_biases_hh[layer].insert(lstm_biases_hh[layer].end(), bias_hh.begin(), bias_hh.end());
    }

    linear_weights = load_vector(root["linear.weight"][0]);
    // assert(linear_weights.size() == hidden_size);
    CHECK(linear_weights.size() == hidden_size) << "Size linear.weight not matched.";
    linear_biases = load_vector(root["linear.bias"]);
    // assert(linear_biases.size() == 1);
    CHECK(linear_biases.size() == 1) << "Size linear.bias not matched.";
#ifdef DEBUG
    debug_vector(linear_biases, "linear.bias");
#endif // DEBUG
}