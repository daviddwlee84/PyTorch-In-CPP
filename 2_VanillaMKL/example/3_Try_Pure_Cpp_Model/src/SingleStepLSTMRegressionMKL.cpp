#include "SingleStepLSTMRegressionMKL.h"
#include <cmath>
#include <iostream>

SingleStepLSTMRegressionMKL::SingleStepLSTMRegressionMKL(int64_t feature_dim, int64_t hidden_size, int64_t num_layers)
    : feature_dim(feature_dim), hidden_size(hidden_size), num_layers(num_layers)
{
    batch_norm_mean.resize(feature_dim, 0.0f);
    batch_norm_var.resize(feature_dim, 1.0f);
    batch_norm_gamma.resize(feature_dim, 1.0f);
    batch_norm_beta.resize(feature_dim, 0.0f);

    lstm_weights.resize(3 * hidden_size * feature_dim, 0.1f);
    lstm_biases.resize(3 * hidden_size, 0.1f);
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

void SingleStepLSTMRegressionMKL::gru_cell(const std::vector<float> &x, const std::vector<float> &h, std::vector<float> &new_h)
{
    std::vector<float> z(hidden_size);
    std::vector<float> r(hidden_size);
    std::vector<float> n(hidden_size);

    // Matrix multiplication
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, hidden_size, 1, feature_dim, 1.0, lstm_weights.data(), feature_dim, x.data(), 1, 0.0, z.data(), 1);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, hidden_size, 1, feature_dim, 1.0, lstm_weights.data() + hidden_size * feature_dim, feature_dim, x.data(), 1, 0.0, r.data(), 1);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, hidden_size, 1, feature_dim, 1.0, lstm_weights.data() + 2 * hidden_size * feature_dim, feature_dim, x.data(), 1, 0.0, n.data(), 1);

    for (int64_t i = 0; i < hidden_size; ++i)
    {
        z[i] += lstm_biases[i];
        r[i] += lstm_biases[i + hidden_size];
        n[i] += lstm_biases[i + 2 * hidden_size];

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
    gru_cell(x_copy, h, new_h);

    std::vector<float> output(1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 1, hidden_size, 1.0, linear_weights.data(), hidden_size, new_h.data(), 1, 0.0, output.data(), 1);
    output[0] += linear_biases[0];

    return std::make_tuple(output, new_h);
}
