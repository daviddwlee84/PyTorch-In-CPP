#ifndef SINGLE_STEP_LSTM_REGRESSION_MKL_H
#define SINGLE_STEP_LSTM_REGRESSION_MKL_H

#include <vector>
#include <tuple>
#include <mkl.h>

class SingleStepLSTMRegressionMKL
{
public:
    SingleStepLSTMRegressionMKL(int64_t feature_dim, int64_t hidden_size = 64, int64_t num_layers = 2);

    std::tuple<std::vector<float>, std::vector<float>> forward(const std::vector<float> &x, const std::vector<float> &h);

private:
    void batch_norm(std::vector<float> &x);
    void gru_cell(const std::vector<float> &x, const std::vector<float> &h, std::vector<float> &new_h);

    int64_t feature_dim;
    int64_t hidden_size;
    int64_t num_layers;
    std::vector<float> batch_norm_mean;
    std::vector<float> batch_norm_var;
    std::vector<float> batch_norm_gamma;
    std::vector<float> batch_norm_beta;
    std::vector<float> lstm_weights;
    std::vector<float> lstm_biases;
    std::vector<float> linear_weights;
    std::vector<float> linear_biases;
};

#endif // SINGLE_STEP_LSTM_REGRESSION_MKL_H
