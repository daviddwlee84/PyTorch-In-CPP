#include <torch/torch.h>
#include <iostream>

struct SingleStepLSTMRegressionImpl : torch::nn::Module {
    SingleStepLSTMRegressionImpl(int64_t feature_dim, int64_t hidden_size = 64, int64_t num_layers = 2)
        : batch_norm(torch::nn::BatchNorm1d(feature_dim)),
          lstm(torch::nn::GRUOptions(feature_dim, hidden_size).num_layers(num_layers).batch_first(true)),
          linear(hidden_size, 1),
          hidden_size(hidden_size),
          num_layers(num_layers) {
        register_module("batch_norm", batch_norm);
        register_module("lstm", lstm);
        register_module("linear", linear);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor h) {
        x = x.squeeze(1);
        x = batch_norm(x);
        x = x.unsqueeze(1);
        std::tie(x, h) = lstm(x, h);
        x = linear(x.squeeze(1));
        return std::make_tuple(x, h);
    }

    torch::nn::BatchNorm1d batch_norm{nullptr};
    torch::nn::GRU lstm{nullptr};
    torch::nn::Linear linear{nullptr};
    int64_t hidden_size;
    int64_t num_layers;
};

TORCH_MODULE(SingleStepLSTMRegression);

int main() {
    int64_t feature_dim = 10; // Example feature dimension
    SingleStepLSTMRegression model(feature_dim);
    
    // Example input
    auto input = torch::randn({32, 1, feature_dim}); // Batch size 32, sequence length 1, feature dimension
    auto h = torch::zeros({model->num_layers, 32, model->hidden_size}); // Initial hidden state
    
    auto output = model->forward(input, h);

    std::cout << std::get<0>(output) << std::endl; // Print output tensor
    return 0;
}
