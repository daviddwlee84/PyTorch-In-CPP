import torch
import torch.nn as nn
from model import PaddedLSTMRegression, SingleStepLSTMRegression

# Instantiate the original model and load pretrained weights
feature_dim = 10
hidden_size = 64
num_layers = 2
batch_first = True

original_model = PaddedLSTMRegression(feature_dim, hidden_size, num_layers, batch_first)

# Initialize with random weights for demonstration purposes
original_model.apply(
    lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None
)

# Save the state dict
state_dict = original_model.state_dict()

# Instantiate the new model
single_step_model = SingleStepLSTMRegression(feature_dim, hidden_size, num_layers)

# Load the state dict into the new model
single_step_model.load_state_dict(state_dict)


# Test code to evaluate whether they are equal
def test_models(
    original_model: PaddedLSTMRegression,
    single_step_model: SingleStepLSTMRegression,
    input_tensor: torch.Tensor,
    lengths: torch.Tensor,
) -> bool:
    original_model.eval()
    single_step_model.eval()

    # Get the original model output
    with torch.no_grad():
        original_output = original_model(input_tensor, lengths)

    # Get the single step model output
    batch_size = input_tensor.size(0)
    h = single_step_model.init_hidden(batch_size)
    single_step_output = []

    for t in range(input_tensor.size(1)):
        x_t = input_tensor[:, t, :]
        output, h = single_step_model(x_t, h)
        single_step_output.append(output)

    single_step_output = torch.cat(single_step_output, dim=1).squeeze(-1)

    # Compare the outputs
    return torch.allclose(original_output, single_step_output, atol=1e-5)


# Generate some sample data
batch_size = 4
seq_len = 5
input_tensor = torch.randn(batch_size, seq_len, feature_dim)
lengths = torch.tensor([seq_len, seq_len, seq_len, seq_len])

# Perform the test
result = test_models(original_model, single_step_model, input_tensor, lengths)
print("Are the outputs from both models equal? ", result)
