import torch
import torch.nn as nn
from model import PaddedLSTMRegression, SingleStepLSTMRegression
from torchinfo import summary
import time
import sys

threads = int(sys.argv[1]) if len(sys.argv) > 1 else -1

if threads > 0:
    # Set the number of threads for intra-op parallelism
    torch.set_num_threads(threads)
    # Set the number of threads for inter-op parallelism
    torch.set_num_interop_threads(threads)

print("intra-op threads:", torch.get_num_threads())
print("inter-op threads:", torch.get_num_interop_threads())

# Instantiate the original model and load pretrained weights
feature_dim = 148
hidden_size = 128
num_layers = 3
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

    total_time = 0
    for t in range(input_tensor.size(1)):
        start_time = time.perf_counter()
        x_t = input_tensor[:, t, :]
        output, h = single_step_model(x_t, h)
        total_time += time.perf_counter() - start_time
        single_step_output.append(output)

    print(
        "Average single tick inference time (sec):", total_time / input_tensor.size(1)
    )

    single_step_output = torch.cat(single_step_output, dim=1).squeeze(-1)

    print("Batch Model")
    summary(original_model, input_data=(input_tensor, lengths))
    print("Single Step Model")
    summary(single_step_model, input_data=(x_t, h))

    # Compare the outputs
    return torch.allclose(original_output, single_step_output, atol=1e-5)


# Generate some sample data
batch_size = 1
seq_len = 4000
input_tensor = torch.randn(batch_size, seq_len, feature_dim)
lengths = torch.tensor([seq_len] * batch_size)

# Perform the test
result = test_models(original_model, single_step_model, input_tensor, lengths)
print("Are the outputs from both models equal? ", result)

# Convert the model to a script module
single_step_script_module = torch.jit.script(single_step_model)
# Save the scripted model to a file
single_step_script_module.save("scripted_single_step_model.pt")
