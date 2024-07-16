import torch
import torchvision
from torchinfo import summary
import time
from model import PaddedLSTMRegression

BATCH_SIZE = 5
HIDDEN_SIZE = 96
FEATURE_DIM = 150
NUM_LAYERS = 5
MAX_LENGTH = 4000

# An instance of your model.
model = PaddedLSTMRegression(
    feature_dim=FEATURE_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    batch_first=True,
)

# Create example input data
print(
    example_input := torch.rand(BATCH_SIZE, MAX_LENGTH, FEATURE_DIM)
)  # Example batch of sequences
# torch.save(example_input, "example_input.pt")
with open("example_input.pt", "wb") as f:
    torch.save(example_input, f)

print(
    example_lengths := (torch.rand(BATCH_SIZE) * MAX_LENGTH).to(dtype=int)
)  # Example lengths of sequences
# torch.save(example_lengths, "example_lengths.pt")
with open("example_lengths.pt", "wb") as f:
    torch.save(example_lengths, f)

print(output := model(example_input, example_lengths))


# Convert the model to TorchScript
scripted_model = torch.jit.script(model)

# Use torch.jit.script to generate a torch.jit.ScriptModule via annotation.
start = time.perf_counter()
script_module = torch.jit.script(model)
print("Generate via tracing cost:", time.perf_counter() - start, "seconds")

# Save the TorchScript model
script_module.save("scripted_gru.pt")

summary(model, input_data=(example_input, example_lengths))
