import torch

# Example input tensor
print(example_input := torch.rand(5, 10, 10))
print(example_lengths := torch.tensor([10, 8, 6, 4, 2]))

# Save the tensors to files
# BUG: terminate called after throwing an instance of 'c10::Error'
#   what():  PytorchStreamReader failed locating file constants.pkl: file not found
# torch.save(example_input, "example_input.pt")
# torch.save(example_lengths, "example_lengths.pt")
# BUG: terminate called after throwing an instance of 'c10::Error'
#   what():  PytorchStreamReader failed reading zip archive: failed finding central directory
# torch.save(example_input, "example_input.pt", _use_new_zipfile_serialization=False)
# torch.save(example_lengths, "example_lengths.pt", _use_new_zipfile_serialization=False)

# Save the tensors to files in a way compatible with LibTorch
with open("example_input.pt", "wb") as f:
    torch.save(example_input, f)

with open("example_lengths.pt", "wb") as f:
    torch.save(example_lengths, f)
