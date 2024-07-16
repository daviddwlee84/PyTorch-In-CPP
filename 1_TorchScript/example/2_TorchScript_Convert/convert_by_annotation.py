import torch
import torchvision
from torchinfo import summary
import time

# An instance of your model.
model = torchvision.models.resnet18()

# Use torch.jit.script to generate a torch.jit.ScriptModule via annotation.
start = time.perf_counter()
script_module = torch.jit.script(model)
print("Generate via tracing cost:", time.perf_counter() - start, "seconds")

script_module.save("scripted_resnet_model.pt")

summary(model)
