import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18()

# Use torch.jit.script to generate a torch.jit.ScriptModule via annotation.
script_module = torch.jit.script(model)

script_module.save("script_resnet_model.pt")
