import torch
import torchvision
from torchinfo import summary
import time

# An instance of your model.
model = torchvision.models.resnet18()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

start = time.perf_counter()
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
print("Generate via tracing cost:", time.perf_counter() - start, "seconds")

traced_script_module.save("traced_resnet_model.pt")

# RuntimeError: Only one of (input_data, input_size) should be specified.
summary(model, input_data=example)
