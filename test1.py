import torch
import intel_extension_for_pytorch as ipex

# Check if AMX is used (indirectly)
print("IPEX version:", ipex.__version__)
print("Device:", torch.xpu if torch.has_xpu else "CPU")

# Simple matmul to trigger AMX under the hood
a = torch.randn(1024, 1024)
b = torch.randn(1024, 1024)

# Optimization step
a, b = a.to(memory_format=torch.channels_last), b.to(memory_format=torch.channels_last)
with torch.no_grad():
    result = torch.matmul(a, b)
    print("Result shape:", result.shape)
