import torch
import torchvision.models as models
import intel_extension_for_pytorch as ipex
import time

# Test parameters
batch_size = 128
warmup_runs = 5
test_runs = 10

# Prepare model and data
model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()
data = torch.rand(batch_size, 3, 224, 224)

print(f"Testing ResNet50 with batch size {batch_size}")
print(f"Warmup runs: {warmup_runs}, Test runs: {test_runs}")
print("-" * 50)

# Test 2: Intel Extension for PyTorch (with ipex.optimize)
print("\n2. Intel Extension for PyTorch (with ipex.optimize)")
model_ipex = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model_ipex.eval()
model_ipex = ipex.optimize(model_ipex)

with torch.no_grad():
    # Warmup
    for _ in range(warmup_runs):
        model_ipex(data)
    
    # Measure
    start_time = time.time()
    for _ in range(test_runs):
        model_ipex(data)
    end_time = time.time()
    
    ipex_time = (end_time - start_time) / test_runs
    print(f"   Average time per inference: {ipex_time:.4f} seconds")

# Test 1: Vanilla PyTorch (without ipex.optimize)
print("1. Vanilla PyTorch (without ipex.optimize)")
model_vanilla = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model_vanilla.eval()

with torch.no_grad():
    # Warmup
    for _ in range(warmup_runs):
        model_vanilla(data)
    
    # Measure
    start_time = time.time()
    for _ in range(test_runs):
        model_vanilla(data)
    end_time = time.time()
    
    vanilla_time = (end_time - start_time) / test_runs
    print(f"   Average time per inference: {vanilla_time:.4f} seconds")

# # Compare results
# print("\n" + "=" * 50)
# print("COMPARISON RESULTS:")
# print("=" * 50)
# print(f"Vanilla PyTorch:     {vanilla_time:.4f} seconds")
# print(f"Intel Extension:     {ipex_time:.4f} seconds")

# if ipex_time < vanilla_time:
#     speedup = vanilla_time / ipex_time
#     print(f"Intel Extension is {speedup:.2f}x FASTER")
# else:
#     slowdown = ipex_time / vanilla_time
#     print(f"Intel Extension is {slowdown:.2f}x SLOWER")

# improvement = ((vanilla_time - ipex_time) / vanilla_time) * 100
# print(f"Performance difference: {improvement:.1f}%")