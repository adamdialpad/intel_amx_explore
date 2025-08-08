import time
import torch
import torchvision
import os
import matplotlib.pyplot as plt

# set the device to cpu
device = 'cpu'
# generate a random image to observe speedup on
image = torch.randn(1, 3, 1200, 1200)
# explore image shape

def load_model_eval_mode():
    """
    Loads model and returns it in eval mode
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights, progress=True,
        num_classes=91, weights_backbone=weights_backbone).to(device)
    model = model.eval()
    
    return model

def get_average_inference_time(model, image):
    """
    does a model warm up and times the model runtime
    """
    with torch.no_grad():
        # warm up
        for _ in range(25):
            model(image)

        # measure
        import time
        start = time.time()
        for _ in range(25):
            output = model(image)
        end = time.time()
        average_inference_time = (end-start)/25*1000
    
    return average_inference_time

def plot_speedup(inference_time_stock, inference_time_optimized):
    """
    Plots a bar chart comparing the time taken by stock PyTorch model and the time taken by
    the model optimized by Intel® Extension for PyTorch* (IPEX)
    """
    data = {'stock_pytorch_time': inference_time_stock, 'optimized_time': inference_time_optimized}
    model_type = list(data.keys())
    times = list(data.values())

    fig = plt.figure(figsize = (10, 5))

    # creating the bar plot
    plt.bar(model_type, times, color ='blue',
            width = 0.4)

    plt.ylabel("Runtime (ms)")
    plt.title(f"Speedup acheived - {inference_time_stock/inference_time_optimized:.2f}x")
    plt.show()

# model configs
weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
weights_backbone = torchvision.models.ResNet50_Weights.DEFAULT

# send the input to the device and pass it through the network to
# get the detections and predictions

model = load_model_eval_mode()

inference_time_stock = get_average_inference_time(model, image)

print(f"time taken for forward pass: {inference_time_stock} ms")

# model = load_model_eval_mode()
# model = model.to(memory_format=torch.channels_last)
# image_channels_last = image.to(memory_format=torch.channels_last)

# inference_time_stock = get_average_inference_time(model, image_channels_last)

# print(f"time taken for forward pass: {inference_time_stock} ms")


model = load_model_eval_mode()
model = model.to(memory_format=torch.channels_last)
image_channels_last = image.to(memory_format=torch.channels_last)
#################### code changes ####################
import intel_extension_for_pytorch as ipex
model = ipex.optimize(model)
######################################################
inference_time_optimized = get_average_inference_time(model, image_channels_last)

print(f"time taken for forward pass: {inference_time_optimized} ms")