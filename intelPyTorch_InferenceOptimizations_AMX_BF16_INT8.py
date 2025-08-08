import os
from time import time
import matplotlib.pyplot as plt
import torch
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
import torchvision
from torchvision import models
# from transformers import BertModel  # Commented out due to PyTorch version incompatibility

SUPPORTED_MODELS = ["resnet50"]   # models supported by this code sample

# ResNet sample data parameters
RESNET_BATCH_SIZE = 64

# BERT sample data parameters
BERT_BATCH_SIZE = 64
BERT_SEQ_LENGTH = 512

# Check if hardware supports Intel® AMX
import sys
# sys.path.append('../../')
# from cpuinfo import get_cpu_info
# info = get_cpu_info()
# flags = info['flags']
amx_supported = True
# for flag in flags:
#     if "amx" in flag:
#         amx_supported = True
#         break
if not amx_supported:
    print("Intel® AMX is not supported on current hardware. Code sample cannot be run.\n")

os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_AMX"
#os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_VNNI"

"""
Function to perform inference on Resnet50 and BERT
"""
def runInference(model, data, modelName="resnet50", dataType="FP32", amx=True):
    """
    Input parameters
        model: the PyTorch model object used for inference
        data: a sample input into the model
        modelName: str representing the name of the model, supported values - resnet50, bert
        dataType: str representing the data type for model parameters, supported values - FP32, BF16, INT8
        amx: set to False to disable Intel® AMX  on BF16, Default: True
    Return value
        inference_time: the time in seconds it takes to perform inference with the model
    """
    
    # Display run case
    if amx:
        isa_text = "AVX512_CORE_AMX"
    else:
        isa_text = "AVX512_CORE_VNNI"
    print("%s %s inference with %s" %(modelName, dataType, isa_text))

    # Special variables for specific models
    batch_size = None
    if "resnet50" == modelName:
        batch_size = RESNET_BATCH_SIZE
    # elif "bert" == modelName:
    #     d = torch.randint(model.config.vocab_size, size=[BERT_BATCH_SIZE, BERT_SEQ_LENGTH]) # sample data input for torchscript and inference
    #     batch_size = BERT_BATCH_SIZE
    else:
        raise Exception("ERROR: modelName %s is not supported. Choose from %s" %(modelName, SUPPORTED_MODELS))

    # Prepare model for inference based on precision (FP32, BF16, INT8)
    if "INT8" == dataType:
        # Quantize model to INT8 if needed (one time)
        model_filename = "quantized_model_%s.pt" %modelName
        if not os.path.exists(model_filename):
            qconfig = ipex.quantization.default_static_qconfig
            prepared_model = prepare(model, qconfig, example_inputs=data, inplace=False)
            converted_model = convert(prepared_model)
            with torch.no_grad():
                if "resnet50" == modelName:
                    traced_model = torch.jit.trace(converted_model, data)
                # elif "bert" == modelName:
                #     traced_model = torch.jit.trace(converted_model, (d,), check_trace=False, strict=False)
                else:
                    raise Exception("ERROR: modelName %s is not supported. Choose from %s" %(modelName, SUPPORTED_MODELS))
                traced_model = torch.jit.freeze(traced_model)
            traced_model.save(model_filename)

        # Load INT8 model for inference
        model = torch.jit.load(model_filename)
        model.eval()
        model = torch.jit.freeze(model)
    elif "BF16" == dataType:
        model = ipex.optimize(model, dtype=torch.bfloat16)
        with torch.no_grad():
            with torch.cpu.amp.autocast():
                if "resnet50" == modelName:
                    model = torch.jit.trace(model, data)
                # elif "bert" == modelName:
                #     model = torch.jit.trace(model, (d,), check_trace=False, strict=False)
                else:
                    raise Exception("ERROR: modelName %s is not supported. Choose from %s" %(modelName, SUPPORTED_MODELS))
                model = torch.jit.freeze(model)
    else: # FP32
        # Apply Intel Extension optimization if AMX is enabled
        if amx:
            model = ipex.optimize(model, dtype=torch.float32)
        
        with torch.no_grad():
            if "resnet50" == modelName:
                model = torch.jit.trace(model, data)
            # elif "bert" == modelName:
            #     model = torch.jit.trace(model, (d,), check_trace=False, strict=False)
            else:
                raise Exception("ERROR: modelName %s is not supported. Choose from %s" %(modelName, SUPPORTED_MODELS))
            model = torch.jit.freeze(model)

    # Run inference
    with torch.no_grad():
        if "BF16" == dataType:
            with torch.cpu.amp.autocast():
                # Warm up
                for i in range(5):
                    model(data)
                
                # Measure latency
                start_time = time()
                model(data)
                end_time = time()
        else:
            # Warm up
            for i in range(5):
                model(data)
            
            # Measure latency
            start_time = time()
            model(data)
            end_time = time()
    inference_time = end_time - start_time
    print("Inference on batch size %d took %.3f seconds" %(batch_size, inference_time))

    return inference_time

"""
Prints out results and displays figures summarizing output.
"""
def summarizeResults(modelName="", results=None, batch_size=1):
    """
    Input parameters
        modelName: a str representing the name of the model
        results: a dict with the run case and its corresponding time in seconds
        batch_size: an integer for the batch size
    Return value
        None
    """

    # Inference time results
    print("\nSummary for %s (Batch Size = %d)" %(modelName, batch_size))
    for key in results.keys():
        print("%s inference time: %.3f seconds" %(key, results[key]))

    # Create bar chart with inference time results
    plt.figure()
    plt.title("%s Inference Time (Batch Size = %d)" %(modelName, batch_size))
    plt.xlabel("Run Case")
    plt.ylabel("Inference Time (seconds)")
    plt.bar(results.keys(), results.values())

    # Calculate speedup when using Intel® AMX
    print("\n")
    # fp32_no_amx_speedup = results["FP32"] / results["FP32_no_AMX"]
    # print("FP32 without Intel® AMX  is %.2fX faster than FP32 with AMX" %fp32_no_amx_speedup)
    bf16_with_amx_speedup = results["FP32"] / results["BF16_with_AMX"]
    print("BF16 with Intel® AMX  is %.2fX faster than FP32" %bf16_with_amx_speedup)
    # int8_with_vnni_speedup = results["FP32"] / results["INT8_with_VNNI"]
    # print("INT8 without Intel® AMX  is %.2fX faster than FP32" %int8_with_vnni_speedup)
    int8_with_amx_speedup = results["FP32"] / results["INT8_with_AMX"]
    print("INT8 with Intel® AMX  is %.2fX faster than FP32" %int8_with_amx_speedup)
    print("\n\n")

    # Create bar chart with speedup results
    plt.figure()
    plt.title("%s Intel® AMX  BF16/INT8 Speedup over FP32" %modelName)
    plt.xlabel("Run Case")
    plt.ylabel("Speedup")
    plt.bar(results.keys(), 
        [1, bf16_with_amx_speedup, int8_with_amx_speedup]
    )

# Record the inference times for INT8 using AVX-512
# int8_with_vnni_resnet_inference_time = 0.033   #TODO: enter in inference time
# int8_with_vnni_bert_inference_time = 0.691    #TODO: enter in inference time

# Set up ResNet50 model and sample data
resnet_model = models.resnet50(pretrained=True)
resnet_data = torch.rand(RESNET_BATCH_SIZE, 3, 224, 224)
resnet_model.eval()
# FP32 (baseline)
fp32_resnet_inference_time = runInference(resnet_model, resnet_data, modelName="resnet50", dataType="FP32", amx=True)
# FP32 without Intel® AMX
# fp32_no_amx_resnet_inference_time = runInference(resnet_model, resnet_data, modelName="resnet50", dataType="FP32", amx=False)

resnet_model = models.resnet50(pretrained=True)
resnet_data = torch.rand(RESNET_BATCH_SIZE, 3, 224, 224)
resnet_model.eval()

# BF16 using Intel® 
bf16_amx_resnet_inference_time = runInference(resnet_model, resnet_data, modelName="resnet50", dataType="BF16", amx=True)


resnet_model = models.resnet50(pretrained=True)
resnet_data = torch.rand(RESNET_BATCH_SIZE, 3, 224, 224)
resnet_model.eval()

# INT8 using Intel® 
int8_amx_resnet_inference_time = runInference(resnet_model, resnet_data, modelName="resnet50", dataType="INT8", amx=True)
# Summarize and display results
results_resnet = {
        "FP32": fp32_resnet_inference_time,
        # "FP32_no_AMX": fp32_no_amx_resnet_inference_time,
        "BF16_with_AMX": bf16_amx_resnet_inference_time,
        # "INT8_with_VNNI": int8_with_vnni_resnet_inference_time,
        "INT8_with_AMX": int8_amx_resnet_inference_time
    }
summarizeResults("ResNet50", results_resnet, RESNET_BATCH_SIZE)
plt.savefig('inference_cmp.png') 