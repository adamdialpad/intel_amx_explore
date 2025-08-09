import time
import torch
import torchvision
from torchvision import models
import os
import matplotlib.pyplot as plt

# set the device to cpu
device = 'cpu'
# generate random sequence for RNN language model (sequence_length, batch_size)
sequence_length = 100
batch_size = 8
vocab_size = 10000  # typical vocabulary size for language models
input_sequence = torch.randint(low=0, high=vocab_size, size=(sequence_length, batch_size))  # Token indices for embedding

# explore image shape

class ComplexRNNModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_size=1024, num_layers=4):
        super(ComplexRNNModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.dropout1 = torch.nn.Dropout(0.2)
        
        # Multi-layer LSTM
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, num_layers, 
                                 dropout=0.3, batch_first=False, bidirectional=True)
        
        # Attention mechanism
        self.attention = torch.nn.Linear(hidden_size * 2, 1)
        
        # Dense layers
        self.fc1 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.dropout2 = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.dropout3 = torch.nn.Dropout(0.2)
        self.fc3 = torch.nn.Linear(hidden_size // 2, vocab_size)
        
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: (seq_len, batch_size)
        embedded = self.embedding(x)  # (seq_len, batch_size, embedding_dim)
        embedded = self.dropout1(embedded)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded)  # (seq_len, batch_size, hidden_size*2)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=0)  # (seq_len, batch_size)
        attention_weights = attention_weights.unsqueeze(-1)  # (seq_len, batch_size, 1)
        
        # Apply attention
        attended = torch.sum(lstm_out * attention_weights, dim=0)  # (batch_size, hidden_size*2)
        
        # Dense layers
        x = self.relu(self.fc1(attended))
        x = self.dropout2(x)
        x = self.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        
        return self.softmax(x)

def load_model_eval_mode():
    """
    Loads model and returns it in eval mode
    """
    model = ComplexRNNModel(vocab_size)
    model = model.eval()
    
    return model

def get_average_inference_time(model, input_data):
    """
    does a model warm up and times the model runtime
    """
    with torch.no_grad():
        # warm up
        for _ in range(5):
            model(input_data)

        # measure
        import time
        start = time.time()
        for _ in range(25):
            output = model(input_data)
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

# send the input to the device and pass it through the network to
# get the detections and predictions

model = load_model_eval_mode()

inference_time_stock = get_average_inference_time(model, input_sequence)

print(f"time taken for forward pass: {inference_time_stock} ms")

# model = load_model_eval_mode()
# model = model.to(memory_format=torch.channels_last)
# image_channels_last = image.to(memory_format=torch.channels_last)

# inference_time_stock = get_average_inference_time(model, image_channels_last)

# print(f"time taken for forward pass: {inference_time_stock} ms")


model = load_model_eval_mode()
#################### code changes ####################
import intel_extension_for_pytorch as ipex
model = ipex.optimize(model)
######################################################
inference_time_optimized = get_average_inference_time(model, input_sequence)

print(f"time taken for forward pass: {inference_time_optimized} ms")