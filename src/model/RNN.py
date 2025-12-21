import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from src.utils import build_sliding_window

class RNNnet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, hidden = self.rnn(x)
        out = self.linear(rnn_out[:, -1, :])
        return out, rnn_out, hidden

        
        
