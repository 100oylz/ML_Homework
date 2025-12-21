import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from src.utils import build_sliding_window


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.3):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # 只有多层时才启用层间 dropout
        )
        self.dropout = nn.Dropout(dropout)  # 额外在输出前加一层 dropout
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # rnn_out: (batch, seq_len, hidden)
        rnn_out, hidden = self.rnn(x)
        # 使用最后一个时间步的输出
        last_output = rnn_out[:, -1, :]          # (batch, hidden)
        last_output = self.dropout(last_output)  # 加 dropout
        out = self.linear(last_output)           # (batch, output_size)
        return out, rnn_out, hidden


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # 多层时启用内置 dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, hidden = self.gru(x)
        last_output = gru_out[:, -1, :]
        last_output = self.dropout(last_output)
        out = self.linear(last_output)
        return out, gru_out, hidden


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # 多层时启用层间 dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        out = self.linear(last_output)
        return out, lstm_out, hidden, cell
