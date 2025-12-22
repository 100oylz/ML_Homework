import torch
from torch import nn
from torch.nn import functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        # 使用经典 sin / cos 编码
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        return x + self.pe[:, :x.size(1)]


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_size,     # 每个时间点的特征维度 F
        hidden_size,    # Transformer 内部维度 d_model
        output_size,    # 输出维度
        num_layers=2,
        num_heads=4,
        dropout=0.1
    ):
        super(TimeSeriesTransformer, self).__init__()

        # 1️⃣ 输入投影：F → d_model
        self.input_proj = nn.Linear(input_size, hidden_size)

        # 2️⃣ 位置编码
        # self.pos_encoder = PositionalEncoding(hidden_size)

        # 3️⃣ Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # 重要！输入用 (B, T, C)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4️⃣ 输出层（预测头）
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x: (B, T, F)
        """
        # 输入映射
        x = self.input_proj(x)  # (B, T, hidden_size)

        # 加位置编码
        # x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.transformer_encoder(x)  # (B, T, hidden_size)

        # 取最后一个时间步（预测未来）
        last_hidden = x[:, -1, :]  # (B, hidden_size)

        # 输出预测
        out = self.fc_out(last_hidden)  # (B, output_size)

        # ⚠️ 为了兼容你现有训练代码，返回 tuple
        return out, None
