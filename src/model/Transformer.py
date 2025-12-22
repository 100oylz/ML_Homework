import torch
from torch import nn
from torch.nn import functional as F
import math

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_size,     # 原始时序每个时间点的特征维度 (F)
        hidden_size,    # Transformer 内部的隐藏层维度 (d_model)
        output_size,    # 最终预测结果的维度
        num_layers=2,   # Encoder 层堆叠数量
        num_heads=4,    # 多头注意力机制的头数
        dropout=0.1     # 丢弃率
    ):
        super(TimeSeriesTransformer, self).__init__()

        # 1️⃣ 输入投影层：将特征空间从 F 映射到 d_model 维度，作为 Transformer 的 Embedding
        self.input_proj = nn.Linear(input_size, hidden_size)

        # 2️⃣ Transformer Encoder：Encoder-only 架构核心
        # 注意：此处未添加位置编码 (Positional Encoding)，侧重于捕捉特征间的上下文交互
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # 输入格式要求为 (Batch, Seq, Feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 3️⃣ 输出层（预测头）：将编码后的隐藏状态映射到目标输出空间
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播逻辑
        输入 x 形状: (B, T, F) -> (Batch_size, Seq_len, Input_size)
        """
        # Step 1: 线性映射，对齐 Transformer 内部维度
        # 输出形状: (B, T, hidden_size)
        x = self.input_proj(x)

        # Step 2: 进入 Transformer Encoder 进行全局上下文特征提取
        # 输出形状: (B, T, hidden_size)
        x = self.transformer_encoder(x)

        # Step 3: 特征提取策略 —— Last-step (取序列最后一个时间步)
        # 理由：在无位置编码的情况下，经过多层 Self-Attention，最后一个步长已聚合了全序列的历史摘要信息
        # 输出形状: (B, hidden_size)
        last_hidden = x[:, -1, :]

        # Step 4: 映射至输出预测值
        # 输出形状: (B, output_size)
        out = self.fc_out(last_hidden)

        # 为了兼容 RNN 等现有训练流程的代码接口，返回结果元组 (预测值, None)
        return out, None