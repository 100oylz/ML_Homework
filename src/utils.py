import torch
import pandas as pd
import numpy as np

def build_sliding_window(data, labels=None, window_size=16):
    """
    将时序数据构建成滑动窗口样本
    data: pandas DataFrame 或 np.array，形状 (n_timesteps, n_features)
    labels: pandas DataFrame/Series 或 np.array，目标值 (n_timesteps, n_targets)
    window_size: int，输入序列长度
    返回: torch.tensor X (n_samples, window_size, n_features), y (n_samples, n_targets)
    """
    # 统一转为 numpy array
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = np.array(data)

    if labels is not None:
        if isinstance(labels, (pd.DataFrame, pd.Series)):
            labels_array = labels.values
        else:
            labels_array = np.array(labels)

        assert len(data_array) == len(
            labels_array), "Data and labels must have the same length"
    else:
        labels_array = None

    if len(data_array) < window_size + 1:
        raise ValueError("Data length must be at least window_size + 1")

    X, y = [], []
    for i in range(len(data_array) - window_size):
        # shape: (window_size, n_features)
        X.append(data_array[i:i + window_size])
        if labels_array is not None:
            y.append(labels_array[i + window_size])    # 预测下一个时刻的值
        else:
            y.append(data_array[i + window_size])       # 无标签时用自身下一个时刻

    X = np.array(X)  # (n_samples, window_size, n_features)
    y = np.array(y)  # (n_samples, n_targets) 或 (n_samples, n_features)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
