import matplotlib.pyplot as plt
import os
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


def plot_training_history(train_history, val_history,
                          save_path=None,
                          title=None,
                          figsize=(10, 6),
                          dataset_name=None,   # ← 新增：数据集名称
                          model_name=None):    # ← 新增：模型名称（如 'RNNnet'）
    """
    可视化训练和验证损失曲线
    
    参数:
    - train_history: list[float]，每个 epoch 的训练损失
    - val_history: list[float]，每个 epoch 的验证损失
    - save_path: str or None，旧方式直接指定完整路径（兼容旧调用）
    - title: str or None，自定义标题
    - figsize: tuple，图片大小
    - dataset_name: str or None，数据集名称，用于新保存路径
    - model_name: str or None，模型类名，用于新保存路径
    """
    epochs = range(1, len(train_history) + 1)

    plt.figure(figsize=figsize)

    # 绘制训练和验证曲线
    plt.plot(epochs, train_history, label='Train Loss',
             color='blue', linewidth=2)
    plt.plot(epochs[:len(val_history)], val_history,
             label='Val Loss', color='orange', linewidth=2)

    # 标注最佳验证损失点
    if len(val_history) > 0:
        best_epoch = val_history.index(min(val_history)) + 1
        best_val_loss = min(val_history)
        plt.scatter(best_epoch, best_val_loss, color='red', s=100,
                    label=f'Best Val (epoch {best_epoch})', zorder=5)
        plt.annotate(f'Best: {best_val_loss:.4f}\nEpoch {best_epoch}',
                     xy=(best_epoch, best_val_loss),
                     xytext=(best_epoch + len(epochs)*0.02, best_val_loss),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontsize=10, color='red')

    # 设置标题和标签
    if title is None:
        title = 'Training and Validation Loss'
    plt.title(title, fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # ==================== 新保存逻辑 ====================
    if dataset_name and model_name:
        # 新方式：自动生成路径 plots/{dataset_name}/{model_name}.png
        auto_save_dir = os.path.join('plots', dataset_name)
        auto_save_path = os.path.join(auto_save_dir, f'{model_name}.png')
        os.makedirs(auto_save_dir, exist_ok=True)
        plt.savefig(auto_save_path, dpi=200, bbox_inches='tight')
        print(f"训练曲线已保存至: {auto_save_path}")
    elif save_path:
        # 兼容旧方式：直接使用传入的 save_path
        os.makedirs(os.path.dirname(save_path),
                    exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"训练曲线已保存至: {save_path}")

    # 显示图片
    # plt.show()
    plt.close()

    # 返回最佳 epoch 和 loss
    if len(val_history) > 0:
        return best_epoch, min(val_history)
    return None, None
