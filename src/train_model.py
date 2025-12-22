from tqdm import tqdm
from torch import optim
from torch import nn
from .utils import build_sliding_window
import torch
import os
import json
import numpy as np


# ======================== 指标计算工具 ========================
def compute_metrics(y_true, y_pred, eps=1e-8):
    """
    计算常见时序回归指标
    y_true, y_pred: numpy array
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    mape = np.mean(np.abs((y_pred - y_true) / (y_true + eps))) * 100

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + eps)

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "R2": float(r2)
    }


# ======================== 验证函数（增强指标） ========================
def validate_model(model, val_data, val_labels, window_size,
                   batch_size=16, loss_function='MSE', collect_metrics=False):
    model.eval()
    device = next(model.parameters()).device

    criterion = nn.MSELoss() if loss_function == "MSE" else nn.CrossEntropyLoss()

    val_inputs, val_labels_tensor = build_sliding_window(
        val_data, val_labels, window_size)
    val_inputs = val_inputs.to(device)
    val_labels_tensor = val_labels_tensor.to(device)

    total_loss = 0.0
    num_batches = 0

    preds = []
    gts = []

    with torch.no_grad():
        for i in range(0, len(val_inputs), batch_size):
            batch_inputs = val_inputs[i:i + batch_size]
            batch_labels = val_labels_tensor[i:i + batch_size]

            outputs = model(batch_inputs)
            pred = outputs[0]

            loss = criterion(pred, batch_labels)
            total_loss += loss.item()
            num_batches += 1

            if collect_metrics:
                preds.append(pred.detach().cpu().numpy())
                gts.append(batch_labels.detach().cpu().numpy())

    avg_loss = total_loss / num_batches

    if collect_metrics:
        preds = np.concatenate(preds, axis=0)
        gts = np.concatenate(gts, axis=0)
        metrics = compute_metrics(gts, preds)
        return avg_loss, metrics

    return avg_loss


# ======================== 训练主函数 ========================
def train_model(model, train_data, train_labels, val_data, val_labels,
                window_size, epochs, batch_size=16,
                learning_rate=0.001, loss_function='MSE', optimizer_name='Adam',
                early_stopping_patience=500,
                dataset_name=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate) \
        if optimizer_name == "Adam" else optim.SGD(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss() if loss_function == "MSE" else nn.CrossEntropyLoss()

    train_inputs, train_labels_tensor = build_sliding_window(
        train_data, train_labels, window_size)
    train_inputs = train_inputs.to(device)
    train_labels_tensor = train_labels_tensor.to(device)

    history_train = []
    history_val = []
    history_metrics = []

    best_val_loss = float('inf')
    early_stopping_counter = 0

    model_name = model.__class__.__name__
    save_dir = os.path.join(
        'checkpoint', dataset_name) if dataset_name else 'checkpoint'
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'{model_name}_best.pth')
    metrics_path = os.path.join(save_dir, f'{model_name}_metrics.json')

    epoch_bar = tqdm(range(epochs), desc=f"Epoch 1/{epochs}", leave=True)

    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_inputs), batch_size):
            batch_inputs = train_inputs[i:i + batch_size]
            batch_labels = train_labels_tensor[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            pred = outputs[0]

            loss = criterion(pred, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches
        history_train.append(avg_train_loss)

        val_loss, metrics = validate_model(
            model, val_data, val_labels, window_size,
            batch_size, loss_function, collect_metrics=True)

        history_val.append(val_loss)
        history_metrics.append({
            "epoch": epoch + 1,
            "val_loss": val_loss,
            **metrics
        })

        # 保存 metrics
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(history_metrics, f, indent=4)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        epoch_bar.set_postfix({
            "Train": f"{avg_train_loss:.4f}",
            "Val": f"{val_loss:.4f}",
            "RMSE": f"{metrics['RMSE']:.4f}",
            "MAE": f"{metrics['MAE']:.4f}"
        })

    print(f"Best model saved to: {save_path}")
    print(f"Metrics saved to: {metrics_path}")

    return model, history_train, history_val, history_metrics
