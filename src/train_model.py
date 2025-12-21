from .model.RNN import RNNnet
from tqdm import tqdm
from torch import optim
from torch import nn
from .utils import build_sliding_window
import torch


def validate_model(model, val_data, val_labels, window_size, batch_size=16, loss_function='MSE'):
    """在验证集上评估模型，返回平均 loss"""
    model.eval()
    device = next(model.parameters()).device

    if loss_function == "MSE":
        criterion = nn.MSELoss()
    elif loss_function == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid loss function specified")

    val_inputs, val_labels_tensor = build_sliding_window(
        val_data, val_labels, window_size)
    val_inputs = val_inputs.to(device)
    val_labels_tensor = val_labels_tensor.to(device)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i in range(0, len(val_inputs), batch_size):
            batch_inputs = val_inputs[i:i + batch_size]
            batch_labels = val_labels_tensor[i:i + batch_size]

            outputs = model(batch_inputs)
            pred = outputs[0]  # 只取第一个输出（预测值），忽略 hidden 等

            loss = criterion(pred, batch_labels)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train_model(model, train_data, train_labels, val_data, val_labels,
                window_size, epochs, batch_size=16,
                learning_rate=0.001, loss_function='MSE', optimizer_name='Adam',
                early_stopping_patience=50):
    # ==================== 设备设置 ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    # ==================== 优化器和损失函数 ====================
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) if optimizer_name == "Adam" \
        else optim.SGD(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss() if loss_function == "MSE" else nn.CrossEntropyLoss()

    # ==================== 训练数据准备 ====================
    train_inputs, train_labels_tensor = build_sliding_window(
        train_data, train_labels, window_size)
    train_inputs = train_inputs.to(device)
    train_labels_tensor = train_labels_tensor.to(device)

    history_train = []
    history_val = []
    best_val_loss = float('inf')
    early_stopping_counter = 0

    epoch_bar = tqdm(range(epochs), desc=f"Epoch 1/{epochs}", leave=True)

    for epoch in epoch_bar:
        epoch_bar.set_description(f"Epoch {epoch+1}/{epochs}")

        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_inputs), batch_size):
            batch_inputs = train_inputs[i:i + batch_size]
            batch_labels = train_labels_tensor[i:i + batch_size]

            optimizer.zero_grad()

            outputs = model(batch_inputs)
            pred = outputs[0]  # 关键：只取 tuple 的第一个元素作为预测输出

            loss = criterion(pred, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches
        history_train.append(avg_train_loss)

        # ==================== 验证阶段 ====================
        val_loss = validate_model(
            model, val_data, val_labels, window_size, batch_size, loss_function)
        history_val.append(val_loss)

        # ==================== 早停与保存最佳模型 ====================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       f'checkpoint/{model.__class__.__name__}_best_model.pth')
            early_stopping_counter = 0
            # print(f"  → New best model saved! Val Loss: {val_loss:.2f}")
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.2f}")
            break

        # 更新进度条
        epoch_bar.set_postfix({
            'Train': f'{avg_train_loss:.2f}',
            'Val': f'{val_loss:.2f}',
            'Best': f'{best_val_loss:.2f}'
        })

    return model, history_train, history_val
