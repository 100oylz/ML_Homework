import json
import os
import pandas as pd
import matplotlib.pyplot as plt


def process_metrics(folder_path):
    all_model_best_metrics = []

    # 1. 遍历文件夹中的所有 JSON 文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            model_name = file_name.replace('_metrics.json', '').replace('.json', '')

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 2. 找到 val_loss 最小的那个 epoch
            # 根据提供的样板，val_loss 越小代表模型在该轮次的验证表现越好
            best_epoch_data = min(data, key=lambda x: x['val_loss'])

            # 添加模型名称以便后续绘图
            best_epoch_data['model'] = model_name
            all_model_best_metrics.append(best_epoch_data)

    return pd.DataFrame(all_model_best_metrics)


def plot_metrics(df,dataset_name,have_Transformer:bool=True):
    # 需要对比的指标列表
    if not have_Transformer:
        # 找到所有模型名称为 TimeSeriesTransformer 的行索引，然后删除
        df.drop(df[df['model'] == "TimeSeriesTransformer"].index, inplace=True)
    metrics = ['MSE', 'RMSE', 'MAE', 'R2']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        # 绘制柱状图
        axes[i].bar(df['model'], df[metric], color='skyblue', edgecolor='black')
        axes[i].set_title(f'Comparison of {metric} (Best Epoch)')
        axes[i].set_ylabel('Value')
        axes[i].tick_params(axis='x', rotation=45)

        # 在柱状图上方标注数值
        for index, value in enumerate(df[metric]):
            axes[i].text(index, value, f'{value:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join("./plots",dataset_name,"Comparsion.png" if have_Transformer else "Comparsion_noTransforms.png"),dpi=300)
    plt.close()

# --- 使用说明 ---
# 1. 将所有模型的 JSON 文件放入一个文件夹（例如名为 'results'）
# 2. 修改下面的 path 变量
# folder_path = './results'
# df_results = process_metrics(folder_path)
# plot_metrics(df_results)