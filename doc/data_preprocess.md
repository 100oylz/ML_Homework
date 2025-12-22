# 数据预处理文档（Data Preprocessing Documentation）

本文档基于项目中的 `Dataset` 类代码，详细记录了对时序数据集的完整预处理流程。所有操作均在 `Dataset` 类初始化时自动执行，确保数据从原始 CSV 到模型输入的标准化、可复现处理。

## 总体流程概述

1. **加载元数据**：从 `./data/kaggle_data/meta_data.json5` 读取配置。
2. **加载原始 CSV**：读取训练集和测试集（或从训练集拆分）。
3. **特征工程**：
   - 时间特征提取（年、月、日、小时、周几、季节、时间段）
   - 分类特征 One-Hot 编码
   - 按时间分组聚合（小时或天级别）
   - 可选：添加滞后特征（lag features）
4. **训练/测试划分**：若无独立测试集，按时间顺序从训练集后部分拆分。
5. **标准化**：使用 `MinMaxScaler` 对聚合列（含 `sum` 或 `mean` 的列）进行缩放，仅在训练集上 fit。
6. **清理冗余列**：移除已提取的原始时间分解列（如 Year、Month 等）。
7. **保存结果**：
   - 处理后的 `train.csv` 和 `test.csv`
   - `scaler.joblib`（用于后续预测反缩放）
   - `meta.json5`（运行时元信息）

处理后数据保存在 `./data/processed_data/{dataset_name}/` 目录下。

## 详细处理步骤

### 1. 元数据驱动配置
- 配置从 `meta_data.json5` 加载，支持多个数据集。
- 关键字段：
  - `train_csv` / `test_csv`：文件路径
  - `need_split` & `split_ratio`：是否/如何拆分
  - `time_column`：时间戳列名
  - `aggregation`：分组聚合方式（mean/sum/first）
  - `lag_features` & `lags`：滞后特征配置（可选）

### 2. 时间特征提取与分类编码
- 解析时间列为 `datetime`。
- 提取：
  - Year, Month, Day（必选）
  - Hour（若 `groupby="hour"`）
  - Weekday & IsWeekend（若配置）
  - TimeOfDay（Morning/Afternoon/Evening/Night）
  - Season（Winter/Spring/Summer/Autumn）
- 分类特征（TimeOfDay、Season 等）进行 One-Hot 编码并转为整数。

### 3. 分组聚合
- 支持按小时或按天聚合。
- 使用配置中的 `aggregation` 字典指定每列的聚合方式（mean/sum/first）。
- 对配置中存在但数据中缺失的列，填充 0（使用 first 聚合）。

### 4. 滞后特征（可选）
- 若 `need_lag=True`，为指定列创建滞后特征（shift）。
- 滞后步数由 `lags` 配置决定。
- 自动删除因 shift 产生的 NaN 行。

### 5. 训练/测试划分
- 若提供独立 `test_csv`：分别处理并对齐列顺序。
- 否则：按时间顺序从训练集尾部切分（比例由 `split_ratio` 决定）。

### 6. 标准化（MinMaxScaler）
- 仅对列名包含 `sum` 或 `mean` 的聚合列进行缩放。
- 训练集：`fit_transform`
- 测试集：`transform`（使用训练集参数）
- scaler 保存为 `scaler.joblib`

### 7. 清理与保存
- 删除原始时间分解列（Year/Month/Day/Hour/Weekday）。
- 保存：
  - `train.csv`、`test.csv`
  - `scaler.joblib`
  - `meta.json5`（包含路径、目标列等信息）

## 使用方式

```python
# 处理数据集（不添加滞后特征）
Dataset('daily-climate-time-series-data', need_lag=False)
Dataset('electric-power-consumption', need_lag=False)

# 添加滞后特征
Dataset('electric-power-consumption', need_lag=True)
```

处理完成后，所有数据位于 `./data/processed_data/{dataset_name}/`，即可直接用于模型训练。

**注意**：当前聚合逻辑针对特定数据集优化（如电力和气候），扩展新数据集时需确保 `meta_data.json5` 配置正确。