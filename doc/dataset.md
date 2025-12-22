# 机器学习时间序列实验数据集指南

## 第一章：电力消耗预测 (Electric Power Consumption)

### 1.1 资源定位

- **数据集 Handle**: `fedesoriano/electric-power-consumption`
- **原始文件路径**: `./data/kaggle_data/electric-power-consumption/powerconsumption.csv`

### 1.2 属性字段详解 (Schema)

| **字段名 (Field Name)**  | **含义 (Description)**       |
| ------------------------ | ---------------------------- |
| `Datetime`               | 数据记录的时间戳             |
| `Temperature`            | 当地摄氏温度                 |
| `Humidity`               | 空气相对湿度                 |
| `WindSpeed`              | 当地风速                     |
| `GeneralDiffuseFlows`    | 广义扩散流量（太阳辐射相关） |
| `DiffuseFlows`           | 扩散流量                     |
| `PowerConsumption_Zone1` | 区域 1 电力消耗数值          |
| `PowerConsumption_Zone2` | 区域 2 电力消耗数值          |
| `PowerConsumption_Zone3` | 区域 3 电力消耗数值          |

### 1.3 任务描述

- **任务类型**：多变量时间序列预测 (Multivariate Time Series Forecasting)
- **目标字段**：`PowerConsumption_Zone1`、`PowerConsumption_Zone2`、`PowerConsumption_Zone3`

## 第二章：日常气候时间序列数据 (Daily Climate Time Series)

### 2.1 资源定位

- **数据集 Handle**: `sumanthvrao/daily-climate-time-series-data`
- **训练集路径**: `./data/kaggle_data/daily-climate/DailyDelhiClimateTrain.csv`
- **测试集路径**: `./data/kaggle_data/daily-climate/DailyDelhiClimateTest.csv`

### 2.2 属性字段详解 (Schema)

| **字段名 (Field Name)** | **含义 (Description)**     |
| ----------------------- | -------------------------- |
| `date`                  | 数据记录的日期（日级粒度） |
| `meantemp`              | 当日平均气温               |
| `humidity`              | 空气相对湿度               |
| `wind_speed`            | 平均风速                   |
| `meanpressure`          | 平均气压                   |

### 2.3 任务描述

- **任务类型**: 多变量时间序列预测 (Multivariate Time Series Forecasting)
- **目标字段**: `meantemp`,`humidity`,`wind_speed`,`meanpressure`

## 第三章：Yahoo 股票价格预测 (Yahoo Stock Price)

### 3.1 资源定位

- **数据集 Handle**: `arashnic/time-series-forecasting-with-yahoo-stock-price`
- **原始文件路径**: `./data/kaggle_data/yahoo-stock/yahoo_stock.csv`

### 3.2 属性字段详解 (Schema)

| **字段名 (Field Name)** | **含义 (Description)**   |
| ----------------------- | ------------------------ |
| `Date`                  | 交易日期（日级粒度）     |
| `High`                  | 当日最高成交价           |
| `Low`                   | 当日最低成交价           |
| `Open`                  | 当日开盘价               |
| `Close`                 | 当日收盘价               |
| `Volume`                | 当日成交量               |
| `Adj Close`             | 经除权除息调整后的收盘价 |

### 3.3 任务描述

- **任务类型**: 多变量时间序列预测 (Multivariate Time Series Forecasting)
- **目标字段**: `Adj Close`,`Volume`

## 第四章：数据集预处理与元数据配置 (Metadata Configuration)

### 4.1 资源定位

- **配置文件路径**: `./meta_data.json5`
- **配置文件格式**: `JSON5`（支持注释与更灵活的语法）

### 4.2 配置字段详解 (Schema)

| **配置字段 (Parameter)** | **含义 (Description)**                                       |
| ------------------------ | ------------------------------------------------------------ |
| `name`                   | 数据集唯一标识符                                             |
| `train_csv / test_csv`   | 训练集与测试集的本地文件路径                                 |
| `need_split`             | 布尔值，标识是否需要从训练集中自动拆分验证/测试集            |
| `split_ratio`            | 拆分比例（如 `0.2` 表示 20% 用于验证）                       |
| `time_column`            | 时间戳所在的列名                                             |
| `groupby`                | 时间聚合粒度（如 `day` 表示按天聚合）                        |
| `aggregation`            | 聚合规则定义，指定各字段按 `sum`、`mean` 或 `first` 汇总     |
| `lag_features`           | 滞后特征列表，指定哪些字段需要生成历史步长特征               |
| `lags`                   | 滞后步长定义（如 `[1, 2, 3]` 表示取过去 1-3 个时间点的数据） |
| `target_columns`         | 模型预测的目标字段列表                                       |

