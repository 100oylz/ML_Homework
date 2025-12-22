import os
import json5
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional
import warnings

from sklearn.preprocessing import MinMaxScaler

# define the time of day
def categorize_time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'


# Define the seasons
def categorize_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'


def grab_col_names(dataframe):
    # Categorical columns
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]

    # Numeric columns:
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    return cat_cols, num_cols


def should_scale(col_names):
    if "sum" in col_names:
        return True
    elif "mean" in col_names:
        return True
    else:
        return False


def remove_noused_columns(df):
    if ("Year" in df.columns):
        df.drop("Year", axis=1, inplace=True)
    if ("Month" in df.columns):
        df.drop("Month", axis=1, inplace=True)
    if ("Day" in df.columns):
        df.drop("Day", axis=1, inplace=True)
    if ("Hour" in df.columns):
        df.drop("Hour", axis=1, inplace=True)
    if ("Weekday" in df.columns):
        df.drop("Weekday", axis=1, inplace=True)


warnings.filterwarnings("ignore")


class Dataset:
    """
    时序数据集加载与预处理类
    负责：加载原始 CSV → 时间聚合 → 时间编码 → 训练/测试划分 → 标准化 → 保存处理后数据
    """

    JSON_FILES = "./data/kaggle_data/meta_data.json5"      # 元数据配置文件路径
    BASIC_PATH = "./data/kaggle_data"                  # 原始数据根目录
    SAVE_PATH = "./data/processed_data"                # 处理后数据保存目录

    @staticmethod
    def _load_meta_data() -> list:
        """加载所有数据集的元数据配置"""
        with open(Dataset.JSON_FILES, 'r', encoding='utf-8') as f:
            return json5.load(f)

    def __init__(self, dataset_name: str,need_lag:bool):
        # print(os.getcwd())
        # 1. 加载元数据并验证数据集名称
        self.meta_data = self._load_meta_data()
        self.meta_item = self._get_meta_item(dataset_name)
        self.dataset_name = dataset_name

        # 2. 加载并聚合原始数据
        train_raw = self._load_csv(self.meta_item["train_csv"])
        test_raw= self._load_csv(self.meta_item["test_csv"]) if self.meta_item["test_csv"] else None
        train_data=self._feature_engineering(train_raw,need_lag=need_lag)
        if(test_raw is None):
            split_ratio=1-self.meta_item["split_ratio"]
            train_data,test_data=self._split_train_test_dataset(train_data,split_ratio=split_ratio)
        else:
            test_data=self._feature_engineering(test_raw,need_lag=need_lag)
            test_data=test_data[train_data.columns]

        scaler_columns=[ col for col in train_data.columns if should_scale(col)]
        self.scaler=None
        # Scaler is None,Fit And Transform train_data
        train_data=self._scale_data(train_data,scaler_columns,self.scaler)
        # Scaler is Not None,Just Transform
        test_data=self._scale_data(test_data,scaler_columns,self.scaler)

        remove_noused_columns(train_data)
        remove_noused_columns(test_data)

        self.train_data=train_data
        self.test_data=test_data
        self._save_data(self.train_data,self.test_data,self.scaler)
        self._save_meta_data()



    # ===================================================================
    # 内部方法
    # ===================================================================

    def _get_meta_item(self, dataset_name: str) -> Dict[str, Any]:
        """根据名称获取单个数据集的配置项"""
        for item in self.meta_data:
            if item["name"] == dataset_name:
                return item
        available = [item["name"] for item in self.meta_data]
        raise ValueError(f"数据集 '{dataset_name}' 不存在。可用数据集: {available}")

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        """加载 CSV 文件"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"未找到文件: {csv_path}")
        return pd.read_csv(csv_path)

    def _feature_engineering(self,df:pd.DataFrame,need_lag:bool) -> pd.DataFrame:
        time_column = self.meta_item["time_column"]
        # Calculate min and sum values by grouping
        aggregation_functions = self.meta_item["aggregation"]
        df_grouped = self._group_by(aggregation_functions,df, time_column)
        # Columns to create lag features
        columns_to_lag = self.meta_item["lag_features"]
        # Lag values (24, 48, 72 hours)
        lags = self.meta_item["lags"]
        if (need_lag):
            df_lagged = self._lag_features(df_grouped, columns_to_lag, lags)
            return df_lagged
        else:
            return df_grouped
    def _lag_features(self, df, columns_to_lag, lags) -> Any:
        # Create lag features
        df_lagged = df.copy()

        for col in columns_to_lag:
            for lag in lags:
                df_lagged[f'{col}_lag{lag}'] = df[col].shift(lag)
        # Remove rows with NaN values due to lagging
        df_lagged = df_lagged.dropna()
        return df_lagged

    def _group_by(self, aggregation_functions: dict[str | Any, list[str] | Any], df: pd.DataFrame, time_column: str) -> pd.DataFrame:
        need_hour=self.meta_item["groupby"]=="hour"
        need_weekday=self.meta_item["needweekday"]
        df[time_column] = pd.to_datetime(df[time_column])
        df['Year'] = df[time_column].dt.year
        df['Month'] = df[time_column].dt.month
        df['Day'] = df[time_column].dt.day
        if(need_hour):
            df['Hour'] = df[time_column].dt.hour
            df['TimeOfDay'] = df['Hour'].apply(categorize_time_of_day)
        if(need_weekday):
            # Add day of the week (0 = Monday, 6 = Sunday)
            df['Weekday'] = df['Datetime'].dt.weekday

            # Add a column to indicate if it's a weekend
            df['IsWeekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
            df['Weekday'] = df['Weekday'].astype(int)
            df['IsWeekend'] = df['IsWeekend'].astype(int)
        df['Season'] = df['Month'].apply(categorize_season)
        # Let's set the data types to numeric
        df['Year'] = df['Year'].astype(int)


        df.drop(columns=time_column, inplace=True)
        cat_cols, num_cols = grab_col_names(df)
        # Convert categorical variables to numerical values using One-Hot Encoding
        df_encoded = pd.get_dummies(df, columns=cat_cols)
        # Convert the columns resulting from One-Hot Encoding to integer values
        df_encoded = df_encoded.astype(int)
        cur_columns=df_encoded.columns
        union_columns={key:value for key,value in aggregation_functions.items() if key in cur_columns}
        nonunion_columns = {
            key: value for key, value in aggregation_functions.items() if key not in cur_columns}
        if(self.meta_item["groupby"]=="hour"):
            df_grouped = df_encoded.groupby(['Year', 'Month', 'Day', 'Hour']).agg(union_columns)
        elif(self.meta_item["groupby"]=="day"):
            df_grouped = df_encoded.groupby(['Year', 'Month', 'Day']).agg(union_columns)
        else:
            raise NotImplementedError
        for key,value in nonunion_columns.items():
            assert value==["first"],f"{key}'s {value} !=first"
            df_grouped[(key,"first")]=0
        # Correct column names
        df_grouped.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df_grouped.columns]
        df_grouped = df_grouped.reset_index()
        return df_grouped

    def _split_train_test_dataset(self,df:pd.DataFrame,split_ratio:float=0.8):
        train_size = int(len(df)*split_ratio)
        train_data=df.iloc[:train_size]
        test_data=df.iloc[train_size:]
        return train_data,test_data

    def _scale_data(self,df,scaler_columns,scaler:MinMaxScaler|None)->pd.DataFrame:
        if(scaler==None):
            scaler = MinMaxScaler()
            df[scaler_columns] = scaler.fit_transform(df[scaler_columns])
            self.scaler = scaler
        else:
            df[scaler_columns] = scaler.transform(df[scaler_columns])
        return df

    def _save_data(self,train_data:pd.DataFrame,test_data:pd.DataFrame,scaler):
        os.makedirs(Dataset.SAVE_PATH, exist_ok=True)
        os.makedirs(os.path.join(Dataset.SAVE_PATH,self.dataset_name), exist_ok=True)
        train_data.to_csv(os.path.join(Dataset.SAVE_PATH,self.dataset_name,"train.csv"),index=False)
        test_data.to_csv(os.path.join(Dataset.SAVE_PATH,self.dataset_name,"test.csv"),index=False)
        joblib.dump(scaler,os.path.join(Dataset.SAVE_PATH,self.dataset_name,"scaler.joblib"))
    def _save_meta_data(self):
        os.makedirs(Dataset.SAVE_PATH, exist_ok=True)
        os.makedirs(os.path.join(Dataset.SAVE_PATH, self.dataset_name), exist_ok=True)
        json_data={
            "dataset_name":self.dataset_name,
            "train_csv":os.path.join(Dataset.SAVE_PATH,self.dataset_name,"train.csv"),
            "test_csv":os.path.join(Dataset.SAVE_PATH,self.dataset_name,"test.csv"),
            "scaler":os.path.join(Dataset.SAVE_PATH,self.dataset_name,"scaler.joblib"),
            "target_columns":self.meta_item["target_columns"],
        }
        with open(os.path.join(Dataset.SAVE_PATH,self.dataset_name,"meta.json5"),"w") as f:
            f.write(json5.dumps(json_data,indent=4))

# ===================================================================
# 测试入口
# ===================================================================
if __name__ == "__main__":
    # 示例运行
    Dataset('daily-climate-time-series-data',need_lag=False)
    Dataset('electric-power-consumption',need_lag=False)
