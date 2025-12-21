import pandas as pd
import os
import json5
import numpy as np
import random
from pprint import pprint
from datetime import datetime


def standardize_data(data, mean=None, std=None):
    mean = np.mean(data, axis=0) if mean is None else mean
    std = np.std(data, axis=0) if std is None else std
    uniform_data = (data-mean)/std
    return uniform_data, mean, std


class Dataset:
    JSON_FILES = "./data/kaggle_data/meta_data.jsonc"
    BASIC_PATH = "./data/kaggle_data"
    SAVE_PATH = "./data/processed_data"

    def _load_meta_data():
        with open(Dataset.JSON_FILES, 'r', encoding='utf-8') as f:
            meta_data = json5.load(f)
        return meta_data

    def __init__(self, dataset_name):
        self.meta_data = Dataset._load_meta_data()
        self.dataset_name_fields = [item['name'] for item in self.meta_data]
        self.meta_item = None
        self.dataset_name = None
        self.set_dataset_name(dataset_name)
        # print(self.meta_item)
        self.train_csv_data = self.load_csv(self.meta_item['train_csv'])
        self.train_csv_data = self.aggregate_data()
        if (self.meta_item["need_split"]):
            self.split_data()
            self.train_csv_data = self.train_data
            self.test_csv_data = self.test_data
        else:
            self.test_csv_data = self.load_csv(self.meta_item['test_csv'])
            self.test_csv_data = self.aggregate_data(is_train=False)
            self.train_data=self.train_csv_data
            self.test_data=self.test_csv_data
        print(f"标准化数据集...")
        self.standardize_data(self.train_data, self.test_data)
        self.convert_time()

        self.save_dataset(self.SAVE_PATH)
        self.save_meta_data(self.SAVE_PATH)
        
    def save_meta_data(self,save_path):
        os.makedirs(save_path, exist_ok=True)
        meta_data={
            "name":self.dataset_name,
            "train_csv":os.path.join(Dataset.SAVE_PATH,self.dataset_name,"train.csv"),
            "test_csv":os.path.join(Dataset.SAVE_PATH,self.dataset_name,"test.csv"),
            "time_columns":["sin_time","cos_time"],
            "target_columns": self.meta_item["target_columns"],
            "feature_columns":list(self.train_csv_data.columns),
        }
        meta_data["epochs"]=300
        meta_data["batch_size"]=64
        meta_data["learning_rate"]=0.001
        meta_data["optimizer"] ="Adam"
        meta_data["loss"] = "MSE" 
        meta_data["window_size"]=64
        meta_data['num_layers']=6
        meta_data["hidden_size"]=64
        with open(os.path.join(save_path, meta_data["name"],"meta_data.jsonc"), 'w', encoding='utf-8') as f:
            json5.dump(meta_data, f, ensure_ascii=False,
                       indent=4, quote_keys=True)

    def convert_time(self):
        time_col = ['time']
        self.train_csv_data['sin_time'] = np.sin(2*np.pi*(self.train_csv_data[time_col]-self.train_csv_data[time_col].min())/(
            self.train_csv_data[time_col].max()-self.train_csv_data[time_col].min()))
        self.test_csv_data['sin_time'] = np.sin(2*np.pi*(self.test_csv_data[time_col]-self.test_csv_data[time_col].min())/(
            self.test_csv_data[time_col].max()-self.test_csv_data[time_col].min()))
        self.train_csv_data['cos_time'] = np.cos(2*np.pi*(self.train_csv_data[time_col]-self.train_csv_data[time_col].min())/(
            self.train_csv_data[time_col].max()-self.train_csv_data[time_col].min()))
        self.test_csv_data['cos_time'] = np.cos(2*np.pi*(self.test_csv_data[time_col]-self.test_csv_data[time_col].min())/(
            self.test_csv_data[time_col].max()-self.test_csv_data[time_col].min()))
        self.train_csv_data = self.train_csv_data.drop(columns=time_col)
        self.test_csv_data = self.test_csv_data.drop(columns=time_col)

    def set_dataset_name(self, dataset_name):
        if (dataset_name not in self.dataset_name_fields):
            self.dataset_name = None
            self.meta_item = None
            raise ValueError(f"数据集 {dataset_name} 不存在于元数据中。"
                             f"可用数据集包括: {self.dataset_name_fields}")
        else:
            self.dataset_name = dataset_name
            self.meta_item = next(
                (item for item in self.meta_data if item['name'] == dataset_name), None)
    # 实现数据集划分功能,目前默认数据经过处理后按照时间先后顺序排列

    def split_data(self):
        split_ratio = self.meta_item["split_ratio"]
        if (split_ratio == None):
            raise ValueError("该数据集不需要划分，请检查元数据中的'split_ratio'字段。")
        else:
            data = self.train_csv_data
            # print(data.head())
            length = len(data)
            test_size = int(length*split_ratio)
            train_data = data.iloc[:-test_size]
            test_data = data.iloc[-test_size:]
            print(
                f"数据集已划分为训练集和测试集，测试集大小为 {test_size} 条数据,训练集大小为{length - test_size} 条数据。")
            
            self.train_data = train_data.copy()
            self.test_data = test_data.copy()

            
    def standardize_data(self,train_data,test_data):
        # 对划分后的数据进行标准化处理
        time_col = ["time"]
        train_values = train_data.drop(
            columns=time_col).values
        test_values = test_data.drop(
            columns=time_col).values
        standardized_train_data, mean, std = standardize_data(train_values)
        standardized_test_data = standardize_data(
            test_values, mean=mean, std=std)[0]
        # 获取除时间列外的所有列名（这些是需要标准化的特征列）
        feature_cols = [
            col for col in train_data.columns if col not in time_col]
        # print(feature_cols)

        # 替换训练集的特征列值为标准化后的值（时间列保持不变）
        self.train_data.loc[:, feature_cols] = standardized_train_data

        self.test_data.loc[:, feature_cols] = standardized_test_data


    def load_csv(self, csv_name):
        if self.dataset_name is None:
            raise ValueError("请先设置有效的数据集名称。")
        dataset_path = csv_name
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"文件 {dataset_path} 在数据集 {self.dataset_name} 中未找到。")
        data = pd.read_csv(dataset_path)

        return data

        # pass
    # TODO:按照方式聚合相同时间段的数据 (目前只确保electric-power-consumption,daily-climate-time-series-data数据集可用)

    def aggregate_data(self, is_train=True):
        # 比较两个datetime对象在指定频率下的大小,目前递归实现到了天
        def _compare_datetime(dt1, dt2, resample_freq):
            def _compare_num(num1, num2):
                if (num1 == num2):
                    return 0
                elif (num1 > num2):
                    return -1
                else:
                    return 1
            if (resample_freq == "Y"):
                return _compare_num(dt1.year, dt2.year)
            elif (resample_freq == "M"):
                if (_compare_datetime(dt1, dt2, "Y") == 0):
                    return _compare_num(dt1.month, dt2.month)
                else:
                    return _compare_datetime(dt1, dt2, "Y")
            elif (resample_freq == "D"):
                if (_compare_datetime(dt1, dt2, "M") == 0):
                    return _compare_num(dt1.day, dt2.day)
                else:
                    return _compare_datetime(dt1, dt2, "M")
            elif (resample_freq == "H"):
                if (_compare_datetime(dt1, dt2, "D") == 0):
                    return _compare_num(dt1.hour, dt2.hour)
                else:
                    return _compare_datetime(dt1, dt2, "D")
            else:
                raise ValueError(f"不支持的重采样频率: {resample_freq}")

        def _group_by_time(data, time_groups, aggreation_column, aggregation_method):
            # time_groups=[(index,datetime),...]
            grouped_data = []
            for group in time_groups:
                group_indices = [item[0] for item in group]
                group_data = data.iloc[group_indices]
                if (aggregation_method == "mean"):
                    aggreagated_value = group_data[aggreation_column].mean()
                elif (aggregation_method == "sum"):
                    aggreagated_value = group_data[aggreation_column].sum()
                else:
                    raise ValueError(f"不支持的聚合方法: {aggregation_method}")
                grouped_data.append(aggreagated_value)
            return grouped_data
        csv_data = self.train_csv_data if is_train else self.test_csv_data
        resample_freq = self.meta_item["resample_freq"]
        time_column = self.meta_item["time_column"]
        time_data = csv_data[time_column]
        # print(time_data)
        if (self.meta_item['name'] == "electric-power-consumption"):
            time_data = [datetime.strptime(t, '%m/%d/%Y %H:%M')
                         for t in time_data]

            # sorted_time_data=[(index,datetime),...]
            sorted_time_data = sorted(enumerate(time_data), key=lambda x: x[1])

            print("开始按时间聚合数据...")
            print(f"时间范围为:{sorted_time_data[0][1]}-{sorted_time_data[-1][1]}")
            cur_time = sorted_time_data[0][1]
            cur_group = []
            group_by_time = []
            for time in sorted_time_data:
                if (_compare_datetime(cur_time, time[1], resample_freq) == 0):
                    cur_group.append(time)
                else:
                    group_by_time.append(cur_group)
                    cur_group = [time]
                    cur_time = time[1]

            group_by_time.append(cur_group)
            print(f"总共有 {len(group_by_time)} 组数据按时间聚合。")
        elif (self.meta_item['name'] == "daily-climate-time-series-data"):
            time_data = [datetime.strptime(
                t, '%Y-%m-%d') for t in time_data]
            # sorted_time_data=[(index,datetime),...]
            sorted_time_data = sorted(enumerate(time_data), key=lambda x: x[1])

            print("开始按时间聚合数据...")
            print(f"时间范围为:{sorted_time_data[0][1]}-{sorted_time_data[-1][1]}")
            cur_time = sorted_time_data[0][1]
            cur_group = []
            group_by_time = []
            for time in sorted_time_data:
                if (_compare_datetime(cur_time, time[1], resample_freq) == 0):
                    cur_group.append(time)
                else:
                    group_by_time.append(cur_group)
                    cur_group = [time]
                    cur_time = time[1]
            # for group in group_by_time[:5]:
            #     print(group)
            group_by_time.append(cur_group)
            print(f"总共有 {len(group_by_time)} 组数据按时间聚合。")
        else:
            raise NotImplementedError(
                "当前仅支持electric-power-consumption,daily-climate-time-series-data 数据集的时间聚合。")

        aggreation_times = [group[0][1].timestamp() for group in group_by_time]

        data_dict = {'time': aggreation_times}
        for column, method in self.meta_item["aggregation"].items():
            # _group_by_time 返回的是该列按时间分组后的聚合结果
            data_dict[column] = _group_by_time(
                csv_data, group_by_time, column, method)

        new_data_frame = pd.DataFrame(data_dict)
        print("数据聚合完成。")
        pd.set_option('display.float_format', '{:.0f}'.format)
        print(new_data_frame.head())
        return new_data_frame

    # TODO:实现保存已划分的数据集的功能

    def save_dataset(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, self.dataset_name), exist_ok=True)
        self.train_csv_data.to_csv(os.path.join(
            save_path, f"{self.dataset_name}/train.csv"), index=False)
        self.test_csv_data.to_csv(os.path.join(
            save_path, f"{self.dataset_name}/test.csv"), index=False)


if __name__ == "__main__":
    dataset = Dataset('daily-climate-time-series-data')
    dataset = Dataset('electric-power-consumption')
    # print(dataset.csv_data.head())
