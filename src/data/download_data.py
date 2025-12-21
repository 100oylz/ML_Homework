import kagglehub
import os
import shutil
from pathlib import Path

# 定义基础数据目录
BASIC_PATH = "../../data/kaggle_data"


def get_kagglehub_dataset(handle: str, base_dir: str):

    print(f"尝试下载数据集: {handle}...")

    # 1. 下载数据集到默认缓存目录
    # path 返回的是数据集在本地缓存的路径 (如: ~/.cache/kagglehub/...)
    source_path = kagglehub.dataset_download(handle)

    # 2. 确定数据集名称和目标路径
    # 例如：electric-power-consumption
    dataset_name = handle.split('/')[-1]

    # 目标路径：./data/kaggle_data/electric-power-consumption
    destination_path = Path(base_dir) / dataset_name

    # 3. 确保目标目录存在
    os.makedirs(destination_path, exist_ok=True)

    # 4. 剪切/移动文件到目标路径
    # 注意：kagglehub下载的是一个包含数据集文件的目录
    # 我们需要将缓存目录下的内容移动到目标目录

    # 检查源路径是否是目录，并获取其下的所有内容
    if os.path.isdir(source_path):
        # 移动源目录下的所有文件/文件夹到目标目录
        for item in os.listdir(source_path):
            s = Path(source_path) / item
            d = Path(destination_path) / item

            # 使用 shutil.move 进行剪切粘贴
            shutil.move(str(s), str(d))

        # 移动完成后，删除空的缓存目录
        os.rmdir(source_path)

    print(f"✅ 成功移动 {handle} 文件！")
    print(f"📦 文件最终保存路径: {destination_path.resolve()}")
    return destination_path


dataset_handle = [
    "rohitsahoo/sales-forecasting", 
    "fedesoriano/electric-power-consumption",
    "arashnic/time-series-forecasting-with-yahoo-stock-price",
    "sumanthvrao/daily-climate-time-series-data"
]

if __name__ == "__main__":

    # ❗ 必须先确保你的 Kaggle 认证文件 (kaggle.json) 配置正确
    print("--- 启动数据集下载任务 ---")
    for handle in dataset_handle:
        try:
            get_kagglehub_dataset(handle, BASIC_PATH)
        except Exception as e:
            print(f"❌ 下载或移动 {handle} 失败。请检查 Kaggle 认证或数据集 ID。错误信息：{e}")
    print("--- 任务完成 ---")
