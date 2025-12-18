import json5
import pandas as pd
from torch.utils.data.dataset import Dataset
import os
class MyDataset(Dataset):
    PROCESSED_PATH="./data/processed_data"
    def __init__(self,dataset_name):
        super().__init__()
        self.dataset_name=dataset_name
        self.train_data=pd.read_csv(os.path.join(MyDataset.PROCESSED_PATH,dataset_name,"train.csv"))
        self.test_data=pd.read_csv(os.path.join(MyDataset.PROCESSED_PATH,dataset_name,"test.csv"))
        
        print(self.train_data.head())
        print(self.test_data.head())
        
        
        
if __name__=="__main__":
    dataset_name="electric-power-consumption"
    dataset=MyDataset(dataset_name)