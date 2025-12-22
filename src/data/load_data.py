import json5
import pandas as pd
from omegaconf import OmegaConf
import os
class MyDataset():
    PROCESSED_PATH= "./data/processed_data"
    def __init__(self,dataset_name):
        super().__init__()
        self.meta_data=json5.load(open(os.path.join(MyDataset.PROCESSED_PATH,dataset_name,"meta.json5"),encoding="utf-8"))
        self.meta_data=OmegaConf.create(self.meta_data)
        self.train_conf=OmegaConf.create(json5.load(open(os.path.join(MyDataset.PROCESSED_PATH,dataset_name,"train_conf.json5"),encoding="utf-8")))
        self.dataset_name=dataset_name
        self.train_data=pd.read_csv(self.meta_data.train_csv)
        self.test_data=pd.read_csv(self.meta_data.test_csv)
        if(self.train_conf.input_size==None):
            self.train_conf.input_size=self.train_data.shape[-1]
        self.train_label=self.train_data[self.meta_data.target_columns]
        self.test_label=self.test_data[self.meta_data.target_columns]




        

        
        
if __name__=="__main__":
    dataset_name="electric-power-consumption"
    dataset=MyDataset(dataset_name)
    print(dataset.train_data.shape)
    print(dataset.test_data.shape)
    print(dataset.train_conf.input_size)
