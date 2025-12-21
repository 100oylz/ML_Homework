import json5
import pandas as pd

import os
class MyDataset():
    PROCESSED_PATH="./data/processed_data"
    def __init__(self,dataset_name):
        super().__init__()
        self.meta_data=json5.load(open(os.path.join(MyDataset.PROCESSED_PATH,dataset_name,"meta_data.jsonc")))
        self.dataset_name=dataset_name
        self.train_data=pd.read_csv(self.meta_data["train_csv"])
        self.test_data=pd.read_csv(self.meta_data["test_csv"])
        self.input_size=len(self.meta_data["feature_columns"])
        self.output_size=len(self.meta_data["target_columns"])
        self.epochs=self.meta_data["epochs"]
        self.batch_size=self.meta_data["batch_size"]
        self.num_layers=self.meta_data["num_layers"]
        self.hidden_size=self.meta_data["hidden_size"]
        self.learning_rate=self.meta_data["learning_rate"]
        self.num_layers=self.meta_data["num_layers"]
        self.window_size=self.meta_data["window_size"]
        self.train_label=self.train_data[self.meta_data["target_columns"]]
        self.test_label=self.test_data[self.meta_data["target_columns"]]

        

        
        
if __name__=="__main__":
    dataset_name="electric-power-consumption"
    dataset=MyDataset(dataset_name)
    print(dataset[0])