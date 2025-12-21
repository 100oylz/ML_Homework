from src.train_model import train_model
from src.model import *
from load_data import MyDataset

DATASET_NAME=[
    "daily-climate-time-series-data",
    "electric-power-consumption"
]

MODEL_NAME=[
    "RNN"
]

def test_train_model():
    for dataset_name in DATASET_NAME:
        # Load the dataset
        dataset = MyDataset(dataset_name)

        # Define the model architecture
        model = RNN(input_size=dataset.input_size, hidden_size=dataset.hidden_size, output_size=dataset.output_size,num_layers=6)

        # Train the model
        train_model(model, dataset.train_data,dataset.train_label,dataset.test_data,dataset.test_label, dataset.window_size,epochs=dataset.epochs, batch_size=dataset.batch_size, learning_rate=0.001)

if __name__ == "__main__":
    test_train_model()