from src.train_model import train_model
from src.model import *
from src.data.load_data import MyDataset
from src.utils import plot_training_history

DATASET_NAME = [
    "daily-climate-time-series-data",
    "electric-power-consumption"
]

MODEL_NAME = [
    "RNN",
    "LSTM",
    "GRU"
]


def test_train_model():
    for dataset_name in DATASET_NAME:
        for model_name in MODEL_NAME:
            # Load the dataset
            dataset = MyDataset(dataset_name)
            if (model_name == "RNN"):
                # Define the model architecture
                model = RNN(input_size=dataset.input_size, hidden_size=dataset.hidden_size,
                            output_size=dataset.output_size, num_layers=dataset.num_layers, dropout=dataset.dropout)
            elif (model_name == "LSTM"):
                model = LSTM(input_size=dataset.input_size, hidden_size=dataset.hidden_size,
                             output_size=dataset.output_size, num_layers=dataset.num_layers, dropout=dataset.dropout)
            elif (model_name == "GRU"):
                model = GRU(input_size=dataset.input_size, hidden_size=dataset.hidden_size,
                            output_size=dataset.output_size, num_layers=dataset.num_layers, dropout=dataset.dropout)
            else:
                raise ValueError("Invalid model name")

            # Train the model
            model, train_history, val_history = train_model(model, dataset.train_data, dataset.train_label, dataset.test_data, dataset.test_label,
                                                            dataset.window_size, epochs=dataset.epochs, batch_size=dataset.batch_size, learning_rate=0.001, dataset_name=dataset_name)
            plot_training_history(train_history, val_history,save_path=None,
                                  dataset_name=dataset_name, model_name=model_name)


if __name__ == "__main__":
    test_train_model()
