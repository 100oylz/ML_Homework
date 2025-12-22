from src.data.load_data import MyDataset
from src.model import *
from src.train_model import train_model
from src.utils import plot_training_history

DATASET_NAME = [
    "daily-climate-time-series-data",
    "electric-power-consumption",
    "time-series-forecasting-with-yahoo-stock-price"
]

MODEL_NAME = [
    "RNN",
    "LSTM",
    "GRU",
    "Transformer"
]


def test_train_model():
    for dataset_name in DATASET_NAME:
        for model_name in MODEL_NAME:
            # Load the dataset
            dataset = MyDataset(dataset_name)
            if (model_name == "RNN"):
                # Define the model architecture
                model = RNN(input_size=dataset.train_conf.input_size, hidden_size=dataset.train_conf.hidden_size,
                            output_size=dataset.train_conf.output_size, num_layers=dataset.train_conf.num_layers, dropout=dataset.train_conf.dropout)
            elif (model_name == "LSTM"):
                model = LSTM(input_size=dataset.train_conf.input_size, hidden_size=dataset.train_conf.hidden_size,
                             output_size=dataset.train_conf.output_size, num_layers=dataset.train_conf.num_layers, dropout=dataset.train_conf.dropout)
            elif (model_name == "GRU"):
                model = GRU(input_size=dataset.train_conf.input_size, hidden_size=dataset.train_conf.hidden_size,
                            output_size=dataset.train_conf.output_size, num_layers=dataset.train_conf.num_layers, dropout=dataset.train_conf.dropout)
            elif (model_name == "Transformer"):
                model = Transformer(input_size=dataset.train_conf.input_size,
                                    hidden_size=dataset.train_conf.hidden_size,
                                    output_size=dataset.train_conf.output_size,
                                    num_layers=dataset.train_conf.num_layers, dropout=dataset.train_conf.dropout,num_heads=dataset.train_conf.n_heads)
            else:
                raise ValueError("Invalid model name")

            # Train the model
            model, train_history, val_history, history_metrics = train_model(model, dataset.train_data, dataset.train_label, dataset.test_data, dataset.test_label,
                                                                             dataset.train_conf.window_size, 
                                                                             epochs=dataset.train_conf.epochs, 
                                                                             batch_size=dataset.train_conf.batch_size, 
                                                                             learning_rate=dataset.train_conf.learning_rate,
                                                                             loss_function=dataset.train_conf.loss, 
                                                                             optimizer_name=dataset.train_conf.optimizer,
                                                                             dataset_name=dataset_name, 
                                                                             early_stopping_patience=dataset.train_conf.early_stopping_patience)
            plot_training_history(train_history, val_history, save_path=None,
                                  dataset_name=dataset_name, model_name=model_name)


if __name__ == "__main__":
    test_train_model()
