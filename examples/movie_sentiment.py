import os

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore

    _ = np.cuda.runtime.getDeviceCount()  # Check if a CUDA device is available
except Exception:
    import numpy as np
import logging

import pandas as pd

from autograd import functional, nn, optim
from autograd.text.utils import create_vocabulary, text_to_one_hot_and_sparse
from autograd.tools.config_schema import GenericTrainingConfig
from autograd.tools.data import (
    SimpleDataLoader,
    train_test_split,
)
from autograd.tools.trainer import SimpleTrainer


def process_data(data: np.ndarray):
    vocab = create_vocabulary(data[:, 0], max_features=6000)
    features, _ = text_to_one_hot_and_sparse(data[:, 0], vocab, max_sequence_length=25)
    labels = np.array([1 if label == "positive" else 0 for label in data[:, 1]])

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)

    return X_train, X_test, y_train, y_test, vocab


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RecurrentBlock(input_size, hidden_size)
        self.batchnorm = nn.BatchNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.rnn(x)
        x = self.batchnorm(x)
        x = self.fc(x)
        return functional.sigmoid(x)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LongShortTermMemoryBlock(
            input_size, hidden_size, dropout_prob=0.5
        )
        self.batchnorm = nn.BatchNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, C_t = self.rnn(x)  # hidden_state, last_cell_state
        x = self.batchnorm(x)
        x = self.fc(x)
        return functional.sigmoid(x)


def main(
    model_cls: type,
    train_data_loader: SimpleDataLoader,
    test_data_loader: SimpleDataLoader,
    config: GenericTrainingConfig,
):
    """
    Trains a movie sentiment analysis model using binary cross-entropy.
    The model is created by the SimpleTrainer using the provided model class,
    optimizer class, loss function and configuration.
    """
    logger.info(
        f"Training {model_cls.__name__} Neural Network for movie sentiment analysis..."
    )
    trainer = SimpleTrainer(
        model_cls=model_cls,
        optimizer_cls=optim.Adam,
        loss_fn=functional.binary_cross_entropy,
        output_type="sigmoid",
        config=config,
    )

    trainer.fit(train_data_loader=train_data_loader, val_data_loader=test_data_loader)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    # Check if data exist; if not, download and extract.
    if not os.path.exists("training_data/IMDB Dataset.csv"):
        print("Downloading data...")
        os.system(
            "curl -L -o training_data/imdb-dataset-of-50k-movie-reviews.zip "
            "https://www.kaggle.com/api/v1/datasets/download/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
        )
        os.system(
            "unzip training_data/imdb-dataset-of-50k-movie-reviews.zip -d training_data"
        )

    # Process the data (assume process_data returns train/test splits and a vocabulary)
    data = pd.read_csv("training_data/IMDB Dataset.csv").to_numpy()
    X_train, X_test, y_train, y_test, vocab = process_data(data)
    train_data_loader = SimpleDataLoader(X_train, y_train, batch_size=32, shuffle=True)
    test_data_loader = SimpleDataLoader(X_test, y_test, batch_size=32, shuffle=False)

    # Train the RNN model.
    config_rnn = GenericTrainingConfig(
        training_run_name="movie_sentiment_rnn",
        total_epochs=15,
        checkpoint_freq=15,
        model_kwargs={"input_size": len(vocab), "hidden_size": 32, "output_size": 1},
        optimizer_kwargs={"lr": 0.001},
    )
    main(RNN, train_data_loader, test_data_loader, config_rnn)

    # Train the LSTM model.
    config_lstm = GenericTrainingConfig(
        training_run_name="movie_sentiment_rnn",
        total_epochs=15,
        checkpoint_freq=15,
        model_kwargs={"input_size": len(vocab), "hidden_size": 64, "output_size": 1},
        optimizer_kwargs={"lr": 0.001, "max_grad_norm": 1.0},
    )
    main(LSTM, train_data_loader, test_data_loader, config_lstm)
