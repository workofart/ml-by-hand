import os
import numpy as np
import pandas as pd
from autograd.tools.data import (
    train_test_split,
)
from autograd.text.utils import create_vocabulary, text_to_one_hot_and_sparse
from autograd.tools.data import SimpleDataLoader
from autograd.tools.metrics import accuracy
from autograd.tools.trainer import SimpleTrainer
from autograd import nn, optim, functional


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
    model: nn.Module,
    train_data_loader: SimpleDataLoader,
    test_data_loader: SimpleDataLoader,
):
    trainer = SimpleTrainer(
        model,
        loss_fn=functional.binary_cross_entropy,
        optimizer=optim.Adam(model.parameters, lr=0.001),
        epochs=15,
        output_type="sigmoid",
    )

    print(
        f"Training {model.__class__.__name__} Neural Network for movie sentiment analysis..."
    )

    trainer.fit(train_loader=train_data_loader, test_loader=test_data_loader)

    # print("Evaluating model...")
    for X_test, y_test in test_data_loader:
        model.eval()
        y_pred = model(X_test).data
        # convert sigmoid to binary
        y_pred = (y_pred > 0.5).astype(int).squeeze()
        print(f"Test Accuracy: {accuracy(y_pred, y_test)}")


if __name__ == "__main__":
    # Check if data exist
    if not os.path.exists("examples/IMDB Dataset.csv"):
        print("Downloading data...")
        os.system(
            "curl -L -o examples/imdb-dataset-of-50k-movie-reviews.zip https://www.kaggle.com/api/v1/datasets/download/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
        )
        os.system("unzip examples/imdb-dataset-of-50k-movie-reviews.zip -d examples")

    data = pd.read_csv("examples/IMDB Dataset.csv").to_numpy()
    X_train, X_test, y_train, y_test, vocab = process_data(data)
    train_data_loader = SimpleDataLoader(X_train, y_train, batch_size=32, shuffle=True)
    test_data_loader = SimpleDataLoader(X_test, y_test, batch_size=32, shuffle=False)

    model = RNN(input_size=len(vocab), hidden_size=32, output_size=1)
    main(model, train_data_loader, test_data_loader)

    model = LSTM(input_size=len(vocab), hidden_size=64, output_size=1)
    main(model, train_data_loader, test_data_loader)
