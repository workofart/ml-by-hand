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
    """
    Processes raw text and sentiment label data for movie sentiment analysis.

    This function creates a vocabulary from the text column of the input data,
    converts the text into one-hot encoded features (with padding/truncation to a fixed length),
    and transforms the sentiment labels into binary values (1 for "positive", 0 otherwise).
    Finally, it splits the features and labels into training and testing sets.

    Args:
        data (np.ndarray): A 2D numpy array where the first column contains text and the second column contains sentiment labels.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
            - X_train: One-hot encoded training features.
            - X_test: One-hot encoded testing features.
            - y_train: Training labels as binary integers.
            - y_test: Testing labels as binary integers.
            - vocab: The vocabulary mapping words to integer indices.
    """
    vocab = create_vocabulary(data[:, 0], max_features=6000)
    features, _ = text_to_one_hot_and_sparse(data[:, 0], vocab, max_sequence_length=25)
    labels = np.array([1 if label == "positive" else 0 for label in data[:, 1]])

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)

    return X_train, X_test, y_train, y_test, vocab


class RNN(nn.Module):
    """
    A simple recurrent neural network (RNN) model for binary sentiment classification.

    The model consists of a recurrent block followed by batch normalization and a fully connected layer.
    A sigmoid activation is applied at the output to produce probabilities.

    Attributes:
        rnn (nn.RecurrentBlock): The recurrent layer for processing sequential input.
        batchnorm (nn.BatchNorm): Batch normalization layer applied to the recurrent output.
        fc (nn.Linear): Linear layer mapping the hidden state to the output.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the RNN model.

        Args:
            input_size (int): The dimensionality of the input features.
            hidden_size (int): The number of hidden units in the recurrent block.
            output_size (int): The dimensionality of the output.
        """
        super().__init__()
        self.rnn = nn.RecurrentBlock(input_size, hidden_size)
        self.batchnorm = nn.BatchNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Compute the forward pass of the RNN model.

        Args:
            x (np.ndarray): The input tensor for the recurrent network.

        Returns:
            np.ndarray: The output probability produced by applying sigmoid to the linear layer.
        """
        x = self.rnn(x)
        x = self.batchnorm(x)
        x = self.fc(x)
        return functional.sigmoid(x)


class LSTM(nn.Module):
    """
    A Long Short-Term Memory (LSTM) model for binary sentiment classification.

    This model employs an LSTM block to process sequential data, applies batch normalization,
    and maps the final hidden state to an output probability via a linear layer and a sigmoid activation.

    Attributes:
        rnn (nn.LongShortTermMemoryBlock): The LSTM block processing the input sequence.
        batchnorm (nn.BatchNorm): Batch normalization layer applied to the LSTM output.
        fc (nn.Linear): Linear layer mapping the hidden state to the output.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the LSTM model.

        Args:
            input_size (int): The size of the input feature vectors.
            hidden_size (int): The number of hidden units in the LSTM block.
            output_size (int): The dimensionality of the output.
        """
        super().__init__()
        self.rnn = nn.LongShortTermMemoryBlock(
            input_size, hidden_size, dropout_prob=0.5
        )
        self.batchnorm = nn.BatchNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Compute the forward pass of the LSTM model.

        Args:
            x (np.ndarray): The input tensor for the LSTM, expected to be sequential.

        Returns:
            np.ndarray: The output probability computed by applying a sigmoid activation.
        """
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
    Trains a movie sentiment analysis model using binary cross-entropy loss.

    The model is instantiated by the SimpleTrainer using the provided model class,
    optimizer (Adam), loss function (binary cross-entropy), and training configuration.
    The trainer then fits the model using the specified training and validation data loaders.

    Args:
        model_cls (type): The neural network model class to instantiate.
        train_data_loader (SimpleDataLoader): Data loader for the training data.
        test_data_loader (SimpleDataLoader): Data loader for the validation/testing data.
        config (GenericTrainingConfig): Configuration for training (epochs, learning rate, etc.).
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
    """
    Main script for training movie sentiment analysis models.

    The script performs the following steps:
      1) Checks if the IMDB Dataset CSV file exists locally; if not, downloads and extracts it.
      2) Reads the dataset using pandas and converts it to a NumPy array.
      3) Processes the data by creating a vocabulary from the review texts, converting the texts to one-hot
         encoded features (with fixed sequence length), and mapping sentiment labels to binary values.
      4) Splits the processed data into training and testing sets.
      5) Creates SimpleDataLoader objects for training and testing.
      6) Constructs training configurations for both RNN and LSTM models using GenericTrainingConfig.
      7) Trains the RNN model followed by the LSTM model using the main training function.
      8) Logs training progress, checkpoints, and evaluation metrics.
    """
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
