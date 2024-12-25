import os
import numpy as np
import pandas as pd
from collections import defaultdict
from autograd.tools.data import train_test_split
from autograd.tools.metrics import accuracy
from autograd.tools.trainer import Trainer
from autograd import nn, optim, functional


class DataLoader:
    def __init__(self, max_features: int, max_sequence_length: int) -> None:
        self.max_features = max_features
        self.max_sequence_length = max_sequence_length

    def create_vocabulary(self, texts):
        """
        Create a vocabulary (word->index) from given texts,
        keeping up to self.max_features most common words.
        """
        word_freq = defaultdict(int)
        for text in texts:
            for word in text.lower().split():
                word_freq[word] += 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        if self.max_features is not None:
            sorted_words = sorted_words[: self.max_features]

        # Create word->index mapping
        vocab = {word: idx for idx, (word, _) in enumerate(sorted_words)}
        return vocab

    def create_feature_matrix(self, texts, vocabulary):
        """
        Convert list of texts into a sequential feature matrix using the vocabulary.
        Shape: (batch_size, sequence_length, vocab_size)
        """
        # Initialize matrix with zeros
        matrix = np.zeros((len(texts), self.max_sequence_length), dtype=np.int32)

        for i, text in enumerate(texts):
            # Split text into words and convert to indices
            words = text.lower().split()
            # Truncate or pad sequence to max_sequence_length
            words = words[: self.max_sequence_length]  # Truncate if too long

            for j, word in enumerate(words):
                if word in vocabulary:
                    matrix[i, j] = vocabulary[word]

        # Convert to one-hot encoding
        # Shape: (batch_size, sequence_length, vocab_size)
        one_hot = np.zeros((len(texts), self.max_sequence_length, len(vocabulary)))
        for i in range(len(texts)):
            for j in range(self.max_sequence_length):
                if matrix[i, j] > 0:
                    one_hot[i, j, matrix[i, j]] = 1

        return one_hot

    def process_data(self):
        data = pd.read_csv("examples/IMDB Dataset.csv").to_numpy()
        vocab = self.create_vocabulary(data[:, 0])
        features = self.create_feature_matrix(data[:, 0], vocab)
        labels = np.array([1 if label == "positive" else 0 for label in data[:, 1]])

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.1
        )

        return X_train, X_test, y_train, y_test, vocab


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RecurrentBlock(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.rnn(x)
        x = self.dropout(x)
        x = self.batchnorm(x)
        x = self.fc(x)
        return functional.sigmoid(x)


if __name__ == "__main__":
    # Check if data exist
    if not os.path.exists("examples/IMDB Dataset.csv"):
        print("Downloading data...")
        url = "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
        os.system(f"kaggle datasets download -d {url} -p examples")
        os.system("unzip examples/imdb-dataset-of-50k-movie-reviews.zip -d examples")

    dl = DataLoader(max_features=4000, max_sequence_length=25)
    X_train, X_test, y_train, y_test, vocab = dl.process_data()

    model = RNN(input_size=len(vocab), hidden_size=32, output_size=1)

    trainer = Trainer(
        model,
        loss_fn=functional.binary_cross_entropy,
        optimizer=optim.Adam(model.parameters, lr=0.001),
        epochs=10,
        output_type="sigmoid",
    )

    print("Training Recurrent Neural Network for movie sentiment analysis...")
    trainer.fit(X_train, y_train)

    print("Evaluating model...")
    model.eval()
    y_pred = model(X_test).data
    # convert sigmoid to binary
    y_pred = (y_pred > 0.5).astype(int).squeeze()
    print(f"Test Accuracy: {accuracy(y_pred, y_test)}")
