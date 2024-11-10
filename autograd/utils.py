import logging
from autograd import nn
import numpy as np

logger = logging.getLogger(__name__)


def train_test_split(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state or 1337)

    num_samples = len(X)
    num_test = int(num_samples * test_size)

    # Create a random permutation of indices
    indices = np.random.permutation(num_samples)

    # Use array indexing to split the data
    X_train, X_test = X[indices[num_test:]], X[indices[:num_test]]
    y_train, y_test = y[indices[num_test:]], y[indices[:num_test]]

    return X_train, X_test, y_train, y_test


def accuracy(y_true, y_pred):
    """
    Accuracy = (True Positives + True Negatives) / Total Predictions
    """
    assert len(y_true) == len(y_pred)
    return np.sum((y_true == y_pred).astype(int)) / len(y_true)


def precision(y_true, y_pred):
    """
    Precision = True Positives / (True Positives + False Positives)
    """
    assert len(y_true) == len(y_pred)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives != 0 else 0.0


def train(
    model: nn.Module, X: np.ndarray, y: np.ndarray, loss_fn, optimizer, epochs=100
):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # shuffle X and y
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        if epoch in range(0, epochs, max(1, epochs // 10)) or epoch == epochs - 1:
            logger.info(f"Epoch: {epoch}, Loss: {loss.data:.2f}")

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
