import numpy as np


def accuracy(y_pred, y_true):
    assert len(y_true) == len(y_pred)
    return np.sum(y_pred == y_true) / len(y_true)


def precision(y_pred, y_true):
    assert len(y_true) == len(y_pred)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives != 0 else 0.0


def mean_squared_error(y_pred, y_true):
    assert len(y_true) == len(y_pred)
    return np.mean((y_pred - y_true) ** 2)
