"""
Metrics for evaluating model predictions.
"""

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore

    _ = np.cuda.runtime.getDeviceCount()  # Check if a CUDA device is available
except Exception:
    import numpy as np


def accuracy(y_pred, y_true):
    """Computes the accuracy of predictions.

    This function calculates the proportion of correctly predicted labels.

    Args:
        y_pred (array-like): Predicted labels.
        y_true (array-like): Ground-truth labels.

    Returns:
        float: The accuracy, defined as the fraction of predictions that match the labels.

    Raises:
        AssertionError: If the length of y_pred and y_true differ.

    Example:
        >>> import numpy as np
        >>> y_pred = np.array([1, 0, 1, 1])
        >>> y_true = np.array([1, 1, 1, 0])
        >>> accuracy(y_pred, y_true)
        0.5
    """
    assert len(y_true) == len(y_pred)
    return np.sum(y_pred == y_true) / len(y_true)


def precision(y_pred, y_true):
    """Computes the precision of binary predictions.

    Precision is defined as the fraction of positive predictions that were actually correct.

    Args:
        y_pred (array-like): Predicted labels (binary).
        y_true (array-like): Ground-truth labels (binary).

    Returns:
        float: The precision, ranging from 0.0 to 1.0. Returns 0.0 if there are no predicted positives.

    Raises:
        AssertionError: If the length of y_pred and y_true differ.

    Example:
        >>> import numpy as np
        >>> y_pred = np.array([1, 0, 1, 1])
        >>> y_true = np.array([1, 1, 1, 0])
        >>> precision(y_pred, y_true)
        0.6666666666666666
    """
    assert len(y_true) == len(y_pred)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives != 0 else 0.0


def mean_squared_error(y_pred, y_true):
    """Computes the mean squared error (MSE) between predictions and ground truth.

    This function calculates the average of the squared differences between y_pred and y_true.

    Args:
        y_pred (array-like): Predicted values.
        y_true (array-like): Ground-truth values.

    Returns:
        float: The mean squared error.

    Raises:
        AssertionError: If the length of y_pred and y_true differ.

    Example:
        >>> import numpy as np
        >>> y_pred = np.array([2.5, 0.0, 2, 8])
        >>> y_true = np.array([3.0, -0.5, 2, 7])
        >>> mean_squared_error(y_pred, y_true)
        0.375
    """
    assert len(y_true) == len(y_pred)
    return np.mean((y_pred - y_true) ** 2)
