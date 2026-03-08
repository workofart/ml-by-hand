"""Metrics for evaluating model predictions."""

from autograd.backend import xp


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
        >>> from autograd.backend import xp
        >>> y_pred = xp.array([1, 0, 1, 1])
        >>> y_true = xp.array([1, 1, 1, 0])
        >>> accuracy(y_pred, y_true)
        0.5
    """
    y_true = xp.array(y_true)
    y_pred = xp.array(y_pred)
    assert len(y_true) == len(y_pred)
    return float(xp.to_scalar(xp.sum(xp.array(y_pred == y_true)))) / len(y_true)


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
        >>> from autograd.backend import xp
        >>> y_pred = xp.array([1, 0, 1, 1])
        >>> y_true = xp.array([1, 1, 1, 0])
        >>> precision(y_pred, y_true)
        0.6666666666666666
    """
    assert len(y_true) == len(y_pred)
    y_true = xp.array(y_true)
    y_pred = xp.array(y_pred)
    true_positives = int(
        float(
            xp.to_scalar(
                xp.sum(xp.logical_and(xp.array(y_true == 1), xp.array(y_pred == 1)))
            )
        )
    )
    predicted_positives = int(float(xp.to_scalar(xp.sum(xp.array(y_pred == 1)))))
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
        >>> from autograd.backend import xp
        >>> y_pred = xp.array([2.5, 0.0, 2, 8])
        >>> y_true = xp.array([3.0, -0.5, 2, 7])
        >>> mean_squared_error(y_pred, y_true)
        0.375
    """
    y_true = xp.array(y_true)
    y_pred = xp.array(y_pred)
    assert len(y_true) == len(y_pred)
    return float(xp.to_scalar(xp.mean((y_pred - y_true) ** 2)))
