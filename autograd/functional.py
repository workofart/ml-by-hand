from typing import Union
from autograd.tensor import Tensor
import numpy as np
import logging

logger = logging.getLogger(__name__)

########### Activation Functions ###############


def relu(x: Tensor) -> Tensor:
    """
    Retified Linear Unit (ReLU) activation function.
    ReLU(x) = max(0, x)
    """

    out = Tensor(np.maximum(0, x.data), prev=(x,))

    def _backward():
        # dL/dx = dL/dy * dy/dx
        x.grad += out.grad * (x.data > 0)

    out._backward = _backward
    return out


def sigmoid(x: Tensor) -> Tensor:
    """
    Sigmoid activation function
    """
    # 709 is the maximum value that can be passed to np.exp without overflowing
    out = Tensor(1 / (1 + np.exp(np.clip(-x.data, -709, 709))), prev=(x,))

    def _backward():
        logger.debug("Sigmoid backward shapes:")
        logger.debug(f"out.grad shape: {out.grad.shape}")
        logger.debug(f"out.data shape: {out.data.shape}")
        logger.debug(f"x.grad shape: {x.grad.shape}")
        # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        x.grad += out.grad * out.data * (1 - out.data)
        logger.debug(f"After backward x.grad shape: {x.grad.shape}")

    out._backward = _backward
    return out


def softmax(x: Tensor) -> Tensor:
    """
    Softmax activation function
    softmax(x) = e^x / sum(e^x)
    """
    # Subtract the maximum value for numerical stability
    exp_x = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
    probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    out = Tensor(probs, prev=(x,))

    def _backward():
        # There are two cases for this gradient because each element in the matrix affects
        # every other elements' gradient due to the fact of sum(e^x) in the denominator.
        # Let's denote i, j as the ith and jth elements in the matrix.
        # Case 1: i == j
        # d(softmax(x))/dx_i = softmax(x)_i * (1[i==j] - softmax(x)_i)
        # Case 2: i != j
        # d(softmax(x))/dx_i = -softmax(x)_i * softmax(x)_j
        # We first create the identify matrix to represent 1[i==j]
        identity_matrix = np.eye(x.data.shape[1])  # number of classes as shape

        # For each sample in batch
        for sample_idx in range(x.data.shape[0]):
            # Compute grad using broadcasting
            grad = probs[sample_idx][:, None] * (
                identity_matrix - probs[sample_idx][None, :]
            )
            x.grad[sample_idx] += out.grad[sample_idx] @ grad

    out._backward = _backward
    return out


###################### Loss Functions #####################
def binary_cross_entropy(y_pred: Tensor, y_true: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Binary Cross Entropy Loss
    Note: We assume the input y_pred contain probabilities not logits.
    -(x * log(y)) + (1 - x) * log(1 - y)
    """
    if y_pred.data.min() < 0 or y_pred.data.max() > 1:
        raise ValueError("y_pred must contain probabilities between 0 and 1")

    y_true = np.array(y_true.data) if isinstance(y_true, Tensor) else y_true
    if y_pred.data.shape[0] != y_true.shape[0]:
        raise ValueError("y_pred and y_true must have the same shape")

    # Clip probabilities to prevent log(0)
    y_pred_prob = np.clip(y_pred.data, 1e-15, 1 - 1e-15)

    # compute the loss
    out = Tensor(
        data=-np.mean(
            y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob)
        ),
        prev=(
            y_pred,
        ),  # this is very important to connect the loss tensor with the y_pred tensor
    )

    def _backward():
        # dL/dpred = -(y/p - (1-y)/(1-p))
        logger.debug(f"y_true shape: {y_true.shape}")
        logger.debug(f"y_pred shape: {y_pred.data.shape}")
        y_pred.grad += -(y_true / y_pred_prob - (1 - y_true) / (1 - y_pred_prob)) / len(
            y_pred_prob
        )
        logger.debug(f"y_pred.grad shape after: {y_pred.grad.shape}")

    out._backward = _backward
    return out


def binary_cross_entropy_with_logits(
    y_pred: Tensor, y_true: Union[Tensor, np.ndarray]
) -> Tensor:
    """
    Binary Cross Entropy Loss with logits input
    Use binary_cross_entropy if y_pred contain probabilities
    -(x * log(y)) + (1 - x) * log(1 - y)
    """
    return binary_cross_entropy(sigmoid(y_pred), y_true)


def sparse_cross_entropy(y_pred: Tensor, y_true: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Sparse Cross Entropy
    Note: this assumes y_pred contains probabilities, not logits
    -y_true * log(y_pred)
    """
    # Input validation
    if y_pred.data.min() < 0 or y_pred.data.max() > 1:
        raise ValueError("y_pred must contain probabilities between 0 and 1")

    y_true = (
        np.array(y_true.data, dtype=np.float64)
        if isinstance(y_true, Tensor)
        else y_true
    )
    n_samples = len(y_true)

    # Clip probabilities to prevent log(0)
    y_pred_prob = np.clip(y_pred.data, 1e-15, 1 - 1e-15)

    # Calculate cross entropy directly using the true class probabilities
    selected_probs = y_pred_prob[range(n_samples), y_true]
    loss = -np.mean(np.log(selected_probs))

    out = Tensor(data=loss, prev=(y_pred,))

    def _backward():
        grad = np.zeros_like(y_pred_prob)
        grad[range(n_samples), y_true] = -1.0 / selected_probs
        y_pred.grad += grad / n_samples

    out._backward = _backward
    return out


def sparse_cross_entropy_with_logits(
    y_pred: Tensor, y_true: Union[Tensor, np.ndarray]
) -> Tensor:
    """
    Sparse Cross Entropy with logits input
    Use sparse_cross_entropy if y_pred contains probabilities
    -y_true * log(y_pred)
    """
    return sparse_cross_entropy(softmax(y_pred), y_true)
