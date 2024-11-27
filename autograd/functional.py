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

    out = x.maximum(0)

    def _backward():
        if out.grad is None:
            return

        # dL/dx = dL/dy * dy/dx
        x._accumulate_grad(out.grad * (x.data > 0))

    out._backward = _backward
    return out


def sigmoid(x: Tensor) -> Tensor:
    """
    Sigmoid activation function
    """
    # 709 is the maximum value that can be passed to np.exp without overflowing
    out = Tensor(1 / (1 + np.exp(np.clip(-x.data, -709, 709))), prev={x})

    def _backward():
        if out.grad is None:
            return

        # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        x._accumulate_grad(out.grad * out.data * (1 - out.data))
        logger.debug("Sigmoid backward shapes:")
        logger.debug(f"out.grad shape: {out.grad.shape}")
        logger.debug(f"out.data shape: {out.data.shape}")
        logger.debug(f"x.grad shape: {x.grad.shape}")
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
    out = Tensor(probs, prev={x})

    def _backward():
        if out.grad is None:
            return
        # There are two cases for this gradient because each element in the matrix affects
        # every other elements' gradient due to the fact of sum(e^x) in the denominator.
        # Let's denote i, j as the ith and jth elements in the matrix.
        # Case 1: i == j
        # d(softmax(x))/dx_i = softmax(x)_i * (1[i==j] - softmax(x)_i)
        # Case 2: i != j
        # d(softmax(x))/dx_i = -softmax(x)_i * softmax(x)_j

        # Vectorized computation using broadcasting
        S = probs[..., None, :]  # Add dimension for broadcasting
        # We create the identify matrix to represent 1[i==j]
        grad = S * (np.eye(x.data.shape[-1]) - S.transpose(0, 2, 1))
        # We use einsum to compute the gradient
        # Accumulate the gradient for the input tensor x by using the Einstein summation convention.
        # The einsum function computes the gradient of the softmax function with respect to its input.
        # 'bi' refers to the batch index (b) and the output index (i) of the gradient (out.grad.data).
        # 'bij' refers to the batch index (b), the output index (i), and the input index (j) of the Jacobian matrix (grad).
        # The result is a tensor of shape (batch_size, j) which represents the accumulated gradient for each input.
        x._accumulate_grad(np.einsum("bi,bij->bj", out.grad.data, grad))

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

    if y_true.ndim == 1 and y_pred.data.ndim == 1:
        pass
    elif y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

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
        if out.grad is None:
            return

        # dL/dpred = -(y/p - (1-y)/(1-p))
        logger.debug(f"y_true shape: {y_true.shape}")
        logger.debug(f"y_pred shape: {y_pred.data.shape}")
        y_pred._accumulate_grad(
            -(y_true / y_pred_prob - (1 - y_true) / (1 - y_pred_prob))
            / len(y_pred_prob)
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
        if out.grad is None:
            return

        # Vectorized gradient computation
        grad = np.zeros_like(y_pred_prob)
        grad[range(n_samples), y_true] = -1.0 / (selected_probs * n_samples)
        y_pred._accumulate_grad(
            grad * out.grad.data[..., None]
        )  # Add dimension for broadcasting

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


def hinge_loss(
    y_pred: Tensor, y_true: Union[Tensor, np.ndarray], reduction: str = "none"
) -> Tensor:
    """
    Hinge Loss
    If the point is correctly classified, y_pred * y_true > 1, the loss is 0 (loss functions typically don't go into the negatives so we take the max of 0 and 1 - y_true * y_pred)
    Otherwise, y_pred * y_true < 1, then the loss is 1 - y_true * y_pred > 0.

    loss = max(0, 1 - y_true * y_pred)

    Objective Function: ||w||^2 + C * sum(max(0, 1 - y_true * y_pred))
    where:
        C is a hyperparameter that controls the trade-off between maximizing the margin (through regularization) and minimizing the loss.
        w is the weight vector (||w||^2 is the regularization term)

    Paper: https://ieeexplore.ieee.org/document/708428
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)

    loss = relu(1 - y_true * y_pred)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction != "none":
        raise ValueError(
            f"Invalid reduction option: {reduction}. Choose from 'none', 'mean', or 'sum'"
        )

    def _backward():
        if loss.grad is None:
            return
        """
        d (1/2||w||^2)/dw = w (we multiple 1/2 because it makes the gradient calculation easier)
        d(C * sum(max(0, 1 - y_true * y_pred)))/dw = C * max(0, 1 - y_true * y_pred)
        = 1 - y_true * y_pred (if y_true * y_pred < 1)
        = 0 (if y_true * y_pred >= 1)
        """
        margin_violated = 1 - y_true.data * y_pred.data > 0
        grad = -y_true.data * margin_violated
        if reduction == "mean":
            grad = grad / len(grad)
        y_pred._accumulate_grad(loss.grad * grad)

    loss._backward = _backward
    return loss
