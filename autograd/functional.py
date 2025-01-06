from typing import Union, Optional
from autograd.tensor import Tensor, Function
import numpy as np
import logging

logger = logging.getLogger(__name__)


########### Activation Functions ###############
def relu(x: Tensor) -> Tensor:
    return Relu.apply(x)


def sigmoid(x: Tensor) -> Tensor:
    return Sigmoid.apply(x)


def softmax(x: Tensor) -> Tensor:
    return Softmax.apply(x)


def tanh(x: Tensor) -> Tensor:
    return Tanh.apply(x)


class Relu(Function):
    """
    Retified Linear Unit (ReLU) activation function.
    ReLU(x) = max(0, x)
    """

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, grad):
        # dL/dx = dL/dy * dy/dx
        return grad * (self.x > 0)


class Sigmoid(Function):
    def forward(self, x):
        # 709 is the maximum value that can be passed to np.exp without overflowing
        self.out = 1 / (1 + np.exp(np.clip(-x, -709, 709)))
        return self.out

    def backward(self, grad):
        # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        return grad * self.out * (1 - self.out)


class Softmax(Function):
    """
    Softmax activation function
    softmax(x) = e^x / sum(e^x)
    """

    def forward(self, x):
        # Subtract the maximum value for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.probs

    def backward(self, grad):
        # There are two cases for this gradient because each element in the matrix affects
        # every other elements' gradient due to the fact of sum(e^x) in the denominator.
        # Let's denote i, j as the ith and jth elements in the matrix.
        # Case 1: i == j
        # d(softmax(x))/dx_i = softmax(x)_i * (1[i==j] - softmax(x)_i)
        # Case 2: i != j
        # d(softmax(x))/dx_i = -softmax(x)_i * softmax(x)_j

        # dL/dx = y * (dL/dy - sum(dL/dy * y, axis=-1, keepdims=True))
        if isinstance(grad, Tensor):
            grad = grad.data
        sum_term = np.sum(grad * self.probs, axis=-1, keepdims=True)
        dLdx = self.probs * (grad - sum_term)
        return dLdx


class Tanh(Function):
    """
    Tanh activation function
    tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    """

    def forward(self, x):
        # For numerical stability, use the fact that tanh(x) = 2*sigmoid(2x) - 1
        # This avoids computing large exponentials directly
        x = 2 * x
        # Clip x to avoid overflow in exp(-x)
        x = np.clip(x, -88.72, 88.72)  # ln(max float32) ≈ 88.72
        sigmoid_2x = 1 / (1 + np.exp(-x))
        self.out = 2 * sigmoid_2x - 1
        return self.out

    def backward(self, grad):
        # d(tanh(x))/dx = 1 - tanh(x)^2
        return grad * (1 - self.out**2)


###################### Loss Functions #####################
class BinaryCrossEntropy(Function):
    """
    Binary Cross Entropy Loss
    Note: We assume the input y_pred contain probabilities not logits.
    -(x * log(y)) + (1 - x) * log(1 - y)
    """

    def forward(self, y_pred, y_true, **kwargs):
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)

        if y_true.ndim == 1 and y_pred.ndim == 1:
            pass
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError("y_pred and y_true must have the same shape")

        self.y_true = y_true
        self.y_pred_prob = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(
            y_true * np.log(np.clip(self.y_pred_prob, 1e-15, None))
            + (1 - y_true) * np.log(np.clip(1 - self.y_pred_prob, 1e-15, None))
        )
        return loss

    def backward(self, grad):
        # dL/dpred = -(y/p - (1-y)/(1-p))
        y_true = self.y_true
        y_pred_prob = self.y_pred_prob

        # Avoid division by zero by clipping probabilities away from 0 and 1
        y_pred_prob = np.clip(y_pred_prob, 1e-7, 1 - 1e-7)
        grad_y_pred = -((y_true / y_pred_prob) - ((1 - y_true) / (1 - y_pred_prob)))
        grad_y_pred /= len(y_pred_prob)
        # Incorporate the upstream gradient
        grad_y_pred *= grad.data

        return grad_y_pred, None  # y_true doesn't need gradient


class SparseCrossEntropy(Function):
    """
    Sparse Cross Entropy
    Note: this assumes y_pred contains probabilities in [0,1].
    If your model outputs logits, use sparse_cross_entropy_with_logits()
    which calls softmax internally.

    -y_true * log(y_pred)
    """

    def forward(self, y_pred, y_true, **kwargs):
        """
        y_pred: (batch_size, feature_dim) or (batch_size, sequence_length, feature_dim) of probabilities
        y_true: (batch_size,) or (batch_size, sequence_length) of integer labels
        """
        # Convert y_true to np array if it's a Tensor
        if isinstance(y_true, Tensor):
            y_true = y_true.data
        y_true = np.asarray(y_true, dtype=np.int64)
        if y_pred.min() < 0 or y_pred.max() > 1:
            raise ValueError("y_pred must contain probabilities between 0 and 1")

        self.y_true_shape = y_true.shape
        self.y_pred_shape = y_pred.shape

        # We'll store clipped probabilities for backward
        y_pred_prob = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_pred_prob = y_pred_prob

        # 2D Case
        # y_pred is (batch_size, feature_dim)
        # y_true is (batch_size,)
        if y_pred.ndim == 2:
            batch_size, feature_dim = y_pred.shape
            if y_true.shape != (batch_size,):
                raise ValueError(
                    f"For 2D y_pred, expect y_true of shape ({batch_size},), got {y_true.shape}"
                )

            # Gather the correct prob per sample
            selected_probs = y_pred_prob[
                np.arange(batch_size), y_true
            ]  # shape (batch_size,)

            loss = -np.mean(np.log(selected_probs))
            self.gather_index = (np.arange(batch_size), y_true)  # store for backward

        # 3D Case
        # y_pred is (batch_size, sequence_length, feature_dim)
        # y_true is (batch_size, sequence_length)
        elif y_pred.ndim == 3:
            batch_size, sequence_length, feature_dim = y_pred.shape
            if y_true.shape != (batch_size, sequence_length):
                raise ValueError(
                    f"For 3D y_pred, expect y_true of shape ({batch_size},{sequence_length}), got {y_true.shape}"
                )

            batch_idx = np.arange(batch_size)[:, None]  # (batch_size,1)
            time_idx = np.arange(sequence_length)[None, :]  # (1,sequence_length)
            selected_probs = y_pred_prob[
                batch_idx, time_idx, y_true
            ]  # shape (batch_size, sequence_length)

            loss = -np.mean(np.log(selected_probs))
            self.gather_index = (batch_idx, time_idx, y_true)  # store for backward

        else:
            raise ValueError("SparseCrossEntropy only supports 2D or 3D y_pred")

        return loss

    def backward(self, grad):
        """
        For 2D: grad_out[batch, label] = -1/(selected_probs * batch_size)
        For 3D: grad_out[batch, time, label] = -1/(selected_probs * batch_size * sequence_len)
        """
        grad = grad.data if isinstance(grad, Tensor) else grad

        # Create grad array with same shape as y_pred
        grad_out = np.zeros_like(self.y_pred_prob)
        y_pred_prob = self.y_pred_prob

        if len(self.y_pred_shape) == 2:
            batch_size, sequence_length = self.y_pred_shape
            # gather_index = (array([0,1,...,B-1]), array([...labels...]))
            selected_probs = y_pred_prob[self.gather_index]
            grad_values = -1.0 / (selected_probs * batch_size)  # average => divide by B
            grad_out[self.gather_index] = grad_values * grad

        elif len(self.y_pred_shape) == 3:
            batch_size, sequence_length, feature_dim = self.y_pred_shape
            # gather_index = (batch_idx, time_idx, labels)
            batch_idx, time_idx, labels = self.gather_index
            selected_probs = y_pred_prob[batch_idx, time_idx, labels]  # shape (B, T)
            grad_values = -1.0 / (
                selected_probs * batch_size * sequence_length
            )  # average => divide by B*T
            # Because gather_index are broadcast shapes, we can assign:
            grad_out[batch_idx, time_idx, labels] = grad_values * grad

        return grad_out, None


class CrossEntropyWithLogits(Function):
    """
    Categorical Cross Entropy with Logits:
    - For multi-class classification with *one-hot* targets
    - Expects logits: shape (Batch Size, # of Classes) or (Batch Size, Sequence Length, # of Classes)
    - Weight (optional): shape (Batch Size, Sequence Length), useful for masking
    - Targets: same shape (Batch Size, # of Classes) or (Batch Size, Sequence Length, # of Classes), each row is one-hot
    """

    def forward(self, y_pred_probs, targets, weight=None, **kwargs):
        self.y_pred_probs = Softmax().forward(y_pred_probs)
        self.targets = targets

        # Cross-entropy = -sum( y * log(probs) ), average
        log_p = np.log(self.y_pred_probs)
        # sum over last dimension
        numerator = -(self.targets * log_p).sum(axis=-1)
        if weight is not None:
            numerator *= weight
        loss_value = numerator.mean()
        return loss_value

    def backward(self, grad):
        """
        dL/dlogits = (probs - y)
        Then average over the batch/time dimension
        """
        # If there's a time dimension or more, we do a total_count = product of all but last dimension
        total_count = int(np.prod(self.y_pred_probs.shape[:-1]))

        # (probs - one_hot)
        grad_logits = (self.y_pred_probs - self.targets) / total_count
        grad_logits *= grad.data
        return grad_logits, None


class HingeLoss(Function):
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

    def forward(self, y_pred, y_true, reduction="none", **kwargs):
        if isinstance(y_true, Tensor):
            y_true = y_true.data
        y_true = np.asarray(y_true, dtype=np.float32)

        # Reshape y_true to match y_pred if needed
        if y_pred.shape != y_true.shape:
            y_true = y_true.reshape(y_pred.shape)

        self.y_true = y_true
        self.y_pred = y_pred
        self.reduction = reduction

        # hinge loss = max(0, 1 - y_true * y_pred)
        self.margins = 1 - y_true * y_pred
        loss_data = np.maximum(0, self.margins)

        if reduction == "mean":
            loss_data = np.mean(loss_data)
        elif reduction == "sum":
            loss_data = np.sum(loss_data)
        elif reduction == "none":
            pass
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
        return loss_data

    def backward(self, grad):
        """
        d (1/2||w||^2)/dw = w (we multiple 1/2 because it makes the gradient calculation easier)
        d(C * sum(max(0, 1 - y_true * y_pred)))/dw = C * max(0, 1 - y_true * y_pred)
        = 1 - y_true * y_pred (if y_true * y_pred < 1)
        = 0 (if y_true * y_pred >= 1)
        """
        grad = grad.data if isinstance(grad, Tensor) else grad
        # Initialize gradient array with same shape as predictions
        grad_y_pred = np.zeros_like(self.y_pred)

        # Where margin > 0, gradient is -y_true
        margin_violated = self.margins > 0
        grad_y_pred = np.where(margin_violated, -self.y_true, 0)

        if self.reduction == "mean":
            grad_y_pred /= self.y_pred.size

        # Handle scalar gradient (from mean/sum reduction)
        if np.isscalar(grad) or grad.size == 1:
            grad_y_pred *= grad
        else:
            # For elementwise gradient
            grad_y_pred *= grad.reshape(grad_y_pred.shape)

        return grad_y_pred, None


class MeanSquaredLoss(Function):
    def forward(self, y_pred, y_true, **kwargs):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, grad):
        """
        dL/dx = 2 * (x - y)
        """
        return 2 * (self.y_pred - self.y_true) * grad.data


def binary_cross_entropy(
    y_pred: Tensor, y_true: Union[Tensor, np.ndarray], **kwargs
) -> Tensor:
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return BinaryCrossEntropy.apply(y_pred, y_true, **kwargs)


def binary_cross_entropy_with_logits(
    y_pred: Tensor, y_true: Union[Tensor, np.ndarray], **kwargs
) -> Tensor:
    """
    Binary Cross Entropy Loss with logits input
    Use binary_cross_entropy if y_pred contain probabilities
    -(x * log(y)) + (1 - x) * log(1 - y)
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return binary_cross_entropy(sigmoid(y_pred), y_true, **kwargs)


def sparse_cross_entropy(
    y_pred: Tensor, y_true: Union[Tensor, np.ndarray], **kwargs
) -> Tensor:
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return SparseCrossEntropy.apply(y_pred, y_true, **kwargs)


def sparse_cross_entropy_with_logits(
    y_pred: Tensor, y_true: Union[Tensor, np.ndarray], **kwargs
) -> Tensor:
    """
    Sparse Cross Entropy with logits input
    Use sparse_cross_entropy if y_pred contains probabilities
    -y_true * log(y_pred)
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return sparse_cross_entropy(softmax(y_pred), y_true, **kwargs)


def cross_entropy_with_logits(
    logits: Tensor,
    y_true: Union[Tensor, np.ndarray],
    weight: Optional[np.ndarray] = None,
    **kwargs,
) -> Tensor:
    """
    For multi-class classification with one-hot y_true
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return CrossEntropyWithLogits.apply(logits, y_true, weight=weight)


def hinge_loss(
    y_pred: Tensor, y_true: Union[Tensor, np.ndarray], reduction: str = "none", **kwargs
) -> Tensor:
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return HingeLoss.apply(y_pred, y_true, reduction=reduction, **kwargs)


def mean_squared_loss(
    y_pred: Tensor, y_true: Union[Tensor, np.ndarray], **kwargs
) -> Tensor:
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return MeanSquaredLoss.apply(y_pred, y_true, **kwargs)
