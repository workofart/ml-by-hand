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
    Sparse cross-entropy for 2D or 3D predictions with optional pad_idx ignoring.
    """

    def forward(self, y_pred, y_true, pad_idx=0, label_smoothing=0.0, **kwargs):
        """
        Args:
            - If y_pred is (batch_size, feature_dim), y_true is (batch_size,)
            - If y_pred is (batch_size, seq_len, feature_dim), y_true is (batch_size, seq_len)
            - y_pred must be probabilities in [0,1], not raw logits.
            - pad_idx (int, optional): The padding index, we will mask this in the loss calculation. Defaults to 0.
            - label_smoothing (float, optional): How much weight to put on non-correct classes. Defaults to 0.0.
                "Rethinking the Inception Architecture for Computer Vision"
                Label Smoothing Paper: https://arxiv.org/abs/1512.00567
        """
        # 1) Ensure y_pred is a valid probability distribution.
        if (y_pred.min() < 0) or (y_pred.max() > 1):
            raise ValueError("y_pred must contain probabilities in [0, 1].")

        # 2) Convert y_true to NumPy if it’s a Tensor.
        if isinstance(y_true, Tensor):
            y_true = y_true.data
        y_true = np.asarray(y_true, dtype=np.int64)

        # 3) Clip probabilities to avoid log(0).
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # 4) Flatten if 3D, store original shape for backward.
        self.original_shape = y_pred.shape
        if y_pred.ndim == 3:
            batch_size, seq_len, num_classes = y_pred.shape
            y_pred = y_pred.reshape(batch_size * seq_len, num_classes)
            y_true = y_true.reshape(batch_size * seq_len)
        else:
            batch_size, num_classes = y_pred.shape

        # 5) Create a mask for non-pad positions.
        self.non_pad_mask = y_true != pad_idx
        non_pad_idx = np.where(self.non_pad_mask)[0]

        # 6) Compute the smoothed cross-entropy loss for each element.
        # $$ L_i = -\left( (1 - \text{label\_smoothing}) \log p_{i,y_i} + \frac{\text{label\_smoothing}}{\text{num\_classes} - 1} \sum_{j \neq y_i} \log p_{i,j} \right) $$
        # $$ = -\left( (1 - \text{label\_smoothing}) \log p_{i,y_i} + \frac{\text{label\_smoothing}}{\text{num\_classes} - 1} \left( \sum_{j=1}^{\text{num\_classes}} \log p_{i,j} - \log p_{i,y_i} \right) \right) $$
        # Essentially, we putting some (label_smoothing) weight on a uniform distribution over the non-correct classes
        # This will make the model prediction les confident, and thus less likely to overfit.
        idx = np.arange(len(y_true))
        log_p = np.log(y_pred)  # shape (batch_size * seq_len, num_classes)
        log_p_correct = log_p[idx, y_true]  # shape (bath_size * seq_len,)
        sum_log_p = np.sum(log_p, axis=1)  # sum over classes
        losses = -(
            (1.0 - label_smoothing) * log_p_correct
            + (label_smoothing / (num_classes - 1)) * (sum_log_p - log_p_correct)
        )

        # 7) Average loss only over non-pad positions.
        loss_val = np.mean(losses[non_pad_idx])

        # 8) Store for backward.
        self.y_pred_prob = y_pred
        self.y_true_flat = y_true
        self.pad_idx = pad_idx
        self.label_smoothing = label_smoothing
        return loss_val

    def backward(self, grad):
        """
        Backprop for:
          $$L_i = -\left( (1 - \text{label\_smoothing}) \log p_{correct} + \frac{\text{label\_smoothing}}{c - 1} \left( \sum_{j=1}^{c} \log p_{i,j} - \log p_{i,y_i} \right) \right) $$
          $$ \partial{L}/\partial{p_{correct}} = -(1-\text{label\_smoothing}) * (1/p_{correct}) $$
          $$ \partial{L}/\partial{p_j} (j \neq c) = -(\text{label\_smoothing}/(c-1)) * (1/p_j) $$
          where c is the number of classes
        Then multiply by (grad / #non_pad).
        """
        grad = grad.data if isinstance(grad, Tensor) else grad
        y_pred = self.y_pred_prob
        y_true = self.y_true_flat
        batch_size_times_seq_length, c = y_pred.shape

        # Prepare output gradient array
        grad_out = np.zeros_like(y_pred)
        non_pad_idx = np.where(self.non_pad_mask)[0]
        count_non_pad = max(1, len(non_pad_idx))

        # For each position i in non-pad, define partial derivatives.
        # We'll do it in 2 steps for clarity:
        # Step 1: For all classes j, add -(label_smoothing/(c-1))*1/p_j
        grad_out[non_pad_idx, :] = (
            -(self.label_smoothing / (c - 1)) / y_pred[non_pad_idx, :]
        )

        # Step 2: For the correct class c_i, add extra -(1 - label_smoothing)*(1/p_correct)
        #         minus the portion we already added in step 1 for that class.
        correct_classes = y_true[non_pad_idx]  # shape (num_non_pad,)
        idx = (non_pad_idx, correct_classes)  # row indices, col indices
        grad_out[idx] += (
            -(1.0 - self.label_smoothing) / y_pred[idx]
        )  # add the correct-class term
        # (No need to "undo" anything because we used +=)

        # Scale
        grad_out *= grad / count_non_pad

        # Reshape to original shape if 3D
        grad_out = grad_out.reshape(self.original_shape)
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
    y_pred: Tensor, y_true: Union[Tensor, np.ndarray], pad_idx: int = None, **kwargs
) -> Tensor:
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return SparseCrossEntropy.apply(y_pred, y_true, pad_idx=pad_idx, **kwargs)


def sparse_cross_entropy_with_logits(
    y_pred: Tensor, y_true: Union[Tensor, np.ndarray], pad_idx=None, **kwargs
) -> Tensor:
    """
    Sparse Cross Entropy with logits input
    Use sparse_cross_entropy if y_pred contains probabilities
    -y_true * log(y_pred)
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return sparse_cross_entropy(softmax(y_pred), y_true, pad_idx=pad_idx, **kwargs)


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
