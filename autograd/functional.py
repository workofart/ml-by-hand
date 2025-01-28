import logging
from typing import Any, Optional, Tuple, Union

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np

from autograd.tensor import Function, Tensor

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


def gelu(x: Tensor) -> Tensor:
    return Gelu.apply(x)


class Relu(Function):
    """
    Retified Linear Unit (ReLU) activation function.
    ReLU(x) = max(0, x)
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(x, 0)

    def backward(self, grad: Tensor) -> np.ndarray:
        # dL/dx = dL/dy * dy/dx
        return grad.data * (self.x > 0)


class Gelu(Function):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    Paper: https://arxiv.org/abs/1606.08415

    GELU(x) = x * P(X <= x) where P(X) ~ Gaussian Distribution with mean 0 and standard deviation 1
    Approximately
        0.5 * x * [1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) )]
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the approximate GELU.
        """
        self.x = x  # Save for backward
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        dGELU/dx = 0.5 * (1 + tanh(alpha)) + 0.5 * x * (1 - tanh(alpha)^2) * alpha'
        where alpha = sqrt(2/pi) * (x + 0.044715 * x^3)
        and alpha' = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
        """
        # Compute alpha(x) = sqrt(2/pi) * (x + 0.044715 * x^3)
        alpha = np.sqrt(2.0 / np.pi) * (self.x + 0.044715 * self.x**3)

        # Compute tanh(alpha)
        tanh_alpha = np.tanh(alpha)

        # Compute derivative of alpha: alpha'(x)
        alpha_prime = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * self.x**2)

        # Derivative of GELU:
        dgelu_dx = (
            0.5 * (1.0 + tanh_alpha)
            + 0.5 * self.x * (1.0 - tanh_alpha**2) * alpha_prime
        )

        # Chain rule: dL/dx = dL/dy * dy/dx
        return grad * dgelu_dx


class Sigmoid(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        # 709 is the maximum value that can be passed to np.exp without overflowing
        self.out = 1 / (1 + np.exp(np.clip(-x, -709, 709)))
        return self.out

    def backward(self, grad: Tensor) -> np.ndarray:
        # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        return grad.data * self.out * (1 - self.out)


class Softmax(Function):
    """
    Softmax activation function
    softmax(x) = e^x / sum(e^x)
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Subtract the maximum value for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.probs

    def backward(self, grad: Tensor) -> np.ndarray:
        # There are two cases for this gradient because each element in the matrix affects
        # every other elements' gradient due to the fact of sum(e^x) in the denominator.
        # Let's denote i, j as the ith and jth elements in the matrix.
        # Case 1: i == j
        # d(softmax(x))/dx_i = softmax(x)_i * (1[i==j] - softmax(x)_i)
        # Case 2: i != j
        # d(softmax(x))/dx_i = -softmax(x)_i * softmax(x)_j

        # dL/dx = y * (dL/dy - sum(dL/dy * y, axis=-1, keepdims=True))
        sum_term = np.sum(grad.data * self.probs, axis=-1, keepdims=True)
        dLdx = self.probs * (grad.data - sum_term)
        return dLdx


class Tanh(Function):
    """
    Tanh activation function
    tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        # For numerical stability, use the fact that tanh(x) = 2*sigmoid(2x) - 1
        # This avoids computing large exponentials directly
        x = 2 * x
        # Clip x to avoid overflow in exp(-x)
        x = np.clip(x, -88.72, 88.72)  # ln(max float32) ≈ 88.72
        sigmoid_2x = 1 / (1 + np.exp(-x))
        self.out = 2 * sigmoid_2x - 1
        return self.out

    def backward(self, grad: Tensor) -> np.ndarray:
        # d(tanh(x))/dx = 1 - tanh(x)^2
        return grad.data * (1 - self.out**2)


###################### Loss Functions #####################
class BinaryCrossEntropy(Function):
    """
    Binary Cross Entropy Loss
    Note: We assume the input y_pred contain probabilities not logits.
    - If you have logits, use `binary_cross_entropy_with_logits`, which applies a sigmoid before computing the loss.

    Forward:
        BCE = -(y_true * log(y_pred)) + (1 - y_true) * log(1 - y_pred)
    """

    def forward(
        self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs: Any
    ) -> np.floating:
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)

        # If labels come in as (batch_size,), explicitly reshaping them to (batch_size, 1) avoids shape mismatch, and certain elementwise operations will broadcast in unintended ways
        if y_true.ndim == 1 and y_pred.ndim == 1:
            pass
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError("y_pred and y_true must have the same shape")

        self.y_true = y_true
        self.y_pred_prob = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(
            y_true * np.log(self.y_pred_prob)
            + (1 - y_true) * np.log(1 - self.y_pred_prob)
        )
        return loss

    def backward(self, grad: Tensor) -> Tuple[np.ndarray, None]:
        # $$ \frac{\partial{L}}{\partial{y_{pred}}} = -(\frac{y_{true}}{y_{pred}} - \frac{1-y_{true}}{1-y_{pred}})$$
        y_true = self.y_true
        y_pred_prob = self.y_pred_prob

        # Avoid division by zero by clipping probabilities away from 0 and 1
        y_pred_prob = np.clip(y_pred_prob, 1e-7, 1 - 1e-7)
        grad_y_pred = -((y_true / y_pred_prob) - ((1 - y_true) / (1 - y_pred_prob)))
        grad_y_pred /= len(y_pred_prob)
        # Incorporate the upstream gradient
        grad_y_pred *= grad.data

        return grad_y_pred, None  # y_true doesn't need gradient


class BinaryCrossEntropyWithLogits(Function):
    """
    Stable implementation of binary cross-entropy with logits.
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
        """
        Args:
            y_pred (np.ndarray): shape (N, ...) unbounded real values
            y_true (np.ndarray): same shape as y_pred in {0, 1}

        Returns:
            float: the binary cross-entropy loss
        """
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)

        # If labels come in as (batch_size,), explicitly reshaping them to (batch_size, 1) avoids shape mismatch, and certain elementwise operations will broadcast in unintended ways
        if y_true.ndim == 1 and y_pred.ndim == 1:
            pass
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have the same shape.")

        # compute loss
        # loss_i = max(y_pred, 0) - y_pred * y_true + log(1 + exp(-|y_pred|))
        loss = np.mean(
            np.maximum(y_pred, 0.0)
            - y_pred * y_true
            + np.log1p(np.exp(-np.abs(y_pred)))
        )

        self.y_pred = y_pred
        self.y_true = y_true
        return loss

    def backward(self, grad: Tensor) -> Tuple[np.ndarray, None]:
        """
        Args:
            grad (Tensor): upstream gradient

        dL/dy_pred = sigmoid(y_pred) - y_true
        """
        # $$ \frac{\partial{L}}{\partial{y_{pred}}} = -(\frac{y_{true}}{y_{pred}} - \frac{1-y_{true}}{1-y_{pred}})$$
        # sigmoid = 1 / (1 + exp(-y_pred))

        # 1) Stable sigmoid computation
        sig = np.empty_like(self.y_pred, dtype=np.float32)

        # For z >= 0, sigmoid(z) = 1 / (1 + exp(-z))
        pos_mask = self.y_pred >= 0
        # clamp to avoid overflow in exp
        z_pos_clamped = np.clip(self.y_pred[pos_mask], -100, 100)
        exp_neg_pos = np.exp(-z_pos_clamped)
        sig[pos_mask] = 1.0 / (1.0 + exp_neg_pos)

        # For z < 0, sigmoid(z) = exp(z) / (1 + exp(z))
        neg_mask = ~pos_mask
        z_neg_clamped = np.clip(self.y_pred[neg_mask], -100, 100)
        exp_pos_neg = np.exp(z_neg_clamped)
        sig[neg_mask] = exp_pos_neg / (1.0 + exp_pos_neg)

        # 2) Compute dL/dy_pred = sigmoid(y_pred) - y_true divided by batch_size
        grad_y_pred = (sig - self.y_true) / np.prod(self.y_pred.shape[0])

        # 3) Multiply by upstream gradient
        grad_y_pred *= grad.data

        # we don't need a gradient for y_true (labels)
        return grad_y_pred, None


class CrossEntropy(Function):
    """
    Cross-entropy for 2D or 3D predictions with optional pad_idx ignoring,
    BUT accepts raw logits (not probabilities).

    Usage is analogous to SparseCrossEntropy, but we do:
    - stable log-softmax inside the forward pass
    - label smoothing if label_smoothing > 0
    """

    def forward(
        self,
        y_pred: np.ndarray,
        y_true: Union[np.ndarray, Tensor],
        pad_idx: int = 0,
        label_smoothing: float = 0.0,
        **kwargs: Any,
    ) -> float:
        """
        Args:
            - y_pred are raw logits (not restricted to [0,1]).
                Shape can be (batch_size, feature_dim) or (batch_size, seq_len, feature_dim).
            - y_true must be integer class indices in [0, feature_dim):
                If y_pred is (batch_size, feature_dim), y_true is (batch_size,).
                If y_pred is (batch_size, seq_len, feature_dim), y_true is (batch_size, seq_len).
            - pad_idx (int, optional): The padding index, which will be masked out from the loss.
              Defaults to 0.
            - label_smoothing (float, optional): How much uniform smoothing to add to the "correct" class.
              Defaults to 0.0.
              (Ref: "Rethinking the Inception Architecture for Computer Vision", https://arxiv.org/abs/1512.00567)

        Returns:
            - A scalar float representing the average cross-entropy loss over non-pad positions.
        """

        # 1. Convert y_true to NumPy if it’s a Tensor, ensure it's int64 for indexing.
        if isinstance(y_true, Tensor):
            y_true = y_true.data
        y_true = np.asarray(y_true, dtype=np.int64)

        # 2. If 3D logits, flatten them for simpler processing.
        self.original_shape = y_pred.shape
        if y_pred.ndim == 3:
            batch_size, seq_len, num_classes = y_pred.shape
            y_pred = y_pred.reshape(batch_size * seq_len, num_classes)
            y_true = y_true.reshape(batch_size * seq_len)
        else:
            batch_size, num_classes = y_pred.shape

        # 3. Create a mask for non-pad positions (where y_true != pad_idx).
        self.non_pad_mask = y_true != pad_idx
        non_pad_idx = np.where(self.non_pad_mask)[0]

        # 4. Compute stable log-softmax:
        # log(softmax(y_pred)) = y_pred - log(sum(exp(y_pred)))
        # However, log(sum(exp(y_pred))) can overflow if y_pred is large.
        # To avoid this, we use the following trick:
        # shifted = y_pred - max(y_pred) along each row
        shifted = y_pred - np.max(y_pred, axis=1, keepdims=True)

        # Going back to the log-softmax formula:
        # log(softmax(y_pred)) = y_pred - log(sum(exp(y_pred)))
        # Instead of computing exp(y_pred), we compute exp(shifted):
        # exp(shifted) = exp(y_pred - max(y_pred)) = exp(y_pred) / exp(max(y_pred))
        # Then we have: log(softmax(shifted)) = shifted - log(sum(exp(shifted)))
        # the largest value in shifted is 0, so sum(exp(shifted)) is safe to compute.
        log_softmax = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))

        # 5. Compute the label-smoothed cross-entropy for each element i.
        # $$ L_i = -\left( (1 - \text{label\_smoothing}) \log p_{i,y_i} + \frac{\text{label\_smoothing}}{\text{num\_classes} - 1} \sum_{j \neq y_i} \log p_{i,j} \right) $$
        # $$ = -\left( (1 - \text{label\_smoothing}) \log p_{i,y_i} + \frac{\text{label\_smoothing}}{\text{num\_classes} - 1} \left( \sum_{j=1}^{\text{num\_classes}} \log p_{i,j} - \log p_{i,y_i} \right) \right) $$
        # Essentially, we are putting some (label_smoothing) weight on a uniform distribution over the non-correct classes
        # This will make the model prediction les confident, and thus less likely to overfit.
        log_p_correct = log_softmax[np.arange(len(y_true)), y_true]
        sum_log_p = np.sum(log_softmax, axis=1)

        losses = -(
            (1.0 - label_smoothing) * log_p_correct
            + (label_smoothing / (num_classes - 1)) * (sum_log_p - log_p_correct)
        )

        # 6) Average the loss only over non-pad positions.
        if len(non_pad_idx) == 0:
            loss_val = 0.0
        else:
            loss_val = np.mean(losses[non_pad_idx])

        # 7) Store for backward pass:
        self.log_softmax = log_softmax
        self.y_true = y_true
        self.pad_idx = pad_idx
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

        return loss_val

    def backward(self, grad: Tensor) -> Tuple[np.ndarray, None]:
        """
        Backprop through the label-smoothed log-softmax cross-entropy.

        p_{i,j} = exp(log_softmax_{i,j})
        dL/dlogits[i,j] = p_{i,j} - T_{i,j}
        $$
        T_{i,j} = (1 - \alpha) * y_{i,j} + \frac{\alpha}{K-1} * (1 - y_{i,j})
        $$
        Then zero out for padded positions, and scale by grad / number of non-padded.
        """

        # 1. Softmax from log-softmax is safe: p_{i,j} = exp(log_sm[i,j])
        softmax_probs = np.exp(self.log_softmax)

        # 2. Prepare the output gradient array with a copy of softmax_probs
        grad_out = np.copy(softmax_probs)

        # 3. Identify which samples are non-padding
        non_pad_idx = np.where(self.non_pad_mask)[0]

        # 4. Label Smoothing
        # When label_smoothing > 0, we subtract (1-label_smoothing) or something smaller
        # than 1, because we don't want to be too confident (1) about the correct class
        # We do this in two steps.
        # Step 1: Incorrect class and correct class: a / (num_class - 1)
        # Step 2: Correct class: 1 - a
        # This is equivalent to:
        # Correct class: 1 - a + a / (num_class - 1) = (1 - a) + a / (num_class - 1)
        # Incorrect class: a / (num_class - 1)

        # Step 1: All classes
        grad_out[non_pad_idx, :] -= self.label_smoothing / (self.num_classes - 1)

        # Step 2: Correct classes
        correct_idx = (non_pad_idx, self.y_true[non_pad_idx])
        grad_out[correct_idx] -= 1.0 - self.label_smoothing

        # 5. For pad positions, set gradient = 0.0
        grad_out[np.where(~self.non_pad_mask)[0], :] = 0.0

        # 6. Multiply by upstream grad and average by non-padded positions
        grad_out *= grad.data / max(1, len(non_pad_idx))

        # 7. Reshape to original shape if 3D
        grad_out = grad_out.reshape(self.original_shape)

        # We do not need grad for y_true
        return grad_out, None


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

    def forward(
        self,
        y_pred: np.ndarray,
        y_true: Union[np.ndarray, Tensor],
        reduction: str = "none",
        **kwargs: Any,
    ) -> Union[np.floating, np.ndarray]:
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

    def backward(self, grad: Tensor) -> Tuple[np.ndarray, None]:
        """
        d (1/2||w||^2)/dw = w (we multiple 1/2 because it makes the gradient calculation easier)
        d(C * sum(max(0, 1 - y_true * y_pred)))/dw = C * max(0, 1 - y_true * y_pred)
        = 1 - y_true * y_pred (if y_true * y_pred < 1)
        = 0 (if y_true * y_pred >= 1)
        """
        # Initialize gradient array with same shape as predictions
        grad_y_pred = np.zeros_like(self.y_pred)

        # Where margin > 0, gradient is -y_true
        margin_violated = self.margins > 0
        grad_y_pred = np.where(margin_violated, -self.y_true, 0)

        if self.reduction == "mean":
            grad_y_pred /= self.y_pred.size

        # Handle scalar gradient (from mean/sum reduction)
        if np.isscalar(grad.data) or grad.data.size == 1:
            grad_y_pred *= grad.data
        else:
            # For elementwise gradient
            grad_y_pred *= grad.data.reshape(grad_y_pred.shape)

        return grad_y_pred, None


class MeanSquaredLoss(Function):
    def forward(
        self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs: Any
    ) -> np.floating:
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, grad: Tensor) -> np.ndarray:
        """
        dL/dx = 2 * (x - y)
        """
        return 2 * (self.y_pred - self.y_true) * grad.data


def binary_cross_entropy(
    y_pred: Tensor, y_true: Union[Tensor, np.ndarray], **kwargs: Any
) -> Tensor:
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return BinaryCrossEntropy.apply(y_pred, y_true, **kwargs)


def binary_cross_entropy_with_logits(
    y_pred: Tensor, y_true: Union[Tensor, np.ndarray], **kwargs: Any
) -> Tensor:
    """
    Binary Cross Entropy Loss with logits input (more stable)
    Use binary_cross_entropy if y_pred contain probabilities
    -(y_true * log(y_pred)) + (1 - y_true) * log(1 - y_pred)
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return BinaryCrossEntropyWithLogits.apply(y_pred, y_true, **kwargs)


def cross_entropy(
    y_pred: Tensor,
    y_true: Union[Tensor, np.ndarray],
    pad_idx: Optional[int] = None,
    label_smoothing: float = 0.0,
    **kwargs: Any,
) -> Tensor:
    """
    For multi-class classification with logits y_pred and y_true should be class indices (not one-hot)
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return CrossEntropy.apply(
        y_pred, y_true, pad_idx=pad_idx, label_smoothing=label_smoothing, **kwargs
    )


def hinge_loss(
    y_pred: Tensor,
    y_true: Union[Tensor, np.ndarray],
    reduction: str = "none",
    **kwargs: Any,
) -> Tensor:
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return HingeLoss.apply(y_pred, y_true, reduction=reduction, **kwargs)


def mean_squared_loss(
    y_pred: Tensor, y_true: Union[Tensor, np.ndarray], **kwargs: Any
) -> Tensor:
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return MeanSquaredLoss.apply(y_pred, y_true, **kwargs)
