import logging
from typing import Any, Optional, Tuple, Union

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore

    _ = np.cuda.runtime.getDeviceCount()  # Check if a CUDA device is available
except Exception:
    import numpy as np
from autograd.tensor import Function, Tensor

logger = logging.getLogger(__name__)


########### Activation Functions ###############
def relu(x: Tensor) -> Tensor:
    """
    Applies the Rectified Linear Unit (ReLU) activation function.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The tensor after applying the ReLU function.
    """
    return Relu.apply(x)


def sigmoid(x: Tensor) -> Tensor:
    """
    Applies the sigmoid activation function.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The tensor after applying the sigmoid function.
    """
    return Sigmoid.apply(x)


def softmax(x: Tensor) -> Tensor:
    """
    Applies the softmax activation function.

    Args:
        x (Tensor): The input tensor containing logits.

    Returns:
        Tensor: The tensor with softmax probabilities.
    """
    return Softmax.apply(x)


def tanh(x: Tensor) -> Tensor:
    """
    Applies the hyperbolic tangent (tanh) activation function.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The tensor after applying the tanh function.
    """
    return Tanh.apply(x)


def gelu(x: Tensor) -> Tensor:
    """
    Applies the Gaussian Error Linear Unit (GELU) activation function.

    This function uses the approximate formula:
    $$
    0.5 * x * (1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) ))
    $$

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The tensor after applying the GELU function.
    """
    return Gelu.apply(x)


class Relu(Function):
    """
    Rectified Linear Unit (ReLU) activation function.

    The ReLU function is defined as:
        $$
        ReLU(x) = max(0, x)
        $$

    Methods:
        forward: Computes the forward pass.
        backward: Computes the gradient with respect to the input.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the ReLU activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: The result of applying ReLU to the input.
        """
        self.x = x
        return np.maximum(x, 0)

    def backward(self, grad: Tensor) -> np.ndarray:
        """
        Computes the backward pass of the ReLU activation function.

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            np.ndarray: The gradient of the loss with respect to the input.
        """
        return grad.data * (self.x > 0)


class Gelu(Function):
    r"""
    Gaussian Error Linear Unit (GELU) activation function.
    GELU(x) = x * P(X <= x) where P(X) ~ Gaussian Distribution with mean 0 and standard deviation 1

    This activation function approximates:
    $$
    0.5 * x * \left(1 + tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715*x^3\right)\right)\right)
    $$

    Paper: https://arxiv.org/abs/1606.08415

    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the GELU activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: The output array after applying GELU.
        """
        self.x = x  # Save for backward
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        r"""
        Computes the backward pass of the GELU activation function.

        The gradient is computed as:
        $$
        \frac{d\,GELU}{dx} = 0.5 \left(1 + tanh(\alpha)\right) + 0.5 \, x \, \left(1 - tanh^2(\alpha)\right) \alpha'
        $$
        where
        $$
        \alpha = \sqrt{\frac{2}{\pi}} (x + 0.044715*x^3)
        $$
        and
        $$
        \alpha' = \sqrt{\frac{2}{\pi}} \left(1 + 3*0.044715*x^2\right)
        $$

        Args:
            grad (np.ndarray): Upstream gradient.

        Returns:
            np.ndarray: The gradient of the loss with respect to the input.
        """
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
    r"""
    Sigmoid activation function.

    The sigmoid function is defined as:
        $$
        sigmoid(x) = \frac{1}{1 + e^{-x}}
        $$
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the sigmoid function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: The output after applying the sigmoid function.
        """
        self.out = 1 / (1 + np.exp(np.clip(-x, -709, 709)))
        return self.out

    def backward(self, grad: Tensor) -> np.ndarray:
        """
        Computes the backward pass of the sigmoid function.

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            np.ndarray: The gradient of the loss with respect to the input.
        """
        return grad.data * self.out * (1 - self.out)


class Softmax(Function):
    r"""
    Softmax activation function.

    The softmax function is defined as:
        $$
        softmax(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
        $$
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the softmax activation function.

        Args:
            x (np.ndarray): Input array of logits.

        Returns:
            np.ndarray: The softmax probabilities.
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.probs

    def backward(self, grad: Tensor) -> np.ndarray:
        """
        Computes the backward pass of the softmax activation function.

        This function computes the gradient of the softmax output with respect to the input logits.

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            np.ndarray: The gradient of the loss with respect to the input logits.
        """
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
    r"""
    Hyperbolic tangent (tanh) activation function.

    The tanh function is defined as:
        $$
        tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
        $$
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the tanh activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: The output after applying the tanh function.
        """
        # For numerical stability, use the fact that tanh(x) = 2*sigmoid(2x) - 1
        # This avoids computing large exponentials directly
        x = 2 * x
        # Clip x to avoid overflow in exp(-x)
        x = np.clip(x, -88.72, 88.72)  # ln(max float32) ≈ 88.72
        sigmoid_2x = 1 / (1 + np.exp(-x))
        self.out = 2 * sigmoid_2x - 1
        return self.out

    def backward(self, grad: Tensor) -> np.ndarray:
        """
        Computes the backward pass of the tanh activation function.
        $$
        d(tanh(x))/dx = 1 - tanh(x)^2
        $$

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            np.ndarray: The gradient of the loss with respect to the input.
        """
        return grad.data * (1 - self.out**2)


###################### Loss Functions #####################
class BinaryCrossEntropy(Function):
    r"""
    Binary Cross Entropy (BCE) Loss.

    This loss function assumes that $y_{pred}$ contains probabilities rather than logits.
    If the input is logits, use :func:`binary_cross_entropy_with_logits`, which applies a sigmoid before computing the loss.

    $$
    BCE = -\left( y_{true} \cdot \log(y_{pred}) + (1 - y_{true}) \cdot \log(1 - y_{pred}) \right)
    $$
    """

    def forward(
        self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs: Any
    ) -> np.floating:
        """
        Computes the binary cross entropy loss.

        Args:
            y_pred (np.ndarray): Predicted probabilities.
            y_true (np.ndarray): True binary labels.
            **kwargs: Additional keyword arguments.

        Returns:
            float: The computed binary cross entropy loss.
        """
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
        r"""
        Computes the gradient of the binary cross entropy loss with respect to $y_{pred}$.

        The gradient is given by:
        $$
        \frac{\partial L}{\partial y_{pred}} = -\left(\frac{y_{true}}{y_{pred}} - \frac{1-y_{true}}{1-y_{pred}}\right)
        $$

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            Tuple[np.ndarray, None]: A tuple containing the gradient with respect to $y_{pred}$ and None for $y_{true}$.
        """
        # Avoid division by zero by clipping probabilities away from 0 and 1
        y_pred_prob = np.clip(self.y_pred_prob, 1e-7, 1 - 1e-7)
        grad_y_pred = -(
            (self.y_true / y_pred_prob) - ((1 - self.y_true) / (1 - y_pred_prob))
        )
        grad_y_pred /= len(y_pred_prob)
        # Incorporate the upstream gradient
        grad_y_pred *= grad.data

        return grad_y_pred, None  # y_true doesn't need gradient


class BinaryCrossEntropyWithLogits(Function):
    """
    Binary Cross Entropy Loss with logits.

    This implementation is numerically stable for logits input.
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
        """
        Computes the binary cross entropy loss with logits input.

        Args:
            y_pred (np.ndarray): shape (N, ...) Unbounded real-valued logits.
            y_true (np.ndarray): True binary labels (0 or 1), same shape as y_pred.

        Returns:
            float: The computed binary cross entropy loss.
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
        r"""
        Computes the gradient of the binary cross entropy loss with logits with respect to $y_{pred}$.

        The gradient is given by:
        $$
        \frac{\partial L}{\partial y_{pred}} = sigmoid(y_{pred}) - y_{true}
        = -(\frac{y_{true}}{y_{pred}} - \frac{1-y_{true}}{1-y_{pred}})
        $$
        Where
        $$
        sigmoid = 1 / (1 + exp(-y_{pred}))
        $$

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            Tuple[np.ndarray, None]: A tuple containing the gradient with respect to $y_{pred}$ and None for $y_{true}$.
        """
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
    Cross-Entropy Loss for multi-dimensional predictions with optional padding and label smoothing.

    This function accepts raw logits (not probabilities) and computes a stable log-softmax internally.
    It supports both 2D and 3D predictions.
    """

    def forward(
        self,
        y_pred: np.ndarray,
        y_true: Union[np.ndarray, Tensor],
        pad_idx: int = 0,
        label_smoothing: float = 0.0,
        **kwargs: Any,
    ) -> float:
        r"""
        Computes the cross-entropy loss with optional padding and label smoothing.

        Args:
            y_pred (np.ndarray): Raw logits. Shape can be $(batch\_size, feature\_dim)$ or $(batch\_size, seq\_len, feature\_dim)$.
            y_true (Union[np.ndarray, Tensor]): True class indices. If $y_{pred}$ is 2D, shape is $(batch\_size,)$; if 3D, shape is $(batch\_size, seq\_len)$.
            pad_idx (int, optional): Padding index to ignore in the loss. Defaults to 0.
            label_smoothing (float, optional): Label smoothing factor. Defaults to 0.0. Label smoothing is applied if $label\_smoothing > 0$
            **kwargs: Additional keyword arguments.

        For label smoothing, the smoothed target distribution is given by:
        $$
        T_{i,j} = (1 - \alpha) \, y_{i,j} + \frac{\alpha}{K-1} (1 - y_{i,j})
        $$
        (Ref: "Rethinking the Inception Architecture for Computer Vision", https://arxiv.org/abs/1512.00567)

        Returns:
            float: The average cross-entropy loss over non-padding positions.
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
        r"""
        Computes the backward pass for the cross-entropy loss with label smoothing.

        The gradient with respect to the logits is given by:
        $$
        \frac{\partial L}{\partial logits_{i,j}} = p_{i,j} - T_{i,j}
        $$
        where
        $$
        p_{i,j} = exp(log\_softmax_{i,j})
        $$
        and the target distribution $T_{i,j}$ is defined as:
        $$
        T_{i,j} = (1 - \alpha) \, y_{i,j} + \frac{\alpha}{K-1} (1 - y_{i,j})
        $$
        The gradient is zeroed out for padding positions and scaled by the upstream gradient divided by the number of non-padding positions.

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            Tuple[np.ndarray, None]: A tuple containing the gradient with respect to the logits and None for $y_{true}$.
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

        # 4. (Step 1): All classes
        grad_out[non_pad_idx, :] -= self.label_smoothing / (self.num_classes - 1)

        # 4. (Step 2): Correct classes
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
    r"""
    Hinge Loss.

    The hinge loss is defined as:
        $$
        loss = max(0, 1 - y_{true} \cdot y_{pred})
        $$
    For correctly classified points ($y_{true} \cdot y_{pred} \geq 1$), the loss is 0; otherwise, it is $1 - y_{true} \cdot y_{pred}$. This is because loss functions typically don't go into the negatives so we take the max of 0 and 1 - y_true * y_pred)

    The objective function typically includes a regularization term:
        $$
        \|w\|^2 + C \sum max(0, 1 - y_{true} \cdot y_{pred})
        $$
    where $C$ is a hyperparameter controlling the trade-off between maximizing the margin (through regularization) and minimizing the loss, and $w$ is the weight vector. ($\|w\|^2$ is the regularization term)

    Paper: https://ieeexplore.ieee.org/document/708428
    """

    def forward(
        self,
        y_pred: np.ndarray,
        y_true: Union[np.ndarray, Tensor],
        reduction: str = "none",
        **kwargs: Any,
    ) -> Union[np.floating, np.ndarray]:
        """
        Computes the hinge loss.

        Args:
            y_pred (np.ndarray): Predicted scores.
            y_true (Union[np.ndarray, Tensor]): True labels.
            reduction (str, optional): Specifies the reduction to apply to the output:
                "none", "mean", or "sum". Defaults to "none".
            **kwargs: Additional keyword arguments.

        Returns:
            Union[float, np.ndarray]: The computed hinge loss. A scalar if reduction is "mean" or "sum", otherwise an array.
        """
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
        r"""
        Computes the gradient of the hinge loss with respect to the predictions.

        For each element, the gradient is:
        $$
        \begin{align}
        \frac{\partial loss}{\partial y_{pred}} \\
        &= \frac{d(C * sum(max(0, 1 - y_{true} * y_{pred})))}{dw} \\
        &= C * max(0, 1 - y_{true} * y_{pred}) \\
        &=\begin{cases}
        -y_{true}, & \text{if } y_{true} \cdot y_{pred} < 1 \\
        0, & \text{otherwise}
        \end{cases}
        \end{align}
        $$
        For the gradient of w in the regularization term, $$\frac{d(\frac{1}{2}\|w\|^2)}{dw} = w$$ (we multiple 1/2 because it makes the gradient calculation easier)

        If the reduction is "mean", the gradient is averaged over the number of elements.

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            Tuple[np.ndarray, None]: A tuple containing the gradient with respect to $y_{pred}$ and None for $y_{true}$.
        """
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
    r"""
    Mean Squared Error (MSE) Loss.

    The MSE loss is defined as:
        $$
        MSE = \frac{1}{N} \sum (y_{pred} - y_{true})^2
        $$
    """

    def forward(
        self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs: Any
    ) -> np.floating:
        """
        Computes the Mean Squared Error loss.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.
            **kwargs: Additional keyword arguments.

        Returns:
            float: The computed MSE loss.
        """
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, grad: Tensor) -> np.ndarray:
        r"""
        Computes the gradient of the Mean Squared Error loss with respect to the predictions.

        The gradient is given by:
        $$
        \frac{\partial L}{\partial y_{pred}} = 2 (y_{pred} - y_{true})
        $$

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            np.ndarray: The gradient with respect to the predictions.
        """
        return 2 * (self.y_pred - self.y_true) * grad.data


def binary_cross_entropy(
    y_pred: Tensor, y_true: Union[Tensor, np.ndarray], **kwargs: Any
) -> Tensor:
    """
    Computes the binary cross entropy loss given predicted probabilities.

    This function wraps the :class:`BinaryCrossEntropy` operation.

    Args:
        y_pred (Tensor): Predicted probabilities.
        y_true (Union[Tensor, np.ndarray]): True binary labels.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The computed binary cross entropy loss.
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return BinaryCrossEntropy.apply(y_pred, y_true, **kwargs)


def binary_cross_entropy_with_logits(
    y_pred: Tensor, y_true: Union[Tensor, np.ndarray], **kwargs: Any
) -> Tensor:
    r"""
    Computes the binary cross entropy loss using logits input for improved numerical stability.

    If $y_{pred}$ contains probabilities, use :func:`binary_cross_entropy` instead.

    The loss is computed as:
        $$
        -\left(y_{true} \cdot \log(y_{pred}) + (1 - y_{true}) \cdot \log(1 - y_{pred})\right)
        $$

    Args:
        y_pred (Tensor): Logits.
        y_true (Union[Tensor, np.ndarray]): True binary labels.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The computed binary cross entropy loss.
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
    Computes the cross-entropy loss for multi-class classification with logits.

    This function expects $y_{pred}$ to be raw logits and $y_{true}$ to be class indices (not one-hot vectors).

    Args:
        y_pred (Tensor): Raw logits.
        y_true (Union[Tensor, np.ndarray]): True class indices.
        pad_idx (Optional[int], optional): Padding index to ignore in the loss. Defaults to None.
        label_smoothing (float, optional): Label smoothing factor. Defaults to 0.0.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The computed cross-entropy loss.
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
    """
    Computes the hinge loss for binary classification.

    Args:
        y_pred (Tensor): Predicted scores.
        y_true (Union[Tensor, np.ndarray]): True labels.
        reduction (str, optional): Specifies the reduction to apply: "none", "mean", or "sum". Defaults to "none".
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The computed hinge loss.
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return HingeLoss.apply(y_pred, y_true, reduction=reduction, **kwargs)


def mean_squared_loss(
    y_pred: Tensor, y_true: Union[Tensor, np.ndarray], **kwargs: Any
) -> Tensor:
    """
    Computes the Mean Squared Error (MSE) loss.

    Args:
        y_pred (Tensor): Predicted values.
        y_true (Union[Tensor, np.ndarray]): True values.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The computed MSE loss.
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return MeanSquaredLoss.apply(y_pred, y_true, **kwargs)
