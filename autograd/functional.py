from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any, Optional, Tuple, Union

from autograd.backend import (
    LOW_PRECISION_FLOAT_DTYPES,
    Array,
    ArrayLike,
    xp,
)
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

    Examples:
        >>> from autograd.tensor import Tensor
        >>> x = Tensor([-1, 0, 2])
        >>> y = relu(x) # Expected output: [0, 0, 2]
    """
    return Relu.apply(x)


def sigmoid(x: Tensor) -> Tensor:
    """
    Applies the sigmoid activation function.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The tensor after applying the sigmoid function.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> x = Tensor([0, 2])
        >>> y = sigmoid(x) # Expected output: [0.5, ~0.88]
    """
    return Sigmoid.apply(x)


def softmax(x: Tensor) -> Tensor:
    """
    Applies the softmax activation function.

    Args:
        x (Tensor): The input tensor containing logits.

    Returns:
        Tensor: The tensor with softmax probabilities.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> import cupy as np
        >>> x = Tensor(xp.array([2.0, 1.0, 0.1]))
        >>> y = softmax(x) # Expected output: probabilities that sum to 1
    """
    return Softmax.apply(x)


def tanh(x: Tensor) -> Tensor:
    """
    Applies the hyperbolic tangent (tanh) activation function.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The tensor after applying the tanh function.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> x = Tensor([0, 1])
        >>> y = tanh(x) # Expected output: [0, tanh(1)]
    """
    return Tanh.apply(x)


def gelu(x: Tensor) -> Tensor:
    r"""
    Applies the Gaussian Error Linear Unit (GELU) activation function.

    This function uses the approximate formula:
    $$
    0.5 * x * \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \, (x + 0.044715*x^3)\right)\right)
    $$

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The tensor after applying the GELU function.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> x = Tensor([1.0, -1.0])
        >>> y = gelu(x) # Expected output: approximate GELU values for the inputs
    """
    return Gelu.apply(x)


def scaled_dot_product_attention_mlx_custom(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    is_training: Optional[bool] = None,
    dropout_prob: float = 0.0,
) -> Tensor:
    """
    Custom-implemented MLX kernel causal attention calculation
    """
    dropout_scale_mask = ScaledDotProductAttentionMLXCustom.build_dropout_scale_mask(
        is_training=is_training,
        dropout_prob=dropout_prob,
        query_shape=query.shape,
        key_shape=key.shape,
    )
    ScaledDotProductAttentionMLXCustom.validate(
        query.data,
        key.data,
        value.data,
        dropout_scale_mask=dropout_scale_mask,
    )
    return ScaledDotProductAttentionMLXCustom.apply(
        query,
        key,
        value,
        dropout_scale_mask=dropout_scale_mask,
    )


class Relu(Function):
    r"""
    Rectified Linear Unit (ReLU) activation function.

    The ReLU function is defined as:
        $$
        ReLU(x) = \max(0, x)
        $$

    Note:
        This class is used internally. For applying ReLU, use the `relu` function.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> x = Tensor([-3, 0, 3])
        >>> y = Relu.apply(x) # Expected output: [0, 0, 3]
    """

    def forward(self, x: Array) -> Array:
        """
        Computes the forward pass of the ReLU activation function.

        Args:
            x (xp.ndarray): Input array.

        Returns:
            xp.ndarray: The result of applying ReLU to the input.
        """
        self.x = x
        return xp.maximum(x, 0)

    def backward(self, grad: Tensor) -> Array:
        """
        Computes the backward pass of the ReLU activation function.

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            xp.ndarray: The gradient of the loss with respect to the input.
        """
        return grad.data * (self.x > 0)


class Gelu(Function):
    r"""
    Gaussian Error Linear Unit (GELU) activation function.
    GELU(x) = x * P(X \le x) where X ~ N(0, 1)

    This activation function approximates:
    $$
    0.5 * x * \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \, (x + 0.044715*x^3)\right)\right)
    $$

    Paper: https://arxiv.org/abs/1606.08415

    Note:
        Use the `gelu` function to apply this activation.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> x = Tensor([0.5, -0.5])
        >>> y = Gelu.apply(x) # Expected output: approximate GELU values
    """

    def forward(self, x: Array) -> Array:
        """
        Computes the forward pass of the GELU activation function.

        Args:
            x (xp.ndarray): Input array.

        Returns:
            xp.ndarray: The output array after applying GELU.
        """
        self.x = x  # Save for backward
        coeff = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + xp.tanh(coeff * (x + 0.044715 * x**3)))

    def backward(self, grad: Array) -> Array:
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
            grad (xp.ndarray): Upstream gradient.

        Returns:
            xp.ndarray: The gradient of the loss with respect to the input.
        """
        coeff = math.sqrt(2.0 / math.pi)
        alpha = coeff * (self.x + 0.044715 * self.x**3)

        # Compute tanh(alpha)
        tanh_alpha = xp.tanh(alpha)

        # Compute derivative of alpha: alpha'(x)
        alpha_prime = coeff * (1.0 + 3.0 * 0.044715 * self.x**2)

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

    Note:
        Use the `sigmoid` function to apply this activation.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> x = Tensor([0, 2])
        >>> y = Sigmoid.apply(x) # Expected output: [0.5, ~0.88]
    """

    def forward(self, x: Array) -> Array:
        """
        Computes the forward pass of the sigmoid function.

        Args:
            x (xp.ndarray): Input array.

        Returns:
            xp.ndarray: The output after applying the sigmoid function.
        """
        self.out = 1 / (1 + xp.exp(xp.clip(-x, -709, 709)))
        return self.out

    def backward(self, grad: Tensor) -> Array:
        """
        Computes the backward pass of the sigmoid function.

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            xp.ndarray: The gradient of the loss with respect to the input.
        """
        return grad.data * self.out * (1 - self.out)


class Softmax(Function):
    r"""
    Softmax activation function.

    The softmax function is defined as:
        $$
        softmax(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
        $$

    Note:
        Use the `softmax` function to apply this activation.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> import cupy as np
        >>> x = Tensor(xp.array([1.0, 2.0, 3.0]))
        >>> y = Softmax.apply(x) # Expected output: probabilities that sum to 1
    """

    def forward(self, x: Array) -> Array:
        """
        Computes the forward pass of the softmax activation function.

        Args:
            x (xp.ndarray): Input array of logits.

        Returns:
            xp.ndarray: The softmax probabilities.
        """
        exp_x = xp.exp(x - xp.max(x, axis=-1, keepdims=True))
        self.probs = exp_x / xp.sum(exp_x, axis=-1, keepdims=True)
        return self.probs

    def backward(self, grad: Tensor) -> Array:
        """
        Computes the backward pass of the softmax activation function.

        This function computes the gradient of the softmax output with respect to the input logits.

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            xp.ndarray: The gradient of the loss with respect to the input logits.
        """
        # There are two cases for this gradient because each element in the matrix affects
        # every other elements' gradient due to the fact of sum(e^x) in the denominator.
        # Let's denote i, j as the ith and jth elements in the matrix.
        # Case 1: i == j
        # d(softmax(x))/dx_i = softmax(x)_i * (1[i==j] - softmax(x)_i)
        # Case 2: i != j
        # d(softmax(x))/dx_i = -softmax(x)_i * softmax(x)_j

        # dL/dx = y * (dL/dy - sum(dL/dy * y, axis=-1, keepdims=True))
        sum_term = xp.sum(grad.data * self.probs, axis=-1, keepdims=True)
        dLdx = self.probs * (grad.data - sum_term)
        return dLdx


class Tanh(Function):
    r"""
    Hyperbolic tangent (tanh) activation function.

    The tanh function is defined as:
        $$
        \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
        $$

    Note:
        Use the `tanh` function to apply this activation.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> x = Tensor([0, 1])
        >>> y = Tanh.apply(x) # Expected output: [0, tanh(1)]
    """

    def forward(self, x: Array) -> Array:
        """
        Computes the forward pass of the tanh activation function.

        Args:
            x (xp.ndarray): Input array.

        Returns:
            xp.ndarray: The output after applying the tanh function.
        """
        # For numerical stability, use the fact that tanh(x) = 2*sigmoid(2x) - 1
        # This avoids computing large exponentials directly
        x = 2 * x
        # Clip x to avoid overflow in exp(-x)
        x = xp.clip(x, -88.72, 88.72)  # ln(max float32) ≈ 88.72
        sigmoid_2x = 1 / (1 + xp.exp(-x))
        self.out = 2 * sigmoid_2x - 1
        return self.out

    def backward(self, grad: Tensor) -> Array:
        """
        Computes the backward pass of the tanh activation function.
        $$
        d(tanh(x))/dx = 1 - tanh(x)^2
        $$

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            xp.ndarray: The gradient of the loss with respect to the input.
        """
        return grad.data * (1 - self.out**2)


class ScaledDotProductAttentionMLXCustom(Function):
    _THREADGROUP = (256, 1, 1)

    @staticmethod
    def _causal_mask(seq_len: int) -> Array:
        # Mask contract: 1.0 means forbidden and 0.0 means allowed.
        return xp.triu(xp.ones((seq_len, seq_len), dtype=xp.float32), k=1).reshape(
            1, 1, seq_len, seq_len
        )

    @staticmethod
    def build_dropout_scale_mask(
        *,
        is_training: Optional[bool],
        dropout_prob: float,
        query_shape: Tuple[int, ...],
        key_shape: Tuple[int, ...],
    ) -> Optional[Array]:
        if not is_training or dropout_prob == 0:
            return None

        dropout_shape = (
            int(query_shape[0]),
            int(query_shape[1]),
            int(query_shape[2]),
            int(key_shape[2]),
        )
        keep_prob = float(1 - dropout_prob)
        if keep_prob <= 0:
            return xp.zeros(dropout_shape, dtype=xp.float32)

        keep_mask = xp.random.bernoulli(keep_prob, shape=dropout_shape)
        return xp.array(keep_mask, dtype=xp.float32) / keep_prob

    @classmethod
    def _causal_attention_probs(
        cls,
        query: Array,
        key: Array,
    ) -> Tuple[Array, float]:
        # Recompute the dense causal probabilities in Python so backward stays
        # aligned with the contract without saving a dense forward tensor.
        scale = float(query.shape[-1]) ** -0.5
        seq_len = int(query.shape[-2])

        r"""
        Transpose keys from ``(..., S_k, D)`` to ``(..., D, S_k)`` for ``QK^T``.
        This helper recomputes the causal probability tensor, not the final
        attention output:
        $$
        \text{attention\_scores} = \frac{QK^T}{\sqrt{key\_dim}}
        $$
        $$
        \text{masked\_scores} =
        \text{attention\_scores} + \text{causal\_mask} \cdot (-10^9)
        $$
        $$
        P = \operatorname{softmax}(\text{masked\_scores})
        $$
        """
        attention_scores = (query @ key.transpose(0, 1, 3, 2)) * scale
        masked_scores = attention_scores + cls._causal_mask(seq_len) * -1e9
        exp_scores = xp.exp(
            masked_scores - xp.max(masked_scores, axis=-1, keepdims=True)
        )
        probs = exp_scores / xp.sum(exp_scores, axis=-1, keepdims=True)
        return probs, scale

    # Cache the compiled Metal kernels so the causal fast path pays the JIT
    # cost once per specialization instead of once per forward call.
    @staticmethod
    # `maxsize=2` keeps both stable variants cached: with and without dropout.
    @lru_cache(maxsize=2)
    def _kernel(use_dropout_scale_mask: bool) -> Any:
        r"""
        Fused causal attention kernel with optional repo-owned post-softmax
        dropout.

        When `use_dropout_scale_mask` is false, the kernel computes:
        $$
        O_{b,h,q,d}
        =
        \sum_{k=0}^{q} p_{b,h,q,k} V_{b,h,k,d}
        $$

        When `use_dropout_scale_mask` is true, the Python caller has already
        sampled the repo's Bernoulli dropout mask and scaled it by
        ``1 / (1 - p)``, so the kernel computes:
        $$
        O_{b,h,q,d}
        =
        \sum_{k=0}^{q} p_{b,h,q,k} m_{b,h,q,k} V_{b,h,k,d}
        $$

        In both cases one thread computes one output element and recomputes the
        causal score row twice: once to find the stable softmax maximum, then a
        second time to accumulate both the softmax denominator and the output
        numerator without materializing the dense score matrix.
        """
        try:
            import mlx.core.fast as mx_fast  # pyright: ignore[reportMissingModuleSource]
        except ModuleNotFoundError as exc:  # pragma: no cover - platform dependent
            raise RuntimeError("mlx_custom requires the MLX backend") from exc

        # MLX kernel signatures are fixed at compile time, so dropout support
        # needs its own cached specialization with one extra input.
        kernel_name = "scaled_dot_product_attention_mlx_custom"
        input_names = ["query", "key", "value"]
        dropout_setup = ""
        output_init = "float numerator = 0.0f;"
        output_update = (
            "numerator += attention_weight * value[key_row_offset + data_idx];"
        )
        if use_dropout_scale_mask:
            kernel_name = "scaled_dot_product_attention_mlx_custom_dropout"
            input_names.append("dropout_scale_mask")
            dropout_setup = """
                // Row-major offset into the [B, H, S_q, S_k] dropout mask.
                int dropout_row_offset =
                    ((batch_idx * num_heads + head_idx) * seq_len + query_idx) * seq_len;
            """
            output_update = """
                float dropout_scale = dropout_scale_mask[dropout_row_offset + key_idx];
                numerator +=
                    attention_weight * dropout_scale * value[key_row_offset + data_idx];
            """
        scaled_dot_product = """
                    float scaled_dot_product = 0.0f;
                    for (int data_offset = 0; data_offset < head_dim; ++data_offset) {
                        scaled_dot_product +=
                            query[query_row_offset + data_offset]
                            * key[key_row_offset + data_offset];
                    }
                    scaled_dot_product *= scale;
        """

        return mx_fast.metal_kernel(
            name=kernel_name,
            input_names=input_names,
            output_names=["out"],
            source=f"""
                // One thread computes one output element O[b, h, q, d].
                uint flat_idx = thread_position_in_grid.x;
                int head_dim = query_shape[3];
                int seq_len = query_shape[2];
                int num_heads = query_shape[1];

                // Map the flat output index back to (batch, head, query position, feature).
                int data_idx = flat_idx % head_dim;
                int query_idx = (flat_idx / head_dim) % seq_len;
                int head_idx = (flat_idx / (head_dim * seq_len)) % num_heads;
                int batch_idx = flat_idx / (head_dim * seq_len * num_heads);

                // Compute row offsets inside the contiguous [B, H, S, D] layout.
                int head_stride = seq_len * head_dim;
                int batch_stride = num_heads * head_stride;
                int head_sequence_offset =
                    batch_idx * batch_stride + head_idx * head_stride;
                int query_row_offset = head_sequence_offset + query_idx * head_dim;
                // Causal masking allows exactly query_idx + 1 visible keys.
                int causal_keys = query_idx + 1;
                {dropout_setup}

                // Equivalent to 1 / sqrt(D).
                float scale = metal::rsqrt(float(head_dim));

                // Pass 1: find the maximum visible logit for stable softmax.
                float max_score = -INFINITY;
                for (int key_idx = 0; key_idx < causal_keys; ++key_idx) {{
                    int key_row_offset = head_sequence_offset + key_idx * head_dim;
                    {scaled_dot_product}
                    max_score = metal::max(max_score, scaled_dot_product);
                }}

                // Pass 2: accumulate the unnormalized softmax denominator and
                // the output numerator in the same loop.
                float softmax_denominator = 0.0f;
                {output_init}
                for (int key_idx = 0; key_idx < causal_keys; ++key_idx) {{
                    int key_row_offset = head_sequence_offset + key_idx * head_dim;
                    {scaled_dot_product}
                    float attention_weight = metal::exp(scaled_dot_product - max_score);
                    softmax_denominator += attention_weight;
                    {output_update}
                }}

                out[flat_idx] = numerator / softmax_denominator;
            """,
        )

    @staticmethod
    def validate(
        query: Array,
        key: Array,
        value: Array,
        *,
        dropout_scale_mask: Optional[Array] = None,
    ) -> None:
        from autograd.backend import NAME

        if NAME != "mlx":
            raise RuntimeError("mlx_custom requires the MLX backend")

        tensors = (query, key, value)
        if any(len(tensor.shape) != 4 for tensor in tensors):
            raise ValueError(
                "mlx_custom expects query, key, and value to be 4D tensors"
            )
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError(
                "mlx_custom requires query, key, and value to share the same shape"
            )
        if any(tensor.dtype != xp.float32 for tensor in tensors):
            raise ValueError("mlx_custom requires float32 query, key, and value")
        if dropout_scale_mask is not None:
            expected_dropout_shape = (
                int(query.shape[0]),
                int(query.shape[1]),
                int(query.shape[2]),
                int(key.shape[2]),
            )
            if dropout_scale_mask.shape != expected_dropout_shape:
                raise ValueError(
                    "mlx_custom dropout mask must match attention probability shape"
                )
            if dropout_scale_mask.dtype != xp.float32:
                raise ValueError("mlx_custom dropout mask must be float32")

    def forward(
        self,
        query: Array,
        key: Array,
        value: Array,
        *,
        dropout_scale_mask: Optional[Array] = None,
    ) -> Array:
        self.validate(
            query,
            key,
            value,
            dropout_scale_mask=dropout_scale_mask,
        )
        self.dropout_scale_mask = dropout_scale_mask

        kernel = self._kernel(dropout_scale_mask is not None)
        inputs = [query, key, value]
        if dropout_scale_mask is not None:
            inputs.append(dropout_scale_mask)
        return kernel(
            inputs=inputs,
            grid=(int(query.size), 1, 1),
            threadgroup=self._THREADGROUP,
            output_shapes=[query.shape],
            output_dtypes=[query.dtype],
        )[0]

    def backward(self, grad: Tensor) -> Tuple[Array, Array, Array]:
        r"""
        The reason we have to define this backward function explicitly is because the kernel's forward function is defined not with our Tensor ops, but with a lower-level kernel ops.
        That's why we don't get the backward implementation for free like other Function classes.

        Backpropagate through the fused causal attention path.

        The no-dropout forward contract is:

        $$
        \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{key\_dim}}\right) V
        $$

        Train mode keeps the repo's post-softmax dropout contract:
        $$
        \widetilde{P} = P \odot M
        $$
        $$
        \text{Attention}_{\text{dropout}}(Q, K, V) = \widetilde{P}V
        $$
        where $M$ is the already-scaled Bernoulli mask sampled in Python.

        We still recompute the causal softmax probabilities inside backward
        instead of saving dense logits/probabilities from the fused forward
        kernel. We will use these intermediate names:
        $$
        \text{attention\_scores} = \frac{QK^T}{\sqrt{key\_dim}}
        $$
        $$
        P = \operatorname{softmax}(\text{attention\_scores}_{\text{causal}})
        $$
        $$
        \widetilde{P} =
        \begin{cases}
        P & \text{if dropout is disabled} \\
        P \odot M & \text{if dropout is enabled}
        \end{cases}
        $$
        $$
        \text{attention\_output} = \widetilde{P}V
        $$

        Let `loss` be the final scalar training objective, and let the upstream
        gradient be:
        $$
        \text{upstream\_grad} = \frac{\partial \text{loss}}{\partial \text{attention\_output}}
        $$

        1. Derivative of `loss` with respect to `V`:
        $$
        \frac{\partial \text{loss}}{\partial V}
        =
        \frac{\partial \text{loss}}{\partial \text{attention\_output}}
        \frac{\partial \text{attention\_output}}{\partial V}
        =
        \widetilde{P}^T \, \text{upstream\_grad}
        $$

        2. Derivative of `loss` with respect to the pre-dropout probabilities:
        $$
        \frac{\partial \text{loss}}{\partial P}
        =
        \left(\text{upstream\_grad} \, V^T\right)
        \odot
        \begin{cases}
        1 & \text{if dropout is disabled} \\
        M & \text{if dropout is enabled}
        \end{cases}
        $$

        3. Derivative of `loss` with respect to `attention_scores`:
        Softmax is applied independently to each query row, so we apply the
        softmax derivative row by row along the key dimension.
        $$
        \frac{\partial \text{loss}}{\partial \text{attention\_scores}}
        =
        \frac{\partial \text{loss}}{\partial P}
        \frac{\partial P}{\partial \text{attention\_scores}}
        =
        P \odot
        \left(
        \frac{\partial \text{loss}}{\partial P}
        -
        \sum_k
        \left(
        \frac{\partial \text{loss}}{\partial P_k}
        P_k
        \right)
        \right)
        $$

        4. Derivative of `loss` with respect to `Q` and `K`, using
        $$
        \text{attention\_scores} = \frac{QK^T}{\sqrt{key\_dim}}
        $$
        $$
        \frac{\partial \text{loss}}{\partial Q}
        =
        \frac{\partial \text{loss}}{\partial \text{attention\_scores}}
        \frac{\partial \text{attention\_scores}}{\partial Q}
        =
        \frac{\partial \text{loss}}{\partial \text{attention\_scores}}
        K \cdot \frac{1}{\sqrt{key\_dim}}
        $$
        $$
        \frac{\partial \text{loss}}{\partial K}
        =
        \frac{\partial \text{loss}}{\partial \text{attention\_scores}}
        \frac{\partial \text{attention\_scores}}{\partial K}
        =
        \left(
        \frac{\partial \text{loss}}{\partial \text{attention\_scores}}
        \right)^T
        Q \cdot \frac{1}{\sqrt{key\_dim}}
        $$
        """
        query, key, value = (tensor.data for tensor in self.tensors)
        probs, scale = self._causal_attention_probs(query, key)

        upstream_grad = grad.data

        dropout_scale_mask = getattr(self, "dropout_scale_mask", None)
        if dropout_scale_mask is not None:
            # Train mode applies dropout after softmax, so use the scaled mask
            # on the probability tensor before transposing to (..., S_k, S_q).
            grad_value = (probs * dropout_scale_mask).transpose(
                0, 1, 3, 2
            ) @ upstream_grad
        else:
            # Eval mode uses the plain causal probabilities. Transpose from
            # (..., S_q, S_k) to (..., S_k, S_q) before multiplying by dL/dO.
            grad_value = probs.transpose(0, 1, 3, 2) @ upstream_grad

        # By the chain rule, the derivative of loss with respect to the
        # pre-dropout attention weights is:
        # d(loss)/d(attention_weights) =
        # d(loss)/d(effective_attention_weights) * dropout_scale_mask
        # Transpose values from (..., S_k, D) to (..., D, S_k).
        grad_probs = upstream_grad @ value.transpose(0, 1, 3, 2)
        if dropout_scale_mask is not None:
            grad_probs = grad_probs * dropout_scale_mask

        # By the chain rule, the derivative of loss with respect to
        # attention_scores comes from the derivative of the softmax output with
        # respect to its input scores.
        # softmax acts on each query row independently, so this reduction is
        # computed row by row along the key dimension.
        grad_scores = probs * (
            grad_probs - xp.sum(grad_probs * probs, axis=-1, keepdims=True)
        )

        # attention_scores = QK^T / sqrt(key_dim)
        # By the chain rule, propagate d(loss)/d(attention_scores) into Q and K.
        grad_query = (grad_scores @ key) * scale
        # Transpose score gradients from (..., S_q, S_k) to (..., S_k, S_q).
        grad_key = (grad_scores.transpose(0, 1, 3, 2) @ query) * scale
        return grad_query, grad_key, grad_value


###################### Loss Functions #####################
class BinaryCrossEntropy(Function):
    r"""
    Binary Cross Entropy (BCE) Loss.

    This loss function assumes that $y_{pred}$ contains probabilities rather than logits.
    If the input is logits, use :func:`binary_cross_entropy_with_logits`.

    The loss is computed as:
    $$
    BCE = -\left( y_{true} \cdot \log(y_{pred}) + (1 - y_{true}) \cdot \log(1 - y_{pred}) \right)
    $$

    Examples:
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor([0.9, 0.2, 0.1])
        >>> y_true = Tensor([1, 0, 0])
        >>> loss = BinaryCrossEntropy.apply(y_pred, y_true) # Expected output: a small loss value
    """

    def forward(self, y_pred: Array, y_true: Array, **kwargs: Any) -> Array:
        """
        Computes the binary cross entropy loss.

        Args:
            y_pred (xp.ndarray): Predicted probabilities.
            y_true (xp.ndarray): True binary labels.
            **kwargs: Additional keyword arguments.

        Returns:
            float: The computed binary cross entropy loss.
        """
        y_true = xp.array(y_true, dtype=xp.float32)
        y_pred = xp.array(y_pred, dtype=xp.float32)

        # If labels come in as (batch_size,), explicitly reshaping them to (batch_size, 1) avoids shape mismatch, and certain elementwise operations will broadcast in unintended ways
        if y_true.ndim == 1 and y_pred.ndim == 1:
            pass
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError("y_pred and y_true must have the same shape")

        self.y_true = y_true
        self.y_pred_prob = xp.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -xp.mean(
            y_true * xp.log(self.y_pred_prob)
            + (1 - y_true) * xp.log(1 - self.y_pred_prob)
        )
        return loss

    def backward(self, grad: Tensor) -> Tuple[Array, None]:
        r"""
        Computes the gradient of the binary cross entropy loss with respect to $y_{pred}$.

        The gradient is given by:
        $$
        \frac{\partial L}{\partial y_{pred}} = -\left(\frac{y_{true}}{y_{pred}} - \frac{1-y_{true}}{1-y_{pred}}\right)
        $$

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            Tuple[xp.ndarray, None]: A tuple containing the gradient with respect to $y_{pred}$ and None for $y_{true}$.
        """
        # Avoid division by zero by clipping probabilities away from 0 and 1
        y_pred_prob = xp.clip(self.y_pred_prob, 1e-7, 1 - 1e-7)
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

    Examples:
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor([2.0, -1.0, -2.0])  # logits
        >>> y_true = Tensor([1, 0, 0])
        >>> loss = BinaryCrossEntropyWithLogits.apply(y_pred, y_true) # Expected output: a loss value computed using logits
    """

    def forward(self, y_pred: Array, y_true: Array) -> Array:
        """
        Computes the binary cross entropy loss with logits input.

        Args:
            y_pred (xp.ndarray): shape (N, ...) Unbounded real-valued logits.
            y_true (xp.ndarray): True binary labels (0 or 1), same shape as y_pred.

        Returns:
            float: The computed binary cross entropy loss.
        """
        y_true = xp.array(y_true, dtype=xp.float32)
        y_pred = xp.array(y_pred, dtype=xp.float32)

        # If labels come in as (batch_size,), explicitly reshaping them to (batch_size, 1) avoids shape mismatch, and certain elementwise operations will broadcast in unintended ways
        if y_true.ndim == 1 and y_pred.ndim == 1:
            pass
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have the same shape.")

        # compute loss
        # loss_i = max(y_pred, 0) - y_pred * y_true + log(1 + exp(-|y_pred|))
        loss = xp.mean(
            xp.maximum(y_pred, 0.0)
            - y_pred * y_true
            + xp.log1p(xp.exp(-xp.abs(y_pred)))
        )

        self.y_pred = y_pred
        self.y_true = y_true
        return loss

    def backward(self, grad: Tensor) -> Tuple[Array, None]:
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
            Tuple[xp.ndarray, None]: A tuple containing the gradient with respect to $y_{pred}$ and None for $y_{true}$.
        """
        # Stable sigmoid without NumPy-style masked mutation.
        sig = xp.where(
            self.y_pred >= 0,
            1.0 / (1.0 + xp.exp(-self.y_pred)),
            xp.exp(self.y_pred) / (1.0 + xp.exp(self.y_pred)),
        )

        # 2) Compute dL/dy_pred = sigmoid(y_pred) - y_true divided by batch_size
        grad_y_pred = (sig - self.y_true) / self.y_pred.shape[0]

        # 3) Multiply by upstream gradient
        grad_y_pred *= grad.data

        # we don't need a gradient for y_true (labels)
        return grad_y_pred, None


IGNORE_INDEX = -100


class CrossEntropy(Function):
    """
    Cross-Entropy Loss for multi-dimensional predictions with optional ignored targets and label smoothing.

    This function accepts raw logits (not probabilities) and computes a stable log-softmax internally.

    Examples:
        >>> import cupy as np
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor(xp.array([[2.0, 1.0, 0.1]]))
        >>> y_true = Tensor(xp.array([0]))
        >>> loss = CrossEntropy.apply(y_pred, y_true, ignore_index=-100, label_smoothing=0.1)
    """

    def forward(
        self,
        y_pred: Array,
        y_true: Array,
        ignore_index: int = IGNORE_INDEX,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> Union[Array, float]:
        r"""
        Computes the cross-entropy loss with optional ignored targets and label smoothing.

        Args:
            y_pred (xp.ndarray): Raw logits. Shape can be $(batch\_size, feature\_dim)$ or $(batch\_size, seq\_len, feature\_dim)$.
            y_true (Union[xp.ndarray, Tensor]): True class indices. If $y_{pred}$ is 2D, shape is $(batch\_size,)$; if 3D, shape is $(batch\_size, seq\_len)$.
            ignore_index (int, optional): Target value to ignore in the loss and gradient.
                Defaults to -100.
            label_smoothing (float, optional): Label smoothing factor. Defaults to 0.0. Label smoothing is applied if $label\_smoothing > 0$
            reduction (str, optional): Either ``"mean"`` or ``"sum"``. ``"mean"``
                divides by the total non-ignored target weight; ``"sum"`` returns
                the summed loss over non-ignored targets.
            For label smoothing, we follow the Inception paper notation.
            Here:
            - $x$ is the current training example
            - $y$ is the ground-truth class index for $x$
            - $k$ is a class index
            - $K$ is the total number of classes
            - $\delta_{k,y}$ is the Kronecker delta, equal to $1$ when $k=y$ and $0$ otherwise
            - $u(k)$ is a prior distribution over classes
            $$
            q'(k \mid x) = (1 - \epsilon)\,\delta_{k,y} + \epsilon\,u(k)
            $$
            and for the uniform prior used in the paper,
            $$
            u(k) = \frac{1}{K}
            $$
            so the smoothed target distribution becomes
            $$
            q'(k \mid x) = (1 - \epsilon)\,\delta_{k,y} + \frac{\epsilon}{K}
            $$
            In per-example / per-class notation, where $y_{i,j}$ is the one-hot
            target entry for example $i$ and class $j$, this is
            $$
            T_{i,j} = (1 - \epsilon)\,y_{i,j} + \frac{\epsilon}{K}
            $$
        (Ref: "Rethinking the Inception Architecture for Computer Vision", https://arxiv.org/abs/1512.00567)

        Returns:
            Union[xp.array, float]: The average cross-entropy loss over non-ignored positions.
        """

        y_true = xp.array(y_true, dtype=xp.int64)
        if y_pred.dtype in LOW_PRECISION_FLOAT_DTYPES:
            y_pred = y_pred.astype(xp.float32)

        # 1. If 3D logits, flatten them for simpler processing while preserving
        # the original shape for the backward pass.
        self.original_shape = y_pred.shape
        if y_pred.ndim == 3:
            batch_size, seq_len, num_classes = y_pred.shape
            y_pred = y_pred.reshape(batch_size * seq_len, num_classes)
            y_true = y_true.reshape(batch_size * seq_len)
        else:
            num_classes = y_pred.shape[-1]  # avoid returning batch_size to save memory

        # 2. Mark which target positions contribute to the loss.
        non_ignored_weights = (y_true != ignore_index).astype(xp.float32)
        safe_y_true = xp.where(non_ignored_weights > 0, y_true, 0)
        non_ignored_count = xp.maximum(
            xp.sum(non_ignored_weights),
            xp.array(1.0, dtype=xp.float32),
        )

        r"""
        3. Shift the logits so the softmax-related terms stay numerically stable.

        This is algebraically the same cross-entropy as the standard
        log-softmax formulation; we are only changing how we materialize the
        intermediate tensors.

        Starting from
        $$
        \log \operatorname{softmax}(y_{\text{pred}})_{i,j}
        = y_{\text{pred}, i,j} - \log \sum_k \exp(y_{\text{pred}, i,k})
        $$
        define
        $$
        \operatorname{shifted}_{i,j}
        = y_{\text{pred}, i,j} - \max_k y_{\text{pred}, i,k}
        $$
        Then
        $$
        \exp(\operatorname{shifted}_{i,j})
        = \exp(y_{\text{pred}, i,j} - \max_k y_{\text{pred}, i,k})
        = \frac{\exp(y_{\text{pred}, i,j})}{\exp(\max_k y_{\text{pred}, i,k})}
        $$
        and because softmax is invariant to subtracting the same constant from
        every entry in a row,
        $$
        \log \operatorname{softmax}(y_{\text{pred}})_{i,j}
        = \operatorname{shifted}_{i,j}
        - \log \sum_k \exp(\operatorname{shifted}_{i,k})
        $$
        $$
        \operatorname{softmax}(y_{\text{pred}})_{i,j}
        = \frac{\exp(\operatorname{shifted}_{i,j})}
        {\sum_k \exp(\operatorname{shifted}_{i,k})}
        $$

        The largest value in each shifted row is 0, so the exponentials are
        numerically safe to compute. We store the probabilities
        $p_{i,j} = \operatorname{softmax}(y_{\text{pred}})_{i,j}$ for backward,
        and reconstruct only the specific log-softmax terms needed by the loss,
        rather than materializing the full log-softmax matrix.
        """
        shifted = y_pred - xp.max(y_pred, axis=1, keepdims=True)
        exp_shifted = xp.exp(shifted)
        denom = xp.sum(exp_shifted, axis=1, keepdims=True)
        probs = exp_shifted / denom
        # TODO: If CE memory becomes a bottleneck, consider rematerializing
        # softmax in backward instead of storing this full probability matrix.
        # denom has shape (N, 1); squeeze the singleton dimension so it can be
        # subtracted from the per-row terms below.
        log_denom = xp.log(denom[:, 0])

        r"""
        4. Compute the label-smoothed cross-entropy for each element i.
        $$
        q'(k \mid x) = (1 - \epsilon)\,\delta_{k,y} + \frac{\epsilon}{K}
        $$
        where $\delta_{k,y}$ is 1 for the correct class and 0 otherwise.
        so each row loss is
        $$
        L_i = -\sum_j q'_{i,j}\log p_{i,j}
        $$
        $$
        = -\left((1 - \epsilon)\log p_{i,y_i} + \frac{\epsilon}{K}\sum_j \log p_{i,j}\right)
        $$
        Here $p_{i,j}$ is the model probability for example i and class j.
        """
        # Since log_softmax[i, j] = shifted[i, j] - log_denom[i], we can compute
        # the two terms needed by the smoothed loss directly without
        # materializing the full log_softmax matrix.
        log_p_correct = shifted[xp.arange(len(y_true)), safe_y_true] - log_denom
        mean_log_p = xp.mean(shifted, axis=1) - log_denom
        losses = -(
            (1.0 - label_smoothing) * log_p_correct + label_smoothing * mean_log_p
        )

        # 5) Reduce the loss over non-ignored positions.
        loss_sum = xp.sum(losses * non_ignored_weights)
        if reduction == "mean":
            loss_val = loss_sum / non_ignored_count
        elif reduction == "sum":
            loss_val = loss_sum
            non_ignored_count = xp.array(1.0, dtype=xp.float32)
        else:
            raise ValueError(f"Unsupported cross_entropy reduction: {reduction!r}")

        # 6) Store for backward pass:
        self.probs = probs
        self.y_true = safe_y_true
        self.non_ignored_weights = non_ignored_weights
        self.non_ignored_count = non_ignored_count
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        return loss_val

    def backward(self, grad: Tensor) -> Tuple[Array, None]:
        r"""
        Computes the backward pass for the cross-entropy loss with label smoothing.

        The gradient with respect to the logits is given by:
        $$
        \frac{\partial L}{\partial logits_{i,j}} = p_{i,j} - T_{i,j}
        $$
        where $p_{i,j}$ is the stored model probability for example $i$ and class
        $j$, and the target distribution $T_{i,j}$ is defined as:
        $$
        T_{i,j} = (1 - \epsilon)\,y_{i,j} + \frac{\epsilon}{K}
        $$
        The gradient is zeroed out for ignored positions and scaled by the upstream gradient divided by the number of non-ignored positions.

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            Tuple[xp.ndarray, None]: A tuple containing the gradient with respect to the logits and None for $y_{true}$.
        """
        # 1. Start from a dense copy of the probabilities.
        # We will transform this buffer into p_{i,j} - T_{i,j} in-place, which
        # avoids building a separate dense target tensor of the same shape.
        grad_out = self.probs * 1.0

        # 2. Expand the non-ignored mask so it can zero out whole rows in the
        # logits-shaped gradient matrix.
        row_mask = xp.expand_dims(self.non_ignored_weights, axis=1)

        r"""
        3. Subtract the label-smoothed target distribution.
        $$
        q'(k \mid x) = (1 - \epsilon)\,\delta_{k,y} + \frac{\epsilon}{K}
        $$
        where $\delta_{k,y}$ is the Kronecker delta.

        Since grad_out starts as $p_{i,j}$, subtracting $\frac{\epsilon}{K}$
        from every class and then subtracting $(1 - \epsilon)$ from the correct
        class gives exactly
        $$
        p_{i,j} - \left((1 - \epsilon)\,\delta_{k,y} + \frac{\epsilon}{K}\right)
        = p_{i,j} - T_{i,j}
        $$
        """
        if self.label_smoothing:
            grad_out -= self.label_smoothing / self.num_classes
        grad_out[
            xp.arange(self.y_true.shape[0]),
            self.y_true,
        ] -= 1.0 - self.label_smoothing

        # 4. Zero ignored rows.
        grad_out *= row_mask

        # 5. Apply the upstream gradient and divide by the number of
        # non-ignored positions to match the forward average.
        grad_out *= grad.data / self.non_ignored_count

        # 6. Reshape to original shape if the forward pass flattened 3D logits.
        grad_out = grad_out.reshape(self.original_shape)

        # We do not need grad for y_true
        return grad_out, None


class HingeLoss(Function):
    r"""
    Hinge Loss.

    The hinge loss is defined as:
        $$
        loss = \max(0, 1 - y_{true} \cdot y_{pred})
        $$

    For correctly classified points ($y_{true} \cdot y_{pred} \geq 1$), the loss is 0; otherwise, it is $1 - y_{true} \cdot y_{pred}$. This is because loss functions typically don't go into the negatives so we take the max of 0 and 1 - y_true * y_pred)

    The objective function typically includes a regularization term:
        $$
        \|w\|^2 + C \sum max(0, 1 - y_{true} \cdot y_{pred})
        $$

    where $C$ is a hyperparameter controlling the trade-off between maximizing the margin (through regularization) and minimizing the loss, and $w$ is the weight vector. ($\|w\|^2$ is the regularization term)

    Paper: https://ieeexplore.ieee.org/document/708428

    Examples:
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor([0.8, -0.5, 0.3])
        >>> y_true = Tensor([1, -1, 1])
        >>> loss = HingeLoss.apply(y_pred, y_true, reduction="mean") # Expected output: average hinge loss
    """

    def forward(
        self,
        y_pred: Array,
        y_true: Array,
        reduction: str = "none",
        **kwargs: Any,
    ) -> Union[Array, float]:
        """
        Computes the hinge loss.

        Args:
            y_pred (xp.ndarray): Predicted scores.
            y_true (Union[xp.ndarray, Tensor]): True labels.
            reduction (str, optional): "none", "mean", or "sum". Defaults to "none".
            **kwargs: Additional keyword arguments.

        Returns:
            Union[xp.array, float]: The computed hinge loss.
        """
        y_true = xp.array(y_true, dtype=xp.float32)

        # Reshape y_true to match y_pred if needed
        if y_pred.shape != y_true.shape:
            y_true = y_true.reshape(y_pred.shape)

        self.y_true = y_true
        self.y_pred = y_pred
        self.reduction = reduction

        # hinge loss = max(0, 1 - y_true * y_pred)
        self.margins = 1 - y_true * y_pred
        loss_data = xp.maximum(0, self.margins)

        if reduction == "mean":
            loss_data = xp.mean(loss_data)
        elif reduction == "sum":
            loss_data = xp.sum(loss_data)
        elif reduction == "none":
            pass
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
        return loss_data

    def backward(self, grad: Tensor) -> Tuple[Array, None]:
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
            Tuple[xp.ndarray, None]: A tuple containing the gradient with respect to $y_{pred}$ and None for $y_{true}$.
        """
        grad_y_pred = xp.zeros_like(self.y_pred)

        # Where margin > 0, gradient is -y_true
        margin_violated = self.margins > 0
        grad_y_pred = xp.where(margin_violated, -self.y_true, 0)

        if self.reduction == "mean":
            grad_y_pred /= self.y_pred.size

        # Handle scalar gradient (from mean/sum reduction)
        if grad.data.ndim == 0 or grad.data.size == 1:
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

    Examples:
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor([3.0, 5.0])
        >>> y_true = Tensor([2.0, 5.0])
        >>> loss = MeanSquaredLoss.apply(y_pred, y_true) # Expected output: 0.5
    """

    def forward(self, y_pred: Array, y_true: Array, **kwargs: Any) -> Array:
        """
        Computes the Mean Squared Error loss.

        Args:
            y_pred (xp.ndarray): Predicted values.
            y_true (xp.ndarray): True values.
            **kwargs: Additional keyword arguments.

        Returns:
            float: The computed MSE loss.
        """
        self.y_pred = y_pred
        self.y_true = y_true
        return xp.mean((y_pred - y_true) ** 2)

    def backward(self, grad: Tensor) -> Array:
        r"""
        Computes the gradient of the Mean Squared Error loss with respect to the predictions.

        The gradient is given by:
        $$
        \frac{\partial L}{\partial y_{pred}} = 2 (y_{pred} - y_{true})
        $$

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            xp.ndarray: The gradient with respect to y_pred.
        """
        return 2 * (self.y_pred - self.y_true) * grad.data


def binary_cross_entropy(
    y_pred: Tensor, y_true: Union[Tensor, ArrayLike], **kwargs: Any
) -> Tensor:
    """
    Computes the binary cross entropy loss given predicted probabilities.

    This function wraps the :class:`BinaryCrossEntropy` operation.

    Args:
        y_pred (Tensor): Predicted probabilities.
        y_true (Union[Tensor, ArrayLike]): True binary labels.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The computed binary cross entropy loss.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor([0.9, 0.2, 0.1])
        >>> y_true = [1, 0, 0]
        >>> loss = binary_cross_entropy(y_pred, y_true)
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return BinaryCrossEntropy.apply(y_pred, y_true, **kwargs)


def binary_cross_entropy_with_logits(
    y_pred: Tensor, y_true: Union[Tensor, ArrayLike], **kwargs: Any
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
        y_true (Union[Tensor, ArrayLike]): True binary labels.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The computed binary cross entropy loss.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor([2.0, -1.0, -2.0])
        >>> y_true = [1, 0, 0]
        >>> loss = binary_cross_entropy_with_logits(y_pred, y_true)
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return BinaryCrossEntropyWithLogits.apply(y_pred, y_true, **kwargs)


def cross_entropy(
    y_pred: Tensor,
    y_true: Union[Tensor, ArrayLike],
    ignore_index: int = IGNORE_INDEX,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Computes the cross-entropy loss for multi-class classification with logits.

    This function expects $y_{pred}$ to be raw logits and $y_{true}$ to be class indices (not one-hot vectors).

    Args:
        y_pred (Tensor): Raw logits.
        y_true (Union[Tensor, ArrayLike]): True class indices.
        ignore_index (int, optional): Target value to ignore in the loss and gradient.
            Defaults to -100.
        label_smoothing (float, optional): Label smoothing factor. Defaults to 0.0.
        reduction (str, optional): Either ``"mean"`` or ``"sum"``. Defaults to
            ``"mean"``.
    Returns:
        Tensor: The computed cross-entropy loss.

    Examples:
        >>> import cupy as np
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor(xp.array([[2.0, 1.0, 0.1]]))
        >>> y_true = Tensor(xp.array([0]))
        >>> loss = cross_entropy(y_pred, y_true, ignore_index=-100, label_smoothing=0.1)
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return CrossEntropy.apply(
        y_pred,
        y_true,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


def hinge_loss(
    y_pred: Tensor,
    y_true: Union[Tensor, ArrayLike],
    reduction: str = "none",
    **kwargs: Any,
) -> Tensor:
    """
    Computes the hinge loss for binary classification.

    Args:
        y_pred (Tensor): Predicted scores.
        y_true (Union[Tensor, ArrayLike]): True labels.
        reduction (str, optional): Specifies the reduction to apply: "none", "mean", or "sum". Defaults to "none".
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The computed hinge loss.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor([0.8, -0.5, 0.3])
        >>> y_true = [1, -1, 1]
        >>> loss = hinge_loss(y_pred, y_true, reduction="mean")
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return HingeLoss.apply(y_pred, y_true, reduction=reduction, **kwargs)


def mean_squared_loss(
    y_pred: Tensor, y_true: Union[Tensor, ArrayLike], **kwargs: Any
) -> Tensor:
    """
    Computes the Mean Squared Error (MSE) loss.

    Args:
        y_pred (Tensor): Predicted values.
        y_true (Union[Tensor, ArrayLike]): True values.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The computed MSE loss.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor([3.0, 5.0])
        >>> y_true = [2.0, 5.0]
        >>> loss = mean_squared_loss(y_pred, y_true) # Expected output: 0.5
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return MeanSquaredLoss.apply(y_pred, y_true, **kwargs)
