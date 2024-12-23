import numpy as np
from .tensor import Tensor
import logging

logger = logging.getLogger(__name__)


class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._is_training = None

    def zero_grad(self):
        for p in self._parameters:
            p.grad = 0

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        """
        Sometimes people like to call model = Module() then call model(x)
        as a forward pass. So this is an alias.
        """
        return self.forward(x)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor):
            self._parameters[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        return self._modules[name]

    @property
    def parameters(self):
        params = self._parameters.copy()

        for k, module in self._modules.items():
            params.update({k: module.parameters})

        return params

    def num_parameters(self):
        """
        Returns the total number of parameters in the module and its submodules.
        """
        # Count parameters in current module
        total = sum(p.data.size for p in self._parameters.values())

        # Recursively count parameters in submodules
        for module in self._modules.values():
            total += module.num_parameters()

        return total

    def train(self):
        for module in self._modules.values():
            module.train()
        self._is_training = True

    def eval(self):
        for module in self._modules.values():
            module.eval()
        self._is_training = False


class Linear(Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__(**kwargs)

        # weight is a matrix of shape (input_size, output_size)
        # Xavier Normal Initialization
        # https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        self._parameters["weight"] = Tensor(
            np.random.uniform(
                low=-np.sqrt(6.0 / (input_size + output_size)),
                high=np.sqrt(6.0 / (input_size + output_size)),
                size=(input_size, output_size),
            ).astype(np.float32)
        )

        # bias is always 1-dimensional
        self._parameters["bias"] = Tensor(np.random.rand(output_size))

    def forward(self, x) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x)

        logger.debug(f"{x.data.shape=}")
        logger.debug(f"Linear forward {self._parameters['weight'].data.shape=}")

        # this is just a linear transformation (dot matrix multiplication)
        return x @ self._parameters["weight"] + self._parameters["bias"]


class Conv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding_mode="valid",
        bias=True,
        **kwargs,
    ):
        """
        Applies a 2D convolution over an input tensor.
        The shape convention is the same as PyTorch:
            input_shape = (N, in_channels, H, W)
            output_shape = (N, out_channels, H', W')

        where:
        - N is the batch size
        - in_channels is the number of input channels
        - out_channels is the number of kernels, where each kernel is convolved with the input tensor in all input channels

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            bias (bool, optional): Whether to add a bias term. Defaults to True.
            padding_mode (str, optional): The amount of padding_mode to add to the input. Defaults to 'valid'.
            - "valid" means no padding.
            - "same" means padding such that the output shape is the same as the input shape.
        """
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.bias = bias

        # The layer contains N number of kernels, which is equivalent to the out_channels
        # Each kernel is of shape (in_channels, H, W)
        # Each kernel needs to be convolved with the input tensor
        # The resulting tensor will have the shape (out_channels, H', W')
        # Xavier Normal Initialization
        # https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        self._parameters["weight"] = Tensor(
            np.random.uniform(
                low=-np.sqrt(
                    6.0
                    / (
                        self.in_channels * self.kernel_size * self.kernel_size
                        + self.out_channels
                    )
                ).astype(np.float32),
                high=np.sqrt(
                    6.0
                    / (
                        self.in_channels * self.kernel_size * self.kernel_size
                        + self.out_channels
                    )
                ).astype(np.float32),
                size=(
                    self.out_channels,
                    self.in_channels,
                    self.kernel_size,
                    self.kernel_size,
                ),
            )
        )
        if bias:
            self._parameters["bias"] = Tensor(
                np.random.rand(self.out_channels)
            )  # one bias per kernel

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        batch_size, in_channels, H, W = x.data.shape
        windows, (H_out, W_out) = extract_windows(
            x, self.kernel_size, self.stride, self.padding_mode
        )
        # windows: (H_out, W_out, N, C, H, W)

        # Permute windows to (N, C, H, W, H_out, W_out)
        windows = windows.permute(2, 3, 4, 5, 0, 1)
        # Now shape is (N, C, H, W, H_out, W_out)

        # Reshape to (N, C * H * W, H_out*W_out)
        windows = windows.reshape(
            batch_size, in_channels * self.kernel_size * self.kernel_size, H_out * W_out
        )

        # Transpose to (N, H_out*W_out, C*H*W) if needed, or directly reshape:
        # Actually, PyTorch uses (N, C*H*W, H_out*W_out) and then it flattens differently.
        # We can directly flatten to (N*H_out*W_out, C*H*W) for multiplication:
        windows = windows.transpose(1, 2).reshape(
            batch_size * H_out * W_out,
            in_channels * self.kernel_size * self.kernel_size,
        )

        # Prepare kernels: (out_channels, in_channels*H*W) then transpose to (C*H*W, out_channels)
        kernel_flat = (
            self._parameters["weight"]
            .reshape(
                self.out_channels, in_channels * self.kernel_size * self.kernel_size
            )
            .transpose(1, 0)
        )

        # Matrix multiply: (N*H_out*W_out, out_channels)
        output = windows @ kernel_flat

        # Reshape and permute to (N, out_channels, H_out, W_out)
        output = output.reshape(batch_size, H_out, W_out, self.out_channels).permute(
            0, 3, 1, 2
        )

        # Add bias if present
        if self.bias:
            output += self._parameters["bias"].reshape(-1, 1, 1)

        return output


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding_mode="valid", **kwargs):
        """
        2D Max Pooling Layer

        Args:
            kernel_size (int): Size of pooling window (2 means 2x2 window)
            stride (int, optional): How many pixels to move the window each time. Defaults to kernel_size.
            padding_mode (str, optional): The type of padding_mode to add to the input. Defaults to 'valid'.
            - "valid" means no padding.
            - "same" means padding such that the output shape is the same as the input shape.
        """
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = (
            stride if stride is not None else kernel_size
        )  # Default to kernel_size
        self.padding_mode = padding_mode

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        windows, (H_out, W_out) = extract_windows(
            x, self.kernel_size, self.stride, self.padding_mode
        )

        H_out, W_out, batch_size, in_channels, H_kernel, W_kernel = windows.shape

        # Reshape windows to match spatial layout
        windows = windows.permute(2, 3, 0, 1, 4, 5)
        # Reorder axes to (batch_size, in_channels, H_out, W_out, kH * kW)
        windows = windows.reshape(batch_size, in_channels, H_out, W_out, -1)

        pooled = windows.max(axis=-1, keepdims=True).reshape(
            batch_size, in_channels, H_out, W_out
        )
        return pooled


class BatchNorm(Module):
    """
    Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

    This layer normalizes the input tensor by subtracting the batch mean and dividing by the batch standard deviation.

    Paper: http://arxiv.org/abs/1502.03167
    """

    def __init__(self, input_size, momentum=0.1, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)

        self.momentum = momentum  # used in running mean and variance calculation
        self.epsilon = epsilon  # small constant for numeric stability

        # Running stats (used for inference)
        self.running_mean = np.zeros(input_size)
        self.running_var = np.ones(input_size)

        # gamma and beta are learnable parameters
        # gamma is responsible for scaling the normalized input
        # beta is responsible for shifting the normalized input
        # self._parameters["weight"] = Tensor(np.ones((1, input_size)))
        # self._parameters["bias"] = Tensor(np.zeros((1, input_size)))
        self._parameters["weight"] = Tensor(np.ones(input_size, dtype=np.float32))
        self._parameters["bias"] = Tensor(np.zeros(input_size, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        """
        Note that the backward pass is implemented via primitive operations in the Tensor class.
        The operations in the forward pass have all been implemented as Tensor-level operations.
        """
        if self._is_training:
            # Compute batch statistics using Tensor operations
            batch_mean = x.mean(axis=0)
            diff = x - batch_mean
            var = (diff**2).sum(axis=0)

            biased_batch_var = var / x.data.shape[0]
            # Unbiased variance (divide by N-1) is based on Bessel's correction
            unbiased_batch_var = var / (x.data.shape[0] - 1)
            std_dev = (biased_batch_var + self.epsilon) ** 0.5

            # Update running statistics
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * unbiased_batch_var.data

            normalized = diff / std_dev
        else:
            normalized = (x - self.running_mean) / np.sqrt(
                self.running_var + self.epsilon
            )

        # Scale and shift
        return normalized * self._parameters["weight"].expand(
            x.data.shape
        ) + self._parameters["bias"].expand(x.data.shape)


class Dropout(Module):
    def __init__(self, p=0.5, **kwargs):
        """
        The Dropout layer randomly sets a fraction of input units to 0 at each update during training time.

        "It prevents overfitting and provides a way of approximately combining exponentially many different
        neural network architectures efficiently."
        Paper: https://arxiv.org/abs/1207.0580

        Args:
            p (float, optional): Fraction of the input units to drop. Defaults to 0.5.
        """
        super().__init__(**kwargs)
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self._is_training:
            mask = np.random.binomial(1, 1 - self.p, size=x.data.shape)
            return (
                x
                * mask
                / (
                    1 - self.p
                )  # we scale the output by 1/(1-p) to keep the expected output the same
                if self.p < 1
                else x * 0  # when p=1, drop everything by multiplying by 0
            )
        return x


########### Utility Functions ###########
def extract_windows(x, kernel_size, stride, padding_mode="valid"):
    """
    Extract windows from input tensor while maintaining computational graph.

    Args:
        x (Tensor): Input tensor of shape (batch_size, channels, height, width)
        kernel_size (int): Size of the sliding window
        stride (int): Step size between windows
        padding_mode (str): Type of padding - "valid" or "same"

    Returns:
        tuple: (windows, output_shape)
            - windows: Stacked tensor of shape (H_out, W_out, batch_size, channels, kernel_size, kernel_size)
            - output_shape: (H_out, W_out) tuple for reshaping
    """
    if not isinstance(x, Tensor):
        x = Tensor(x)

    batch_size, in_channels, H, W = x.data.shape

    if padding_mode == "same":
        pad_h = pad_w = kernel_size // 2
        x_padded = x.pad(
            pad_width=((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=0,
        )
    else:
        x_padded = x

    # Get windows using strided_windows
    windows = x_padded.strided_windows(kernel_size, stride)

    # Calculate output dimensions
    H_out = (x_padded.shape[2] - kernel_size) // stride + 1
    W_out = (x_padded.shape[3] - kernel_size) // stride + 1

    return windows, (H_out, W_out)
