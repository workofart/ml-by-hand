"""
Initialization methods for weights of the neural network
"""

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np

from autograd.tensor import Tensor


def xavier_uniform(tensor: Tensor):
    """
    In-place Xavier Uniform Initialization
    weight is a matrix of shape (input_size, output_size, additional_dimensions...)
    https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
    fan_in, fan_out = compute_fan_in_out(tensor.data)
    limit = np.sqrt(6.0 / (fan_in + fan_out))

    tensor.data[...] = np.random.uniform(
        low=-limit, high=limit, size=tensor.data.shape
    ).astype(tensor.data.dtype)
    return tensor


def compute_fan_in_out(tensor: Tensor):
    """
    Compute the fan in and fan out of the tensor

    For convolution kernels:
    tensor.size(0) = out_channels
    tensor.size(1) = in_channels
    The rest of the dimensions are the spatial kernel sizes (e.g., kernel_height, kernel_width).
    """
    tensor_shape = tensor.shape
    if len(tensor_shape) < 2:
        raise ValueError("Tensor must have at least 2 dimensions")

    receptive_field_size = 1
    if len(tensor_shape) > 2:
        for s in tensor_shape[2:]:
            receptive_field_size *= s

    fan_in = tensor_shape[0] * receptive_field_size
    fan_out = tensor_shape[-1] * receptive_field_size
    return fan_in, fan_out
